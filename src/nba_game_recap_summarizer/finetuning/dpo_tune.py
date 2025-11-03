import os
import json
import time
import pandas as pd
from datasets import Dataset
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig

from nba_game_recap_summarizer.finetuning.utils.tokenization_utils import (
    preprocess_text,
    add_custom_tokens_to_tokenizer,
)


def build_prompt(recap: str) -> str:
    return (
        "You are an NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n"
        "### NBA Game Recap ###\n" + recap + "\n\n### Recap Summary ###\n"
    )


def _load_pairs(csv_path: str, max_prompt_len: int) -> Dataset:
    df = pd.read_csv(csv_path)
    required = {"game_recap", "chosen", "rejected"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DPO CSV: {missing}")

    df = df.dropna(subset=["game_recap", "chosen", "rejected"]).copy()
    df["game_recap"] = df["game_recap"].astype(str).map(preprocess_text)
    df["chosen"] = df["chosen"].astype(str).map(preprocess_text)
    df["rejected"] = df["rejected"].astype(str).map(preprocess_text)
    df["prompt"] = df["game_recap"].apply(build_prompt)
    df["prompt"] = df["prompt"].str.slice(0, max(128, max_prompt_len * 4))
    pairs = df[["prompt", "chosen", "rejected", "game_recap"]].copy()
    return Dataset.from_pandas(pairs.reset_index(drop=True))


def dpo_tune(cfg) -> str:
    """Run DPO tuning with QLoRA and export FP16 merged aligned model."""
    logger.info("Starting DPO tuning")

    pairs_csv = cfg.data.preference_pairs_csv
    output_root = cfg.outputs.output_dir
    os.makedirs(output_root, exist_ok=True)

    base_dir = os.environ.get("BASE_MODEL_DIR")
    prefer_local = base_dir and os.path.exists(base_dir) and os.path.exists(os.path.join(base_dir, "config.json"))
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_dir if prefer_local else cfg.model.name)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from base model dir: {e}. Falling back to {cfg.model.name}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer = add_custom_tokens_to_tokenizer(tokenizer)

    # Build model loading kwargs - only include quantization_config if quantization is enabled
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
    if cfg.model.quantization and cfg.model.quantization_type == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_dir if prefer_local else cfg.model.name,
        **model_kwargs,
    )

    if base_model.get_input_embeddings().num_embeddings != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))

    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora)

    dataset = _load_pairs(pairs_csv, cfg.lengths.max_prompt_length)
    split = dataset.train_test_split(test_size=cfg.data.test_size, seed=cfg.data.split_seed)
    dpo_train = split["train"]
    dpo_eval = split["test"]

    # Initialize W&B (optional)
    run = None
    try:
        import wandb
        run = wandb.init(project=f"{cfg.project_name}-{os.getenv('ENV','dev')}", name="dpo_tune", tags=["dpo"], config={
            "model": cfg.model.name,
            "quantization": f"{cfg.model.quantization}:{cfg.model.quantization_type}",
            "lora": dict(r=cfg.lora.r, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout),
            "beta": cfg.dpo.beta,
            "epochs": cfg.training.max_epochs,
            "batch_size": cfg.training.batch_size,
            "grad_accum": cfg.training.accumulate_grad_batches,
        })
    except Exception as e:
        logger.warning(f"W&B init skipped: {e}")

    dpo_args = DPOConfig(
        output_dir=output_root,
        beta=cfg.dpo.beta,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.accumulate_grad_batches,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=cfg.training.max_epochs,
        logging_steps=cfg.training.log_every_n_steps,
        save_steps=10_000,
        bf16=False,
        fp16=False,
        eval_steps=0,
        optim="adamw_torch",
        max_grad_norm=cfg.training.gradient_clip_val,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        max_length=cfg.lengths.max_prompt_length + cfg.lengths.max_completion_length,
        report_to=["wandb"] if run else [],  # Enable console logging (TRL logs to stdout)
        gradient_checkpointing=False,
    )

    logger.info("Initializing DPOTrainer")
    logger.info(f"Training configuration:")
    logger.info(f"  Batch size: {cfg.training.batch_size}, Grad accum: {cfg.training.accumulate_grad_batches} (effective: {cfg.training.batch_size * cfg.training.accumulate_grad_batches})")
    logger.info(f"  Learning rate: {cfg.training.learning_rate}, Epochs: {cfg.training.max_epochs}")
    logger.info(f"  Training pairs: {len(dpo_train)}, Eval pairs: {len(dpo_eval)}")
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dpo_train,
        eval_dataset=dpo_eval,
        processing_class=tokenizer,
    )

    # Log loss evolution during training (like Colab)
    class LossCallback(TrainerCallback):
        def __init__(self):
            self.losses = []
            self.steps = []
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                step = state.global_step
                loss = logs["loss"]
                self.losses.append(loss)
                self.steps.append(step)
                # Print like Colab table format
                logger.info(f"Step {step:3d} | Training Loss: {loss:.6f}")
                # Also log other DPO metrics if available
                for key in ["rewards/chosen", "rewards/rejected", "rewards/accuracies"]:
                    if key in logs:
                        logger.debug(f"  {key}: {logs[key]:.4f}")
            return control
            
    loss_callback = LossCallback()
    trainer.add_callback(loss_callback)

    start = time.time()
    trainer.train()
    train_time = time.time() - start
    
    # Get all loss values from trainer state (more comprehensive than callback)
    all_losses = []
    all_steps = []
    if hasattr(trainer.state, 'log_history'):
        for entry in trainer.state.log_history:
            if 'loss' in entry:
                all_losses.append(entry['loss'])
                all_steps.append(entry.get('step', len(all_steps)))
    
    # Use callback losses if available, otherwise fall back to trainer state
    losses_to_use = loss_callback.losses if loss_callback.losses else all_losses
    steps_to_use = loss_callback.steps if loss_callback.steps else all_steps
    
    # Print summary like Colab
    if losses_to_use:
        logger.info("\n" + "="*60)
        logger.info("DPO Training Loss Summary")
        logger.info("="*60)
        logger.info(f"Initial Loss: {losses_to_use[0]:.6f}")
        logger.info(f"Final Loss: {losses_to_use[-1]:.6f}")
        if losses_to_use[0] > 0:
            reduction = ((losses_to_use[0] - losses_to_use[-1]) / losses_to_use[0] * 100)
            logger.info(f"Loss Reduction: {reduction:.1f}%")
        logger.info(f"Average Loss: {sum(losses_to_use) / len(losses_to_use):.6f}")
        logger.info(f"Total Steps: {len(losses_to_use)}")
        logger.info("="*60)

    logger.info("Merging and saving aligned model (FP16)")
    if isinstance(model, PeftModel):
        merged = model.merge_and_unload()
        if hasattr(merged, "generation_config"):
            merged.generation_config.pad_token_id = tokenizer.pad_token_id
        aligned_fp16_dir = os.path.join(output_root, "hf_model_merged_aligned")
        os.makedirs(aligned_fp16_dir, exist_ok=True)
        tokenizer.save_pretrained(aligned_fp16_dir)
        merged.save_pretrained(aligned_fp16_dir)
    else:
        aligned_fp16_dir = os.path.join(output_root, "hf_model_merged_aligned")
        os.makedirs(aligned_fp16_dir, exist_ok=True)
        tokenizer.save_pretrained(aligned_fp16_dir)
        model.save_pretrained(aligned_fp16_dir)
    
    # Optionally quantize the merged model for efficient inference
    # This is useful if you trained in FP16 but want 4-bit quantized model for deployment
    if cfg.outputs.get("save_hf_model_merged_quantized", False):
        logger.info("Quantizing merged model for efficient inference...")
        try:
            quantized_dir = os.path.join(output_root, "hf_model_merged_aligned_quantized")
            os.makedirs(quantized_dir, exist_ok=True)
            
            # Load the FP16 merged model
            fp16_model = AutoModelForCausalLM.from_pretrained(
                aligned_fp16_dir,
                torch_dtype=torch.float16,
                device_map="cpu",  # Load on CPU first
            )
            
            # Apply quantization
            quant_type = cfg.model.quantization_type if cfg.model.quantization_type else "4bit"
            if quant_type == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    aligned_fp16_dir,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            else:
                logger.warning(f"Quantization type {quant_type} not yet supported for post-training quantization")
                quantized_model = None
            
            if quantized_model:
                # For quantized models, we save the config but the model needs to be loaded with quantization
                tokenizer.save_pretrained(quantized_dir)
                # Save quantization config
                with open(os.path.join(quantized_dir, "quantization_config.json"), "w") as f:
                    json.dump({
                        "quantization_type": quant_type,
                        "load_in_4bit": quant_type == "4bit",
                        "bnb_4bit_compute_dtype": "float16",
                    }, f, indent=2)
                logger.success(f"Quantized model configuration saved to: {quantized_dir}")
                logger.info("Note: Load quantized model using AutoModelForCausalLM.from_pretrained() with BitsAndBytesConfig")
            
            # Clean up
            del fp16_model
            if quantized_model:
                del quantized_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            logger.warning(f"Could not quantize model: {e}")

    with open(os.path.join(output_root, "train_config.json"), "w") as f:
        json.dump(cfg.to_container(resolve=True) if hasattr(cfg, 'to_container') else {}, f, indent=2)

    # Export training loss history and plot
    try:
        log_hist = getattr(trainer.state, "log_history", [])
        loss_entries = [e for e in log_hist if "loss" in e]
        with open(os.path.join(output_root, "training_loss.json"), "w") as f:
            json.dump(loss_entries, f, indent=2)
        # Plot
        import matplotlib.pyplot as plt
        steps = [e.get("step", i) for i, e in enumerate(loss_entries)]
        losses = [e["loss"] for e in loss_entries]
        plt.figure(figsize=(10,4))
        plt.plot(steps, losses, marker='o', markersize=2)
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title('DPO Training Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_root, "training_loss.png"), dpi=150, bbox_inches='tight')
        if run:
            try:
                import wandb
                wandb.log({"train/loss_curve": wandb.Image(os.path.join(output_root, "training_loss.png"))})
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Could not save/plot training loss: {e}")

    # Log summary to W&B
    if run:
        try:
            import wandb
            run.log({
                "train/time_sec": train_time,
                "data/train_size": len(dpo_train),
                "data/eval_size": len(dpo_eval),
            })
            run.finish()
        except Exception:
            pass

    logger.success(f"DPO tuning complete. Artifacts at: {output_root}")
    return aligned_fp16_dir


