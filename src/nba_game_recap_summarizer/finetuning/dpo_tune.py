import os
import json
import time
import pandas as pd
from datasets import Dataset
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

    bnb_cfg = None
    if cfg.model.quantization and cfg.model.quantization_type == "4bit":
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_dir if prefer_local else cfg.model.name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_cfg,
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
        logging_steps=10,
        save_steps=10_000,
        bf16=False,
        fp16=False,
        eval_steps=0,
        optim="adamw_torch",
        max_grad_norm=cfg.training.gradient_clip_val,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        max_length=cfg.lengths.max_prompt_length + cfg.lengths.max_completion_length,
        report_to="none",
        gradient_checkpointing=False,
    )

    # Compatibility shim: accommodate transformers Trainer calling get_batch_samples with an extra device arg
    class PatchedDPOTrainer(DPOTrainer):
        def get_batch_samples(self, epoch_iterator, num_batches, device=None):  # type: ignore[override]
            # TRL's DPOTrainer expects (epoch_iterator, num_batches); ignore device if passed
            return DPOTrainer.get_batch_samples(self, epoch_iterator, num_batches)

    logger.info("Initializing DPOTrainer")
    trainer = PatchedDPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dpo_train,
        eval_dataset=dpo_eval,
        tokenizer=tokenizer,
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

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


