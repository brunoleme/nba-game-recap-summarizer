import os

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
import wandb
from torch.utils.data import DataLoader

from nba_game_recap_summarizer.finetuning.data.nba_recap_dataset import NBARecapDataModule
from nba_game_recap_summarizer.finetuning.utils.logger import setup_logger
from nba_game_recap_summarizer.finetuning.utils.load_models import load_class_from_path, MODEL_CLASSES
from nba_game_recap_summarizer.finetuning.models.trainer import SummarizationModelTrainer

def train(cfg: DictConfig):
    """Pure PyTorch training function - replaces PyTorch Lightning implementation."""
    setup_logger(cfg.logging.log_path)
    logger.info("Starting pure PyTorch training pipeline")

    env_folder = os.getenv("ENV", "no-env")
    pipeline_run_id = os.getenv("PIPELINE_RUN_ID", "no-pipeline-id")

    # Setup data
    datamodule = NBARecapDataModule(
        model_name=cfg.model.name,
        source_data_path=cfg.data.source_data_path,
        preprocessed_input_data_folder=cfg.data.preprocessed_input_data_folder,
        env_folder=env_folder,
        batch_size=cfg.training.batch_size,
        max_length=cfg.model.max_length,
        num_workers=cfg.training.num_workers,
        shuffle=cfg.data.shuffle,
        shuffle_seed=cfg.data.shuffle_seed,
    )
    datamodule.setup()
    dataloaders = datamodule.get_dataloaders()

    # Setup model
    model_class_path = MODEL_CLASSES[cfg.model.type]
    ModelClass = load_class_from_path(model_class_path)

    model = ModelClass(
        model_name=cfg.model.name,
        model_type=cfg.model.type,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        weight_decay=cfg.training.weight_decay,
        use_quantization=cfg.model.quantization,
        quantization_type=cfg.model.quantization_type,
        peft_method=cfg.model.peft_method,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params:,}")

    # Initialize wandb (skip in test environments)
    if not os.getenv("SKIP_WANDB", "false").lower() == "true":
        wandb.init(
            project=f"{cfg.project_name}-training-{env_folder}", 
            name=f"{cfg.model.name}-{cfg.model.peft_method}", 
            tags=[f"pipeline:{pipeline_run_id}"]
        )
        wandb.config.update(dict(cfg))

    # Create trainer and train
    trainer = SummarizationModelTrainer(model, dataloaders, cfg)
    trainer.train()

    # Save final model in Hugging Face format - BOTH merged and unmerged for KTO compatibility
    logger.info("Saving model in Hugging Face format")
    hf_root = os.path.join(cfg.training.model_artifact_dir, f"{pipeline_run_id}")
    base_dir = os.path.join(hf_root, "hf_model_base")
    adapters_dir = os.path.join(hf_root, "hf_model_adapters")
    merged_dir = os.path.join(hf_root, "hf_model_merged")

    # (A) Always save tokenizer once
    model.tokenizer.save_pretrained(base_dir)
    logger.info("Tokenizer saved to base directory")

    # (B) Save UNMERGED for future fine-tuning (KTO training)
    from peft import PeftModel
    if isinstance(model.model, PeftModel):
        # CRITICAL: Save the original unquantized model for KTO training
        if hasattr(model, 'original_model') and model.original_model is not None:
            logger.info("Saving original unquantized model for KTO training...")
            # Fix generation_config before saving
            if hasattr(model.original_model, 'generation_config'):
                model.original_model.generation_config.pad_token_id = model.tokenizer.pad_token_id
            model.original_model.save_pretrained(base_dir)
            logger.info("✅ Original unquantized base model saved for KTO training")
        else:
            # Fallback: save the quantized model (not ideal for KTO)
            logger.warning("No original model found - saving quantized model (may cause KTO loading issues)")
            base_model = model.model.get_base_model()
            # Fix generation_config before saving
            if hasattr(base_model, 'generation_config'):
                base_model.generation_config.pad_token_id = model.tokenizer.pad_token_id
            base_model.save_pretrained(base_dir)
            logger.warning("NOTE: Base model is quantized - may cause loading issues for KTO training")
        
        # Save adapters only
        # CRITICAL: Set inference_mode=False before saving for KTO compatibility
        for adapter_name, peft_config in model.model.peft_config.items():
            peft_config.inference_mode = False
        model.model.save_pretrained(adapters_dir)
        logger.info("LoRA adapters saved for KTO training (inference_mode=False)")
    else:
        # No PEFT: just save base
        if hasattr(model, 'original_model') and model.original_model is not None:
            # Fix generation_config before saving
            if hasattr(model.original_model, 'generation_config'):
                model.original_model.generation_config.pad_token_id = model.tokenizer.pad_token_id
            model.original_model.save_pretrained(base_dir)
            logger.info("Original unquantized base model saved (no PEFT)")
        else:
            # Fix generation_config before saving
            if hasattr(model.model, 'generation_config'):
                model.model.generation_config.pad_token_id = model.tokenizer.pad_token_id
            model.model.save_pretrained(base_dir)
            logger.info("Base model saved (no PEFT)")

    # (C) Also save MERGED copy for inference
    try:
        if isinstance(model.model, PeftModel):
            merged = model.model.merge_and_unload()
            # Fix generation_config before saving
            if hasattr(merged, 'generation_config'):
                merged.generation_config.pad_token_id = model.tokenizer.pad_token_id
            merged.save_pretrained(merged_dir)
            logger.info("Merged model saved for inference")
        else:
            # Fix generation_config before saving
            if hasattr(model.model, 'generation_config'):
                model.model.generation_config.pad_token_id = model.tokenizer.pad_token_id
            model.model.save_pretrained(merged_dir)
            logger.info("Model saved for inference")
    except Exception as e:
        logger.warning(f"Could not save merged model: {e}")

    logger.success("Model saved successfully - both training and inference formats available")

    # Log final metrics (skip in test environments)
    if not os.getenv("SKIP_WANDB", "false").lower() == "true":
        wandb.log({
            "final_val_loss": trainer.best_val_loss,
            "final_epoch": trainer.current_epoch,
            "sagemaker_pipeline_run_id": pipeline_run_id
        })
        wandb.finish()
