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

    # Initialize wandb
    wandb.init(
        project=f"{cfg.project_name}-training-{env_folder}", 
        name=f"{cfg.model.name}-{cfg.model.peft_method}", 
        tags=[f"pipeline:{pipeline_run_id}"]
    )
    wandb.config.update(dict(cfg))

    # Create trainer and train
    trainer = SummarizationModelTrainer(model, dataloaders, cfg)
    trainer.train()

    # Save final model in Hugging Face format
    logger.info("Saving model in Hugging Face format")
    hf_save_path = os.path.join(cfg.training.model_artifact_dir, f"{pipeline_run_id}/hf_model")

    if getattr(model, "peft_method", None) == "lora":
        try:
            merged = model.model.merge_and_unload()  # merge LoRA into base weights
            merged.save_pretrained(hf_save_path)
            logger.info("Merged LoRA into base and saved.")
        except Exception as e:
            logger.warning(f"merge_and_unload failed ({e}). Saving base + adapters separately.")
            model.model.save_pretrained(hf_save_path)      # base
    else:
        model.model.save_pretrained(hf_save_path)

    model.tokenizer.save_pretrained(hf_save_path)
    logger.success("Model saved successfully")

    # Log final metrics
    wandb.log({
        "final_val_loss": trainer.best_val_loss,
        "final_epoch": trainer.current_epoch,
        "sagemaker_pipeline_run_id": pipeline_run_id
    })
    wandb.finish()
