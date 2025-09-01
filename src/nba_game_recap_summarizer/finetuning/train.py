import os

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
import wandb

from nba_game_recap_summarizer.finetuning.data.nba_recap_dataset import NBARecapDataModule
from nba_game_recap_summarizer.finetuning.utils.logger import setup_logger

MODEL_CLASSES = {
    "llama": "nba_game_recap_summarizer.finetuning.models.llama_model.LlamaRecapSummarizationModel",
}

class GPUMonitoringCallback(Callback):
    """Callback to monitor GPU usage during training"""
    
    def on_train_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            logger.info(f"🚀 Training started on GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory at training start: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    def on_train_epoch_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            logger.info(f"📊 Epoch {trainer.current_epoch} - GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    def on_train_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            logger.info(f"✅ Epoch {trainer.current_epoch} completed - GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    def on_validation_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            logger.info(f"🔍 Validation started - GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    def on_validation_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            logger.info(f"🔍 Validation completed - GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    def on_train_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            logger.info(f"🏁 Training completed - Final GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            logger.info(f"Peak GPU Memory used: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

def train(cfg: DictConfig):
    setup_logger(cfg.logging.log_path)
    logger.info(f"Starting training pipeline")

    env_folder = os.getenv("ENV", "no-env")
    pipeline_run_id = os.getenv("PIPELINE_RUN_ID", "no-pipeline-id")

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

    model_class_path = MODEL_CLASSES[cfg.model.type]
    ModelClass = hydra.utils.get_class(model_class_path)

    model = ModelClass(
        model_name=cfg.model.name,
        model_type=cfg.model.type,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        weight_decay=cfg.training.weight_decay,
        use_quantization=cfg.model.quantization,
        peft_method=cfg.model.peft_method,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params:,}")
    
    # Log GPU information and verify model placement
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Check if model is on GPU
        model_device = next(model.parameters()).device
        logger.info(f"Model device: {model_device}")
        if model_device.type == 'cuda':
            logger.info(f"✅ Model successfully loaded on GPU: {torch.cuda.get_device_name(model_device.index)}")
            logger.info(f"GPU Memory after model load: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        else:
            logger.warning(f"⚠️ Model is on {model_device}, not on GPU as expected")
    else:
        logger.warning("CUDA not available, training will use CPU")


    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.training.model_artifact_dir, f"{pipeline_run_id}/checkpoints"),
        filename=f"best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        auto_insert_metric_name=False,
    )
    logger.debug(f"Checkpoint directory: {checkpoint_callback.dirpath}")
    logger.debug(f"Checkpoint filename: {checkpoint_callback.filename}")

    callbacks = [
        checkpoint_callback,
        EarlyStopping(monitor="val_loss", patience=cfg.training.patience, mode="min"),
        GPUMonitoringCallback(),
    ]

    wandb.init(project=f"{cfg.project_name}-training-{env_folder}", name=f"{cfg.model.name}-{cfg.model.peft_method}", tags=[f"pipeline:{pipeline_run_id}"])
    wandb_logger = WandbLogger(project=cfg.project_name, log_model=False)
    wandb_logger.experiment.config.update(dict(cfg))

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices=cfg.training.devices,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=cfg.training.gradient_clip_val,
        precision=cfg.training.precision,   # e.g. "bf16-mixed" or "16-mixed"
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        val_check_interval=1.0,  # Validate every epoch (more efficient for small datasets)
        check_val_every_n_epoch=1,  # Validate every epoch
        # Memory optimization settings
        enable_progress_bar=True,
        enable_model_summary=False,  # Disable to save memory
        log_every_n_steps=1,  # Log every step for small datasets
    )
    if getattr(cfg.training, "gradient_checkpointing", False):
        try:
            model.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing.")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")

    logger.info("Starting model training")
    trainer.fit(model, datamodule)
    logger.success("Training completed successfully")

    logger.info("Saving model in hf format")
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

    wandb_logger.experiment.summary["best_val_loss"] = trainer.callback_metrics["val_loss"].item()
    wandb_logger.experiment.summary["best_epoch"] = trainer.current_epoch
    wandb_logger.experiment.summary["sagemaker_pipeline_run_id"] = pipeline_run_id
    wandb.finish()
