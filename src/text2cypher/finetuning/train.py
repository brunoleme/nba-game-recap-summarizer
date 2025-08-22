import os

import hydra
from loguru import logger
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from text2cypher.finetuning.data.notechat_dataset import NoteChatDataModule
from text2cypher.finetuning.utils.logger import setup_logger

MODEL_CLASSES = {
    "t5": "text2cypher.finetuning.models.t5_model.T5NoteGenerationModel",
}

def train(cfg: DictConfig):
    setup_logger(cfg.logging.log_path)
    logger.info(f"Starting training pipeline")

    env_folder = os.getenv("ENV", "no-env")
    pipeline_run_id = os.getenv("PIPELINE_RUN_ID", "no-pipeline-id")

    datamodule = NoteChatDataModule(
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
    )

    logger.info("Starting model training")
    trainer.fit(model, datamodule=datamodule)
    logger.success("Training completed successfully")

    logger.info("Saving model in hf format")
    hf_save_path = os.path.join(cfg.training.model_artifact_dir, f"{pipeline_run_id}/hf_model")
    model.model.merge_and_unload().save_pretrained(hf_save_path)
    model.tokenizer.save_pretrained(hf_save_path)
    logger.success("Model saved successfully")

    wandb_logger.experiment.summary["best_val_loss"] = trainer.callback_metrics["val_loss"].item()
    wandb_logger.experiment.summary["best_epoch"] = trainer.current_epoch
    wandb_logger.experiment.summary["sagemaker_pipeline_run_id"] = pipeline_run_id
    wandb.finish()
