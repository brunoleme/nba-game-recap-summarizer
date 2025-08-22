import os

import hydra
from loguru import logger
from omegaconf import DictConfig

def pre_train(cfg: DictConfig):
    logger.info(f"Creating artifact folder structure")
    pipeline_run_id = os.getenv("PIPELINE_RUN_ID", "no-pipeline-id")
    root = os.path.join(cfg.training.model_artifact_dir, pipeline_run_id)
    os.makedirs(os.path.join(root, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(root, "hf_model"), exist_ok=True)
    # os.makedirs(os.path.join(root, "reports"), exist_ok=True)
    logger.info(f"Resolved root path: {root}")
    logger.info(f"Contents of {root}: {os.listdir(root)}")   
    logger.info(f"Created artifact folders under: {root}")
