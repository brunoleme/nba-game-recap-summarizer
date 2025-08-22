import os
import tarfile
import shutil
from loguru import logger
from omegaconf import DictConfig
from text2cypher.api import inference  # Keeps import for packaging logic validation

def package_model(cfg: DictConfig):
    pipeline_run_id = os.getenv("PIPELINE_RUN_ID", "no-pipeline-id")
    model_dir = os.path.join(cfg.training.model_artifact_dir, f"{pipeline_run_id}/hf_model")
    tar_path = os.path.join(cfg.training.model_artifact_dir, f"{pipeline_run_id}/model.tar.gz")
    code_dir = os.path.join(model_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    # Copy inference.py into code/
    inference_src_path = os.path.abspath("src/text2cypher/api/inference.py")
    inference_dst_path = os.path.join(code_dir, "inference.py")

    logger.info(f"Copying inference.py to: {inference_dst_path}")
    shutil.copy(inference_src_path, inference_dst_path)

    # Package everything under hf_model
    logger.info(f"Packing {model_dir} into {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                fullpath = os.path.join(root, file)
                arcname = os.path.relpath(fullpath, model_dir)
                tar.add(fullpath, arcname=arcname)

    logger.success(f"Model packaged successfully at: {tar_path}")
