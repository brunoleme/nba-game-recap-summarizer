import os
from pathlib import Path
import hydra
from text2cypher.finetuning.train import train
from tests.utils import run_preprocessing_for_tests


def test_training_pipeline():
    config_name = f"config.test"
    config_path = os.path.abspath("tests/resources/config")

    with hydra.initialize_config_dir(config_dir=config_path):
        cfg = hydra.compose(config_name=config_name)

    run_preprocessing_for_tests()

    env_folder = "dev"
    os.environ["ENV"] = env_folder
    os.environ["PIPELINE_RUN_ID"] = "pipeline_id"
    train(cfg)

    ckpts = list(Path(cfg.training.model_artifact_dir + "/pipeline_id/checkpoints").rglob("*.ckpt"))
    assert len(ckpts) > 0
    assert ckpts[0].stat().st_size > 0
