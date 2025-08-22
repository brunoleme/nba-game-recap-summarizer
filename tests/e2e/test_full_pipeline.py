import os
from pathlib import Path
import hydra
from text2cypher.finetuning.preprocessing import preprocessing
from text2cypher.finetuning.train import train
from text2cypher.finetuning.evaluate_model import evaluate_model

def test_full_pipeline():
    config_name = f"config.test"
    config_path = os.path.abspath("tests/resources/config")

    with hydra.initialize_config_dir(config_dir=config_path):
        cfg = hydra.compose(config_name=config_name)

    env_folder = "dev"
    os.environ["ENV"] = env_folder
    os.environ["PIPELINE_RUN_ID"] = "pipeline_id"

    preprocessing(cfg)
    train(cfg)
    evaluate_model(cfg)

    model_artifacts_dir = Path(cfg.training.model_artifact_dir) / "pipeline_id"
    report_file = model_artifacts_dir / "reports/eval_metrics.json"
    assert report_file.exists()
    assert report_file.read_text()