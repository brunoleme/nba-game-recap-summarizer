import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import hydra

from text2cypher.finetuning.evaluate_model import evaluate_model
from tests.utils import run_preprocessing_for_tests


def test_evaluation_pipeline():
    config_name = f"config.test"
    config_path = os.path.abspath("tests/resources/config")

    with hydra.initialize_config_dir(config_dir=config_path):
        cfg = hydra.compose(config_name=config_name)

    run_preprocessing_for_tests()

    env_folder = "dev"
    os.environ["ENV"] = env_folder

    # Create dummy checkpoint file
    model_artifacts_dir = Path(cfg.training.model_artifact_dir) / "pipeline_id"
    model_artifacts_dir.mkdir(parents=True, exist_ok=True)
    dummy_checkpoint = model_artifacts_dir / "best_model.ckpt"
    dummy_checkpoint.touch()

    os.environ["PIPELINE_RUN_ID"] = "pipeline_id"

    with (
        patch("text2cypher.finetuning.evaluate_model.load_model") as mock_load_model,
        patch("text2cypher.finetuning.evaluate_model.compute_group_metrics") as mock_compute_metrics,
        patch("text2cypher.finetuning.evaluate_model.calculate_average_latency", return_value=0.1),
        patch("text2cypher.finetuning.evaluate_model.calculate_model_size_in_params", return_value=1234567)
    ):
        # Mock model and metric result
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_compute_metrics.side_effect = [
            pd.DataFrame({"rouge_score": [0.0], "bleu_score": [0.0]}),
            pd.DataFrame({"bert_score": [0.847899]}),
            pd.DataFrame({
                "factual_consistency": [3.0],
                "relevance": [5.0],
                "completeness": [3.0],
                "conciseness": [3.666667],
                "clarity": [3.0],
            })
        ]

        evaluate_model(cfg)

    report_file = model_artifacts_dir / "reports/eval_metrics.json"
    assert report_file.exists()
    assert report_file.read_text()
