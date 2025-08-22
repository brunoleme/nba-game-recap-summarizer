import pytest
from unittest import mock
import wandb


@pytest.fixture(autouse=True)
def mock_wandb(monkeypatch):
    """Avoid real wandb API calls during tests."""
    monkeypatch.setattr(wandb, "init", lambda *args, **kwargs: mock.MagicMock())
    monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(wandb, "finish", lambda *args, **kwargs: None)
    monkeypatch.setattr(wandb, "Table", lambda *args, **kwargs: None)


@pytest.fixture
def temp_output_dirs(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preprocessed").mkdir(parents=True, exist_ok=True)

    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "preprocessed").mkdir(parents=True, exist_ok=True)

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return {
        "preprocessed_output_data_folder": output_dir,
        "preprocessed_input_data_folder": input_dir,
        "model_artifact_dir": artifacts_dir,
        "reports_dir": reports_dir,
    }
