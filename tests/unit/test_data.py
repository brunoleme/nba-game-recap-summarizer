import pytest
import pandas as pd
from datasets import Dataset
from nba_game_recap_summarizer.finetuning.data.nba_recap_dataset import NBARecapDataModule
from nba_game_recap_summarizer.finetuning.data.nba_recap_preprocessing import NBARecapDataPreprocessingModule

source_data_folder = "tests/resources"
source_data_path = "source_data/game_recaps_with_summaries_sample.csv"
preprocessed_output_data_folder = "tests/resources"
preprocessed_input_data_folder = "tests/resources"
env_folder = 'dev'

def test_dataset_initialization() -> None:
    dataset = NBARecapDataModule(model_name="meta-llama/Llama-3.2-3B-Instruct", source_data_path=source_data_path, env_folder=env_folder, preprocessed_input_data_folder=preprocessed_input_data_folder, batch_size=2, max_length=128, )
    assert dataset.model_name == "meta-llama/Llama-3.2-3B-Instruct"
    assert dataset.batch_size == 2
    assert dataset.max_length == 128

def test_preprocess_function() -> None:
    dataset_preprocessing = NBARecapDataPreprocessingModule(model_name="meta-llama/Llama-3.2-3B-Instruct", source_data_folder=source_data_folder, source_data_path=source_data_path, preprocessed_output_data_folder=preprocessed_output_data_folder, env_folder=env_folder)
    examples = {
        "game_recap": ["Tonight in the oppening game, Lakers beats Suns in the overtime with a Bryant game winner."],
        "game_recap_summary": ["Lakers beats Suns"],
    }
    result = dataset_preprocessing.preprocess_function(
        examples,
        dataset_preprocessing.tokenizer,
        max_source_length=128,
        max_target_length=128,
    )
    assert "input_ids" in result
    assert "labels" in result
    assert "source_lengths" in result
    assert "target_lengths" in result
    assert "game_recap" in result
    assert "game_recap_summary" in result

def test_data_splitting_with_mocker(mocker) -> None:
    dummy_data = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "labels": [[1, 2, 3]],
    }
    dummy_ds = Dataset.from_pandas(pd.DataFrame(dummy_data))
    dummy_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    mocker.patch("datasets.load_dataset", return_value={"train": dummy_ds})

    module = NBARecapDataModule(model_name="meta-llama/Llama-3.2-3B-Instruct", source_data_path=source_data_path, env_folder=env_folder, preprocessed_input_data_folder=preprocessed_input_data_folder, train_samples=10, val_samples=2, test_samples=2)
    module.prepare_data()
    module.setup()

    assert len(module.train_dataset) == 10
    assert len(module.val_dataset) == 2
    assert len(module.test_dataset) == 2

def test_dataloader_creation_with_mock(mocker) -> None:
    dummy_data = Dataset.from_dict({
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "labels": [[1, 2, 3]],
    })
    dummy_ds = Dataset.from_pandas(pd.DataFrame(dummy_data))
    dummy_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    mocker.patch("datasets.load_dataset", return_value={"train": dummy_ds})

    module = NBARecapDataModule(model_name="meta-llama/Llama-3.2-3B-Instruct", source_data_path=source_data_path, env_folder=env_folder, preprocessed_input_data_folder=preprocessed_input_data_folder, train_samples=5)
    module.setup()

    loader = module.train_dataloader()
    batch = next(iter(loader))
    assert "input_ids" in batch
    assert "labels" in batch
