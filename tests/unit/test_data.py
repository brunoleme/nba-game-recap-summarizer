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
    dataset = NBARecapDataModule(model_name="hf-internal-testing/tiny-random-LlamaForCausalLM", source_data_path=source_data_path, env_folder=env_folder, preprocessed_input_data_folder=preprocessed_input_data_folder, batch_size=2, max_length=128, )
    assert dataset.model_name == "hf-internal-testing/tiny-random-LlamaForCausalLM"
    assert dataset.batch_size == 2
    assert dataset.max_length == 128

def test_preprocess_function() -> None:
    dataset = NBARecapDataModule(model_name="hf-internal-testing/tiny-random-LlamaForCausalLM", source_data_path=source_data_path, env_folder=env_folder, preprocessed_input_data_folder=preprocessed_input_data_folder, batch_size=2, max_length=128, )
    examples = {
        "game_recap": ["Tonight in the oppening game, Lakers beats Suns in the overtime with a Bryant game winner."],
        "game_recap_summary": ["Lakers beats Suns"],
    }
    result = dataset.preprocess_function(
        examples,
        dataset.tokenizer,
        max_source_length=128,  # Keep test values small
        max_target_length=128,  # Keep test values small
    )
    assert "input_ids" in result
    assert "labels" in result
    assert "source_lengths" in result
    assert "target_lengths" in result
    assert "game_recap" in result
    assert "game_recap_summary" in result

def test_data_splitting_with_mocker(mocker) -> None:
    # Create dummy datasets with the right structure for NBA recap data
    dummy_data = {
        "date": ["2024-01-01"] * 20,
        "home_team": ["Lakers"] * 20,
        "away_team": ["Suns"] * 20,
        "query": ["test"] * 20,
        "recap_link": ["http://test.com"] * 20,
        "game_recap": ["Test game recap text"] * 20,
        "game_recap_summary": ["Test summary"] * 20,
    }
    
    # Create datasets with enough samples for our test
    train_ds = Dataset.from_pandas(pd.DataFrame({k: v[:10] for k, v in dummy_data.items()}))
    val_ds = Dataset.from_pandas(pd.DataFrame({k: v[10:15] for k, v in dummy_data.items()}))
    test_ds = Dataset.from_pandas(pd.DataFrame({k: v[15:20] for k, v in dummy_data.items()}))
    
    # Mock the datasets.load_dataset to return the right splits
    def mock_load_dataset(data_files, **kwargs):
        if "train" in data_files:
            return {"train": train_ds}
        elif "val" in data_files:
            return {"train": val_ds}
        elif "test" in data_files:
            return {"train": test_ds}
        else:
            return {"train": train_ds}
    
    mocker.patch("datasets.load_dataset", side_effect=mock_load_dataset)

    module = NBARecapDataModule(model_name="hf-internal-testing/tiny-random-LlamaForCausalLM", source_data_path=source_data_path, env_folder=env_folder, preprocessed_input_data_folder=preprocessed_input_data_folder, train_samples=10, val_samples=5, test_samples=5)
    module.setup()

    assert len(module.train_dataset) == 10
    assert len(module.val_dataset) == 5
    assert len(module.test_dataset) == 5

def test_dataloader_creation_with_mock(mocker) -> None:
    dummy_data = Dataset.from_dict({
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "labels": [[1, 2, 3]],
    })
    dummy_ds = Dataset.from_pandas(pd.DataFrame(dummy_data))
    dummy_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    mocker.patch("datasets.load_dataset", return_value={"train": dummy_ds})

    module = NBARecapDataModule(model_name="hf-internal-testing/tiny-random-LlamaForCausalLM", source_data_path=source_data_path, env_folder=env_folder, preprocessed_input_data_folder=preprocessed_input_data_folder, train_samples=5)
    module.setup()

    loader = module.train_dataloader()
    batch = next(iter(loader))
    assert "input_ids" in batch
    assert "labels" in batch
