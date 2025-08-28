import os
from pathlib import Path
import hydra
from nba_game_recap_summarizer.finetuning.preprocessing import preprocessing


def test_preprocessing_pipeline():
    config_name = f"config.test"
    config_path = os.path.abspath("tests/resources/config")

    with hydra.initialize_config_dir(config_dir=config_path):
        cfg = hydra.compose(config_name=config_name)

    env_folder = "dev"
    os.environ["ENV"] = env_folder
    
    preprocessing(cfg)

    out_root = Path(cfg.data.preprocessed_output_data_folder)
    for name in [
        "preprocessed/game_recaps_with_summaries_sample_train.parquet",
        "preprocessed/game_recaps_with_summaries_sample_val.parquet",
        "preprocessed/game_recaps_with_summaries_sample_test.parquet",
    ]:
        assert (out_root / name).exists()
