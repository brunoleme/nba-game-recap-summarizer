import os
from pathlib import Path
import hydra
from text2cypher.finetuning.preprocessing import preprocessing


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
        "preprocessed/notechat_sample_dataset_train.parquet",
        "preprocessed/notechat_sample_dataset_val.parquet",
        "preprocessed/notechat_sample_dataset_test.parquet",
    ]:
        assert (out_root / name).exists()
