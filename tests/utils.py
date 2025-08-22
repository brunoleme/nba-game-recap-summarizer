import os
from pathlib import Path

from text2cypher.finetuning.data.notechat_preprocessing import NoteChatDataPreprocessingModule

def run_preprocessing_for_tests():
    source_data_folder = "tests/resources"
    preprocessed_input_data_folder = "tests/resources"
    source_data_path = "source_data/notechat_sample_dataset.csv"
    env_folder = "dev"

    preprocessingmodule = NoteChatDataPreprocessingModule(
        model_name="t5-small",
        source_data_folder=source_data_folder,
        source_data_path=source_data_path,
        preprocessed_output_data_folder=preprocessed_input_data_folder,
        env_folder=env_folder,
        max_length=128,
    )

    preprocessingmodule.run()

