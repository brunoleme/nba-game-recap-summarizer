import os
from pathlib import Path

from nba_game_recap_summarizer.finetuning.data.nba_recap_preprocessing import NBARecapDataPreprocessingModule

def run_preprocessing_for_tests():
    source_data_folder = "tests/resources"
    preprocessed_input_data_folder = "tests/resources"
    source_data_path = "source_data/game_recaps_with_summaries_sample.csv"
    env_folder = "dev"

    preprocessingmodule = NBARecapDataPreprocessingModule(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        source_data_folder=source_data_folder,
        source_data_path=source_data_path,
        preprocessed_output_data_folder=preprocessed_input_data_folder,
        env_folder=env_folder,
        max_length=128,
    )

    preprocessingmodule.run()

