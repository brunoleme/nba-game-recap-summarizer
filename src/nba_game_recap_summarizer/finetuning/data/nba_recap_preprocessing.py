import os
from functools import partial
from typing import Dict, Any

import pandas as pd
from datasets import Dataset
from loguru import logger

class NBARecapDataPreprocessingModule():
    def __init__(
        self,
        model_name: str,
        source_data_folder: str,
        source_data_path: str,
        preprocessed_output_data_folder: str,
        env_folder: str,
        max_length: int = 2048,
        max_source_length: int = 1536,  # Reduced from 3072 to save memory
        max_target_length: int = 512,   # Reduced from 1024 to save memory
        train_samples: int = -1,
        val_samples: int = -1,
        test_samples: int = -1,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle: bool = True,
        shuffle_seed: int = 42,
    ):
        self.model_name = model_name
        self.source_data_folder = source_data_folder
        self.source_data_path = source_data_path
        self.preprocessed_output_data_folder = preprocessed_output_data_folder
        self.env_folder = env_folder
        self.max_length = max_length
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed


    def load_data(self) -> None:
        source_data_path = f"{self.source_data_folder}/{self.source_data_path}"
        logger.info(f"Reading CSV from S3 in the mounted path: {source_data_path}")
        df = pd.read_csv(source_data_path)
        assert {"game_recap", "game_recap_summary"}.issubset(df.columns), "CSV must contain 'game_recap' and 'game_recap_summary' columns"
        self.dataset = Dataset.from_pandas(df)
        logger.info(f"Loaded {len(self.dataset)} rows from S3.")

    def setup(self) -> None:
        logger.info(f"Setting up dataset...")

        self.load_data()

        full_dataset = self.dataset.shuffle(seed=self.shuffle_seed)
        logger.debug(f"Dataset columns: {full_dataset.column_names}")

        n = len(full_dataset)
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "Train/val/test splits must sum to 1.0"

        train_end = int(n * self.train_split)
        val_end = train_end + int(n * self.val_split)

        self.train_dataset = full_dataset.select(range(0, train_end))
        self.val_dataset = full_dataset.select(range(train_end, val_end))
        self.test_dataset = full_dataset.select(range(val_end, n))

        if self.train_samples > 0:
            self.train_dataset = self.train_dataset.select(
                range(min(self.train_samples, len(self.train_dataset)))
            )

        if self.val_samples > 0:
            self.val_dataset =self.val_dataset.select(
                range(min(self.val_samples, len(self.val_dataset)))
            )

        if self.test_samples > 0:
            self.test_dataset = self.test_dataset.select(
                range(min(self.test_samples, len(self.test_dataset)))
            )

        self._log_dataset_statistics(self.train_dataset, "train")
        self._log_dataset_statistics(self.val_dataset, "validation")
        self._log_dataset_statistics(self.test_dataset, "test")

        logger.info(
            f"Train size: {len(self.train_dataset)}, "
            f"Validation size: {len(self.val_dataset)}, "
            f"Test size: {len(self.test_dataset)}"
        )

    def _log_dataset_statistics(self, dataset, split_name: str) -> None:
        recap_lengths = [len(x.split()) for x in dataset["game_recap"]]
        summaries_lengths = [len(x.split()) for x in dataset["game_recap_summary"]]

        logger.info(f"{split_name.capitalize()} set statistics:")
        logger.info(
            f"Recap lengths - Min: {min(recap_lengths)}, "
            f"Max: {max(recap_lengths)}, "
            f"Avg: {sum(recap_lengths)/len(recap_lengths):.2f}"
        )
        logger.info(
            f"Summary lengths - Min: {min(summaries_lengths)}, "
            f"Max: {max(summaries_lengths)}, "
            f"Avg: {sum(summaries_lengths)/len(summaries_lengths):.2f}"
        )

    def export_cleaned_data(self) -> None:

        filename = os.path.basename(self.source_data_path)

        train_filename = filename.replace(".csv", "_train.parquet")
        val_filename = filename.replace(".csv", "_val.parquet")
        test_filename = filename.replace(".csv", "_test.parquet")

        output_folder = f"{self.preprocessed_output_data_folder}/preprocessed"

        train_data_dest_path = f"{output_folder}/{train_filename}"
        val_data_dest_path = f"{output_folder}/{val_filename}"
        test_data_dest_path = f"{output_folder}/{test_filename}"

        self.train_dataset.to_parquet(train_data_dest_path)
        self.val_dataset.to_parquet(val_data_dest_path)
        self.test_dataset.to_parquet(test_data_dest_path)


    def clean_output_folder(self):
        output_folder = f"{self.preprocessed_output_data_folder}/preprocessed"
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


    def run(self) -> None:
        self.clean_output_folder()
        self.setup()
        self.export_cleaned_data()
