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
        # Data quality filtering parameters
        apply_quality_filters: bool = False,
        min_summary_length: int = 10,
        min_recap_length: int = 50,
        max_length_ratio: float = 0.8,
        min_length_ratio: float = 0.01,
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
        # Data quality filtering parameters
        self.apply_quality_filters = apply_quality_filters
        self.min_summary_length = min_summary_length
        self.min_recap_length = min_recap_length
        self.max_length_ratio = max_length_ratio
        self.min_length_ratio = min_length_ratio


    def load_data(self) -> None:
        source_data_path = f"{self.source_data_folder}/{self.source_data_path}"
        logger.info(f"Reading CSV from S3 in the mounted path: {source_data_path}")
        df = pd.read_csv(source_data_path)
        assert {"game_recap", "game_recap_summary"}.issubset(df.columns), "CSV must contain 'game_recap' and 'game_recap_summary' columns"
        
        # Clean data: remove rows with None values in critical columns
        initial_count = len(df)
        df = df.dropna(subset=["game_recap", "game_recap_summary"])
        cleaned_count = len(df)
        if initial_count != cleaned_count:
            logger.warning(f"Removed {initial_count - cleaned_count} rows with None values in game_recap or game_recap_summary")
        
        # Apply data quality filtering
        df = self._filter_low_quality_data(df)
        
        self.dataset = Dataset.from_pandas(df)
        logger.info(f"Loaded {len(self.dataset)} rows from S3 after quality filtering.")

    def setup(self) -> None:
        logger.info(f"Setting up dataset...")

        self.load_data()

        full_dataset = self.dataset.shuffle(seed=self.shuffle_seed)
        logger.debug(f"Dataset columns: {full_dataset.column_names}")

        n = len(full_dataset)
        
        # Use explicit sample counts instead of percentages
        if self.train_samples > 0:
            train_size = min(self.train_samples, n)
        else:
            train_size = n
            
        if self.val_samples > 0:
            val_size = min(self.val_samples, n - train_size)
        else:
            val_size = max(0, n - train_size) // 2
            
        if self.test_samples > 0:
            test_size = min(self.test_samples, n - train_size - val_size)
        else:
            test_size = max(0, n - train_size - val_size)

        # Ensure we don't exceed total dataset size
        total_allocated = train_size + val_size + test_size
        if total_allocated > n:
            # Scale down proportionally
            scale_factor = n / total_allocated
            train_size = int(train_size * scale_factor)
            val_size = int(val_size * scale_factor)
            test_size = n - train_size - val_size

        self.train_dataset = full_dataset.select(range(0, train_size))
        self.val_dataset = full_dataset.select(range(train_size, train_size + val_size))
        self.test_dataset = full_dataset.select(range(train_size + val_size, train_size + val_size + test_size))

        self._log_dataset_statistics(self.train_dataset, "train")
        self._log_dataset_statistics(self.val_dataset, "validation")
        self._log_dataset_statistics(self.test_dataset, "test")

        logger.info(
            f"Train size: {len(self.train_dataset)}, "
            f"Validation size: {len(self.val_dataset)}, "
            f"Test size: {len(self.test_dataset)}"
        )

    def _log_dataset_statistics(self, dataset, split_name: str) -> None:
        # Handle None values in game_recap
        recap_lengths = [len(x.split()) if x is not None else 0 for x in dataset["game_recap"]]
        # Handle None values in game_recap_summary
        summaries_lengths = [len(x.split()) if x is not None else 0 for x in dataset["game_recap_summary"]]

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


    def _filter_low_quality_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out low-quality data samples based on various criteria."""
        if not self.apply_quality_filters:
            logger.info("Data quality filtering disabled, skipping...")
            return df
            
        initial_count = len(df)
        logger.info(f"Starting data quality filtering with {initial_count} samples...")
        
        # Calculate text metrics
        df['recap_length'] = df['game_recap'].str.len()
        df['summary_length'] = df['game_recap_summary'].str.len()
        df['recap_word_count'] = df['game_recap'].str.split().str.len()
        df['summary_word_count'] = df['game_recap_summary'].str.split().str.len()
        df['length_ratio'] = df['summary_length'] / df['recap_length']
        
        # Filter 1: Remove very short summaries (likely just scores)
        short_summary_mask = df['summary_word_count'] < self.min_summary_length
        short_summary_count = short_summary_mask.sum()
        if short_summary_count > 0:
            logger.warning(f"Removing {short_summary_count} samples with very short summaries (< {self.min_summary_length} words)")
            df = df[~short_summary_mask]
        
        # Filter 2: Remove very short recaps (likely corrupted data)
        short_recap_mask = df['recap_word_count'] < self.min_recap_length
        short_recap_count = short_recap_mask.sum()
        if short_recap_count > 0:
            logger.warning(f"Removing {short_recap_count} samples with very short recaps (< {self.min_recap_length} words)")
            df = df[~short_recap_mask]
        
        # Filter 3: Remove extreme length ratios (likely data corruption)
        extreme_low_ratio_mask = df['length_ratio'] < self.min_length_ratio
        extreme_high_ratio_mask = df['length_ratio'] > self.max_length_ratio
        extreme_ratio_count = (extreme_low_ratio_mask | extreme_high_ratio_mask).sum()
        if extreme_ratio_count > 0:
            logger.warning(f"Removing {extreme_ratio_count} samples with extreme length ratios")
            df = df[~(extreme_low_ratio_mask | extreme_high_ratio_mask)]
        
        # Filter 4: Remove samples with HTML tags (data corruption)
        html_recap_mask = df['game_recap'].str.contains('<[^>]+>', regex=True, na=False)
        html_summary_mask = df['game_recap_summary'].str.contains('<[^>]+>', regex=True, na=False)
        html_count = (html_recap_mask | html_summary_mask).sum()
        if html_count > 0:
            logger.warning(f"Removing {html_count} samples with HTML tags")
            df = df[~(html_recap_mask | html_summary_mask)]
        
        # Filter 5: Remove samples with corrupted recap content
        corrupted_recap_mask = df['game_recap'].str.contains(r'\[No meaningful paragraphs found\]', regex=False, na=False)
        corrupted_count = corrupted_recap_mask.sum()
        if corrupted_count > 0:
            logger.warning(f"Removing {corrupted_count} samples with corrupted recap content")
            df = df[~corrupted_recap_mask]
        
        # Filter 6: Remove duplicate recaps (keep first occurrence)
        duplicate_recap_mask = df.duplicated(subset=['game_recap'], keep='first')
        duplicate_count = duplicate_recap_mask.sum()
        if duplicate_count > 0:
            logger.warning(f"Removing {duplicate_count} duplicate recaps")
            df = df[~duplicate_recap_mask]
        
        # Filter 7: Remove samples where summary is just a score (pattern matching)
        score_patterns = [
            r'^\d+-\d+$',  # Just numbers like "73-104"
            r'^\w+ - \d+\s*\n\w+ - \d+$',  # Team - Score format
            r'^\w+ wins \d+-\d+',  # "Team wins 107-100"
            r'^Score: \w+ \d+-\d+$',  # "Score: Team 93-85"
        ]
        
        score_summary_mask = df['game_recap_summary'].str.match('|'.join(score_patterns), case=False, na=False)
        score_count = score_summary_mask.sum()
        if score_count > 0:
            logger.warning(f"Removing {score_count} samples where summary is just a score")
            df = df[~score_summary_mask]
        
        # Clean up temporary columns
        df = df.drop(columns=['recap_length', 'summary_length', 'recap_word_count', 'summary_word_count', 'length_ratio'])
        
        final_count = len(df)
        removed_count = initial_count - final_count
        removal_percentage = (removed_count / initial_count) * 100
        
        logger.info(f"Data quality filtering complete:")
        logger.info(f"  Initial samples: {initial_count}")
        logger.info(f"  Removed samples: {removed_count} ({removal_percentage:.1f}%)")
        logger.info(f"  Final samples: {final_count}")
        
        # Store filtering statistics for later use
        self.filtering_stats = {
            'initial_count': initial_count,
            'removed_count': removed_count,
            'final_count': final_count,
            'removal_percentage': removal_percentage
        }
        
        return df

    def run(self) -> None:
        self.clean_output_folder()
        self.setup()
        self.export_cleaned_data()
