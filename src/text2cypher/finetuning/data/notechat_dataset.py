import os
from functools import partial
from typing import Dict, Optional

import pandas as pd
import pytorch_lightning as pl
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from text2cypher.finetuning.utils.text_utils import replace_first_dash


class NoteChatDataModule(pl.LightningDataModule):
    def __init__(
        self,
        source_data_path: str,
        preprocessed_input_data_folder: str,
        env_folder: str,
        model_name: str,
        batch_size: int = 8,
        max_length: int = 128,
        max_source_length: int = 384,
        max_target_length: int = 128,
        num_workers: int = 4,
        train_samples: int = -1,
        val_samples: int = -1,
        test_samples: int = -1,
        shuffle: bool = True,
        shuffle_seed: int = 42,
    ):
        super().__init__()
        self.model_name = model_name
        self.source_data_path = source_data_path
        self.preprocessed_input_data_folder = preprocessed_input_data_folder
        self.env_folder = env_folder
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_workers = num_workers
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        logger.info(f"Initializing tokenizer with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(replace_first_dash(model_name))

        special_tokens = {
            "additional_special_tokens": [
                "<conversation>", "</conversation>",
                "<note>", "</note>",
                "<speaker>Doctor:</speaker>",
                "<speaker>Patient:</speaker>",
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=model_name,
            padding="longest",
            max_length=self.max_length,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )

    def load_data(self):
        filename = os.path.basename(self.source_data_path)

        train_filename = filename.replace(".csv", "_train.parquet")
        val_filename = filename.replace(".csv", "_val.parquet")
        test_filename = filename.replace(".csv", "_test.parquet")

        input_train_path = f"{self.preprocessed_input_data_folder}/preprocessed/{train_filename}"
        logger.info(f"Reading train parquet data from S3: {input_train_path}")
        self.train_dataset = load_dataset("parquet", data_files=input_train_path)["train"]
        self.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        logger.info(f"Loaded {len(self.train_dataset)} train rows from S3.")

        input_val_path = f"{self.preprocessed_input_data_folder}/preprocessed/{val_filename}"
        logger.info(f"Reading valid parquet data from S3: {input_val_path}")
        self.val_dataset = load_dataset("parquet", data_files=input_val_path)["train"]
        self.val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        logger.info(f"Loaded {len(self.val_dataset)} valid rows from S3.")

        input_test_path = f"{self.preprocessed_input_data_folder}/preprocessed/{test_filename}"
        logger.info(f"Reading test parquet data from S3: {input_test_path}")
        self.test_dataset = load_dataset("parquet", data_files=input_test_path)["train"]
        self.test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        logger.info(f"Loaded {len(self.test_dataset)} test rows from S3.")



    def setup(self, stage: Optional[str] = None):
        logger.info(f"Setting up dataset for stage: {stage}")

        self.load_data()

        if self.train_samples > 0 and self.train_samples < len(self.train_dataset):
            self.train_dataset = self.train_dataset.shuffle(seed=self.shuffle_seed).select(range(self.train_samples))

        if self.val_samples > 0 and self.val_samples < len(self.val_dataset):
            self.val_dataset = self.val_dataset.shuffle(seed=self.shuffle_seed).select(range(self.val_samples))

        if self.test_samples > 0 and self.test_samples < len(self.test_dataset):
            self.test_dataset = self.test_dataset.shuffle(seed=self.shuffle_seed).select(range(self.test_samples))

        logger.info(
            f"Train size: {len(self.train_dataset)}, "
            f"Validation size: {len(self.val_dataset)}, "
            f"Test size: {len(self.test_dataset)}"
        )

    def _create_dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=self.data_collator,
            persistent_workers=self.num_workers > 0,
            pin_memory=False,
        )

    def train_dataloader(self) -> DataLoader:
        logger.info("Creating train dataloader")
        return self._create_dataloader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        logger.info("Creating validation dataloader")
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        logger.info("Creating test dataloader")
        return self._create_dataloader(self.test_dataset, shuffle=False)
