import os
from functools import partial
from typing import Dict, Optional, Any

from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from dataclasses import dataclass
from typing import List, Dict
import torch

@dataclass
class CausalLMCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = 8
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        labels = [f["labels"] for f in features]
        allowed = {"input_ids", "attention_mask"}
        inputs = []
        for f in features:
            item = {k: v for k, v in f.items() if k in allowed}
            if not item:
                raise ValueError("Batch not tokenized: expected 'input_ids' (and 'attention_mask').")
            inputs.append(item)

        # Use the tokenizer's padding_side setting instead of hardcoded "longest"
        batch = self.tokenizer.pad(
            inputs,
            padding=True,  # Use True to respect tokenizer.padding_side
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            if isinstance(lab, torch.Tensor):
                lab = lab.tolist()
            pad_len = max_len - len(lab)
            padded_labels.append(lab + [self.label_pad_token_id] * pad_len)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


class NBARecapDataModule:
    def __init__(
        self,
        source_data_path: str,
        preprocessed_input_data_folder: str,
        env_folder: str,
        model_name: str,
        batch_size: int = 8,
        max_length: int = 2048,
        max_source_length: int = 1536,  # Reduced from 3072 to save memory
        max_target_length: int = 512,   # Reduced from 1024 to save memory
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.data_collator = CausalLMCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8
        )


    def load_data(self):
        filename = os.path.basename(self.source_data_path)

        train_filename = filename.replace(".csv", "_train.parquet")
        val_filename = filename.replace(".csv", "_val.parquet")
        test_filename = filename.replace(".csv", "_test.parquet")

        source_data_columns = ['date', 'home_team', 'away_team', 'query', 'recap_link', 'game_recap', 'game_recap_summary']

        input_train_path = f"{self.preprocessed_input_data_folder}/preprocessed/{train_filename}"
        logger.info(f"Reading train parquet data from S3: {input_train_path}")
        self.train_dataset = load_dataset("parquet", data_files=input_train_path)["train"]
        self.train_dataset.set_format(type="torch", columns=source_data_columns)
        logger.info(f"Loaded {len(self.train_dataset)} train rows from S3.")

        input_val_path = f"{self.preprocessed_input_data_folder}/preprocessed/{val_filename}"
        logger.info(f"Reading valid parquet data from S3: {input_val_path}")
        self.val_dataset = load_dataset("parquet", data_files=input_val_path)["train"]
        self.val_dataset.set_format(type="torch", columns=source_data_columns)
        logger.info(f"Loaded {len(self.val_dataset)} valid rows from S3.")

        input_test_path = f"{self.preprocessed_input_data_folder}/preprocessed/{test_filename}"
        logger.info(f"Reading test parquet data from S3: {input_test_path}")
        self.test_dataset = load_dataset("parquet", data_files=input_test_path)["train"]
        self.test_dataset.set_format(type="torch", columns=source_data_columns)
        logger.info(f"Loaded {len(self.test_dataset)} test rows from S3.")


    def setup(self, stage: Optional[str] = None):
        logger.info(f"Setting up dataset for stage: {stage}")

        self.load_data()

        self.train_dataset = self.preprocess_dataset(self.train_dataset)
        self.val_dataset = self.preprocess_dataset(self.val_dataset)
        self.test_dataset = self.preprocess_dataset(self.test_dataset)

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



    def preprocess_dataset(self, dataset) -> Any:
        preprocess_fn = partial(
            self.preprocess_function,
            tokenizer=self.tokenizer,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
        )

        return dataset.map(
            preprocess_fn,
            remove_columns=dataset.column_names,
            desc="Processing data...",
            batched=True,
            batch_size=1000,
            num_proc=None if os.environ.get("TOKENIZERS_PARALLELISM") == "false" else 4,
        )

    @staticmethod
    def preprocess_function(examples, tokenizer, max_source_length, max_target_length) -> Dict:
        """
        Build a single sequence: PROMPT + TARGET + EOS
        Labels = same as input_ids, with prompt positions masked to -100.
        """
        game_recaps = examples["game_recap"]
        game_recap_summaries = examples["game_recap_summary"]

        prompts = [
            "You are an NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n"
            f"### NBA Game Recap ###\n{game_recap}\n\n### Recap Summary ###\n"
            for game_recap in game_recaps
        ]

        # Tokenize full text (prompt + target + eos)
        full_texts = [
            p + game_recap_summary + (tokenizer.eos_token or "")
            for p, game_recap_summary in zip(prompts, game_recap_summaries)
        ]
        enc_full = tokenizer(
            full_texts,
            padding=False,
            truncation=True,
            max_length=max_source_length + max_target_length
        )

        # Tokenize prompts alone to get the prefix length for masking
        enc_prompt = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=max_source_length,
        )

        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]
        prompt_ids = enc_prompt["input_ids"]

        labels = []
        source_lengths = []
        target_lengths = []
        for ids, pids in zip(input_ids, prompt_ids):
            start = len(pids)
            lab = ids.copy()
            # mask prompt tokens
            for i in range(min(start, len(lab))):
                lab[i] = -100
            labels.append(lab)
            source_lengths.append(min(start, len(ids)))
            target_lengths.append(max(0, len(ids) - min(start, len(ids))))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source_lengths": source_lengths,
            "target_lengths": target_lengths,
            "game_recap": game_recaps,
            "game_recap_summary": game_recap_summaries,
        }

    def _create_dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=self.data_collator,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """Get all dataloaders as a dictionary."""
        return {
            'train': self._create_dataloader(self.train_dataset, shuffle=self.shuffle),
            'val': self._create_dataloader(self.val_dataset, shuffle=False),
            'test': self._create_dataloader(self.test_dataset, shuffle=False)
        }

    def train_dataloader(self) -> DataLoader:
        logger.info("Creating train dataloader")
        return self._create_dataloader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        logger.info("Creating validation dataloader")
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        logger.info("Creating test dataloader")
        return self._create_dataloader(self.test_dataset, shuffle=False)
