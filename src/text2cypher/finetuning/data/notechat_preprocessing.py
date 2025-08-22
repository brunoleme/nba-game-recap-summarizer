import os
from functools import partial
from typing import Dict, Any

import pandas as pd
from datasets import Dataset
from loguru import logger
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from text2cypher.finetuning.utils.text_utils import replace_first_dash

class NoteChatDataPreprocessingModule():
    def __init__(
        self,
        model_name: str,
        source_data_folder: str,
        source_data_path: str,
        preprocessed_output_data_folder: str,
        env_folder: str,
        max_length: int = 128,
        max_source_length: int = 384,
        max_target_length: int = 128,
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

    @staticmethod
    def format_conversation(conversation: str) -> str:
        turns = conversation.split("\n")
        formatted_turns = []
        for turn in turns:
            if turn.startswith("Doctor:"):
                formatted_turns.append(f"<speaker>Doctor:</speaker>{turn[7:]}")
            elif turn.startswith("Patient:"):
                formatted_turns.append(f"<speaker>Patient:</speaker>{turn[8:]}")
        return f"summarize: <conversation>{' '.join(formatted_turns)}</conversation>"

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
        conversations = examples["conversation"]
        clinical_notes = examples["data"]

        formatted_conversations = [
            NoteChatDataPreprocessingModule.format_conversation(conv) for conv in conversations
        ]

        model_inputs = tokenizer(
            formatted_conversations,
            padding=False,
            max_length=max_source_length,
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                clinical_notes,
                padding=False,
                max_length=max_target_length,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["source_lengths"] = [len(x) for x in model_inputs["input_ids"]]
        model_inputs["target_lengths"] = [len(x) for x in labels["input_ids"]]
        model_inputs["conversation"] = conversations
        model_inputs["clinical_note"] = clinical_notes
        return model_inputs


    def load_data(self) -> None:
        source_data_path = f"{self.source_data_folder}/{self.source_data_path}"
        logger.info(f"Reading CSV from S3 in the mounted path: {source_data_path}")
        df = pd.read_csv(source_data_path)
        assert {"conversation", "data"}.issubset(df.columns), "CSV must contain 'conversation' and 'data' columns"
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

        self.train_dataset = self.preprocess_dataset(self.train_dataset)
        self.val_dataset = self.preprocess_dataset(self.val_dataset)
        self.test_dataset = self.preprocess_dataset(self.test_dataset)

        self._log_dataset_statistics(self.train_dataset, "train")
        self._log_dataset_statistics(self.val_dataset, "validation")
        self._log_dataset_statistics(self.test_dataset, "test")

        logger.info(
            f"Train size: {len(self.train_dataset)}, "
            f"Validation size: {len(self.val_dataset)}, "
            f"Test size: {len(self.test_dataset)}"
        )

    def _log_dataset_statistics(self, dataset, split_name: str) -> None:
        conv_lengths = [len(x.split()) for x in dataset["conversation"]]
        note_lengths = [len(x.split()) for x in dataset["clinical_note"]]

        logger.info(f"{split_name.capitalize()} set statistics:")
        logger.info(
            f"Conversation lengths - Min: {min(conv_lengths)}, "
            f"Max: {max(conv_lengths)}, "
            f"Avg: {sum(conv_lengths)/len(conv_lengths):.2f}"
        )
        logger.info(
            f"Note lengths - Min: {min(note_lengths)}, "
            f"Max: {max(note_lengths)}, "
            f"Avg: {sum(note_lengths)/len(note_lengths):.2f}"
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
