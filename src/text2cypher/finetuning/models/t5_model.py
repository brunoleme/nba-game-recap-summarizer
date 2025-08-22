import os
from typing import Any, Dict, List, Optional

from loguru import logger
from peft import get_peft_model
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


from text2cypher.finetuning.utils.text_utils import replace_first_dash
from .base_model import BaseNoteGenerationModel


class T5NoteGenerationModel(BaseNoteGenerationModel):
    def __init__(self, model_name: str = "t5-small", model=None, tokenizer=None, **kwargs):
        super().__init__(model_name=model_name, **kwargs)

        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer

    def _initialize_model(
        self,
        model_name: str,
        model_type: str,
        use_quantization: bool,
        peft_method: Optional[str] = None,
        **kwargs,
    ):
        try:
            if use_quantization and self.quantization_config:
                logger.info(f"Initializing T5 model with: {model_name}, quantization type: {self.quantization_type}")
                model = T5ForConditionalGeneration.from_pretrained(
                    replace_first_dash(model_name),
                    device_map="auto",
                    quantization_config=self.quantization_config
                )
            else:
                logger.info(f"Initializing T5 model with: {model_name}, no quantization")
                model = T5ForConditionalGeneration.from_pretrained(
                    replace_first_dash(model_name),
                    device_map="cpu"
                )
        except ImportError as e:
            logger.warning(f"Quantization failed: {str(e)}. Falling back to CPU mode")
            model = T5ForConditionalGeneration.from_pretrained(
                replace_first_dash(model_name),
                device_map="cpu"
            )

        tokenizer = AutoTokenizer.from_pretrained(replace_first_dash(model_name))

        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [
                "<conversation>", "</conversation>",
                "<note>", "</note>",
                "<speaker>Doctor:</speaker>",
                "<speaker>Patient:</speaker>",
            ]
        }
        tokenizer.add_special_tokens(special_tokens)

        # Apply PEFT if configured
        if peft_method and self.peft_config:
            logger.info(f"Applying PEFT: {peft_method}")
            model = get_peft_model(model, self.peft_config)

        model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def forward(self, **inputs) -> Any:
        logger.debug(f"Forward pass with input keys: {inputs.keys()}")
        return self.model(**inputs)

    def generate_note(self, conversation: str, max_length: Optional[int] = None) -> str:
        if max_length is None:
            max_length = 512

        logger.info("Generating clinical note")
        logger.debug(f"Input conversation length: {len(conversation)}")

        inputs = self.tokenizer(
            conversation,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length + 2,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_notes(self, dataloader: DataLoader, max_length: Optional[int] = None) -> List[str]:
        if max_length is None:
            max_length = 512

        logger.info("Generating clinical notes for batch")
        results = []

        self.model.eval()
        with torch.no_grad():

            for batch_idx, batch in enumerate(dataloader):
                try:
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length + 2,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                    )

                    decoded = [
                        self.tokenizer.decode(output, skip_special_tokens=True)
                        for output in outputs
                    ]
                    results.extend(decoded)
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue

        return results

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = T5ForConditionalGeneration.from_pretrained(path)

        special_tokens = {
            "additional_special_tokens": [
                "<conversation>", "</conversation>",
                "<note>", "</note>",
                "<speaker>Doctor:</speaker>",
                "<speaker>Patient:</speaker>",
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))

        return cls(model_name=path, tokenizer=tokenizer, model=model, **kwargs)

    @staticmethod
    def load_model_from_checkpoint(
        checkpoint_path: str,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        peft_method: Optional[str] = None,
    ) -> "T5NoteGenerationModel":
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = T5NoteGenerationModel.load_from_checkpoint(checkpoint_path)

            # Fallback to checkpoint hyperparams if not provided
            model_name = model_name or checkpoint.hparams.model_name
            model_type = model_type or checkpoint.hparams.model_type
            peft_method = peft_method or checkpoint.hparams.peft_method

            model = T5NoteGenerationModel(
                model_name=model_name,
                model_type=model_type,
                peft_method=peft_method,
                use_quantization=False,
            )

            checkpoint_state = T5NoteGenerationModel.load_from_checkpoint(
                checkpoint_path,
                model_name=model_name,
                model_type=model_type,
                peft_method=peft_method,
            ).state_dict()

            model.load_state_dict(checkpoint_state)
            logger.success("Model restored successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to restore model: {str(e)}")
            raise RuntimeError(f"Restore failed: {e}")

    def prefill(self, input_text):
        input_enc = self.tokenizer(
            "summarize: " + input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                input_ids=input_enc["input_ids"],
                attention_mask=input_enc["attention_mask"]
            )

        return {
            "encoder_outputs": BaseModelOutput(last_hidden_state=encoder_outputs.last_hidden_state),
            "attention_mask": input_enc["attention_mask"]
        }

    def decode(
        self,
        encoder_outputs,  # this should be a BaseModelOutput object
        attention_mask,
        max_length: int = 1024
    ) -> str:
        attention_mask = torch.tensor(attention_mask).to(self.model.device)

        outputs = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
