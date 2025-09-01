import os
from typing import Any, Dict, List, Optional

from loguru import logger
from peft import get_peft_model, prepare_model_for_kbit_training
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_model import BaseRecapSummarizationModel


class LlamaRecapSummarizationModel(BaseRecapSummarizationModel):
    """
    Expects batches to already contain:
      - input_ids, attention_mask (and optionally labels during training)
    For supervised fine-tuning, labels should mask prompt tokens with -100.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", model=None, tokenizer=None, **kwargs):
        # IMPORTANT: pass model_type="llama" (or "decoder-only") so Base can set flags/PEFT correctly
        if "model_type" not in kwargs:
            kwargs["model_type"] = "llama"
        super().__init__(model_name=model_name, **kwargs)

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            # tokenizer hygiene for decoder-only
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

    # ---------- init ----------
    def _initialize_model(
        self,
        model_name: str,
        model_type: str,
        use_quantization: bool,
        peft_method: Optional[str] = None,
        **kwargs,
    ):
        logger.info(f"Initializing tokenizer with model: {model_name}")

        name = model_name

        # Tokenizer first (so we can pass ids to model if needed)
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
        # LLaMA tokenizers often lack pad_token; set to eos for batching
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # DO NOT add arbitrary special tokens by default for LLaMA.
        # If you truly need your XML-like tags, you can add them, but then call model.resize_token_embeddings.

        try:
            if use_quantization and self.quantization_config:
                logger.info(f"Initializing LLaMA with: {name}, quantization: {self.quantization_type}")
                model = AutoModelForCausalLM.from_pretrained(
                    name,
                    device_map="auto",
                    quantization_config=self.quantization_config,
                    torch_dtype="auto",
                )
                if self.peft_method == "lora":
                    model = prepare_model_for_kbit_training(model)
            else:
                logger.info(f"Initializing LLaMA with: {name}, no quantization")
                model = AutoModelForCausalLM.from_pretrained(
                    name,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    torch_dtype="auto",
                )
        except ImportError as e:
            logger.warning(f"Quantization failed: {str(e)}. Falling back to CPU mode")
            model = AutoModelForCausalLM.from_pretrained(
                name,
                device_map="cpu",
                torch_dtype="auto",
            )

        # Apply PEFT if configured (Base sets proper CAUSAL_LM + target_modules for LLaMA)
        if peft_method and self.peft_config:
            logger.info(f"Applying PEFT: {peft_method}")
            model = get_peft_model(model, self.peft_config)

        # If (and only if) you explicitly added extra tokens above, then:
        # model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    # ---------- forward ----------
    def forward(self, **inputs) -> Any:
        """
        Expect keys like: input_ids, attention_mask, labels (labels optional for inference).
        For training, labels should already mask prompt tokens with -100.
        """
        logger.debug(f"Forward pass with input keys: {inputs.keys()}")
        return self.model(**inputs)

    # ---------- single-recap summarization ----------
    def summarize_recap(self, game_recap: str, max_length: Optional[int] = None) -> str:
        """
        Builds a prompt, generates continuation, then strips the prompt.
        """
        if max_length is None:
            max_length = 2048

        logger.info("Generating game recap summary(LLaMA)")
        logger.debug(f"Input game_recap length: {len(game_recap)}")

        # Use plain delimiters to avoid changing tokenizer vocab
        prompt = (
            "You are a NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n"
            "### NBA Game Recap ###\n"
            f"{game_recap}\n\n"
            "### Recap Summary ###\n"
        )

        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.tokenizer.model_max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_length,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                typical_p=None,
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                pad_token_id=self.tokenizer.pad_token_id,
                no_repeat_ngram_size=3,
                repetition_penalty=1.1,
            )

        # Strip the prompt portion
        gen_ids = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # ---------- batch generation ----------
    def summarize_recaps(self, dataloader: DataLoader, max_length: Optional[int] = None) -> List[str]:
        if max_length is None:
            max_length = 2048

        logger.info("Generating recap summaries for batch (LLaMA)")
        results: List[str] = []
        self.model.eval()
        
        # Collect all inputs first for better batching
        all_inputs = []
        all_prompt_lengths = []
        
        logger.info(f"Processing dataloader with batch_size={dataloader.batch_size}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                logger.info(f"Processing batch {batch_idx}, batch keys: {list(batch.keys())}")
                try:
                    # Support two styles:
                    #  A) pre-tokenized batches (input_ids/attention_mask)
                    #  B) raw text batches with "game_recap" field
                    if "input_ids" in batch:
                        logger.info(f"Batch {batch_idx}: Using pre-tokenized input, shape: {batch['input_ids'].shape}")
                        inputs = {k: v.to(self.device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}
                        prompt_lengths = inputs["input_ids"].shape[1] * torch.ones(inputs["input_ids"].shape[0], dtype=torch.long, device=self.device)
                    else:
                        logger.info(f"Batch {batch_idx}: Using raw text input, game_recap count: {len(batch['game_recap'])}")
                        convs: List[str] = batch["game_recap"]
                        prompts = [
                            "You are an NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n"
                            "### NBA Game Recap ###\n"
                            f"{c}\n\n"
                            "### Recap Summary ###\n"
                            for c in convs
                        ]
                        enc = self.tokenizer(
                            prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=self.tokenizer.model_max_length,
                        )
                        inputs = {k: v.to(self.device) for k, v in enc.items()}
                        prompt_lengths = inputs["input_ids"].ne(self.tokenizer.pad_token_id).sum(dim=1)
                    
                    all_inputs.append(inputs)
                    all_prompt_lengths.append(prompt_lengths)
                    logger.info(f"Batch {batch_idx}: Successfully processed, inputs shape: {inputs['input_ids'].shape}")
                    
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    logger.error(f"Batch {batch_idx} content: {batch}")
                    continue
            
            # Process each batch individually to avoid tensor size mismatches
            if all_inputs:
                logger.info(f"Processing {len(all_inputs)} batches individually")
                
                for batch_idx, (inputs, prompt_lengths) in enumerate(zip(all_inputs, all_prompt_lengths)):
                    try:
                        logger.info(f"Generating for batch {batch_idx}, input shape: {inputs['input_ids'].shape}")
                        # Generate summaries for this batch
                        out = self.model.generate(
                            **inputs,
                            max_new_tokens=max_length,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            top_k=None,
                            typical_p=None,
                            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                            pad_token_id=self.tokenizer.pad_token_id,
                            no_repeat_ngram_size=3,
                            repetition_penalty=1.1,
                        )
                        
                        logger.info(f"Generated output for batch {batch_idx}, shape: {out.shape}")
                        
                        # Decode results for this batch
                        for i in range(out.shape[0]):
                            gen_ids = out[i][prompt_lengths[i]:]
                            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                            results.append(decoded)
                            logger.info(f"Batch {batch_idx}, sample {i}: Generated summary of length {len(decoded)}")
                            
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_idx}: {e}")
                        logger.error(f"Batch {batch_idx} inputs: {inputs}")
                        logger.error(f"Batch {batch_idx} prompt_lengths: {prompt_lengths}")
                        # Add empty results for failed batches to maintain indexing
                        batch_size = inputs["input_ids"].shape[0]
                        results.extend([""] * batch_size)
            else:
                logger.warning("No inputs collected from dataloader")

        return results

    # ---------- loading ----------
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype="auto",
        )

        # If you trained with PEFT, caller should wrap with get_peft_model afterwards (or load adapters).

        return cls(model_name=path, tokenizer=tokenizer, model=model, **kwargs)

    @staticmethod
    def load_model_from_checkpoint(
        checkpoint_path: str,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        peft_method: Optional[str] = None,
    ) -> "LlamaRecapSummarizationModel":
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            # Load checkpoint only once
            checkpoint = LlamaRecapSummarizationModel.load_from_checkpoint(checkpoint_path)

            # Fallback to checkpoint hyperparams if not provided
            model_name = model_name or checkpoint.hparams.model_name
            model_type = model_type or checkpoint.hparams.model_type
            peft_method = peft_method or checkpoint.hparams.peft_method

            # Create new model instance
            model = LlamaRecapSummarizationModel(
                model_name=model_name,
                model_type=model_type or "llama",
                peft_method=peft_method,
                use_quantization=False,
            )

            # Use the already loaded checkpoint state
            checkpoint_state = checkpoint.state_dict()
            model.load_state_dict(checkpoint_state, strict=False)  # allow for PEFT heads etc.
            
            logger.success("Model restored successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to restore model: {str(e)}")
            raise RuntimeError(f"Restore failed: {e}")
