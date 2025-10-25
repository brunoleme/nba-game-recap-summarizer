import os
from typing import Any, Dict, List, Optional
from loguru import logger
from peft import get_peft_model, prepare_model_for_kbit_training
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_model import BaseRecapSummarizationModel
from ..utils.tokenization_utils import (
    preprocess_text, 
    postprocess_text, 
    add_custom_tokens_to_tokenizer
)


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
        # CRITICAL: Set padding_side to "left" for decoder-only models
        tokenizer.padding_side = "left"

        # Add custom NBA tokens for better performance
        tokenizer = add_custom_tokens_to_tokenizer(tokenizer)

        try:
            if use_quantization and self.quantization_config:
                logger.info(f"Initializing LLaMA with: {name}, quantization: {self.quantization_type}")
                
                # CRITICAL: Store the original unquantized model for KTO training
                logger.info("Loading original unquantized model for KTO compatibility...")
                original_model = AutoModelForCausalLM.from_pretrained(
                    name,
                    device_map="cpu",  # Load on CPU first to avoid memory issues
                    torch_dtype="auto",
                )
                # Store reference to original model
                self.original_model = original_model
                logger.info("Original unquantized model stored for KTO training")
                
                # Now load the quantized model for training
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
                # No quantization, so original model is the same
                self.original_model = None
        except ImportError as e:
            logger.warning(f"Quantization failed: {str(e)}. Falling back to CPU mode")
            model = AutoModelForCausalLM.from_pretrained(
                name,
                device_map="cpu",
                torch_dtype="auto",
            )
            self.original_model = None

        # Apply PEFT if configured (Base sets proper CAUSAL_LM + target_modules for LLaMA)
        if peft_method and self.peft_config:
            logger.info(f"Applying PEFT: {peft_method}")
            model = get_peft_model(model, self.peft_config)

        # Resize token embeddings to accommodate custom tokens
        if len(tokenizer) > model.get_input_embeddings().num_embeddings:
            logger.info(f"Resizing token embeddings from {model.get_input_embeddings().num_embeddings} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

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

        # Preprocess the input text for better tokenization
        preprocessed_recap = preprocess_text(game_recap)

        # Use plain delimiters to avoid changing tokenizer vocab
        prompt = (
            "You are an NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n"
            "### NBA Game Recap ###\n"
            f"{preprocessed_recap}\n\n"
            "### Recap Summary ###\n"
        )

        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.tokenizer.model_max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_length,
                do_sample=True,           # Enable sampling for better quality
                temperature=0.7,          # Add randomness to avoid repetition
                top_p=0.9,               # Nucleus sampling
                top_k=50,                # Limit vocabulary for better quality
                typical_p=None,
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                pad_token_id=self.tokenizer.pad_token_id,
                no_repeat_ngram_size=0,   # Disable n-gram blocking to prevent word merging
                repetition_penalty=1.1,   # Lower penalty to prevent word merging
            )

        # Strip the prompt portion
        gen_ids = out[0][enc["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        # Postprocess the generated text to restore proper formatting
        return postprocess_text(generated_text)

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
                        # Preprocess each game recap for better tokenization
                        preprocessed_convs = [preprocess_text(c) for c in convs]
                        prompts = [
                            "You are an NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n"
                            "### NBA Game Recap ###\n"
                            f"{c}\n\n"
                            "### Recap Summary ###\n"
                            for c in preprocessed_convs
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
                        
                        # Log GPU memory before generation
                        if torch.cuda.is_available():
                            gpu_memory_before = torch.cuda.memory_allocated(0) / 1024**3
                            logger.info(f"GPU Memory before generation: {gpu_memory_before:.2f} GB")
                        
                        # Generate summaries for this batch
                        # Limit output length to reasonable summary length based on target distribution
                        max_new_tokens = min(max_length, 300)  # Cap at 300 tokens (covers P95 and max)
                        logger.info(f"Generating with max_new_tokens={max_new_tokens}")
                        
                        out = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,           # Enable sampling for better quality
                            temperature=0.7,          # Add randomness to avoid repetition
                            top_p=0.9,               # Nucleus sampling
                            top_k=50,                # Limit vocabulary for better quality
                            typical_p=None,
                            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                            pad_token_id=self.tokenizer.pad_token_id,
                            no_repeat_ngram_size=0,   # Disable n-gram blocking to prevent word merging
                            repetition_penalty=1.1,   # Lower penalty to prevent word merging
                        )
                        
                        logger.info(f"Generated output for batch {batch_idx}, shape: {out.shape}")
                        
                        # Log GPU memory after generation
                        if torch.cuda.is_available():
                            gpu_memory_after = torch.cuda.memory_allocated(0) / 1024**3
                            logger.info(f"GPU Memory after generation: {gpu_memory_after:.2f} GB")
                            logger.info(f"GPU Memory used for generation: {gpu_memory_after - gpu_memory_before:.2f} GB")
                        
                        # Decode results for this batch
                        for i in range(out.shape[0]):
                            gen_ids = out[i][prompt_lengths[i]:]
                            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                            # Postprocess the generated text to restore proper formatting
                            postprocessed = postprocess_text(decoded)
                            results.append(postprocessed)
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

        try:
            # Handle S3 URLs by downloading first
            if checkpoint_path.startswith("s3://"):
                import boto3
                import tempfile
                import os
                
                s3_client = boto3.client('s3')
                s3_path_parts = checkpoint_path.replace("s3://", "").split("/", 1)
                bucket_name = s3_path_parts[0]
                object_key = s3_path_parts[1]
                
                # Download to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt") as temp_file:
                    logger.info(f"Downloading checkpoint from s3://{bucket_name}/{object_key}")
                    s3_client.download_file(bucket_name, object_key, temp_file.name)
                    local_checkpoint_path = temp_file.name
                
                # Load from local file
                checkpoint_data = torch.load(local_checkpoint_path, map_location="cpu")
                
                # Clean up temporary file
                os.unlink(local_checkpoint_path)
            else:
                # Load checkpoint data directly from local path
                checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            hparams = checkpoint_data.get("hyper_parameters", {})

            # Create model with same parameters as training
            model = LlamaRecapSummarizationModel(
                model_name=hparams.get("model_name", model_name or "meta-llama/Llama-3.2-1B-Instruct"),
                model_type=model_type or "llama",
                use_quantization=hparams.get("use_quantization", True),
                quantization_type=hparams.get("quantization_type", "4bit"),
                peft_method=hparams.get("peft_method", peft_method),
                lora_r=hparams.get("lora_r", 16),
                lora_alpha=hparams.get("lora_alpha", 32),
                lora_dropout=hparams.get("lora_dropout", 0.1),
            )

            # Check if this is a LoRA-only checkpoint
            if "lora_state_dict" in checkpoint_data:
                logger.info("Loading LoRA-only checkpoint")
                # Load LoRA weights directly into the PEFT model
                model.load_state_dict(checkpoint_data["lora_state_dict"], strict=False)
            else:
                # Load full model state dict
                model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
            
            logger.success("Model restored successfully from checkpoint")
            return model

        except Exception as e:
            # If loading fails due to quantization metadata mismatch, try loading with strict=False
            if "Unexpected key(s) in state_dict" in str(e) and "absmax" in str(e):
                logger.warning(f"Quantization metadata mismatch detected, attempting to load with strict=False: {str(e)}")
                try:
                    # Load the checkpoint manually to extract hyperparameters
                    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
                    hparams = checkpoint_data.get("hyper_parameters", {})
                    
                    # Create model with same parameters as training
                    model = LlamaRecapSummarizationModel(
                        model_name=hparams.get("model_name", model_name or "meta-llama/Llama-3.2-1B-Instruct"),
                        model_type=model_type or "llama",
                        use_quantization=hparams.get("use_quantization", True),
                        quantization_type=hparams.get("quantization_type", "4bit"),
                        peft_method=hparams.get("peft_method", peft_method),
                        lora_r=hparams.get("lora_r", 16),
                        lora_alpha=hparams.get("lora_alpha", 32),
                        lora_dropout=hparams.get("lora_dropout", 0.1),
                    )
                    
                    # Check if this is a LoRA-only checkpoint
                    if "lora_state_dict" in checkpoint_data:
                        logger.info("Loading LoRA-only checkpoint with strict=False")
                        model.load_state_dict(checkpoint_data["lora_state_dict"], strict=False)
                    else:
                        # Load full model state dict with strict=False
                        model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
                    logger.success("Model restored successfully from checkpoint with strict=False")
                    return model
                    
                except Exception as e2:
                    logger.error(f"Failed to restore model even with strict=False: {str(e2)}")
                    raise RuntimeError(f"Restore failed: {e2}")
            else:
                logger.error(f"Failed to restore model: {str(e)}")
                raise RuntimeError(f"Restore failed: {e}")

    def is_loaded(self) -> bool:
        """Check if the model is properly loaded and ready for inference."""
        try:
            return (
                self.model is not None and 
                self.tokenizer is not None and
                hasattr(self, 'model') and
                hasattr(self, 'tokenizer')
            )
        except Exception:
            return False
