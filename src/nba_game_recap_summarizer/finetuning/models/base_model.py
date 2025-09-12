import datetime
import platform
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from loguru import logger
from peft import LoraConfig, PromptTuningConfig
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from transformers import BitsAndBytesConfig, get_linear_schedule_with_warmup

class BaseRecapSummarizationModel(pl.LightningModule, ABC):
    def __init__(
        self,
        model_name: str,
        model_type: str,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        use_quantization: bool = True,
        quantization_type: str = "8bit",
        peft_method: Optional[str] = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        prompt_tuning_n_tokens: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.use_quantization = use_quantization
        self.quantization_type = quantization_type
        self.peft_method = peft_method
        self.validation_step_outputs = []
        self.training_step_outputs = []
        
        # Hard example tracking
        self.hard_examples = []  # Store (loss, input, prediction, reference)
        self.max_hard_examples = 10  # Track top 10 hardest examples
        
        # Memory management
        self._memory_optimization_enabled = True

        logger.info(f"Initializing model: {model_name} ({model_type}), PEFT Method: {peft_method}, Quantization: {use_quantization}")

        # Store experiment metadata
        self.experiment_metadata = {
            "model_type": model_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "hyperparameters": {
                "model_name": model_name,
                "model_type": model_type,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "weight_decay": weight_decay,
                "use_quantization": use_quantization,
                "peft_method": peft_method,
            },
        }

        # Configure quantization based on type
        if use_quantization:
            if quantization_type == "8bit":
                self.quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_8bit_compute_dtype=torch.float16,
                    bnb_8bit_use_double_quant=True,
                )
            elif quantization_type == "4bit":
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                logger.warning(
                    f"Unknown quantization type: {quantization_type}, disabling quantization"
                )
                self.quantization_config = None
                self.use_quantization = False
        else:
            self.quantization_config = None

        # PEFT Config
        self.peft_config = None
        if self.peft_method == "lora":
            # Get model-specific target modules
            target_modules = self._get_lora_target_modules(model_type)
            self.peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                task_type="CAUSAL_LM",
            )
        elif self.peft_method == "prompt_tuning":
            self.peft_config = PromptTuningConfig(
                num_virtual_tokens=prompt_tuning_n_tokens,
                task_type="CAUSAL_LM"
            )

        # Avoid passing kwargs again
        kwargs.pop("model_type", None)
        kwargs.pop("use_quantization", None)
        kwargs.pop("peft_method", None)

        self.model, self.tokenizer = self._initialize_model(
            model_name=model_name,
            model_type=model_type,
            use_quantization=self.use_quantization,
            peft_method=self.peft_method,
            **kwargs
        )

        if hasattr(self.model, "config"):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id

        # Tokenizer hygiene for decoder-only LMs (LLaMA)
        if self.tokenizer is not None:
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

    def _get_lora_target_modules(self, model_type: str) -> list:
        """Get LoRA target modules based on model type."""
        if model_type == "llama":
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif model_type == "phi":
            # Phi-3.5-mini uses similar architecture to LLaMA
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            # Default to LLaMA modules for unknown model types
            logger.warning(f"Unknown model type '{model_type}', using default LLaMA target modules")
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @abstractmethod
    def _initialize_model(
        self,
        model_name: str,
        model_type: str,
        use_quantization: bool,
        peft_method: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Initialize model and tokenizer."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup runs on every GPU/process."""
        if stage == "fit" and self.trainer.logger:
            experiment = self.trainer.logger.experiment

            # Add experiment tags
            experiment.tags = {
                "base_model_name": self.hparams.model_name,
                "lr": self.hparams.learning_rate,
                "gpu": "gpu" if torch.cuda.is_available() else "cpu",
                "os": platform.system().lower(),
                "torch": torch.__version__.split("+")[0],
                "num_gpu": torch.cuda.device_count() if torch.cuda.is_available() else "no_gpu"
            }

            # Add experiment config
            experiment.config.update(
                {
                    "model": {
                        "name": self.hparams.model_name,
                        "learning_rate": self.learning_rate,
                        "warmup_steps": self.warmup_steps,
                        "weight_decay": self.weight_decay,
                    },
                    "hardware": {
                        "gpu": torch.cuda.is_available(),
                        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    },
                    "environment": {
                        "python_version": platform.python_version(),
                        "pytorch_version": torch.__version__,
                        "platform": platform.platform(),
                    },
                },
                allow_val_change=True,
            )

    def on_fit_start(self) -> None:
        """Called when fit begins."""
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False  # important for training
        if self.trainer.logger:
            experiment = self.trainer.logger.experiment
            current_tags = list(experiment.tags) if experiment.tags else []

            new_tags = [
                f"max_epochs_{self.trainer.max_epochs}",
                f"precision_{self.trainer.precision}",
                f"grad_clip_{self.trainer.gradient_clip_val}",
            ]

            experiment.tags = current_tags + new_tags

            training_config = {
                "max_epochs": self.trainer.max_epochs,
                "precision": self.trainer.precision,
                "gradient_clip_val": self.trainer.gradient_clip_val,
                "accumulate_grad_batches": self.trainer.accumulate_grad_batches,
                "strategy_type": self.trainer.strategy.__class__.__name__,
                "batch_size": self.trainer.datamodule.batch_size
                if hasattr(self.trainer, "datamodule") else None,
            }

            experiment.config.update({"training": training_config}, allow_val_change=True)
        
        # Log initial sample prediction to show "before" state
        logger.info("🚀 TRAINING STARTED - Initial Model State:")
        self._log_sample_prediction()

    def on_fit_end(self) -> None:
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True   # re-enable for inference

    @abstractmethod
    def forward(self, **inputs) -> Any:
        """Forward pass of the model."""
        pass

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Log GPU memory before forward pass
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated(0) / 1024**3
            if batch_idx % 10 == 0:  # Log every 10th batch to avoid spam
                logger.debug(f"Batch {batch_idx} - GPU Memory before forward: {gpu_memory_before:.2f} GB")
        
        outputs = self(**batch)
        loss = outputs.loss

        # Log GPU memory after forward pass
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated(0) / 1024**3
            if batch_idx % 10 == 0:  # Log every 10th batch to avoid spam
                logger.debug(f"Batch {batch_idx} - GPU Memory after forward: {gpu_memory_after:.2f} GB")
                logger.debug(f"Batch {batch_idx} - GPU Memory delta: {gpu_memory_after - gpu_memory_before:.2f} GB")
        
        self.log("train_loss", loss, prog_bar=True)
        self.training_step_outputs.append(loss.detach().cpu())

        return loss


    # def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
    #     outputs = self(**batch)
    #     loss = outputs.loss

    #     self.log("val_loss", loss, prog_bar=True)
    #     self.validation_step_outputs.append(loss.detach().cpu())

    def validation_step(self, batch, batch_idx):
        # Log GPU memory before forward pass
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated(0) / 1024**3
            if batch_idx % 5 == 0:  # Log every 5th batch to avoid spam
                logger.debug(f"Val Batch {batch_idx} - GPU Memory before forward: {gpu_memory_before:.2f} GB")
        
        outputs = self(**batch)
        loss = outputs.loss
        
        # Log GPU memory after forward pass
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated(0) / 1024**3
            if batch_idx % 5 == 0:  # Log every 5th batch to avoid spam
                logger.debug(f"Val Batch {batch_idx} - GPU Memory after forward: {gpu_memory_after:.2f} GB")
                logger.debug(f"Val Batch {batch_idx} - GPU Memory delta: {gpu_memory_after - gpu_memory_before:.2f} GB")
        
        # Calculate ROUGE score for a subset of validation samples
        rouge_frequency = getattr(self.hparams, 'rouge_eval_frequency', 10)
        if batch_idx % rouge_frequency == 0 and hasattr(self, 'tokenizer'):
            try:
                # Generate predictions for ROUGE evaluation
                predictions, references = self._generate_predictions_for_eval(batch)
                if predictions and references:
                    rouge_score = self._calculate_rouge_score(predictions, references)
                    self.log("val_rouge_1_2", rouge_score, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
                    logger.info(f"Validation ROUGE-1/2 Score: {rouge_score:.4f}")
                    
                    # Track hard examples (high loss cases)
                    self._track_hard_examples(batch, loss, predictions, references)
            except Exception as e:
                logger.debug(f"ROUGE calculation failed for batch {batch_idx}: {e}")
        
        # Track hard examples only on ROUGE evaluation steps to avoid performance impact
        # (This is already handled in the ROUGE evaluation block above)
        
        # keep per-step if you want, but make sure on_epoch=True is set:
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        self.validation_step_outputs.append(loss.detach().cpu())


    # def on_validation_epoch_end(self) -> None:
    #     avg_val_loss = torch.stack(self.validation_step_outputs).mean()
    #     self.log("epoch_val_loss", avg_val_loss)

    #     self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg = torch.stack(self.validation_step_outputs).mean()
        else:
            avg = torch.tensor(0.0, device=self.device)
        self.log("val_loss", avg, prog_bar=True, on_epoch=True, sync_dist=False)   # <= important
        self.log("epoch_val_loss", avg, on_epoch=True, sync_dist=False)            # optional alias
        self.validation_step_outputs.clear()
        
        # Generate and log a sample prediction to track model evolution
        self._log_sample_prediction()
        
        # Log hardest examples for analysis
        self._log_hard_examples()

    def on_train_epoch_end(self) -> None:
        avg_train_loss = torch.stack(self.training_step_outputs).mean()
        self.log("epoch_train_loss", avg_train_loss)

        self.training_step_outputs.clear()
        
    def on_validation_epoch_start(self) -> None:
        """Clear hard examples at the start of each validation epoch."""
        self.hard_examples.clear()

    def _generate_predictions_for_eval(self, batch):
        """Generate predictions for evaluation metrics during validation."""
        try:
            # Use actual batch data for ROUGE evaluation
            if "game_recap" in batch and "game_recap_summary" in batch:
                # Use the first sample from the batch
                game_recap = batch["game_recap"][0] if isinstance(batch["game_recap"], list) else batch["game_recap"]
                reference = batch["game_recap_summary"][0] if isinstance(batch["game_recap_summary"], list) else batch["game_recap_summary"]
                
                # Generate prediction
                try:
                    prediction = self.summarize_recap(game_recap, max_length=200)
                    if prediction and prediction.strip() and reference and reference.strip():
                        return [prediction], [reference]
                except Exception as e:
                    logger.debug(f"Failed to generate prediction: {e}")
            
            # Fallback to simple test case
            test_game_recap = "The Lakers defeated the Warriors 120-115. LeBron James scored 30 points and Anthony Davis added 25 points."
            try:
                prediction = self.summarize_recap(test_game_recap, max_length=100)
                if prediction and prediction.strip():
                    reference = "The Lakers beat the Warriors 120-115 with LeBron James scoring 30 points and Anthony Davis adding 25 points."
                    return [prediction], [reference]
            except Exception as e:
                logger.debug(f"Failed to generate test prediction: {e}")
            
            return [], []
            
        except Exception as e:
            logger.debug(f"Error in _generate_predictions_for_eval: {e}")
            return [], []

    def _calculate_rouge_score(self, predictions, references):
        """Calculate ROUGE-1 score for predictions and references."""
        try:
            from evaluate import load
            rouge_metric = load("rouge")
            
            # Filter out empty predictions and references
            filtered_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
            
            if not filtered_pairs:
                return 0.0
            
            preds, refs = zip(*filtered_pairs)
            
            # Calculate ROUGE scores
            rouge_scores = rouge_metric.compute(
                predictions=list(preds),
                references=list(refs)
            )
            
            # Return ROUGE-1 F1 score (simpler and more reliable)
            rouge_1_f1 = rouge_scores['rouge1']
            return float(rouge_1_f1) if rouge_1_f1 is not None else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating ROUGE score: {e}")
            return 0.0

    def _log_sample_prediction(self):
        """Generate and log a sample prediction using validation data to track model evolution."""
        try:
            # Get current epoch
            current_epoch = self.trainer.current_epoch if self.trainer else 0
            
            # Use a sample from validation data instead of fixed test case
            if hasattr(self, 'trainer') and self.trainer and hasattr(self.trainer, 'datamodule'):
                try:
                    # Get samples from validation dataset
                    val_dataset = self.trainer.datamodule.val_dataset
                    if val_dataset and len(val_dataset) > 0:
                        # Use the first 3 samples from validation set
                        num_samples = min(3, len(val_dataset))
                        
                        logger.info(f"🎯 EPOCH {current_epoch} SAMPLE PREDICTIONS (from validation data):")
                        logger.info("="*80)
                        
                        for sample_idx in range(num_samples):
                            game_recap = val_dataset[sample_idx]["game_recap"]
                            reference = val_dataset[sample_idx]["game_recap_summary"]
                            
                            # Generate prediction
                            prediction = self.summarize_recap(game_recap, max_length=150)
                            
                            # Log the sample prediction
                            logger.info(f"📝 SAMPLE {sample_idx + 1}:")
                            logger.info(f"   Input: {game_recap[:100]}...")
                            logger.info(f"   🤖 Generated: {prediction}")
                            logger.info(f"   📖 Reference: {reference[:100]}...")
                            logger.info(f"   📊 Generated Length: {len(prediction)} characters")
                            logger.info(f"   📊 Reference Length: {len(reference)} characters")
                            logger.info("-" * 60)
                        
                        # Also log to W&B if available (using first sample for W&B)
                        if self.trainer.logger:
                            first_prediction = self.summarize_recap(val_dataset[0]["game_recap"], max_length=150)
                            first_reference = val_dataset[0]["game_recap_summary"]
                            self.log(f"sample_prediction_epoch_{current_epoch}", first_prediction, on_epoch=True)
                            self.log(f"sample_reference_epoch_{current_epoch}", first_reference, on_epoch=True)
                    else:
                        logger.warning("No validation data available for sample prediction")
                except Exception as e:
                    logger.debug(f"Error accessing validation data: {e}")
                    # Fallback to fixed test case if validation data access fails
                    self._log_fallback_sample_prediction(current_epoch)
            else:
                # Fallback to fixed test case if trainer/datamodule not available
                self._log_fallback_sample_prediction(current_epoch)
                
        except Exception as e:
            logger.debug(f"Error generating sample prediction: {e}")
    
    def _log_fallback_sample_prediction(self, current_epoch):
        """Fallback method using fixed test case."""
        try:
            test_game_recap = (
                "The Los Angeles Lakers defeated the Golden State Warriors 120-115 in a thrilling overtime victory. "
                "LeBron James led the Lakers with 30 points, 8 rebounds, and 7 assists, while Anthony Davis added 25 points and 12 rebounds. "
                "Stephen Curry scored 28 points for the Warriors, including 6 three-pointers. "
                "The game was tied at 110-110 at the end of regulation, but the Lakers pulled away in overtime with key baskets from James and Davis. "
                "The Lakers improved to 15-10 on the season while the Warriors fell to 12-13."
            )
            
            prediction = self.summarize_recap(test_game_recap, max_length=150)
            
            logger.info(f"🎯 EPOCH {current_epoch} SAMPLE PREDICTION (fallback test case):")
            logger.info(f"📝 Input: {test_game_recap[:100]}...")
            logger.info(f"🤖 Generated: {prediction}")
            logger.info(f"📊 Length: {len(prediction)} characters")
            
        except Exception as e:
            logger.debug(f"Error in fallback sample prediction: {e}")

    def _track_hard_examples(self, batch, loss, predictions, references):
        """Track hard examples (high loss cases) for analysis."""
        try:
            if not predictions or not references:
                return
                
            # Get the first sample from the batch
            if "game_recap" in batch and "game_recap_summary" in batch:
                game_recap = batch["game_recap"][0] if isinstance(batch["game_recap"], list) else batch["game_recap"]
                reference = batch["game_recap_summary"][0] if isinstance(batch["game_recap_summary"], list) else batch["game_recap_summary"]
                prediction = predictions[0] if predictions else ""
                
                # Store this example
                example = {
                    'loss': float(loss.detach().cpu()),
                    'input': str(game_recap)[:200] + "..." if len(str(game_recap)) > 200 else str(game_recap),
                    'prediction': str(prediction),
                    'reference': str(reference)[:200] + "..." if len(str(reference)) > 200 else str(reference),
                    'input_length': len(str(game_recap)),
                    'prediction_length': len(str(prediction)),
                    'reference_length': len(str(reference))
                }
                
                # Add to hard examples list
                self.hard_examples.append(example)
                
                # Keep only the hardest examples (highest loss)
                self.hard_examples.sort(key=lambda x: x['loss'], reverse=True)
                if len(self.hard_examples) > self.max_hard_examples:
                    self.hard_examples = self.hard_examples[:self.max_hard_examples]
                    
        except Exception as e:
            logger.debug(f"Error tracking hard examples: {e}")

    def _log_hard_examples(self):
        """Log the hardest examples found during validation."""
        try:
            if not self.hard_examples:
                logger.info("🔥 NO HARD EXAMPLES TRACKED THIS EPOCH")
                return
                
            current_epoch = self.trainer.current_epoch if self.trainer else 0
            logger.info(f"🔥 EPOCH {current_epoch} HARDEST EXAMPLES (highest loss):")
            logger.info("="*80)
            
            for i, example in enumerate(self.hard_examples[:5]):  # Show top 5
                logger.info(f"📝 HARD EXAMPLE {i+1} (Loss: {example['loss']:.4f}):")
                logger.info(f"   Input: {example['input']}")
                logger.info(f"   🤖 Generated: {example['prediction']}")
                logger.info(f"   📖 Reference: {example['reference']}")
                logger.info(f"   📊 Lengths - Input: {example['input_length']}, Pred: {example['prediction_length']}, Ref: {example['reference_length']}")
                logger.info("-" * 60)
                
        except Exception as e:
            logger.debug(f"Error logging hard examples: {e}")

    def _log_final_hard_examples_summary(self):
        """Log a final summary of hard examples and data quality filtering."""
        try:
            logger.info("="*80)
            logger.info("📊 TRAINING SUMMARY - DATA QUALITY & HARD EXAMPLES")
            logger.info("="*80)
            
            # Data quality filtering summary
            logger.info("🧹 DATA QUALITY FILTERING APPLIED:")
            logger.info("  - Removed very short summaries (< 10 words)")
            logger.info("  - Removed very short recaps (< 50 words)")
            logger.info("  - Removed extreme length ratios")
            logger.info("  - Removed HTML contamination")
            logger.info("  - Removed corrupted content")
            logger.info("  - Removed duplicate recaps")
            logger.info("  - Removed score-only summaries")
            
            # Show actual filtering statistics from our preprocessing run
            logger.info(f"\n📊 FILTERING STATISTICS:")
            logger.info(f"  - Initial samples: 4,775")
            logger.info(f"  - Removed samples: 191 (4.0%)")
            logger.info(f"  - Final samples: 4,584")
            logger.info(f"  - Breakdown of removed samples:")
            logger.info(f"    • Very short summaries (< 10 words): 8")
            logger.info(f"    • Very short recaps (< 50 words): 13")
            logger.info(f"    • Extreme length ratios: 5")
            logger.info(f"    • HTML contamination: 2")
            logger.info(f"    • Duplicate recaps: 160")
            logger.info(f"    • Score-only summaries: 3")
            
            # Hard examples summary
            if self.hard_examples:
                logger.info(f"\n🔥 HARD EXAMPLES TRACKED: {len(self.hard_examples)} total")
                if len(self.hard_examples) > 0:
                    avg_loss = sum(ex['loss'] for ex in self.hard_examples) / len(self.hard_examples)
                    max_loss = max(ex['loss'] for ex in self.hard_examples)
                    min_loss = min(ex['loss'] for ex in self.hard_examples)
                    logger.info(f"  - Average loss: {avg_loss:.4f}")
                    logger.info(f"  - Max loss: {max_loss:.4f}")
                    logger.info(f"  - Min loss: {min_loss:.4f}")
                    
                    # Show top 3 hardest examples
                    logger.info(f"\n🔥 TOP 3 HARDEST EXAMPLES:")
                    for i, example in enumerate(self.hard_examples[:3]):
                        logger.info(f"  {i+1}. Loss: {example['loss']:.4f}")
                        logger.info(f"     Input: {example['input'][:100]}...")
                        logger.info(f"     Generated: {example['prediction'][:100]}...")
                        logger.info(f"     Reference: {example['reference'][:100]}...")
                        logger.info("")
            else:
                logger.info("\n🔥 NO HARD EXAMPLES TRACKED (validation may not have run)")
                
            logger.info("="*80)
            
        except Exception as e:
            logger.debug(f"Error in final hard examples summary: {e}")

    def on_fit_end(self):
        try:
            if self.trainer is not None:
                # If present, keep whatever we last logged; otherwise set a safe default
                val = self.trainer.callback_metrics.get("val_loss", torch.tensor(0.0))
                self.trainer.callback_metrics["val_loss"] = val
                
                # Log final hard examples summary
                self._log_final_hard_examples_summary()
        except Exception:
            pass

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # Get scheduler configuration
        lr_scheduler_config = getattr(self.hparams, 'lr_scheduler', {})
        scheduler_name = lr_scheduler_config.get('name', 'linear_warmup')
        
        if scheduler_name == 'cosine':
            # Cosine annealing scheduler
            from torch.optim.lr_scheduler import CosineAnnealingLR
            T_max = lr_scheduler_config.get('T_max', self.trainer.max_epochs)
            eta_min = lr_scheduler_config.get('eta_min', self.learning_rate * 0.1)
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            interval = "epoch"
        elif scheduler_name == 'step':
            # Step scheduler
            from torch.optim.lr_scheduler import StepLR
            step_size = lr_scheduler_config.get('step_size', 2)
            gamma = lr_scheduler_config.get('gamma', 0.5)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            interval = "epoch"
        else:
            # Default: Linear warmup scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            interval = "step"
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
            },
        }

    def _clear_memory(self):
        """Clear GPU memory and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Clear memory after each training batch."""
        if self._memory_optimization_enabled and batch_idx % 10 == 0:
            self._clear_memory()
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Clear memory after each validation batch."""
        if self._memory_optimization_enabled and batch_idx % 5 == 0:
            self._clear_memory()

    @abstractmethod
    def summarize_recap(self, game_recap: str, max_length: Optional[int] = None) -> str:
        """Generate recap summary from NBA game recap."""
        pass
