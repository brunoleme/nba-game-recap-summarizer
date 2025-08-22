import datetime
import platform
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from loguru import logger
from peft import LoraConfig, PromptTuningConfig
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BitsAndBytesConfig, get_linear_schedule_with_warmup

class BaseNoteGenerationModel(pl.LightningModule, ABC):
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
            self.peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q", "v"],
                task_type="SEQ_2_SEQ_LM",
            )
        elif self.peft_method == "prompt_tuning":
            self.peft_config = PromptTuningConfig(
                num_virtual_tokens=prompt_tuning_n_tokens,
                task_type="SEQ_2_SEQ_LM"
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

    @abstractmethod
    def forward(self, **inputs) -> Any:
        """Forward pass of the model."""
        pass

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(**batch)
        loss = outputs.loss

        self.log("train_loss", loss, prog_bar=True)
        self.training_step_outputs.append(loss.detach().cpu())

        return loss


    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(**batch)
        loss = outputs.loss

        self.log("val_loss", loss, prog_bar=True)
        self.validation_step_outputs.append(loss.detach().cpu())


    def on_validation_epoch_end(self) -> None:
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("epoch_val_loss", avg_val_loss)

        self.validation_step_outputs.clear()

    def on_train_epoch_end(self) -> None:
        avg_train_loss = torch.stack(self.training_step_outputs).mean()
        self.log("epoch_train_loss", avg_train_loss)

        self.training_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    @abstractmethod
    def generate_note(self, conversation: str, max_length: Optional[int] = None) -> str:
        """Generate clinical note from conversation."""
        pass
