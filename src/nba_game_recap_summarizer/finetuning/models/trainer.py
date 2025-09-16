import os
import time
from typing import Dict, Any

import torch
from loguru import logger
from omegaconf import DictConfig
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator


class SummarizationModelTrainer:
    """Pure PyTorch trainer for summarization models without PyTorch Lightning dependencies."""
    
    def __init__(self, model, dataloaders: Dict[str, DataLoader], config: DictConfig):
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        
        # Setup training
        self.model.setup_training()
        
        # Setup Accelerate for multi-GPU training
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # Get optimizer and scheduler
        num_training_steps = len(dataloaders['train']) * config.training.max_epochs
        self.optimizer, self.scheduler = model.get_optimizer_and_scheduler(
            num_training_steps=num_training_steps,
            lr_scheduler_config=config.training.get('lr_scheduler', {})
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Prepare model, optimizer, and dataloaders with Accelerate
        self.model, self.optimizer, self.dataloaders['train'], self.dataloaders['val'] = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloaders['train'], self.dataloaders['val']
        )
        
        logger.info(f"Training on device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.dataloaders['train']):
            # Forward pass with Accelerate
            with self.accelerator.accumulate(self.model):
                loss = self.model.compute_loss(batch)
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if hasattr(self.config.training, 'gradient_clip_val'):
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_val)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Clear memory if needed
            self.model.clear_memory_if_needed(batch_idx, is_validation=False)
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"train_loss": avg_loss}

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_rouge = 0.0
        num_batches = 0
        rouge_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloaders['val']):
                # Accelerate handles device placement automatically
                
                # Compute metrics
                metrics = self.model.compute_validation_metrics(batch, batch_idx)
                
                # Update metrics
                total_loss += metrics.get("val_loss", 0.0)
                if "val_rouge_1_2" in metrics:
                    total_rouge += metrics["val_rouge_1_2"]
                    rouge_count += 1
                
                num_batches += 1
                
                # Clear memory if needed
                self.model.clear_memory_if_needed(batch_idx, is_validation=True)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_rouge = total_rouge / rouge_count if rouge_count > 0 else 0.0
        
        return {
            "val_loss": avg_loss,
            "val_rouge_1_2": avg_rouge
        }

    def save_checkpoint(self, checkpoint_path: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'hyper_parameters': {
                'model_name': self.model.model_name,
                'model_type': self.model.model_type,
                'learning_rate': self.model.learning_rate,
                'warmup_steps': self.model.warmup_steps,
                'weight_decay': self.model.weight_decay,
                'use_quantization': self.model.use_quantization,
                'quantization_type': self.model.quantization_type,
                'peft_method': self.model.peft_method,
            }
        }
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = checkpoint_path.replace('.ckpt', '_best.ckpt')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")

    def train(self):
        """Main training loop."""
        logger.info("Starting training")
        
        for epoch in range(self.config.training.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics['epoch'] = epoch
            
            # Log metrics
            logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['train_loss']:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
            if 'val_rouge_1_2' in val_metrics:
                logger.info(f"Epoch {epoch} - Val ROUGE-1/2: {val_metrics['val_rouge_1_2']:.4f}")
            
            # Log to wandb
            wandb.log(all_metrics)
            
            # Check for best model
            val_loss = val_metrics['val_loss']
            is_best = val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                logger.info(f"New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                self.config.training.model_artifact_dir, 
                f"{os.getenv('PIPELINE_RUN_ID', 'pipeline_id')}/checkpoints/model_epoch_{epoch}.ckpt"
            )
            self.save_checkpoint(checkpoint_path, is_best=is_best)
            
            # Also save as best_model.ckpt if this is the best model
            if is_best:
                best_checkpoint_path = os.path.join(
                    self.config.training.model_artifact_dir, 
                    f"{os.getenv('PIPELINE_RUN_ID', 'pipeline_id')}/checkpoints/best_model.ckpt"
                )
                self.save_checkpoint(best_checkpoint_path, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        # Setup for inference
        self.model.setup_inference()
        logger.info("Training completed successfully")
