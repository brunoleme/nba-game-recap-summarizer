import os
import torch
import gc
import hydra
from omegaconf import OmegaConf

from nba_game_recap_summarizer.finetuning.train import train
from nba_game_recap_summarizer.finetuning.pre_training import pre_train

# Set memory optimization environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def clear_memory():
    """Clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    env = os.environ.get("ENV", "dev")
    config_name = f"config.{env}"
    config_path = os.path.abspath("src/nba_game_recap_summarizer/finetuning/config")

    with hydra.initialize_config_dir(config_dir=config_path):
        cfg = hydra.compose(config_name=config_name)

    print(OmegaConf.to_yaml(cfg))
    
    # Clear memory before starting
    clear_memory()
    
    try:
        # Set additional memory optimizations
        torch.backends.cudnn.benchmark = False  # Disable for memory stability
        torch.backends.cudnn.deterministic = True
        
        pre_train(cfg)
        train(cfg)
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA OOM Error: {e}")
        print("💡 Suggestions:")
        print("   - Reduce batch_size further (try 1)")
        print("   - Reduce max_length further (try 1024)")
        print("   - Increase accumulate_grad_batches")
        print("   - Use gradient_checkpointing=True")
        raise
    finally:
        # Clean up memory
        clear_memory()


if __name__ == "__main__":
    main()
