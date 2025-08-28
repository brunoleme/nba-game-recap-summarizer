import os
import hydra
from omegaconf import OmegaConf
from loguru import logger

from nba_game_recap_summarizer.finetuning.train import train
from nba_game_recap_summarizer.finetuning.evaluate_model import evaluate_model


def main():
    env = os.environ.get("ENV", "dev")
    config_name = f"config.{env}"
    config_path = os.path.abspath("src/nba_game_recap_summarizer/finetuning/config")

    with hydra.initialize_config_dir(config_dir=config_path):
        cfg = hydra.compose(config_name=config_name)

    logger.info(f"🚀 Starting pipeline with config:\n{OmegaConf.to_yaml(cfg)}")

    logger.info("📦 Training model...")
    train(cfg)

    logger.info("📊 Running evaluation...")
    evaluate_model(cfg)

    logger.success("✅ Pipeline completed!")


if __name__ == "__main__":
    main()
