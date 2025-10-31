import os
import argparse
import hydra
from omegaconf import OmegaConf

from nba_game_recap_summarizer.finetuning.evaluate_dpo import evaluate_dpo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="src/nba_game_recap_summarizer/finetuning/config")
    parser.add_argument("--config-name", default=None)
    args = parser.parse_args()

    env = os.environ.get("ENV", "dev")
    config_name = args.config_name or f"config.{env}"
    config_path = os.path.abspath(args.config_path)

    with hydra.initialize_config_dir(config_dir=config_path):
        cfg = hydra.compose(config_name=config_name)

    print(OmegaConf.to_yaml(cfg))
    evaluate_dpo(cfg)


if __name__ == "__main__":
    main()


