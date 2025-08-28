import os
import hydra
from omegaconf import OmegaConf

from nba_game_recap_summarizer.finetuning.train import train
from nba_game_recap_summarizer.finetuning.pre_training import pre_train
from nba_game_recap_summarizer.finetuning.post_training import package_model


def main():
    env = os.environ.get("ENV", "dev")
    config_name = f"config.{env}"
    config_path = os.path.abspath("src/nba_game_recap_summarizer/finetuning/config")

    with hydra.initialize_config_dir(config_dir=config_path):
        cfg = hydra.compose(config_name=config_name)

    print(OmegaConf.to_yaml(cfg))
    pre_train(cfg)
    train(cfg)
    package_model(cfg)


if __name__ == "__main__":
    main()
