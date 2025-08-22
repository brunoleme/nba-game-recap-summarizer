import os
import hydra
from omegaconf import OmegaConf
from text2cypher.finetuning.preprocessing import preprocessing


def main():
    env = os.environ.get("ENV", "dev")
    config_name = f"config.{env}"
    config_path = os.path.abspath("src/text2cypher/finetuning/config")

    with hydra.initialize_config_dir(config_dir=config_path):
        cfg = hydra.compose(config_name=config_name)

    print(OmegaConf.to_yaml(cfg))
    preprocessing(cfg)


if __name__ == "__main__":
    main()
