from loguru import logger
from omegaconf import DictConfig
import os


from text2cypher.finetuning.data.notechat_preprocessing import NoteChatDataPreprocessingModule
from text2cypher.finetuning.utils.logger import setup_logger


def preprocessing(cfg: DictConfig):
    setup_logger(cfg.logging.log_path)
    logger.info(f"Starting preprocessing pipeline")

    env_folder = os.getenv("ENV", "no-env")
    preprocessingmodule = NoteChatDataPreprocessingModule(
        model_name=cfg.model.name,
        source_data_folder=cfg.data.source_data_folder,
        source_data_path=cfg.data.source_data_path,
        env_folder = env_folder,
        preprocessed_output_data_folder=cfg.data.preprocessed_output_data_folder,
        max_length=cfg.model.max_length,
        train_samples=cfg.data.train_samples,
        val_samples=cfg.data.val_samples,
        test_samples=cfg.data.test_samples,
        shuffle=cfg.data.shuffle,
        shuffle_seed=cfg.data.shuffle_seed,
    )

    preprocessingmodule.run()
    logger.info(f"Data preprocessing finished")


if __name__ == "__main__":
    preprocessing()
