import os
import gc
import json
import pandas as pd
import torch
from omegaconf import DictConfig

from loguru import logger
from typing import Dict
import wandb

from text2cypher.finetuning.data.notechat_dataset import NoteChatDataModule
from text2cypher.finetuning.utils.load_models import load_model
from text2cypher.finetuning.eval.metrics import (
    calculate_rouge, calculate_bleu,
    calculate_bertscore,
    calculate_factual_consistency, calculate_relevance,
    calculate_completeness, calculate_conciseness, calculate_clarity,
    compute_group_metrics, calculate_average_latency, calculate_model_size_in_params
)

def setup_dataloader(cfg, samples, env_folder):
    data_module = NoteChatDataModule(
        model_name=cfg.model.name,
        preprocessed_input_data_folder=cfg.data.preprocessed_input_data_folder,
        source_data_path=cfg.data.source_data_path,
        env_folder=env_folder,
        batch_size=cfg.training.batch_size,
        max_length=cfg.model.max_length,
        num_workers=cfg.training.num_workers,
        train_samples=1,
        val_samples=1,
        test_samples=samples,
        shuffle=cfg.data.shuffle,
        shuffle_seed=cfg.data.shuffle_seed,
    )
    data_module.setup()
    return data_module.test_dataloader()

def run_metric_evaluation(title, cfg, dataloader_samples, metrics_dict, model, device, env_folder):
    logger.info(f"Starting {title} evaluation")
    dataloader = setup_dataloader(cfg, dataloader_samples, env_folder)
    results_df = compute_group_metrics(model, dataloader, device, cfg.model.max_length, metrics_dict)
    del dataloader
    gc.collect()
    return results_df

def evaluate_model(cfg: DictConfig):
    env_folder = os.getenv("ENV", "no-env")
    pipeline_run_id = os.getenv("PIPELINE_RUN_ID", "no-pipeline-id")

    with wandb.init(project=f"{cfg.project_name}-evaluation-{env_folder}", name=f"{cfg.model.name}-{cfg.model.peft_method}", tags=[f"pipeline:{pipeline_run_id}"]) as run:
        logger.info("Starting computing evaluation metrics")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Defining evaluation metrics to be computed")
        lexical_metrics_dict = {
            "rouge_score": calculate_rouge,
            "bleu_score": calculate_bleu,
        }
        semantical_metrics_dict = {
            "bert_score": calculate_bertscore
        }
        ai_as_a_judge_metrics_dict = {
            "factual_consistency": calculate_factual_consistency,
            "relevance": calculate_relevance,
            "completeness": calculate_completeness,
            "conciseness": calculate_conciseness,
            "clarity": calculate_clarity,
        }

        logger.info("Loading model from checkpoint")

        model_ckpt = f"{cfg.evaluation.model_artifact_dir}/{pipeline_run_id}/checkpoints/best_model.ckpt"

        model_name = cfg.model.name
        model_type = cfg.model.type
        peft_method = cfg.model.peft_method

        model = load_model(model_ckpt, model_name, model_type, device, peft_method)

        results = []

        model_name_df = pd.DataFrame({"pipeline_run_id": [pipeline_run_id]})
        results.append(model_name_df)

        # Lexical metrics #def run_metric_evaluation(title, cfg, dataloader_samples, metrics_dict, model, device):
        logger.info("Computing lexical metrics")
        lexical_metrics_results_df = run_metric_evaluation(
            "lexical metrics", cfg, cfg.evaluation.test_samples_lexical_metrics, lexical_metrics_dict, model, device, env_folder
        )
        results.append(lexical_metrics_results_df)

        logger.info("Computing semantic metrics")
        semantical_metrics_results_df = run_metric_evaluation(
            "semantical metrics", cfg, cfg.evaluation.test_samples_semantic_metrics, semantical_metrics_dict, model, device, env_folder
        )
        results.append(semantical_metrics_results_df)

        logger.info("Computing AI as a Judge metrics")
        ai_as_a_judge_metrics_results_df = run_metric_evaluation(
            "ai as a judge metrics", cfg, cfg.evaluation.test_samples_ai_as_judge_metrics, ai_as_a_judge_metrics_dict, model, device, env_folder
        )
        results.append(ai_as_a_judge_metrics_results_df)

        logger.info("Computing system metrics")
        dataloader = setup_dataloader(cfg, cfg.evaluation.test_samples_semantic_metrics, env_folder)
        latency = calculate_average_latency(model, dataloader, cfg.model.max_length)
        size_params = calculate_model_size_in_params(model)
        system_metrics_df = pd.DataFrame({
            "model_size_params": [size_params],
            "avg_latency_sec": [latency],
        })
        results.append(system_metrics_df)

        del dataloader, model
        gc.collect()

        model_metrics_df = pd.concat(results, axis=1)
        wandb.log({"evaluation_results": wandb.Table(dataframe=model_metrics_df)})
        reports_folder = f'{cfg.training.model_artifact_dir}/{pipeline_run_id}/reports'
        os.makedirs(reports_folder, exist_ok=True)
        logger.info(f"Contents of {f'{cfg.training.model_artifact_dir}/{pipeline_run_id}'}: {os.listdir(f'{cfg.training.model_artifact_dir}/{pipeline_run_id}')}")  
        model_metrics_df.to_json(f'{reports_folder}/eval_metrics.json', lines=True, orient='records')
        run.finish()
        logger.info("Finished computing evaluation metrics")

if __name__ == "__main__":
    evaluate_model()
