import os
import gc
import json
import time
import pandas as pd
import torch
from omegaconf import DictConfig

from loguru import logger
from typing import Dict
import wandb

from nba_game_recap_summarizer.finetuning.data.nba_recap_dataset import NBARecapDataModule
from nba_game_recap_summarizer.finetuning.utils.load_models import load_model
from nba_game_recap_summarizer.finetuning.eval.metrics import (
    calculate_rouge, calculate_bleu,
    calculate_bertscore,
    calculate_factual_consistency, calculate_relevance,
    calculate_completeness, calculate_conciseness, calculate_clarity,
    compute_group_metrics, calculate_average_latency, calculate_model_size_in_params
)

def setup_dataloader(cfg, samples, env_folder):
    data_module = NBARecapDataModule(
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

def run_batch_metric_evaluation(cfg, model, device, env_folder):
    """Run all metrics on the same dataloader to avoid reloading data and model"""
    start_time = time.time()
    
    logger.info("Starting batch metric evaluation")
    
    # Use the maximum number of samples needed for any metric
    max_samples = max(
        cfg.evaluation.test_samples_lexical_metrics,
        cfg.evaluation.test_samples_semantic_metrics,
        cfg.evaluation.test_samples_ai_as_judge_metrics
    )
    
    logger.info(f"Setting up dataloader for {max_samples} samples")
    dataloader_start = time.time()
    dataloader = setup_dataloader(cfg, max_samples, env_folder)
    dataloader_time = time.time() - dataloader_start
    logger.info(f"Dataloader setup completed in {dataloader_time:.2f}s")
    
    # Get all predictions at once (this is the expensive part)
    logger.info("Generating predictions for all samples")
    prediction_start = time.time()
    predictions = model.summarize_recaps(dataloader, max_length=cfg.model.max_length)
    prediction_time = time.time() - prediction_start
    logger.info(f"Text generation completed in {prediction_time:.2f}s for {len(predictions)} samples")
    if len(predictions) > 0:
        logger.info(f"Average time per sample: {prediction_time/len(predictions):.2f}s")
    else:
        logger.warning("No predictions generated, cannot calculate average time per sample")
    
    # Extract references and instructions
    all_references = []
    all_instructions = []
    
    for batch in dataloader:
        inputs = {key: value.to(device) for key, value in batch.items()}
        
        batch_references = [
            model.tokenizer.decode(reference[reference != -100], skip_special_tokens=True)
            for reference in inputs["labels"]
        ]
        all_references.extend(batch_references)
        
        batch_instructions = [
            model.tokenizer.decode(input_id, skip_special_tokens=True)
            for input_id in inputs["input_ids"]
        ]
        all_instructions.extend(batch_instructions)
    
    # Limit to the actual number of predictions generated
    actual_samples = len(predictions)
    all_references = all_references[:actual_samples]
    all_instructions = all_instructions[:actual_samples]
    
    logger.info(f"Generated {actual_samples} predictions, processing metrics")
    
    # Compute all metrics using the same predictions
    results = {}
    
    # Lexical metrics
    if cfg.evaluation.test_samples_lexical_metrics > 0:
        lexical_samples = min(cfg.evaluation.test_samples_lexical_metrics, actual_samples)
        lexical_metrics_dict = {
            "rouge_score": calculate_rouge,
            "bleu_score": calculate_bleu,
        }
        lexical_predictions = predictions[:lexical_samples]
        lexical_references = all_references[:lexical_samples]
        lexical_instructions = all_instructions[:lexical_samples]
        
        for metric_name, metric_fn in lexical_metrics_dict.items():
            results[f"lexical_{metric_name}"] = metric_fn(lexical_predictions, lexical_references, lexical_instructions)
    
    # Semantic metrics
    if cfg.evaluation.test_samples_semantic_metrics > 0:
        semantic_samples = min(cfg.evaluation.test_samples_semantic_metrics, actual_samples)
        semantic_metrics_dict = {
            "bert_score": calculate_bertscore
        }
        semantic_predictions = predictions[:semantic_samples]
        semantic_references = all_references[:semantic_samples]
        semantic_instructions = all_instructions[:semantic_samples]
        
        for metric_name, metric_fn in semantic_metrics_dict.items():
            results[f"semantic_{metric_name}"] = metric_fn(semantic_predictions, semantic_references, semantic_instructions)
    
    # AI as a Judge metrics
    if cfg.evaluation.test_samples_ai_as_judge_metrics > 0:
        ai_samples = min(cfg.evaluation.test_samples_ai_as_judge_metrics, actual_samples)
        ai_metrics_dict = {
            "factual_consistency": calculate_factual_consistency,
            "relevance": calculate_relevance,
            "completeness": calculate_completeness,
            "conciseness": calculate_conciseness,
            "clarity": calculate_clarity,
        }
        ai_predictions = predictions[:ai_samples]
        ai_references = all_references[:ai_samples]
        ai_instructions = all_instructions[:ai_samples]
        
        for metric_name, metric_fn in ai_metrics_dict.items():
            results[f"ai_judge_{metric_name}"] = metric_fn(ai_predictions, ai_references, ai_instructions)
    
    del dataloader
    gc.collect()
    
    total_time = time.time() - start_time
    logger.info(f"Batch metric evaluation completed in {total_time:.2f}s")
    
    return pd.DataFrame(results, index=[0])

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

        # Run all metrics in batch to avoid reloading model and data
        logger.info("Computing all metrics in batch")
        batch_metrics_results_df = run_batch_metric_evaluation(cfg, model, device, env_folder)
        results.append(batch_metrics_results_df)

        logger.info("Computing system metrics")
        system_start = time.time()
        
        size_start = time.time()
        size_params = calculate_model_size_in_params(model)
        size_time = time.time() - size_start
        logger.info(f"Model size calculation completed in {size_time:.2f}s")
        
        # Calculate latency using the already-generated predictions for efficiency
        latency_start = time.time()
        if len(predictions) > 0:
            # Use a small subset of existing predictions to calculate latency
            latency_samples = min(3, len(predictions))
            logger.info(f"Calculating latency using {latency_samples} existing predictions")
            
            # Calculate latency from the batch generation we already did
            # This avoids reloading the model and regenerating text
            latency = prediction_time / len(predictions)  # Average time per sample from batch generation
        else:
            logger.warning("No predictions available for latency calculation")
            latency = 0.0
            
        latency_time = time.time() - latency_start
        logger.info(f"Latency calculation completed in {latency_time:.2f}s")
        
        system_time = time.time() - system_start
        logger.info(f"All system metrics completed in {system_time:.2f}s")
        
        system_metrics_df = pd.DataFrame({
            "model_size_params": [size_params],
            "avg_latency_sec": [latency],
        })
        results.append(system_metrics_df)
        
        del latency_dataloader

        del model
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
