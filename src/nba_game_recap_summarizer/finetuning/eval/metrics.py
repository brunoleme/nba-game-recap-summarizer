from evaluate import load
import time
import pandas as pd
from typing import Dict, Callable
import numpy as np
import os

from torch.utils.data import DataLoader
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from nba_game_recap_summarizer.finetuning.eval.prompts import (
    factual_consistency_prompt,
    relevance_prompt,
    completeness_prompt,
    conciseness_prompt,
    clarity_prompt,
)

# Load evaluation metrics
rouge_metric = load("rouge")
bleu_metric = load("bleu")
bertscore_metric = load("bertscore")

# --- Helper classes & functions ---
class ResponseFormatter(BaseModel):
    score: int = Field(description="The score measured by the evaluation model")

def init_grader(prompt_template) -> Callable:
    api_key = os.getenv("OPENAI_API_KEY")
    # Add timeout to prevent hanging (30 seconds per request)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0, 
        api_key=api_key,
        timeout=30.0,  # 30 second timeout per request
        max_retries=2  # Retry up to 2 times
    ).bind_tools([ResponseFormatter])
    return prompt_template | llm

def evaluate_with_grader(grader, format_fn, *data_streams, metric_name="unknown"):
    """Evaluate with grader, adding progress logging and timeout handling."""
    from loguru import logger
    
    # Return 0 if grader is None (no API key available)
    if grader is None:
        return 0.0
    
    # Convert to lists to avoid consuming iterator
    data_lists = [list(stream) for stream in data_streams] if data_streams else []
    total = len(data_lists[0]) if data_lists else 0
    
    if total == 0:
        return 0.0
    
    logger.info(f"Calculating {metric_name} metric for {total} samples (this may take a few minutes)...")
    
    grades = []
    for idx in range(total):
        args = tuple(stream[idx] for stream in data_lists)
        prompt_input = format_fn(*args)
        try:
            # Log progress every 5 samples
            if (idx + 1) % 5 == 0 or idx == 0:
                logger.info(f"  {metric_name}: Processing sample {idx + 1}/{total}")
            
            response = grader.invoke(prompt_input)
            if hasattr(response, "tool_calls") and response.tool_calls:
                score = float(response.tool_calls[0]["args"]["score"])
                if 1 <= score <= 5:
                    grades.append(score)
        except Exception as e:
            logger.warning(f"  {metric_name}: Failed to evaluate sample {idx + 1}: {e}")
            continue
    
    result = np.mean(grades) if grades else 0.0
    logger.info(f"  {metric_name}: Completed {len(grades)}/{total} samples, average score: {result:.2f}")
    return result

# --- Classical metrics ---
def calculate_rouge(predictions, references, _):
    try:
        scores = rouge_metric.compute(predictions=predictions, references=references)
        # Use ROUGE-1 and ROUGE-2 F1 scores (more reliable than ROUGE-L)
        rouge_1_f1 = scores['rouge1']
        rouge_2_f1 = scores['rouge2']
        return (rouge_1_f1 + rouge_2_f1) / 2.0
    except Exception:
        return 0

def calculate_bleu(predictions, references, _):
    try:
        return bleu_metric.compute(predictions=predictions, references=references)['bleu']
    except Exception:
        return 0

def calculate_bertscore(predictions, references, _):
    try:
        return np.mean(bertscore_metric.compute(predictions=predictions, references=references, lang="en")['f1'])
    except Exception:
        return 0

# --- LLM-as-a-Judge metrics ---
# Initialize graders lazily to avoid API key issues during import
RELEVANCE_GRADER = None
FACTUAL_GRADER = None
COMPLETENESS_GRADER = None
CLARITY_GRADER = None
CONCISENESS_GRADER = None

def _get_grader(grader_name):
    """Get or initialize a grader lazily."""
    global RELEVANCE_GRADER, FACTUAL_GRADER, COMPLETENESS_GRADER, CLARITY_GRADER, CONCISENESS_GRADER
    
    # Check if API key is available before initializing
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Return a mock grader for testing
        return None
    
    if grader_name == "relevance" and RELEVANCE_GRADER is None:
        RELEVANCE_GRADER = init_grader(relevance_prompt)
    elif grader_name == "factual" and FACTUAL_GRADER is None:
        FACTUAL_GRADER = init_grader(factual_consistency_prompt)
    elif grader_name == "completeness" and COMPLETENESS_GRADER is None:
        COMPLETENESS_GRADER = init_grader(completeness_prompt)
    elif grader_name == "clarity" and CLARITY_GRADER is None:
        CLARITY_GRADER = init_grader(clarity_prompt)
    elif grader_name == "conciseness" and CONCISENESS_GRADER is None:
        CONCISENESS_GRADER = init_grader(conciseness_prompt)
    
    return {
        "relevance": RELEVANCE_GRADER,
        "factual": FACTUAL_GRADER,
        "completeness": COMPLETENESS_GRADER,
        "clarity": CLARITY_GRADER,
        "conciseness": CONCISENESS_GRADER
    }[grader_name]

def calculate_relevance(predictions, references, instructions):
    return evaluate_with_grader(
        _get_grader("relevance"),
        lambda pred, ref, instr: {
            "instruction": instr,
            "ground_truth_recap_summary": ref,
            "generated_recap_summary": pred,
        },
        predictions, references, instructions,
        metric_name="relevance"
    )

def calculate_factual_consistency(predictions, _, instructions):
    return evaluate_with_grader(
        _get_grader("factual"),
        lambda pred, instr: {
            "instruction": instr,
            "generated_recap_summary": pred,
        },
        predictions, instructions,
        metric_name="factual_consistency"
    )

def calculate_completeness(predictions, _, instructions):
    return evaluate_with_grader(
        _get_grader("completeness"),
        lambda pred, instr: {
            "instruction": instr,
            "generated_recap_summary": pred,
        },
        predictions, instructions,
        metric_name="completeness"
    )

def calculate_clarity(predictions, _, instructions):
    return evaluate_with_grader(
        _get_grader("clarity"),
        lambda pred, instr: {
            "instruction": instr,
            "generated_recap_summary": pred,
        },
        predictions, instructions,
        metric_name="clarity"
    )

def calculate_conciseness(predictions, _, instructions):
    return evaluate_with_grader(
        _get_grader("conciseness"),
        lambda pred, instr: {
            "instruction": instr,
            "generated_recap_summary": pred,
        },
        predictions, instructions,
        metric_name="conciseness"
    )

# --- Group Evaluation Entry Point ---
def compute_group_metrics(model, dataloader: DataLoader, device: str, max_length: int, metrics_list: Dict):
    tokenizer = model.tokenizer
    predictions = model.summarize_recap(dataloader, max_length=max_length)

    all_references = []
    all_instructions = []

    for batch in dataloader:
        inputs = {key: value.to(device) for key, value in batch.items()}

        batch_references = [
            tokenizer.decode(reference[reference != -100], skip_special_tokens=True)
            for reference in inputs["labels"]
        ]
        all_references.extend(batch_references)

        batch_instructions = [
            tokenizer.decode(input_id, skip_special_tokens=True)
            for input_id in inputs["input_ids"]
        ]
        all_instructions.extend(batch_instructions)

    result = {
        metric_name: metric_fn(predictions, all_references, all_instructions)
        for metric_name, metric_fn in metrics_list.items()
    }
    return pd.DataFrame(result, index=[0])

# --- Misc Profiling Helpers ---
def calculate_model_size(model_ckpt_path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(model_ckpt_path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return round(total / (1024 * 1024), 2)

def calculate_model_size_in_params(model) -> int:
    return sum(p.numel() for p in model.model.parameters())

def calculate_average_latency(model, dataloader, max_length: int) -> float:
    latencies = []
    for batch in dataloader:
        instructions = [
            model.tokenizer.decode(input_id, skip_special_tokens=True)
            for input_id in batch["input_ids"]
        ]
        _ = model.summarize_recap(game_recap=instructions[0], max_length=max_length)  # warm-up

        for instr in instructions:
            start = time.time()
            _ = model.summarize_recap(game_recap=instr, max_length=max_length)
            latencies.append(time.time() - start)

    return round(sum(latencies) / len(latencies), 4) if latencies else None
