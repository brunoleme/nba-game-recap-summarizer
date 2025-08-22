from evaluate import load
import time
import pandas as pd
from typing import Dict, Callable
import numpy as np
import os

from torch.utils.data import DataLoader
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from text2cypher.finetuning.eval.prompts import (
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
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key).bind_tools([ResponseFormatter])
    return prompt_template | llm

def evaluate_with_grader(grader, format_fn, *data_streams):
    grades = []
    for args in zip(*data_streams):
        prompt_input = format_fn(*args)
        try:
            response = grader.invoke(prompt_input)
            if hasattr(response, "tool_calls") and response.tool_calls:
                score = float(response.tool_calls[0]["args"]["score"])
                if 1 <= score <= 5:
                    grades.append(score)
        except Exception:
            continue
    return np.mean(grades) if grades else 0

# --- Classical metrics ---
def calculate_rouge(predictions, references, _):
    try:
        scores = rouge_metric.compute(predictions=predictions, references=references)
        return sum(scores.values()) / len(scores)
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
RELEVANCE_GRADER = init_grader(relevance_prompt)
FACTUAL_GRADER = init_grader(factual_consistency_prompt)
COMPLETENESS_GRADER = init_grader(completeness_prompt)
CLARITY_GRADER = init_grader(clarity_prompt)
CONCISENESS_GRADER = init_grader(conciseness_prompt)

def calculate_relevance(predictions, references, instructions):
    return evaluate_with_grader(
        RELEVANCE_GRADER,
        lambda pred, ref, instr: {
            "instruction": instr,
            "ground_truth_note": ref,
            "generated_note": pred,
        },
        predictions, references, instructions
    )

def calculate_factual_consistency(predictions, _, instructions):
    return evaluate_with_grader(
        FACTUAL_GRADER,
        lambda pred, instr: {
            "instruction": instr,
            "generated_note": pred,
        },
        predictions, instructions
    )

def calculate_completeness(predictions, _, instructions):
    return evaluate_with_grader(
        COMPLETENESS_GRADER,
        lambda pred, instr: {
            "instruction": instr,
            "generated_note": pred,
        },
        predictions, instructions
    )

def calculate_clarity(predictions, _, instructions):
    return evaluate_with_grader(
        CLARITY_GRADER,
        lambda pred, instr: {
            "instruction": instr,
            "generated_note": pred,
        },
        predictions, instructions
    )

def calculate_conciseness(predictions, _, instructions):
    return evaluate_with_grader(
        CONCISENESS_GRADER,
        lambda pred, instr: {
            "instruction": instr,
            "generated_note": pred,
        },
        predictions, instructions
    )

# --- Group Evaluation Entry Point ---
def compute_group_metrics(model, dataloader: DataLoader, device: str, max_length: int, metrics_list: Dict):
    tokenizer = model.tokenizer
    predictions = model.generate_notes(dataloader, max_length=max_length)

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
        _ = model.generate_note(conversation=instructions[0], max_length=max_length)  # warm-up

        for instr in instructions:
            start = time.time()
            _ = model.generate_note(conversation=instr, max_length=max_length)
            latencies.append(time.time() - start)

    return round(sum(latencies) / len(latencies), 4) if latencies else None
