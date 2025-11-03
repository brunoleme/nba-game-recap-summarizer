import os
import json
import pandas as pd
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

from nba_game_recap_summarizer.finetuning.utils.tokenization_utils import (
    preprocess_text,
    postprocess_text,
)


def _load_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"game_recap", "chosen", "rejected"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DPO CSV: {missing}")
    df = df.dropna(subset=["game_recap", "chosen", "rejected"]).copy()
    return df


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            use_cache=True,
            early_stopping=True,
            num_beams=1,
        )
    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return postprocess_text(text.strip())


def evaluate_dpo(cfg) -> str:
    logger.info("Starting DPO evaluation")
    pairs_csv = cfg.data.preference_pairs_csv
    df = _load_pairs(pairs_csv)

    model_root = "/opt/ml/processing/input/model-artifacts"
    pipeline_run_id = os.getenv("PIPELINE_RUN_ID", "")
    
    # Try multiple path structures (with and without pipeline_run_id)
    candidates = []
    if pipeline_run_id:
        # Path structure: /opt/ml/processing/input/model-artifacts/{PIPELINE_RUN_ID}/dpo/hf_model_merged_aligned
        candidates.append(os.path.join(model_root, pipeline_run_id, "dpo", "hf_model_merged_aligned"))
        candidates.append(os.path.join(model_root, pipeline_run_id, "hf_model_merged_aligned"))
    # Fallback paths (without pipeline_run_id)
    candidates.extend([
        os.path.join(model_root, "dpo", "hf_model_merged_aligned"),
        os.path.join(model_root, "hf_model_merged_aligned"),
        os.path.join(model_root, "hf_model_merged_unquantized_aligned"),
    ])
    
    aligned_dir = None
    for c in candidates:
        if os.path.exists(c) and os.path.exists(os.path.join(c, "config.json")):
            aligned_dir = c
            logger.info(f"Found aligned model at: {aligned_dir}")
            break
    
    if aligned_dir is None or not os.path.exists(aligned_dir):
        logger.error(f"Aligned model not found. Tried paths: {candidates}")
        raise FileNotFoundError(f"Aligned model not found in model-artifacts input. Checked: {candidates}")

    tokenizer = AutoTokenizer.from_pretrained(aligned_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(aligned_dir, torch_dtype=torch.float16, device_map="auto")

    # Lazy import sentence-transformers to avoid hard dependency if unavailable
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.warning(f"sentence-transformers not available: {e}. Falling back: using tokenizer embeddings for cosine (weaker metric).")
        st = None
    run = None
    try:
        import wandb
        run = wandb.init(project=f"{cfg.project_name}-{os.getenv('ENV','dev')}", name="dpo_eval", tags=["dpo","eval"]) 
    except Exception as e:
        logger.warning(f"W&B init skipped: {e}")
    n = min(getattr(cfg.evaluation, 'num_samples', 20), len(df))
    samples = df.sample(n)

    results = {
        "preference_accuracy": 0,
        "alignment_scores": [],
        "semantic_preservation": [],
        "examples": [],
    }

    for _, row in samples.iterrows():
        prompt = (
            "You are an NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n"
            "### NBA Game Recap ###\n" + preprocess_text(str(row["game_recap"])) + "\n\n### Recap Summary ###\n"
        )
        gen = _generate(model, tokenizer, prompt, max_new_tokens=cfg.lengths.max_new_tokens)

        if st is not None:
            chosen_sim = cosine_similarity(
                st.encode([gen]),
                st.encode([str(row["chosen"])])
            )[0][0]
            rejected_sim = cosine_similarity(
                st.encode([gen]),
                st.encode([str(row["rejected"])])
            )[0][0]
        else:
            # very rough fallback using token overlap
            import math
            import collections
            def _vec(s):
                toks = s.lower().split()
                c = collections.Counter(toks)
                return c
            def _cos(a,b):
                keys = set(a)|set(b)
                va = [a.get(k,0) for k in keys]
                vb = [b.get(k,0) for k in keys]
                dot = sum(x*y for x,y in zip(va,vb))
                na = math.sqrt(sum(x*x for x in va)); nb = math.sqrt(sum(x*x for x in vb))
                return float(dot/(na*nb)) if na>0 and nb>0 else 0.0
            chosen_sim = _cos(_vec(gen), _vec(str(row["chosen"])))
            rejected_sim = _cos(_vec(gen), _vec(str(row["rejected"])))

        results["preference_accuracy"] += 1 if chosen_sim > rejected_sim else 0
        results["alignment_scores"].append(float(chosen_sim))
        if st is not None:
            prompt_sim = cosine_similarity(
                st.encode([gen]),
                st.encode([str(row["game_recap"])])
            )[0][0]
        else:
            prompt_sim = _cos(_vec(gen), _vec(str(row["game_recap"])))
        results["semantic_preservation"].append(float(prompt_sim))
        results["examples"].append({
            "prompt": prompt[:200] + "...",
            "generated": gen[:200] + "...",
            "chosen_sim": float(chosen_sim),
            "rejected_sim": float(rejected_sim),
            "prompt_sim": float(prompt_sim),
        })

    n_safe = max(1, n)
    results["preference_accuracy"] = float(results["preference_accuracy"]) / n_safe
    results["avg_alignment"] = float(sum(results["alignment_scores"]) / len(results["alignment_scores"])) if results["alignment_scores"] else 0.0
    results["avg_semantic_preservation"] = float(sum(results["semantic_preservation"]) / len(results["semantic_preservation"])) if results["semantic_preservation"] else 0.0

    out_root = "/opt/ml/processing/output/model-artifacts"
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success("Saved evaluation_results.json")

    # Save examples CSV
    try:
        ex_rows = []
        for ex in results["examples"]:
            ex_rows.append(ex)
        if ex_rows:
            ex_df = pd.DataFrame(ex_rows)
            ex_df.to_csv(os.path.join(out_root, "dpo_evaluation_examples.csv"), index=False)
    except Exception as e:
        logger.warning(f"Could not save examples CSV: {e}")

    # Write pipeline report file (for PropertyFile)
    try:
        reports_dir = getattr(cfg.evaluation, "reports_dir", None)
        if reports_dir:
            os.makedirs(reports_dir, exist_ok=True)
            with open(os.path.join(reports_dir, "eval_metrics.json"), "w") as f:
                json.dump({
                    "preference_accuracy": results["preference_accuracy"],
                    "avg_alignment": results.get("avg_alignment", 0.0),
                    "avg_semantic_preservation": results.get("avg_semantic_preservation", 0.0),
                }, f, indent=2)
            logger.info("Saved reports/eval_metrics.json for pipeline")
    except Exception as e:
        logger.warning(f"Could not write pipeline report file: {e}")

    # W&B logging
    if run:
        try:
            import wandb
            run.log({
                "eval/preference_accuracy": results["preference_accuracy"],
                "eval/avg_alignment": results.get("avg_alignment", 0.0),
                "eval/avg_semantic_preservation": results.get("avg_semantic_preservation", 0.0),
            })
            # Log a small table of examples
            if results["examples"]:
                table = wandb.Table(columns=list(results["examples"][0].keys()))
                for ex in results["examples"]:
                    table.add_data(*[ex[c] for c in table.columns])
                run.log({"eval/examples": table})
            run.finish()
        except Exception:
            pass
    return out_path


