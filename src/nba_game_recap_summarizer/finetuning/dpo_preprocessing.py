import os
import pandas as pd
from loguru import logger

from nba_game_recap_summarizer.finetuning.utils.tokenization_utils import preprocess_text


def _find_source_csv(source_dir: str) -> str:
    """Find the DPO pairs CSV in the mounted source-data directory."""
    candidate = os.path.join(source_dir, "game_recaps_with_rewritten_curated_summaries.csv")
    if os.path.exists(candidate):
        return candidate
    # fallback: first CSV
    for fn in os.listdir(source_dir):
        if fn.endswith(".csv"):
            return os.path.join(source_dir, fn)
    raise FileNotFoundError("No CSV found in source-data directory")


def dpo_preprocessing(cfg) -> str:
    """
    Stage and normalize the DPO pairs CSV for training.

    Returns the output CSV path.
    """
    logger.info("Starting DPO preprocessing")
    source_dir = "/opt/ml/processing/input/source-data"
    output_dir = "/opt/ml/processing/output/preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = _find_source_csv(source_dir)
    logger.info(f"Reading DPO pairs from: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"game_recap", "chosen", "rejected"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DPO CSV missing columns: {missing}")

    # Drop NA and duplicates
    before = len(df)
    df = df.dropna(subset=["game_recap", "chosen", "rejected"]).copy()
    df = df.drop_duplicates(subset=["game_recap", "chosen", "rejected"])  # conservative
    logger.info(f"Dropped {before - len(df)} rows with NA/dupes")

    # Normalize text with existing utils
    for col in ["game_recap", "chosen", "rejected"]:
        df[col] = df[col].astype(str).map(preprocess_text)

    # Ensure chosen != rejected
    neq_mask = df["chosen"].ne(df["rejected"]) & df["chosen"].str.strip().ne("") & df["rejected"].str.strip().ne("")
    filtered = df[neq_mask].copy()
    logger.info(f"Filtered {len(df) - len(filtered)} rows where chosen == rejected or empty")

    out_path = os.path.join(output_dir, "game_recaps_with_rewritten_curated_summaries.csv")
    filtered.to_csv(out_path, index=False)
    logger.success(f"Wrote preprocessed DPO CSV to: {out_path}")

    # Write a small preprocessing report for observability
    try:
        import json
        import numpy as np
        # Basic stats
        report = {
            "rows_before": int(before),
            "rows_after": int(len(filtered)),
            "na_counts": {
                c: int(df[c].isna().sum()) if c in df.columns else 0 for c in ["game_recap","chosen","rejected"]
            },
        }
        # Length summaries
        for c in ["game_recap","chosen","rejected"]:
            lengths = filtered[c].astype(str).str.len()
            report[f"len_{c}"] = {
                "min": int(lengths.min()) if len(lengths) else 0,
                "max": int(lengths.max()) if len(lengths) else 0,
                "mean": float(lengths.mean()) if len(lengths) else 0.0,
            }
        with open(os.path.join(output_dir, "dpo_preprocessing_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved dpo_preprocessing_report.json")
    except Exception as e:
        logger.warning(f"Could not write preprocessing report: {e}")
    return out_path


