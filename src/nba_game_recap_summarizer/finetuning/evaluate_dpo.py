import os
import json
import re
import math
import pandas as pd
from loguru import logger
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

from nba_game_recap_summarizer.finetuning.utils.tokenization_utils import (
    preprocess_text,
    postprocess_text,
)
from nba_game_recap_summarizer.finetuning.utils.text_utils import clean_game_recap


def _load_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"game_recap", "chosen", "rejected"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DPO CSV: {missing}")
    df = df.dropna(subset=["game_recap", "chosen", "rejected"]).copy()
    return df


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """
    Generate text from model, matching LlamaRecapSummarizationModel.summarize_recap parameters.
    
    This function replicates the exact generation pipeline used in production inference
    to ensure evaluation results are comparable.
    """
    # Use tokenizer's max_length for consistency with model.summarize_recap
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding=True,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=10,  # Ensure minimum generation length
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            typical_p=None,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            no_repeat_ngram_size=0,  # Disable n-gram blocking (matches model.summarize_recap)
            repetition_penalty=1.1,  # Lower penalty (matches model.summarize_recap)
            use_cache=True,
            early_stopping=False,  # Don't stop early - let it generate full length
            num_beams=1,
        )
    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    result = postprocess_text(text.strip())
    
    # Log if generation is suspiciously short
    if len(result) < 20:
        logger.warning(f"Generated text is very short ({len(result)} chars): {result[:100]}")
    
    return result


class NarrativeStyleEvaluator:
    """Evaluate narrative style of text summaries (from Colab implementation)."""
    def __init__(self, embed_model_name='sentence-transformers/all-mpnet-base-v2', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embed_model_name, device=self.device)
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer for narrative evaluator: {e}")
            self.model = None
        self.connector_list = [
            'however','despite','while','as','after','because','therefore','meanwhile','although','though',
            'whereas','furthermore','moreover','consequently','thus','hence','additionally','similarly','conversely'
        ]
    
    @staticmethod
    def _sentences(text):
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in parts if s]
    
    @staticmethod
    def _token_count(s):
        return len(re.findall(r"\w+|\S", s))
    
    @staticmethod
    def _bulletiness(text):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return 1.0
        bullet = sum(1 for l in lines if l.startswith(('-', '•', '*')) or re.match(r'^(Score|Top\s*Performers|Outcome|Key|Stats?):', l, re.I))
        return bullet / max(1, len(lines))
    
    def _discourse_prop(self, text):
        tl = text.lower()
        hits = sum(conn in tl for conn in self.connector_list)
        tokens = max(1, len(re.findall(r"[A-Za-z']+", tl)))
        return min(1.0, hits / tokens * 12)
    
    def calculate_narrative_structure_score(self, text):
        sents = self._sentences(text)
        n = len(sents)
        if n < 3:
            return 0.0
        r_count = 1.0 if 3 <= n <= 7 else math.exp(-abs(n-5)/2)
        avg_len = np.mean([self._token_count(s) for s in sents]) if sents else 0
        r_lenband = 1.0 if 12 <= avg_len <= 30 else max(0.0, 1 - abs(avg_len - 21) / 21)
        r_disc = self._discourse_prop(text)
        score = 0.5 * r_count + 0.35 * r_lenband + 0.15 * r_disc
        return float(np.clip(score, 0.0, 1.0))
    
    def calculate_coherence_score(self, text):
        if self.model is None:
            logger.debug("SentenceTransformer model not available for coherence calculation")
            return 0.0
        sents = self._sentences(text)
        if len(sents) < 2:
            return 0.0
        try:
            embs = self.model.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
            sims = (embs[:-1] * embs[1:]).sum(axis=1)
            return float(np.mean(sims)) if sims.size else 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate coherence score: {e}")
            return 0.0
    
    def calculate_coverage_score(self, original, summary):
        if self.model is None:
            logger.debug("SentenceTransformer model not available for coverage calculation")
            return 0.0
        if not original or not summary:
            return 0.0
        try:
            embs = self.model.encode([original, summary], convert_to_numpy=True, normalize_embeddings=True)
            return float(np.dot(embs[0], embs[1]))
        except Exception as e:
            logger.warning(f"Failed to calculate coverage score: {e}")
            return 0.0
    
    def evaluate(self, original, summary):
        b = self._bulletiness(summary)
        s = self.calculate_narrative_structure_score(summary)
        coh = self.calculate_coherence_score(summary)
        cov = self.calculate_coverage_score(original, summary)
        composite = ((1 - b) * 0.30 + s * 0.35 + coh * 0.20 + cov * 0.15)
        score = float(np.clip(composite, 0.0, 1.0)) * 5.0
        return {
            'bulletiness_score': float(b),
            'structure_score': float(s),
            'coherence_score': float(coh),
            'coverage_score': float(cov),
            'narrative_style_score': round(score, 2),
        }


def _find_model_path(model_root, pipeline_run_id, subdirs):
    """Find model path by trying multiple candidates."""
    candidates = []
    if pipeline_run_id:
        for subdir in subdirs:
            candidates.append(os.path.join(model_root, pipeline_run_id, subdir))
    candidates.extend([os.path.join(model_root, sd) for sd in subdirs])
    
    for candidate in candidates:
        if os.path.exists(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    return None


def evaluate_dpo(cfg) -> str:
    """
    Comprehensive DPO evaluation comparing pre-DPO (base) and post-DPO (aligned) models.
    
    Metrics Calculated:
    -------------------
    For BOTH pre-DPO and post-DPO models:
    
    1. **Preference Metrics**:
       - preference_accuracy: % of examples where generated output is more similar to "chosen" than "rejected"
       - alignment_scores (chosen_sim): Cosine similarity to "chosen" summary
       - semantic_preservation (prompt_sim): Cosine similarity to original game recap
    
    2. **Narrative Style** (NarrativeStyleEvaluator):
       - narrative_style_score: Composite score (0-5) based on structure, coherence, coverage, bulletiness
       - structure_score, coherence_score, coverage_score, bulletiness_score
    
    3. **Classical Metrics**:
       - rouge_score: ROUGE-1/2 F1 average
       - bert_score: BERTScore F1 average
    
    4. **AI-as-Judge Metrics** (requires OPENAI_API_KEY):
       - factual_consistency: Factual accuracy (1-5 scale)
       - relevance: Relevance to ground truth (1-5 scale)
       - completeness: Completeness of information (1-5 scale)
       - conciseness: Conciseness rating (1-5 scale)
       - clarity: Clarity rating (1-5 scale)
    
    Output Files:
    ------------
    - alignment_evaluation_results.json: Full results with all metrics for both models
    - alignment_eval_metrics.json: Summary metrics only (for pipeline PropertyFile)
    
    Model Locations:
    ---------------
    - Post-DPO (aligned): {PIPELINE_RUN_ID}/hf_model_merged_aligned/ (saved directly in pipeline root)
    - Pre-DPO (base): Loaded from BASE_MODEL_PATH env var or previous pipeline run
    """
    logger.info("Starting comprehensive DPO evaluation (pre vs post-DPO)")
    pairs_csv = cfg.data.preference_pairs_csv
    df = _load_pairs(pairs_csv)

    model_root = "/opt/ml/processing/input/model-artifacts"
    pipeline_run_id = os.getenv("PIPELINE_RUN_ID", "")
    
    # Find post-DPO (aligned) model - saved directly in pipeline root, not in dpo/ subfolder
    aligned_dir = _find_model_path(
        model_root, pipeline_run_id,
        ["hf_model_merged_aligned", "hf_model_merged_unquantized_aligned"]
    )
    
    if aligned_dir is None:
        logger.error(f"Aligned model not found in {model_root}")
        raise FileNotFoundError(f"Aligned model not found in model-artifacts input")
    
    logger.info(f"Found post-DPO model at: {aligned_dir}")

    # Find pre-DPO (base) model
    base_model_path = os.getenv("BASE_MODEL_PATH", "")
    base_dir = None
    
    if base_model_path:
        # BASE_MODEL_PATH is an S3 path, download it
        # Format: s3://bucket/output/artifacts/{PIPELINE_ID}/hf_model_merged
        if "s3://" in base_model_path:
            logger.info(f"Downloading base model from S3: {base_model_path}")
            try:
                import boto3
                import tempfile
                
                s3_client = boto3.client('s3')
                s3_path = base_model_path.replace("s3://", "")
                bucket_name, key_prefix = s3_path.split("/", 1)
                
                # Create temporary directory for model
                base_dir = tempfile.mkdtemp(prefix="base_model_")
                logger.info(f"Downloading model to: {base_dir}")
                
                # Download all files from S3 prefix
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket_name, Prefix=key_prefix)
                
                files_downloaded = 0
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            file_key = obj['Key']
                            # Remove the prefix to get relative path
                            relative_path = file_key.replace(key_prefix, "").lstrip("/")
                            if relative_path:  # Skip empty paths
                                local_file_path = os.path.join(base_dir, relative_path)
                                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                                s3_client.download_file(bucket_name, file_key, local_file_path)
                                files_downloaded += 1
                
                if files_downloaded > 0:
                    logger.success(f"Downloaded {files_downloaded} files from base model S3 path")
                    
                    # Download tokenizer files from hf_model_base directory (pre-alignment models don't include them)
                    # Extract pipeline ID from S3 path to build tokenizer path
                    # Format: s3://bucket/output/artifacts/{PIPELINE_ID}/hf_model_merged
                    path_parts = key_prefix.split("/")
                    if "artifacts" in path_parts:
                        artifacts_idx = path_parts.index("artifacts")
                        if artifacts_idx + 1 < len(path_parts):
                            pipeline_id = path_parts[artifacts_idx + 1]
                            tokenizer_prefix = f"output/artifacts/{pipeline_id}/hf_model_base/"
                            logger.info(f"Downloading tokenizer files from: s3://{bucket_name}/{tokenizer_prefix}")
                            
                            tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.model']
                            tokenizer_files_downloaded = 0
                            for tokenizer_file in tokenizer_files:
                                # Check if file already exists (some models might have it)
                                local_file_path = os.path.join(base_dir, tokenizer_file)
                                if not os.path.exists(local_file_path):
                                    try:
                                        s3_key = f"{tokenizer_prefix}{tokenizer_file}"
                                        s3_client.download_file(bucket_name, s3_key, local_file_path)
                                        tokenizer_files_downloaded += 1
                                        logger.info(f"  Downloaded tokenizer file: {tokenizer_file}")
                                    except Exception as e:
                                        logger.warning(f"  Could not download {tokenizer_file}: {e}")
                                else:
                                    logger.debug(f"  Tokenizer file already exists: {tokenizer_file}")
                            
                            if tokenizer_files_downloaded > 0:
                                logger.info(f"Downloaded {tokenizer_files_downloaded} tokenizer files from hf_model_base")
                            else:
                                logger.info("All tokenizer files already present or not available")
                else:
                    logger.warning(f"No files found at S3 path: {base_model_path}")
                    base_dir = None
            except Exception as e:
                logger.error(f"Failed to download base model from S3: {e}")
                base_dir = None
        else:
            # Local path
            base_dir = base_model_path if os.path.exists(base_model_path) else None
    
    # Fallback: try to find base model in same pipeline run
    if base_dir is None:
        logger.info("Trying to find base model in current pipeline run...")
        base_dir = _find_model_path(
            model_root, pipeline_run_id,
            ["hf_model_merged", "hf_model_merged_unquantized"]
        )
    
    if base_dir is None or not os.path.exists(os.path.join(base_dir, "config.json")):
        logger.warning("Pre-DPO (base) model not found. Will only evaluate post-DPO model.")
        logger.warning(f"BASE_MODEL_PATH was: {base_model_path}")
        base_model = None
        base_tokenizer = None
    else:
        logger.info(f"Found pre-DPO model at: {base_dir}")
        base_tokenizer = AutoTokenizer.from_pretrained(base_dir)
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base_dir, torch_dtype=torch.float16, device_map="auto"
        )

    # Load post-DPO model
    aligned_tokenizer = AutoTokenizer.from_pretrained(aligned_dir)
    if aligned_tokenizer.pad_token is None:
        aligned_tokenizer.pad_token = aligned_tokenizer.eos_token
    aligned_model = AutoModelForCausalLM.from_pretrained(
        aligned_dir, torch_dtype=torch.float16, device_map="auto"
    )

    # Initialize evaluators
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.warning(f"sentence-transformers not available: {e}. Some metrics will be skipped.")
        st = None
    
    narrative_evaluator = NarrativeStyleEvaluator(
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Import metrics
    try:
        from nba_game_recap_summarizer.finetuning.eval.metrics import (
            calculate_rouge, calculate_bertscore,
            calculate_factual_consistency, calculate_relevance,
            calculate_completeness, calculate_conciseness, calculate_clarity
        )
        metrics_available = True
    except Exception as e:
        logger.warning(f"Could not import evaluation metrics: {e}. Some metrics will be skipped.")
        metrics_available = False

    # Initialize W&B
    run = None
    try:
        import wandb
        run = wandb.init(project=f"{cfg.project_name}-{os.getenv('ENV','dev')}", name="dpo_eval", tags=["dpo","eval"]) 
    except Exception as e:
        logger.warning(f"W&B init skipped: {e}")

    n = min(getattr(cfg.evaluation, 'num_samples', 20), len(df))
    samples = df.sample(n, random_state=42)

    results = {
        "pre_dpo": {},
        "post_dpo": {},
        "improvements": {},
        "examples": [],
    }

    # Generate predictions for all samples
    logger.info(f"Generating predictions for {len(samples)} samples...")
    prompts = []
    pre_generations = []
    post_generations = []
    game_recaps = []
    chosen_summaries = []
    rejected_summaries = []

    for _, row in samples.iterrows():
        # Clean and preprocess the game recap (matching inference API pipeline)
        raw_recap = str(row["game_recap"])
        cleaned_recap = clean_game_recap(raw_recap)  # First clean (removes control chars, normalizes)
        preprocessed_recap = preprocess_text(cleaned_recap)  # Then preprocess (converts scores, etc.)
        
        prompt = (
            "You are an NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n"
            "### NBA Game Recap ###\n" + preprocessed_recap + "\n\n### Recap Summary ###\n"
        )
        prompts.append(prompt)
        game_recaps.append(raw_recap)  # Store original for metrics
        chosen_summaries.append(str(row["chosen"]))
        rejected_summaries.append(str(row["rejected"]))
        
        # Generate with pre-DPO model
        if base_model is not None:
            pre_gen = _generate(base_model, base_tokenizer, prompt, max_new_tokens=cfg.lengths.max_new_tokens)
        else:
            pre_gen = None
        pre_generations.append(pre_gen)
        
        # Generate with post-DPO model
        post_gen = _generate(aligned_model, aligned_tokenizer, prompt, max_new_tokens=cfg.lengths.max_new_tokens)
        post_generations.append(post_gen)

    # Calculate metrics for both models
    def calculate_model_metrics(model_name, generations, tokenizer, model_obj):
        """Calculate all metrics for a model."""
        metrics = {}
        
        if generations is None or all(g is None for g in generations):
            return metrics
        
        valid_gens = [g for g in generations if g is not None]
        valid_chosen = [chosen_summaries[i] for i, g in enumerate(generations) if g is not None]
        valid_prompts = [prompts[i] for i, g in enumerate(generations) if g is not None]
        
        # Preference metrics (similarity-based)
        if st is not None:
            chosen_sims = []
            rejected_sims = []
            prompt_sims = []
            for i, gen in enumerate(valid_gens):
                if gen and len(gen.strip()) > 0:  # Only process non-empty generations
                    try:
                        # Encode all texts at once for efficiency
                        gen_emb = st.encode([gen], normalize_embeddings=True, convert_to_numpy=True)
                        chosen_emb = st.encode([valid_chosen[i]], normalize_embeddings=True, convert_to_numpy=True)
                        rejected_idx = i % len(rejected_summaries)
                        rejected_emb = st.encode([rejected_summaries[rejected_idx]], normalize_embeddings=True, convert_to_numpy=True)
                        game_recap_emb = st.encode([game_recaps[i]], normalize_embeddings=True, convert_to_numpy=True)
                        
                        # Calculate cosine similarity (dot product since embeddings are normalized)
                        chosen_sim = float(np.dot(gen_emb[0], chosen_emb[0]))
                        rejected_sim = float(np.dot(gen_emb[0], rejected_emb[0]))
                        prompt_sim = float(np.dot(gen_emb[0], game_recap_emb[0]))
                        
                        chosen_sims.append(chosen_sim)
                        rejected_sims.append(rejected_sim)
                        prompt_sims.append(prompt_sim)
                    except Exception as e:
                        logger.warning(f"Failed to calculate similarity for sample {i}: {e}")
                        continue
            
            if chosen_sims:
                metrics["preference_accuracy"] = float(sum(1 for cs, rs in zip(chosen_sims, rejected_sims) if cs > rs) / len(chosen_sims))
                metrics["avg_alignment"] = float(np.mean(chosen_sims))
                metrics["avg_semantic_preservation"] = float(np.mean(prompt_sims))
                logger.debug(f"Preference metrics: accuracy={metrics['preference_accuracy']:.3f}, alignment={metrics['avg_alignment']:.3f}")
            else:
                logger.warning("No valid generations for preference metrics calculation")
        else:
            logger.warning("SentenceTransformer not available, skipping preference metrics")
        
        # Narrative style scores
        narrative_scores = []
        structure_scores = []
        coherence_scores = []
        coverage_scores = []
        for i, gen in enumerate(valid_gens):
            if gen:
                narrative_result = narrative_evaluator.evaluate(game_recaps[i], gen)
                narrative_scores.append(narrative_result['narrative_style_score'])
                structure_scores.append(narrative_result['structure_score'])
                coherence_scores.append(narrative_result['coherence_score'])
                coverage_scores.append(narrative_result['coverage_score'])
        
        if narrative_scores:
            metrics["avg_narrative_style_score"] = float(np.mean(narrative_scores))
            metrics["avg_structure_score"] = float(np.mean(structure_scores))
            metrics["avg_coherence_score"] = float(np.mean(coherence_scores))
            metrics["avg_coverage_score"] = float(np.mean(coverage_scores))
        
        # Classical metrics
        if metrics_available and valid_chosen:
            try:
                metrics["rouge_score"] = float(calculate_rouge(valid_gens, valid_chosen, valid_prompts))
            except Exception as e:
                logger.warning(f"Could not calculate ROUGE: {e}")
                metrics["rouge_score"] = 0.0
            
            try:
                metrics["bert_score"] = float(calculate_bertscore(valid_gens, valid_chosen, valid_prompts))
            except Exception as e:
                logger.warning(f"Could not calculate BERTScore: {e}")
                metrics["bert_score"] = 0.0
        
        # AI-as-Judge metrics (requires OPENAI_API_KEY)
        # These can take a long time (20 samples × 5 metrics = 100 API calls, ~10-20 minutes)
        # Can be disabled via config: evaluation.skip_ai_judge_metrics = true
        skip_ai_judge = getattr(cfg.evaluation, "skip_ai_judge_metrics", False)
        if skip_ai_judge:
            logger.info("Skipping AI-as-Judge metrics (disabled in config)")
        elif metrics_available and os.getenv("OPENAI_API_KEY"):
            try:
                logger.info(f"Calculating AI-as-Judge metrics for {len(valid_gens)} samples (this may take 10-20 minutes)...")
                logger.info("  Calculating factual_consistency...")
                metrics["factual_consistency"] = float(calculate_factual_consistency(valid_gens, None, valid_prompts))
                logger.info("  Calculating relevance...")
                metrics["relevance"] = float(calculate_relevance(valid_gens, valid_chosen, valid_prompts))
                logger.info("  Calculating completeness...")
                metrics["completeness"] = float(calculate_completeness(valid_gens, None, valid_prompts))
                logger.info("  Calculating conciseness...")
                metrics["conciseness"] = float(calculate_conciseness(valid_gens, None, valid_prompts))
                logger.info("  Calculating clarity...")
                metrics["clarity"] = float(calculate_clarity(valid_gens, None, valid_prompts))
                logger.info("AI-as-Judge metrics completed")
            except Exception as e:
                logger.warning(f"Could not calculate AI-as-Judge metrics: {e}")
                logger.warning("Continuing evaluation without AI-as-Judge metrics")
        else:
            logger.info("Skipping AI-as-Judge metrics (OPENAI_API_KEY not set or metrics not available)")
        
        return metrics
    
    # Calculate metrics for both models
    logger.info("Calculating metrics for pre-DPO model...")
    results["pre_dpo"] = calculate_model_metrics("pre_dpo", pre_generations, base_tokenizer, base_model)
    
    logger.info("Calculating metrics for post-DPO model...")
    results["post_dpo"] = calculate_model_metrics("post_dpo", post_generations, aligned_tokenizer, aligned_model)
    
    # Calculate improvements
    for metric in results["post_dpo"]:
        if metric in results["pre_dpo"]:
            pre_val = results["pre_dpo"][metric]
            post_val = results["post_dpo"][metric]
            if pre_val > 0:
                improvement_pct = ((post_val - pre_val) / pre_val) * 100
            else:
                improvement_pct = float('inf') if post_val > 0 else 0.0
            results["improvements"][metric] = {
                "absolute": float(post_val - pre_val),
                "percentage": float(improvement_pct)
            }
        else:
            results["improvements"][metric] = {
                "absolute": float(results["post_dpo"][metric]),
                "percentage": float('inf')
            }
    
    # Store examples with full details
    for i, (prompt, pre_gen, post_gen, game_recap, chosen, rejected) in enumerate(
        zip(prompts, pre_generations, post_generations, game_recaps, chosen_summaries, rejected_summaries)
    ):
        example = {
            "index": i,
            "game_recap": game_recap[:500] + "..." if len(game_recap) > 500 else game_recap,
            "chosen_summary": chosen[:500] + "..." if len(chosen) > 500 else chosen,
            "rejected_summary": rejected[:500] + "..." if len(rejected) > 500 else rejected,
            "pre_dpo_generation": pre_gen[:500] + "..." if pre_gen and len(pre_gen) > 500 else (pre_gen or "N/A"),
            "post_dpo_generation": post_gen[:500] + "..." if post_gen and len(post_gen) > 500 else (post_gen or "N/A"),
        }
        
        # Add narrative scores for both
        if pre_gen:
            pre_narrative = narrative_evaluator.evaluate(game_recap, pre_gen)
            example["pre_dpo_narrative_score"] = pre_narrative['narrative_style_score']
        else:
            example["pre_dpo_narrative_score"] = None
        
        if post_gen:
            post_narrative = narrative_evaluator.evaluate(game_recap, post_gen)
            example["post_dpo_narrative_score"] = post_narrative['narrative_style_score']
        else:
            example["post_dpo_narrative_score"] = None
        
        results["examples"].append(example)

    # Get reports directory from config (should include pipeline_run_id)
    reports_dir = getattr(cfg.evaluation, "reports_dir", None)
    if not reports_dir:
        # Fallback to default location
        pipeline_run_id = os.getenv("PIPELINE_RUN_ID", "unknown")
        reports_dir = f"/opt/ml/processing/output/model-artifacts/{pipeline_run_id}/reports"
    
    os.makedirs(reports_dir, exist_ok=True)
    logger.info(f"Saving evaluation results to: {reports_dir}")
    
    # Save full evaluation results JSON (renamed to distinguish from supervised fine-tuning)
    results_path = os.path.join(reports_dir, "alignment_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Saved alignment_evaluation_results.json to: {reports_dir}")

    # Write pipeline report file (for PropertyFile - summary metrics only)
    summary_metrics = {
        "preference_accuracy": {
            "pre_dpo": results["pre_dpo"].get("preference_accuracy", 0.0),
            "post_dpo": results["post_dpo"].get("preference_accuracy", 0.0),
            "improvement": results["improvements"].get("preference_accuracy", {}).get("absolute", 0.0)
        },
        "avg_alignment": {
            "pre_dpo": results["pre_dpo"].get("avg_alignment", 0.0),
            "post_dpo": results["post_dpo"].get("avg_alignment", 0.0),
            "improvement": results["improvements"].get("avg_alignment", {}).get("absolute", 0.0)
        },
        "avg_narrative_style_score": {
            "pre_dpo": results["pre_dpo"].get("avg_narrative_style_score", 0.0),
            "post_dpo": results["post_dpo"].get("avg_narrative_style_score", 0.0),
            "improvement": results["improvements"].get("avg_narrative_style_score", {}).get("absolute", 0.0)
        },
        "rouge_score": {
            "pre_dpo": results["pre_dpo"].get("rouge_score", 0.0),
            "post_dpo": results["post_dpo"].get("rouge_score", 0.0),
            "improvement": results["improvements"].get("rouge_score", {}).get("absolute", 0.0)
        },
        "bert_score": {
            "pre_dpo": results["pre_dpo"].get("bert_score", 0.0),
            "post_dpo": results["post_dpo"].get("bert_score", 0.0),
            "improvement": results["improvements"].get("bert_score", {}).get("absolute", 0.0)
        },
    }
    
    # Add AI-as-Judge metrics if available
    for metric in ["factual_consistency", "relevance", "completeness", "conciseness", "clarity"]:
        if metric in results["post_dpo"]:
            summary_metrics[metric] = {
                "pre_dpo": results["pre_dpo"].get(metric, 0.0),
                "post_dpo": results["post_dpo"].get(metric, 0.0),
                "improvement": results["improvements"].get(metric, {}).get("absolute", 0.0)
            }
    
    try:
        summary_path = os.path.join(reports_dir, "alignment_eval_metrics.json")
        with open(summary_path, "w") as f:
            json.dump(summary_metrics, f, indent=2)
        logger.info(f"Saved alignment_eval_metrics.json (summary) to: {reports_dir}")
    except Exception as e:
        logger.warning(f"Could not write pipeline report file: {e}")

    # W&B logging
    if run:
        try:
            import wandb
            # Log pre-DPO metrics
            for metric, value in results["pre_dpo"].items():
                run.log({f"eval/pre_dpo_{metric}": value})
            
            # Log post-DPO metrics
            for metric, value in results["post_dpo"].items():
                run.log({f"eval/post_dpo_{metric}": value})
            
            # Log improvements
            for metric, improvement in results["improvements"].items():
                run.log({f"eval/improvement_{metric}_absolute": improvement.get("absolute", 0.0)})
                run.log({f"eval/improvement_{metric}_percentage": improvement.get("percentage", 0.0)})
            
            run.finish()
        except Exception:
            pass

    logger.success(f"DPO evaluation complete. Results saved to: {reports_dir}")
    return results_path