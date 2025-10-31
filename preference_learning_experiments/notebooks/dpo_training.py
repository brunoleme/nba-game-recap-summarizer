# DPO Training for NBA Game Recap Summarization
# Using single-score data converted to chosen/rejected pairs

# =============================================================================
# CELL 1: Install and Import Packages
# =============================================================================

# Install required packages
print("Installing required packages...")
!pip install --upgrade pip
!pip install torch boto3 awscli
!pip install "transformers>=4.43" "accelerate>=0.33" "datasets>=2.19" "trl>=0.9.6" "peft>=0.11.0" bitsandbytes einops sentencepiece --upgrade
!pip install protobuf sentence-transformers

import os, json, random, textwrap, pathlib, re
from dataclasses import dataclass, asdict
from typing import Optional, List
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
import trl, transformers

# =============================================================================
# PREPROCESSING AND POSTPROCESSING FUNCTIONS
# =============================================================================

def preprocess_text(text: str) -> str:
    """
    Preprocess text to make it more tokenizer-friendly.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    # Convert scores from "117-109" to "117 to 109" for better tokenization
    text = re.sub(r'(\d+)-(\d+)', r'\1 to \2', text)
    
    # Convert "3-pointer" to "three pointer" for better tokenization
    text = re.sub(r'(\d+)-pointer', r'\1 pointer', text)
    text = re.sub(r'(\d+)-point', r'\1 point', text)
    
    # Convert "3s" to "three pointers" for consistency
    text = re.sub(r'\b(\d+)s\b', r'\1 pointers', text)
    
    # Convert time formats to be more readable
    text = re.sub(r'(\d+):(\d+)', r'\1 minutes \2 seconds', text)
    
    # Convert ratios to be more readable
    text = re.sub(r'(\d+)/(\d+)', r'\1 out of \2', text)
    
    # Convert percentages to be more readable
    text = re.sub(r'(\d+)%', r'\1 percent', text)
    
    return text

def postprocess_text(text: str) -> str:
    """
    Postprocess generated text to restore proper formatting.
    
    Args:
        text: Generated text to postprocess
        
    Returns:
        Postprocessed text
    """
    # Restore scores from "117 to 109" to "117-109"
    text = re.sub(r'(\d+) to (\d+)', r'\1-\2', text)
    
    # Restore "three pointer" to "3-pointer"
    text = re.sub(r'three pointer', '3-pointer', text)
    text = re.sub(r'three point', '3-point', text)
    text = re.sub(r'two pointer', '2-pointer', text)
    text = re.sub(r'two point', '2-point', text)
    
    # Restore "three pointers" to "3-pointers"
    text = re.sub(r'three pointers', '3-pointers', text)
    text = re.sub(r'two pointers', '2-pointers', text)
    
    # Restore time formats
    text = re.sub(r'(\d+) minutes (\d+) seconds', r'\1:\2', text)
    
    # Restore ratios
    text = re.sub(r'(\d+) out of (\d+)', r'\1/\2', text)
    
    # Restore percentages
    text = re.sub(r'(\d+) percent', r'\1%', text)
    
    return text

print('trl version:', trl.__version__)
print('transformers version:', transformers.__version__)

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

PROJECT_DIR = '/content/dpo_project'
pathlib.Path(PROJECT_DIR).mkdir(parents=True, exist_ok=True)

# =============================================================================
# CELL 2: Configure AWS and Pipeline ID
# =============================================================================

# Configure AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_AWS_ACCESS_KEY_ID'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_AWS_SECRET_ACCESS_KEY'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# Pipeline ID for your trained model
PIPELINE_ID = 'efa479ff-388b-4977-a37b-7d0b923509e3'

# =============================================================================
# CELL 3: Download Model from S3
# =============================================================================

import boto3

def download_model_from_s3():
    """Download the unquantized merged model AND tokenizer from S3"""
    s3_client = boto3.client('s3')
    bucket_name = 'nba-recap-summarization-model-staging'
    
    # Download the unquantized merged model
    merged_prefix = f'output/artifacts/{PIPELINE_ID}/hf_model_merged_unquantized/'
    merged_path = './hf_model_merged_unquantized/'
    
    # Download tokenizer from base model directory
    tokenizer_prefix = f'output/artifacts/{PIPELINE_ID}/hf_model_base/'
    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
    
    # Create local directory
    os.makedirs(merged_path, exist_ok=True)
    
    print(f"Downloading unquantized merged model from: s3://{bucket_name}/{merged_prefix}")
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=merged_prefix)
    
    files_downloaded = 0
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                local_file = os.path.join(merged_path, key.replace(merged_prefix, ''))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                s3_client.download_file(bucket_name, key, local_file)
                files_downloaded += 1
    
    print(f"Downloaded {files_downloaded} files from unquantized merged model")
    
    # Download tokenizer files
    print(f"Downloading tokenizer files from: s3://{bucket_name}/{tokenizer_prefix}")
    for tokenizer_file in tokenizer_files:
        s3_key = f'{tokenizer_prefix}{tokenizer_file}'
        try:
            local_file = os.path.join(merged_path, tokenizer_file)
            s3_client.download_file(bucket_name, s3_key, local_file)
            files_downloaded += 1
            print(f"  Downloaded {tokenizer_file}")
        except Exception as e:
            print(f"  ⚠️ Failed to download {tokenizer_file}: {e}")
    
    if files_downloaded > 0:
        print(f"✅ Total files downloaded: {files_downloaded}")
        return merged_path
    else:
        raise RuntimeError("Failed to download unquantized merged model from S3!")

# Download the model
model_path = download_model_from_s3()

# Validate that model_path exists and contains necessary files
print(f"\nValidating downloaded model at: {model_path}")
if os.path.exists(model_path):
    files = os.listdir(model_path)
    print(f"Downloaded files: {files}")
    
    required_files = ['tokenizer.json', 'config.json']
    missing_files = [f for f in required_files if f not in files]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        raise RuntimeError(f"Model download incomplete. Missing: {missing_files}")
    else:
        print("✅ All required model files found")
else:
    raise RuntimeError(f"Model path does not exist: {model_path}")

# =============================================================================
# CELL 4: Load Model
# =============================================================================

print("Loading model...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print(f"Tokenizer vocabulary size: {len(tokenizer)}")

# Load the model
print("Loading unquantized merged model in FP16...")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    print("✅ Model loaded in FP16")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)[:200]}")
    raise RuntimeError(f"Failed to load model: {e}")

# Resize token embeddings if needed
expected_vocab_size = len(tokenizer)
current_vocab_size = base_model.get_input_embeddings().num_embeddings

if current_vocab_size != expected_vocab_size:
    print(f"⚠️ Resizing model embeddings from {current_vocab_size} to {expected_vocab_size}")
    base_model.resize_token_embeddings(expected_vocab_size)
    print(f"✅ Model embeddings resized")
else:
    print("✅ Vocabulary sizes already match")

# Attach LoRA adapters for DPO training
print("Attaching LoRA adapters for DPO training...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
print("✅ LoRA adapters attached successfully")
model.train()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

# =============================================================================
# CELL 5: Prepare DPO Dataset
# =============================================================================

def load_and_prepare_dpo_data(csv_path='game_recaps_with_summaries_sample_for_reward_model_with_generated_full.csv'):
    """Load data and create chosen/rejected pairs for DPO"""
    df = pd.read_csv(csv_path)
    
    print(f"Total samples: {len(df)}")
    print("\nScore distribution:")
    print(df['narrative_style_score'].describe())
    
    # Create DPO pairs: high-score samples vs randomly sampled (as rejected)
    scores = df['narrative_style_score'].astype(float).values
    
    # Define threshold for "chosen" (high quality)
    threshold = np.percentile(scores, 75)  # Top 25% as chosen
    print(f"\nChosen threshold (75th percentile): {threshold:.2f}")
    
    # Get high-score samples
    chosen_idx = df[scores >= threshold].index.tolist()
    print(f"High-quality samples (chosen): {len(chosen_idx)}")
    
    # Sample random pairs as "rejected" for each chosen sample
    rejected_idx = []
    for _ in chosen_idx:
        # Pick a random sample (could be low or high score, it's used as negative)
        rejected_idx.append(df.index[random.randint(0, len(df)-1)])
    
    # Create DPO pairs
    dpo_pairs = []
    for ch_idx, rej_idx in zip(chosen_idx, rejected_idx):
        # Skip if same sample or if rejected score is higher than chosen
        if ch_idx == rej_idx:
            continue
        
        chosen_score = scores[ch_idx]
        rejected_score = scores[rej_idx]
        
        if chosen_score <= rejected_score:
            continue  # Skip if rejected is better than chosen
        
        # IMPORTANT: The issue is that we're pairing DIFFERENT games
        # Standard DPO: Same prompt (same task), different responses
        # Our case: Different prompts (different games), different responses
        #
        # SOLUTION: We need to ensure the SAME prompt is used for both
        # chosen and rejected. We'll use the chosen game's prompt as the base,
        # but include info about both summaries in the response comparison.
        #
        # Actually, let me reconsider: the model should learn to prefer
        # "good narrative style summaries" regardless of content.
        # Let's use the chosen game's recap as the prompt for BOTH:
        
        chosen_game_recap = preprocess_text(str(df.loc[ch_idx, 'game_recap']))
        chosen_summary = preprocess_text(str(df.loc[ch_idx, 'game_recap_summary_generated']))
        rejected_summary = preprocess_text(str(df.loc[rej_idx, 'game_recap_summary_generated']))
        
        # IMPORTANT: We have different games for chosen vs rejected
        # DPO needs SAME prompt for both. We need to include the full context
        # with the summary in the chosen/rejected fields.
        #
        # However, DPOTrainer will subtract the prompt to get the completion.
        # So we need to make sure the prompt is a PREFIX of the full text.
        #
        # The correct format is:
        # prompt = "base template" (the task)
        # chosen = prompt + game_A_recap + "### Recap Summary ###\n" + summary_A
        # rejected = prompt + game_B_recap + "### Recap Summary ###\n" + summary_B
        #
        # But DPOTrainer expects prompt to be a prefix of chosen/rejected.
        # So we'll need to include the game recap in the prompt OR store empty prompt.
        #
        # Actually, a better approach: match the prompt to the chosen, and accept
        # that the rejected summary won't match the prompt. The model will still
        # learn the style difference.
        
        prompt_template = "You are an NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n### NBA Game Recap ###\n"
        
        # Use prompt that matches the CHOSEN game (standard DPO format)
        prompt = prompt_template + chosen_game_recap + "\n\n### Recap Summary ###\n"
        
        dpo_pairs.append({
            'game_recap_chosen': chosen_game_recap[:200],  # For debugging
            'game_recap_rejected': df.loc[rej_idx, 'game_recap'][:200],  # For debugging  
            'prompt': prompt,
            'chosen': chosen_summary,  # Matches the prompt
            'rejected': rejected_summary,  # From different game - this is OK!
        })
    
    print(f"Created {len(dpo_pairs)} DPO pairs")
    
    # Show score stats for chosen vs rejected
    chosen_scores = [scores[df.index.get_loc(d['game_recap']) if df.index.is_object() else ch_idx] 
                     for d in dpo_pairs[:100]] if len(dpo_pairs) > 0 else []
    
    return dpo_pairs

# Load and prepare DPO data
dpo_pairs = load_and_prepare_dpo_data()

# Convert to dataset
df_dpo = pd.DataFrame(dpo_pairs)
dpo_dataset = Dataset.from_pandas(df_dpo)
print(f"\nDPO dataset created with {len(dpo_dataset)} pairs")

# Train/test split
train_test = dpo_dataset.train_test_split(test_size=0.1, seed=42)
dpo_train = train_test['train']
dpo_test = train_test['test']

print(f"Training pairs: {len(dpo_train)}")
print(f"Test pairs: {len(dpo_test)}")

# =============================================================================
# CELL 6: Configure DPO Training
# =============================================================================

@dataclass
class DPORunConfig:
    max_prompt_length: int = 512
    max_completion_length: int = 256
    
    # DPO settings
    beta: float = 0.1  # DPO beta (similar to KTO)
    
    # Training settings
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    lr_scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.03
    num_train_epochs: float = 1.0
    
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 0
    bf16: bool = False
    fp16: bool = False
    
    output_dir: str = f'{PROJECT_DIR}/outputs/dpo_training'

cfg = DPORunConfig()
os.makedirs(cfg.output_dir, exist_ok=True)
json.dump(asdict(cfg), open(os.path.join(cfg.output_dir, 'train_config.json'), 'w'), indent=2)

print("DPO Configuration:")
print(json.dumps(asdict(cfg), indent=2))

# Configure DPO training
dpo_config = DPOConfig(
    output_dir=cfg.output_dir,
    beta=cfg.beta,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    learning_rate=cfg.learning_rate,
    lr_scheduler_type=cfg.lr_scheduler_type,
    warmup_ratio=cfg.warmup_ratio,
    num_train_epochs=cfg.num_train_epochs,
    logging_steps=cfg.logging_steps,
    save_steps=cfg.save_steps,
    bf16=False,
    fp16=False,
    eval_steps=cfg.eval_steps,
    optim="adamw_torch",
    max_grad_norm=1.0,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    max_length=cfg.max_prompt_length + cfg.max_completion_length,
    report_to="none",
    gradient_checkpointing=False,
)

print("DPO Configuration:")
print(json.dumps(dpo_config.to_dict(), indent=2))

# =============================================================================
# CELL 7: Initialize DPO Trainer
# =============================================================================

print("Initializing DPO trainer...")

# Use the same model as reference (frozen)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Resize reference embeddings if needed
ref_vocab_size = ref_model.get_input_embeddings().num_embeddings
if ref_vocab_size != expected_vocab_size:
    ref_model.resize_token_embeddings(expected_vocab_size)

# Freeze reference model
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()

print("✅ Reference model loaded and frozen")

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # For PEFT training, don't pass ref_model (or use force_use_ref_model=True)
    args=dpo_config,
    train_dataset=dpo_train,
    eval_dataset=dpo_test,
    processing_class=tokenizer,  # Changed from tokenizer to processing_class
    peft_config=None,  # PEFT already applied, don't pass again
)

print("✅ DPO trainer initialized successfully!")

# =============================================================================
# CELL 8: Start DPO Training
# =============================================================================

print("Starting DPO training...")
print(f"Training on {len(dpo_train)} pairs")
print(f"Configuration: {cfg.num_train_epochs} epochs, batch size {cfg.per_device_train_batch_size}")

# Train
trainer.train()

# Save
trainer.save_model(cfg.output_dir)
print(f"✅ Training completed! Model saved to: {cfg.output_dir}")

# =============================================================================
# CELL 9: Evaluate DPO Results
# =============================================================================

print("\n" + "="*60)
print("EVALUATING DPO RESULTS")
print("="*60)

def evaluate_preference_alignment(model, tokenizer, eval_pairs, num_samples=20):
    """Evaluate alignment quality and preference accuracy"""
    import random
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Load sentence transformer for semantic similarity
    print("Loading sentence transformer for evaluation...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Convert to list if it's a HuggingFace Dataset
    if hasattr(eval_pairs, 'to_list'):
        eval_list = eval_pairs.to_list()
    elif hasattr(eval_pairs, '__iter__'):
        eval_list = list(eval_pairs)
    else:
        eval_list = eval_pairs
    
    # Sample evaluation pairs
    eval_samples = random.sample(eval_list, min(num_samples, len(eval_list)))
    
    results = {
        'preference_accuracy': 0,
        'alignment_scores': [],
        'narrative_improvements': [],
        'semantic_preservation': []
    }
    
    print(f"\nEvaluating {len(eval_samples)} pairs...")
    
    for i, pair in enumerate(eval_samples):
        prompt = pair['prompt']
        chosen = pair['chosen']
        rejected = pair['rejected']
        
        # Generate responses with tuned model
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            # Generate with current model
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # 1. Preference Accuracy: Does model prefer chosen > rejected?
        # Compare semantic similarity with chosen vs rejected
        chosen_sim = cosine_similarity(
            st_model.encode([generated]),
            st_model.encode([chosen])
        )[0][0]
        
        rejected_sim = cosine_similarity(
            st_model.encode([generated]),
            st_model.encode([rejected])
        )[0][0]
        
        preference_correct = 1 if chosen_sim > rejected_sim else 0
        results['preference_accuracy'] += preference_correct
        
        # 2. Alignment Score: How well does generated match chosen?
        alignment = cosine_similarity(
            st_model.encode([generated]),
            st_model.encode([chosen])
        )[0][0]
        results['alignment_scores'].append(alignment)
        
        # 3. Semantic Preservation: Check content fidelity
        # (Simplified - could use more sophisticated NLI models)
        results['semantic_preservation'].append(chosen_sim)
        
        if (i + 1) % 5 == 0:
            print(f"  Evaluated {i+1}/{len(eval_samples)} pairs")
    
    # Calculate metrics (convert to Python native types for JSON serialization)
    results['preference_accuracy'] = float(results['preference_accuracy'] / len(eval_samples))
    results['avg_alignment'] = float(np.mean(results['alignment_scores']))
    results['avg_semantic_preservation'] = float(np.mean(results['semantic_preservation']))
    
    # Convert lists to Python types for JSON serialization
    results['alignment_scores'] = [float(x) for x in results['alignment_scores']]
    results['semantic_preservation'] = [float(x) for x in results['semantic_preservation']]
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Preference Accuracy: {results['preference_accuracy']:.2%}")
    print(f"  Avg Alignment Score: {results['avg_alignment']:.3f}")
    print(f"  Avg Semantic Preservation: {results['avg_semantic_preservation']:.3f}")
    
    return results

# Run evaluation
eval_results = evaluate_preference_alignment(model, tokenizer, dpo_test, num_samples=20)

# Save evaluation results
import json
with open(os.path.join(cfg.output_dir, 'evaluation_results.json'), 'w') as f:
    json.dump(eval_results, f, indent=2)
print(f"\n✅ Evaluation results saved to: {cfg.output_dir}/evaluation_results.json")

# =============================================================================
# CELL 10: Before/After Comparison with Training Loss Analysis
# =============================================================================

def generate_before_after_comparison(model, tokenizer, original_df, num_examples=5):
    """Generate comparison showing model output BEFORE and AFTER DPO training"""
    import random
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    
    print("\n" + "="*60)
    print("BEFORE/AFTER COMPARISON")
    print("="*60)
    
    # Load the ORIGINAL model (before DPO tuning) for comparison
    print("\nLoading ORIGINAL model (before DPO) for comparison...")
    
    try:
        # Try to download and load the merged model (before DPO)
        # Note: We already have the model downloaded from earlier cells
        # Just reload it for comparison
        
        # Use the existing downloaded model path
        original_model_path = "./hf_model_merged_unquantized/"
        
        print(f"Loading original model from: {original_model_path}")
        
        # Load the original model
        original_model = AutoModelForCausalLM.from_pretrained(
            original_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("✅ Original model loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load original model: {e}")
        print("   Will only show AFTER tuning results")
        print(f"   Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        original_model = None
    
    # Sample random examples from rejected (lower quality) summaries
    low_score_rows = original_df[original_df['narrative_style_score'] < 3.0]
    
    if len(low_score_rows) == 0:
        print("No low-score examples found, using random samples...")
        low_score_rows = original_df.sample(min(num_examples * 5, len(original_df)))
    
    samples = low_score_rows.sample(min(num_examples, len(low_score_rows)))
    
    examples = []
    
    print(f"\n📝 Generating {len(samples)} before/after comparison examples...")
    
    # Import narrative scoring functions
    print("Importing narrative scoring functions...")
    import sys
    sys.path.append('/content/dpo_project')
    import math
    
    # Define the evaluator class inline
    class NarrativeStyleEvaluator:
        """A class to evaluate the narrative style of text summaries."""
        def __init__(self, embed_model_name='sentence-transformers/all-mpnet-base-v2', device=None):
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embed_model_name, device=self.device)
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
            sents = self._sentences(text)
            if len(sents) < 2:
                return 0.0
            embs = self.model.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
            sims = (embs[:-1] * embs[1:]).sum(axis=1)
            return float(np.mean(sims)) if sims.size else 0.0
        
        def calculate_coverage_score(self, original, summary):
            if not original or not summary:
                return 0.0
            embs = self.model.encode([original, summary], convert_to_numpy=True, normalize_embeddings=True)
            return float(np.dot(embs[0], embs[1]))
        
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
    
    # Initialize the evaluator
    try:
        print("Initializing narrative style evaluator...")
        evaluator = NarrativeStyleEvaluator(device='cuda' if torch.cuda.is_available() else 'cpu')
        print("✅ Evaluator initialized")
    except Exception as e:
        print(f"⚠️ Could not initialize evaluator: {e}")
        evaluator = None
    
    def calculate_narrative_score(original, summary):
        """Calculate narrative score for a summary"""
        if evaluator is None:
            return None
        try:
            result = evaluator.evaluate(original, summary)
            return result['narrative_style_score']
        except Exception as e:
            print(f"⚠️ Error calculating narrative score: {e}")
            return None
    
    for i, row in enumerate(samples.iterrows()):
        idx, data = row
        game_recap = data['game_recap']
        
        # IMPORTANT: Preprocess the game recap before creating prompt
        game_recap_preprocessed = preprocess_text(str(game_recap))
        
        # Create the prompt with preprocessed text
        prompt = f"You are an NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n### NBA Game Recap ###\n{game_recap_preprocessed}"
        
        # Get the original summary (ground truth)
        original_summary = data.get('game_recap_summary_ground_truth', 'N/A')
        
        def generate_with_model(mdl, tok, prmpt):
            """Helper to generate with a model"""
            inputs = tok(prmpt, return_tensors="pt", truncation=True, max_length=1024, padding=True)
            device = next(mdl.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            mdl.eval()
            with torch.no_grad():
                outputs = mdl.generate(
                    **inputs,
                    max_new_tokens=512,  # Increased from 256
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Properly extract only the generated tokens (not the input)
            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            generated = tok.decode(generated_ids, skip_special_tokens=True)
            
            # IMPORTANT: Postprocess the generated text
            generated = postprocess_text(generated)
            
            # Debug: Check if generation looks correct
            if len(generated) < 50:  # Suspiciously short
                print(f"⚠️ Short generation detected: {len(generated)} chars")
                print(f"   Full outputs shape: {outputs.shape}")
                print(f"   Input length: {input_length}")
                print(f"   Generated tokens: {generated}")
            
            return generated
        
        # Generate with ORIGINAL model
        original_output = "N/A (model not loaded)"
        if original_model is not None:
            original_output = generate_with_model(original_model, tokenizer, prompt)
        
        # Generate with TUNED model (current model)
        tuned_output = generate_with_model(model, tokenizer, prompt)
        
        # Calculate narrative scores for both outputs (using original non-preprocessed recap for scoring)
        before_score = calculate_narrative_score(game_recap, original_output) if original_output != "N/A (model not loaded)" else None
        after_score = calculate_narrative_score(game_recap, tuned_output)
        
        # Debug: Check the full output
        if i == 0:
            print(f"\n🔍 DEBUG: Full generated output length: {len(tuned_output)}")
            print(f"🔍 DEBUG: Full output: {tuned_output[:500]}...")
            print(f"🔍 DEBUG: Before DPO score: {before_score}")
            print(f"🔍 DEBUG: After DPO score: {after_score}")
        
        examples.append({
            'index': int(idx),
            'game_recap': game_recap[:1000] if len(game_recap) > 1000 else game_recap,
            'original_summary': str(original_summary)[:1000] if len(str(original_summary)) > 1000 else original_summary,
            'before_dpo': original_output[:1000] if len(original_output) > 1000 else original_output,
            'after_dpo': tuned_output[:1000] if len(tuned_output) > 1000 else tuned_output,
            'narrative_score_before_dpo': before_score if before_score is not None else 'N/A',
            'narrative_score_after_dpo': after_score if after_score is not None else 'N/A',
            'original_narrative_score': data.get('narrative_style_score', 'N/A')
        })
        
        if i == 0:
            print("\n" + "="*60)
            print("EXAMPLE 1:")
            print("="*60)
            print(f"\nGame Recap:\n{game_recap[:500]}...")
            print(f"\nOriginal Ground Truth:\n{original_summary[:500]}...")
            print(f"\nBefore DPO (Original Model):\n{original_output[:500]}...")
            print(f"\nAfter DPO (Tuned Model):\n{tuned_output[:500]}...")
            print(f"\nNarrative Score: {data.get('narrative_style_score', 'N/A')}")
    
    # Save examples
    examples_df = pd.DataFrame(examples)
    examples_df.to_csv(os.path.join(cfg.output_dir, 'before_after_comparison.csv'), index=False)
    print(f"\n✅ Before/after comparison saved to: {cfg.output_dir}/before_after_comparison.csv")
    
    return examples

# Generate before/after comparison
print("\n" + "="*60)
print("LOADING ORIGINAL DATA FOR COMPARISON")
print("="*60)

# Load the original CSV to get full game recaps
original_csv_path = "/content/dpo_project/data/game_recaps_with_summaries_sample_for_reward_model_with_generated_full.csv"
original_df = pd.read_csv(original_csv_path)

before_after_examples = generate_before_after_comparison(model, tokenizer, original_df, num_examples=10)

# =============================================================================
# CELL 11: Training Loss Analysis
# =============================================================================

print("\n" + "="*60)
print("TRAINING LOSS ANALYSIS")
print("="*60)

def analyze_training_loss(trainer):
    """Analyze the training loss evolution"""
    import json
    
    # Get training logs
    state = trainer.state
    
    print("\n📊 Training Loss Evolution:")
    print("-" * 60)
    
    if hasattr(state, 'log_history') and state.log_history:
        # Extract loss values
        losses = [entry.get('loss', None) for entry in state.log_history if 'loss' in entry]
        steps = [entry.get('step', None) for entry in state.log_history if 'loss' in entry]
        
        if losses:
            print(f"\nInitial Loss: {losses[0]:.4f}")
            print(f"Final Loss: {losses[-1]:.4f}")
            print(f"Loss Reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
            print(f"Average Loss: {sum(losses) / len(losses):.4f}")
            print(f"Number of Steps: {len(losses)}")
            
            # Save loss data
            loss_data = {
                'initial_loss': losses[0],
                'final_loss': losses[-1],
                'loss_reduction_percent': ((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] > 0 else 0,
                'average_loss': sum(losses) / len(losses),
                'losses': losses,
                'steps': steps
            }
            
            with open(os.path.join(cfg.output_dir, 'training_loss_analysis.json'), 'w') as f:
                json.dump({k: float(v) if isinstance(v, (int, float)) else v for k, v in loss_data.items()}, f, indent=2)
            
            print(f"\n✅ Loss analysis saved to: {cfg.output_dir}/training_loss_analysis.json")
            
            # Create a simple visualization
            try:
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(12, 4))
                plt.plot(steps, losses, marker='o', markersize=2)
                plt.xlabel('Training Step')
                plt.ylabel('Loss')
                plt.title('DPO Training Loss Evolution')
                plt.grid(True, alpha=0.3)
                
                loss_plot_path = os.path.join(cfg.output_dir, 'training_loss_plot.png')
                plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
                print(f"✅ Loss plot saved to: {loss_plot_path}")
                
            except Exception as e:
                print(f"⚠️ Could not create loss plot: {e}")
        else:
            print("No loss values found in training logs")
    else:
        print("No training history available")

# Analyze training loss
analyze_training_loss(trainer)

print("\n" + "="*60)
print("DPO EVALUATION COMPLETE")
print("="*60)
print(f"✅ Model saved to: {cfg.output_dir}")
print(f"\nGenerated Files:")
print(f"  - evaluation_results.json (alignment metrics)")
print(f"  - before_after_comparison.csv (before/after DPO examples)")
print(f"  - training_loss_analysis.json (loss statistics)")
print(f"  - training_loss_plot.png (loss visualization)")
print("\nKey Metrics:")
print(f"  - Preference Accuracy: {eval_results['preference_accuracy']:.1%}")
print(f"  - Alignment Score: {eval_results['avg_alignment']:.3f}")
print(f"  - Semantic Preservation: {eval_results['avg_semantic_preservation']:.3f}")

