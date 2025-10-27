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

import os, json, random, textwrap, pathlib
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
        
        dpo_pairs.append({
            'game_recap': df.loc[ch_idx, 'game_recap'],
            'prompt': "You are an NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n### NBA Game Recap ###\n" + str(df.loc[ch_idx, 'game_recap']) + "\n\n### Recap Summary ###\n",
            'chosen': df.loc[ch_idx, 'game_recap_summary_generated'],
            'rejected': df.loc[rej_idx, 'game_recap_summary_generated'],
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
# CELL 10: Generate Comparison Examples
# =============================================================================

def generate_comparison_examples(model, tokenizer, test_pairs, num_examples=5):
    """Generate before/after comparison examples"""
    import random
    
    # Convert to list if it's a HuggingFace Dataset
    if hasattr(test_pairs, 'to_list'):
        pairs_list = test_pairs.to_list()
    elif hasattr(test_pairs, '__iter__'):
        pairs_list = list(test_pairs)
    else:
        pairs_list = test_pairs
    
    examples = []
    samples = random.sample(pairs_list, min(num_examples, len(pairs_list)))
    
    print(f"\n📝 Generating {len(samples)} comparison examples...")
    
    for i, pair in enumerate(samples):
        prompt = pair['prompt']
        chosen = pair['chosen']
        rejected = pair['rejected']
        
        # Generate with current model
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        examples.append({
            'prompt': prompt[:200] + "...",
            'chosen': chosen[:200] + "...",
            'rejected': rejected[:200] + "...",
            'generated': generated[:200] + "...",
        })
        
        if i == 0:
            print("\nExample 1:")
            print(f"Prompt: {prompt[:150]}...")
            print(f"Generated: {generated[:200]}...")
            print(f"Chosen: {chosen[:200]}...")
            print(f"Rejected: {rejected[:200]}...")
    
    # Save examples
    examples_df = pd.DataFrame(examples)
    examples_df.to_csv(os.path.join(cfg.output_dir, 'comparison_examples.csv'), index=False)
    print(f"\n✅ Comparison examples saved to: {cfg.output_dir}/comparison_examples.csv")
    
    return examples

# Generate comparison examples
comparison_examples = generate_comparison_examples(model, tokenizer, dpo_test, num_examples=10)

print("\n" + "="*60)
print("DPO EVALUATION COMPLETE")
print("="*60)
print(f"✅ Model saved to: {cfg.output_dir}")
print(f"✅ Evaluation results: {cfg.output_dir}/evaluation_results.json")
print(f"✅ Comparison examples: {cfg.output_dir}/comparison_examples.csv")
print("\nKey Metrics:")
print(f"  - Preference Accuracy: {eval_results['preference_accuracy']:.1%}")
print(f"  - Alignment Score: {eval_results['avg_alignment']:.3f}")
print(f"  - Semantic Preservation: {eval_results['avg_semantic_preservation']:.3f}")

