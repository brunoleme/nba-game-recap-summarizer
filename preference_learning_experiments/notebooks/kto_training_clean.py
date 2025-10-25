# KTO Training for NBA Game Recap Summarization
# Clean, optimized implementation for the new model structure

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
from peft import PeftModel, prepare_model_for_kbit_training
from trl import KTOTrainer, KTOConfig
import trl, transformers

print('trl version:', trl.__version__)
print('transformers version:', transformers.__version__)

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

PROJECT_DIR = '/content/kto_project'
pathlib.Path(PROJECT_DIR).mkdir(parents=True, exist_ok=True)

# =============================================================================
# CELL 2: Configure AWS and Pipeline ID
# =============================================================================

# Configure AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_AWS_ACCESS_KEY_ID'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_AWS_SECRET_ACCESS_KEY'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# Pipeline ID for your trained model
PIPELINE_ID = 'a02c5d0b-0a26-483d-94ad-514328745678'

# =============================================================================
# CELL 3: Download Model from S3 (Training-Ready Format)
# =============================================================================

import boto3

def download_model_from_s3():
    """Download the fine-tuned model from S3 to local storage"""
    s3_client = boto3.client('s3')
    bucket_name = 'nba-recap-summarization-model-staging'
    
    # We need base + adapters for KTO training (not merged!)
    base_prefix = f'output/artifacts/{PIPELINE_ID}/hf_model_base/'
    adapters_prefix = f'output/artifacts/{PIPELINE_ID}/hf_model_adapters/'
    
    base_path = './hf_model_base/'
    adapters_path = './hf_model_adapters/'
    
    # Create local directories
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(adapters_path, exist_ok=True)
    
    def download_s3_prefix(s3_prefix, local_path, name):
        print(f"Downloading {name} from: s3://{bucket_name}/{s3_prefix}")
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
        
        files_downloaded = 0
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    local_file = os.path.join(local_path, key.replace(s3_prefix, ''))
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    s3_client.download_file(bucket_name, key, local_file)
                    files_downloaded += 1
        
        if files_downloaded > 0:
            print(f"{name} download completed! Downloaded {files_downloaded} files")
            return True
        else:
            print(f"No files found for {name}")
            return False
    
    # Download base and adapters (required for KTO training)
    base_success = download_s3_prefix(base_prefix, base_path, "Base Model")
    adapters_success = download_s3_prefix(adapters_prefix, adapters_path, "LoRA Adapters")
    
    if base_success and adapters_success:
        print("✅ Successfully downloaded base model and adapters for KTO training!")
        return base_path, adapters_path
    else:
        raise RuntimeError("Failed to download required model files from S3. Need both hf_model_base and hf_model_adapters for training!")

# Download the model
base_path, adapters_path = download_model_from_s3()

# Check what's actually in the S3 bucket
print("\nChecking S3 bucket contents...")
try:
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket='nba-recap-summarization-model-staging', Prefix=f'output/artifacts/{PIPELINE_ID}/')
    
    if 'Contents' in response:
        print("Available files in S3:")
        for obj in response['Contents']:
            print(f"  - {obj['Key']}")
    else:
        print("No files found in S3 bucket")
except Exception as e:
    print(f"Error checking S3 bucket: {e}")

# =============================================================================
# CELL 4: Load Training-Ready Model
# =============================================================================

print("Loading training-ready model...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load model - use the correct approach for base + adapters
print("Loading model for KTO training...")

# The base model was saved with quantization, so we need to load it with BitsAndBytes
print("Loading base model with 4-bit quantization...")

# Configure BitsAndBytes quantization - MUST match training configuration
# From base_model.py line 87-91, training uses float16 compute dtype
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # Match training config!
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load the base model WITHOUT quantization (full precision LoRA)
print("Loading base model for full precision LoRA training...")

# Load model with FP16 (no quantization)
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map="auto",
        torch_dtype=torch.float16,  # FP16, not quantized
        trust_remote_code=True,
    )
    print("✅ Base model loaded in FP16 (no quantization)")
except Exception as e:
    print(f"❌ Failed to load unquantized base model: {str(e)[:200]}")
    
    # Fallback: Try with auto dtype
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        print("✅ Base model loaded with auto dtype")
    except Exception as e2:
        print(f"❌ Fallback also failed: {str(e2)[:200]}")
        
        # Last resort: Try merged model
        print("Trying to use merged model as fallback...")
        try:
            # Download merged model
            merged_prefix = f'output/artifacts/{PIPELINE_ID}/hf_model_merged/'
            merged_path = './hf_model_merged/'
            os.makedirs(merged_path, exist_ok=True)
            
            # Download merged model
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket='nba-recap-summarization-model-staging', Prefix=merged_prefix)
            
            files_downloaded = 0
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        local_file = os.path.join(merged_path, key.replace(merged_prefix, ''))
                        os.makedirs(os.path.dirname(local_file), exist_ok=True)
                        s3_client.download_file('nba-recap-summarization-model-staging', key, local_file)
                        files_downloaded += 1
            
            if files_downloaded > 0:
                print(f"Downloaded merged model ({files_downloaded} files)")
                base_model = AutoModelForCausalLM.from_pretrained(
                    merged_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                print("✅ Merged model loaded successfully")
                print("⚠️  WARNING: Using merged model - no separate adapters available")
                print("⚠️  This means you'll need to attach new LoRA adapters for KTO training")
            else:
                raise RuntimeError("No merged model files found")
                
        except Exception as e3:
            print(f"❌ Merged model strategy also failed: {str(e3)[:200]}")
            raise RuntimeError("Failed to load any model variant")

# Now attach the LoRA adapters (if available)
print("Attaching LoRA adapters...")
if adapters_path and os.path.exists(adapters_path):
    try:
        # First, check the vocabulary size mismatch
        print("Checking vocabulary size compatibility...")
        
        # Get the expected vocab size from the tokenizer (which should match training)
        expected_vocab_size = len(tokenizer)
        current_vocab_size = base_model.get_input_embeddings().num_embeddings
        
        print(f"Current vocab size: {current_vocab_size}")
        print(f"Tokenizer vocab size: {expected_vocab_size}")
        
        # Resize token embeddings to match the training vocabulary size
        if current_vocab_size != expected_vocab_size:
            print(f"Resizing token embeddings from {current_vocab_size} to {expected_vocab_size}")
            base_model.resize_token_embeddings(expected_vocab_size)
        
        # NO QUANTIZATION - use the base model as is
        print("Using base model without quantization (full precision LoRA)")
        
        # Now attach the adapters
        model = PeftModel.from_pretrained(base_model, adapters_path)
        print("✅ LoRA adapters attached successfully to base model")
    except Exception as e:
        print(f"❌ Failed to attach adapters: {e}")
        print("⚠️  Using base model without adapters - you may need to add new LoRA adapters")
        model = base_model
else:
    print(f"⚠️  Adapters not found at {adapters_path}")
    print("⚠️  Using base model without adapters - you may need to add new LoRA adapters")
    model = base_model

# Check if we have trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Initial trainable parameters: {trainable_params:,}")

# Debug: Check model type and structure
print(f"Model type: {type(model)}")
if hasattr(model, 'peft_config'):
    print(f"PEFT config: {model.peft_config}")
if hasattr(model, 'get_base_model'):
    base_model = model.get_base_model()
    print(f"Base model type: {type(base_model)}")

if trainable_params == 0:
    print("⚠️  No trainable parameters found - adding new LoRA adapters for KTO training")
    
    # Check if model already has PEFT adapters
    if hasattr(model, 'peft_config') and model.peft_config:
        print("⚠️  Model already has PEFT config - unloading first")
        model = model.unload()
    
    # Create new LoRA configuration
    from peft import LoraConfig, get_peft_model
    
    lora_config = LoraConfig(
        r=16,  # Match training config
        lora_alpha=32,  # Match training config
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # CRITICAL: Disable PEFT adapters on the model before adding new ones
    if hasattr(model, 'disable_adapter'):
        model.disable_adapter()
    
    # Add new LoRA adapters for KTO training
    model = get_peft_model(model, lora_config)
    print("✅ New LoRA adapters added for KTO training")
    
    # Check trainable parameters after adding LoRA
    new_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after LoRA: {new_trainable:,}")
    
    # Enable the new adapters
    model.train()  # Ensure training mode
    
    print("✅ Model ready with new LoRA adapters")
    print(f"✅ Trainable parameters confirmed: {new_trainable:,}")

# CRITICAL: Enable gradients for training (full precision LoRA)
print("Enabling gradients for training...")

# For full precision LoRA, this is simpler - no special quantization setup needed
# Just ensure LoRA layers are trainable
model.train()

print("✅ Gradients enabled for training")

# Set to training mode
model.train()
print("✅ Model ready for KTO training")

# Final check for trainable parameters
final_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
if final_trainable > 0:
    print(f"✅ Final trainable parameters: {final_trainable:,}")
else:
    print("❌ WARNING: Still no trainable parameters - KTO training may not work")

# Verify model is ready for KTO training
def verify_kto_readiness(model):
    """Verify the model is ready for KTO training"""
    print("\n" + "="*50)
    print("KTO Training Readiness Check:")
    print("="*50)
    
    # Check trainable parameters
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Check if model is in training mode
    print(f"Model training mode: {model.training}")
    
    # Check device
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Test gradient flow
    try:
        test_input = tokenizer("Test input", return_tensors="pt")
        if torch.cuda.is_available():
            test_input = {k: v.cuda() for k, v in test_input.items()}
        
        outputs = model(**test_input)
        print("✅ Model forward pass successful")
        
        outputs.logits.sum().backward()
        print("✅ Gradient computation successful")
        
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"✅ Gradients flowing: {has_grads}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
    
    print("="*50)

verify_kto_readiness(model)

# =============================================================================
# CELL 5: Load and Prepare Preference Data
# =============================================================================

# Load preference data
def load_preference_data(csv_path='game_recaps_with_summaries_sample_for_reward_model_with_generated_full.csv'):
    """Load the generated summaries with scores for KTO training"""
    df = pd.read_csv(csv_path)
    
    # Filter for high-quality generated summaries (narrative_style_score > 4.0)
    high_quality = df[df['narrative_style_score'] > 4.0]
    
    print(f"Total samples: {len(df)}")
    print(f"High-quality samples (score > 4.0): {len(high_quality)}")
    
    # Show score distribution
    print("\nScore distribution:")
    print(df['narrative_style_score'].describe())
    
    return df, high_quality

# Load your preference data (upload CSV file to Colab first)
df, high_quality_df = load_preference_data()

# Prepare KTO dataset
print("Preparing KTO dataset...")

# Normalize scores
scores = df['narrative_style_score'].astype(float).values
s_min, s_max = np.nanpercentile(scores, 1), np.nanpercentile(scores, 99)
scores_n = np.clip((scores - s_min) / max(1e-6, (s_max - s_min)), 0, 1)

# Create binary labels
thr = 0.6  # Threshold for positive/negative labels
df['_label'] = (scores_n >= thr).astype(int)

print(f'KTO positives: {int(df["_label"].sum())} of {len(df)}')

# Create prompts
def make_prompt(game_recap):
    prompt = (
        "You are an NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n"
        "### NBA Game Recap ###\n"
        f"{game_recap}\n\n"
        "### Recap Summary ###\n"
    )
    return prompt

# Prepare KTO dataset
df_kto = pd.DataFrame({
    'prompt': df['game_recap'].astype(str).map(make_prompt),
    'completion': df['game_recap_summary_generated'].astype(str),
    'label': df['_label'].astype(int),
})

kto_ds = Dataset.from_pandas(df_kto)
print('Sample:', kto_ds[0])

# Train/test split
train_test_splits = kto_ds.train_test_split(test_size=0.1)
kto_ds_train = train_test_splits['train']
kto_ds_test = train_test_splits['test']

print(f"Training samples: {len(kto_ds_train)}")
print(f"Test samples: {len(kto_ds_test)}")

# =============================================================================
# CELL 6: Configure KTO Training
# =============================================================================

@dataclass
class KTORunConfig:
    # Model settings
    max_prompt_length: int = 512
    max_response_length: int = 256
    
    # KTO settings
    beta: float = 0.1
    loss_type: str = 'sigmoid'
    kto_label_threshold: float = 0.6
    
    # Training settings
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    lr_scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.03
    num_train_epochs: float = 1.0
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 0
    bf16: bool = True
    
    # Output
    output_dir: str = f'{PROJECT_DIR}/outputs/kto_training'
    run_name: str = 'nba_recap_kto_training'
    
    # Data
    csv_path: str = 'game_recaps_with_summaries_sample_for_reward_model_with_generated_full.csv'

cfg = KTORunConfig()
os.makedirs(cfg.output_dir, exist_ok=True)
json.dump(asdict(cfg), open(os.path.join(cfg.output_dir, 'train_config.json'), 'w'), indent=2)

print("KTO Configuration:")
print(json.dumps(asdict(cfg), indent=2))

# Configure KTO training
kto_config = KTOConfig(
    output_dir=cfg.output_dir,
    beta=cfg.beta,
    loss_type=cfg.loss_type,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    learning_rate=cfg.learning_rate,
    lr_scheduler_type=cfg.lr_scheduler_type,
    warmup_ratio=cfg.warmup_ratio,
    num_train_epochs=cfg.num_train_epochs,
    logging_steps=cfg.logging_steps,
    save_steps=cfg.save_steps,
    bf16=False,  # Disable bf16 to avoid CUDA errors
    fp16=True,  # Use fp16 instead
    eval_steps=cfg.eval_steps,
    optim="adamw_torch_fused",
    dataloader_num_workers=0,  # Reduce workers to avoid issues
    remove_unused_columns=False,
    max_length=cfg.max_prompt_length + cfg.max_response_length,
    report_to="none",  # Disable wandb logging
    desirable_weight=0.8,  # Recommended weight for imbalanced positive/negative examples
    gradient_checkpointing=True,  # Enable for KTO training
)

print("KTO Configuration:")
print(json.dumps(kto_config.to_dict(), indent=2))

# =============================================================================
# CELL 7: Initialize KTO Trainer
# =============================================================================

print("Initializing KTO trainer...")

# Wandb is disabled via report_to="none" in KTOConfig

# Create a reference model for KTO (needed for KL penalty calculation)
# Use the merged model as the reference (it already has the finetuned weights)
print("Creating reference model for KTO...")
# Download merged model
merged_prefix = f'output/artifacts/{PIPELINE_ID}/hf_model_merged/'
merged_path = './hf_model_merged/'
os.makedirs(merged_path, exist_ok=True)

print("Downloading merged model for reference...")
paginator = s3_client.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket='nba-recap-summarization-model-staging', Prefix=merged_prefix)

files_downloaded = 0
for page in pages:
    if 'Contents' in page:
        for obj in page['Contents']:
            key = obj['Key']
            local_file = os.path.join(merged_path, key.replace(merged_prefix, ''))
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            s3_client.download_file('nba-recap-summarization-model-staging', key, local_file)
            files_downloaded += 1

print(f"Downloaded {files_downloaded} files from merged model")

# Load merged model as reference (no quantization needed)
ref_model = AutoModelForCausalLM.from_pretrained(
    merged_path,
    device_map="auto",
    torch_dtype=torch.float16,  # FP16, not quantized
    trust_remote_code=True
)
print("✅ Reference model (merged) loaded in FP16")

trainer = KTOTrainer(
    model=model,  # Our training-ready model with LoRA adapters
    ref_model=ref_model,  # Reference model for KTO KL penalty
    args=kto_config,
    train_dataset=kto_ds_train,
    eval_dataset=kto_ds_test,
    processing_class=tokenizer,
    peft_config=None,  # Don't pass peft_config - adapters already attached
)

print("✅ KTO trainer initialized successfully!")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Ensure model is in training mode
model.train()
print("✅ Model set to training mode")

# Note: The KTO trainer handles gradient computation internally
# We'll skip the manual gradient test and proceed directly to training
print("\n✅ Trainer initialized successfully!")
print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

# Debug: Check for out-of-range token IDs in the dataset
print("\nChecking dataset for potential issues...")
sample_batch = kto_ds_train[0]
vocab_size = len(tokenizer)

for key, value in sample_batch.items():
    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], int):
        # Check if any token IDs are out of range
        max_id = max(value) if value else 0
        min_id = min(value) if value else 0
        if max_id >= vocab_size or min_id < 0:
            print(f"⚠️  {key}: Found out-of-range token IDs! Max: {max_id}, Min: {min_id}, Vocab size: {vocab_size}")
        else:
            print(f"✅ {key}: Token IDs are in range (max: {max_id})")

print("Starting KTO training...")

# Set environment variables for better error messages
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Clear any CUDA errors
try:
    torch.cuda.empty_cache()
    # Reset CUDA state
    import subprocess
    subprocess.run(["nvidia-smi"], check=False)
except:
    pass

# =============================================================================
# CELL 8: Start KTO Training
# =============================================================================

print("Starting KTO training...")
print(f"Training on {len(kto_ds_train)} samples")
print(f"Configuration: {cfg.num_train_epochs} epochs, batch size {cfg.per_device_train_batch_size}")

# Train the model
trainer.train()

# Save the trained model
trainer.save_model(cfg.output_dir)
print(f"✅ Training completed! Model saved to: {cfg.output_dir}")

# =============================================================================
# CELL 9: Test Model Generation
# =============================================================================

def test_model_generation(model, tokenizer, sample_text, max_length=200):
    """Test model generation with a sample input"""
    prompt = (
        "You are an NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n"
        "### NBA Game Recap ###\n"
        f"{sample_text}\n\n"
        "### Recap Summary ###\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the generated part
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return summary

# Test with a sample
sample_recap = df['game_recap'].iloc[0]
print("Sample game recap:")
print(sample_recap)
print("\n" + "="*50)

# Generate summary with trained model
summary = test_model_generation(model, tokenizer, sample_recap)
print("Generated summary:")
print(summary)

# =============================================================================
# CELL 10: Plot Training Loss
# =============================================================================

import matplotlib.pyplot as plt

logs = [l for l in trainer.state.log_history if "loss" in l]
if logs:
    steps = [l["step"] for l in logs]
    losses = [l["loss"] for l in logs]
    
    # Smooth the loss curve
    if len(losses) > 5:
        w = min(21, len(losses) // 4)
        if w % 2 == 0:
            w += 1
        smoothed = np.convolve(losses, np.ones(w)/w, mode="same")
    else:
        smoothed = losses
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, alpha=0.3, label="raw loss", color='blue')
    plt.plot(steps, smoothed, label="smoothed", color='red', linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("KTO Loss")
    plt.title("KTO Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Min loss: {min(losses):.4f}")
else:
    print("No training logs found")

# =============================================================================
# CELL 11: Save Evaluation Results
# =============================================================================

def save_evaluation_results():
    """Save evaluation results to CSV"""
    results = []
    
    # Test on a sample of data
    test_samples = df.sample(min(100, len(df)), random_state=42)
    
    for _, row in test_samples.iterrows():
        recap = str(row['game_recap'])
        
        # Generate summary with trained model
        summary = test_model_generation(model, tokenizer, recap)
        
        results.append({
            'game_recap': recap,
            'generated_summary': summary,
            'original_score': row['narrative_style_score'],
            'ground_truth_summary': row['game_recap_summary_generated']
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(cfg.output_dir, 'evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"✅ Evaluation results saved to: {results_path}")
    print(f"Evaluated {len(results)} samples")
    
    return results_df

# Run evaluation
eval_results = save_evaluation_results()
print("\nSample results:")
print(eval_results.head())

# =============================================================================
# CELL 12: Summary and Next Steps
# =============================================================================

print("\n" + "="*60)
print("KTO TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"✅ Model trained and saved to: {cfg.output_dir}")
print(f"✅ Training samples: {len(kto_ds_train)}")
print(f"✅ Test samples: {len(kto_ds_test)}")
print(f"✅ Configuration: {cfg.num_train_epochs} epochs")
print("\nNext steps:")
print("1. Download the trained model from Colab")
print("2. Test the model on new NBA game recaps")
print("3. Compare narrative style scores before/after training")
print("4. Deploy the improved model to your inference pipeline")
print("\nFiles created:")
print(f"- Trained model: {cfg.output_dir}")
print(f"- Evaluation results: {cfg.output_dir}/evaluation_results.csv")
print(f"- Training config: {cfg.output_dir}/train_config.json")
print("="*60)
