# Google Colab Notebook Code for Loading Fine-tuned Model
# Pipeline ID: 15b93b90-b1d0-4237-8c0c-8fca5a9190fe
# S3 Path: s3://nba-recap-summarization-model-staging/output/artifacts/15b93b90-b1d0-4237-8c0c-8fca5a9190fe/hf_model_merged/

# Install required packages
print("Installing required packages...")
!pip install --upgrade pip
!pip install transformers torch datasets accelerate peft trl
!pip install boto3 awscli
!pip install bitsandbytes
!pip install sentencepiece protobuf

# Verify installations
import importlib
required_packages = ['transformers', 'torch', 'datasets', 'accelerate', 'peft', 'trl', 'boto3', 'bitsandbytes']
for package in required_packages:
    try:
        importlib.import_module(package)
        print(f"✓ {package} installed successfully")
    except ImportError as e:
        print(f"✗ {package} installation failed: {e}")

print("Package installation completed!")

# Configure AWS credentials (you'll need to add your credentials)
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_AWS_ACCESS_KEY_ID'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_AWS_SECRET_ACCESS_KEY'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'  # or your region

# Method 1: Download from S3 and load locally
import boto3
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model_from_s3():
    """Download the fine-tuned model from S3 to local storage"""
    s3_client = boto3.client('s3')
    bucket_name = 'nba-recap-summarization-model-staging'
    base_prefix = 'output/artifacts/15b93b90-b1d0-4237-8c0c-8fca5a9190fe/hf_model_base/'
    adapters_prefix = 'output/artifacts/15b93b90-b1d0-4237-8c0c-8fca5a9190fe/hf_model_adapters/'
    merged_prefix = 'output/artifacts/15b93b90-b1d0-4237-8c0c-8fca5a9190fe/hf_model_merged/'
    
    base_path = './hf_model_base/'
    adapters_path = './hf_model_adapters/'
    merged_path = './hf_model_merged/'
    
    # Create local directories
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(adapters_path, exist_ok=True)
    os.makedirs(merged_path, exist_ok=True)
    
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
                    
                    # Create subdirectories if needed
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    
                    # Download file
                    print(f"Downloading {key}...")
                    s3_client.download_file(bucket_name, key, local_file)
                    files_downloaded += 1
        
        if files_downloaded > 0:
            print(f"{name} download completed! Downloaded {files_downloaded} files")
            return True
        else:
            print(f"No files found for {name}")
            return False
    
    # Try to download all three formats
    base_success = download_s3_prefix(base_prefix, base_path, "Base Model")
    adapters_success = download_s3_prefix(adapters_prefix, adapters_path, "LoRA Adapters")
    merged_success = download_s3_prefix(merged_prefix, merged_path, "Merged Model")
    
    if base_success and adapters_success:
        print("✅ Successfully downloaded base model and adapters for KTO training!")
        return base_path, adapters_path, merged_path
    elif merged_success:
        print("⚠️ Only merged model available - will need to attach new LoRA for KTO")
        return None, None, merged_path
    else:
        raise RuntimeError("No model files found in S3")

# Download the model
base_path, adapters_path, merged_path = download_model_from_s3()

# Load the model and tokenizer for KTO training
print("Loading model and tokenizer for KTO training...")

if base_path and adapters_path:
    # Load training-ready model (base + adapters)
    print("Loading training-ready model (base + LoRA adapters)...")
    
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Configure quantization for training
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4"
    )
    
    # Load base model with quantization
    base = AutoModelForCausalLM.from_pretrained(
        base_path, 
        device_map="auto", 
        quantization_config=bnb_cfg, 
        torch_dtype=torch.bfloat16
    )
    base = prepare_model_for_kbit_training(base)
    
    # Attach the LoRA adapters
    model = PeftModel.from_pretrained(base, adapters_path)
    
    # Critical: enable gradients for training
    inner = model.get_base_model() if hasattr(model, "get_base_model") else getattr(model, "model", model)
    if hasattr(inner, "enable_input_require_grads"):
        inner.enable_input_require_grads()
    
    model.train()
    print("✅ Training-ready model loaded with LoRA adapters!")
    
else:
    # Fallback: load merged model (will need new LoRA for training)
    print("Loading merged model (will attach new LoRA for training)...")
    
    tokenizer = AutoTokenizer.from_pretrained(merged_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load merged model
    model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    print("⚠️ Merged model loaded - will need to attach new LoRA for KTO training")

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
        # Create a simple test input
        test_input = tokenizer("Test input", return_tensors="pt")
        if torch.cuda.is_available():
            test_input = {k: v.cuda() for k, v in test_input.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**test_input)
        
        print("✅ Model forward pass successful")
        
        # Test gradient computation
        outputs.logits.sum().backward()
        print("✅ Gradient computation successful")
        
        # Check if gradients are flowing
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"✅ Gradients flowing: {has_grads}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
    
    print("="*50)

print("Model loaded successfully!")
verify_kto_readiness(model)

# Test the model with a sample input
def test_model():
    """Test the loaded model with a sample game recap"""
    sample_text = """
    The Lakers faced off against the Warriors in a thrilling matchup. LeBron James scored 30 points and grabbed 12 rebounds, while Stephen Curry led the Warriors with 28 points. The game went into overtime, with the Lakers ultimately winning 115-112.
    """
    
    # Tokenize input
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=512)
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"Model device: {device}")
    print(f"Input device: {inputs['input_ids'].device}")
    
    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_beams=4,
            early_stopping=True,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated Summary:")
    print(summary)

# Run test
test_model()

# Alternative Method 2: Load directly from S3 using transformers (if supported)
# This might not work depending on your model format, but worth trying
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try loading directly from S3 (this might not work for all model types)
try:
    model_name = "s3://nba-recap-summarization-model-staging/output/artifacts/15b93b90-b1d0-4237-8c0c-8fca5a9190fe/hf_model_merged/"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Successfully loaded model directly from S3!")
except Exception as e:
    print(f"Direct S3 loading failed: {e}")
    print("Use Method 1 (download first) instead.")
"""

# For KTO training, you'll also need to load your preference data
import pandas as pd

def load_preference_data():
    """Load the generated summaries with scores for KTO training"""
    # Upload your CSV file to Colab first, then load it
    df = pd.read_csv('game_recaps_with_summaries_sample_for_reward_model_with_generated_full.csv')
    
    # Filter for high-quality generated summaries (narrative_style_score > 4.0)
    high_quality = df[df['narrative_style_score'] > 4.0]
    
    print(f"Total samples: {len(df)}")
    print(f"High-quality samples (score > 4.0): {len(high_quality)}")
    
    # Show score distribution
    print("\nScore distribution:")
    print(df['narrative_style_score'].describe())
    
    return df, high_quality

def prepare_kto_data(df, min_score=3.0):
    """Prepare data for KTO training with preference scores"""
    # Filter samples with sufficient quality
    filtered_df = df[df['narrative_style_score'] >= min_score].copy()
    
    # Create KTO format: (prompt, response, preference_score)
    kto_data = []
    for _, row in filtered_df.iterrows():
        prompt = f"Summarize this NBA game recap: {row['game_recap']}"
        response = row['game_recap_summary_generated']
        preference_score = row['narrative_style_score'] / 5.0  # Normalize to 0-1
        
        kto_data.append({
            'prompt': prompt,
            'response': response,
            'preference_score': preference_score,
            'original_score': row['narrative_style_score']
        })
    
    print(f"Prepared {len(kto_data)} samples for KTO training")
    return kto_data

# Load your preference data
# df, high_quality_df = load_preference_data()
# kto_data = prepare_kto_data(df)

# Example KTO training setup (uncomment when ready)
"""
from trl import KTOTrainer, KTOConfig
from datasets import Dataset

# Convert to HuggingFace Dataset format
dataset = Dataset.from_list(kto_data)

# Configure KTO training
kto_config = KTOConfig(
    model_name="meta-llama/Llama-3.2-1B-Instruct",  # Base model
    learning_rate=5e-6,
    batch_size=4,
    num_train_epochs=3,
    max_length=512,
    beta=0.1,  # KTO beta parameter
    loss_type="kto",  # Use KTO loss
)

# Initialize trainer
trainer = KTOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=kto_config,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./kto_finetuned_model")
"""

print("\n" + "="*50)
print("Model loading setup complete!")
print("Next steps for KTO training:")
print("1. Upload your CSV file to Colab")
print("2. Run load_preference_data() to load your data")
print("3. Run prepare_kto_data() to format for KTO training")
print("4. Uncomment and run the KTO training example above")
print("5. Use the narrative_style_score as your preference signal")
print("="*50)
