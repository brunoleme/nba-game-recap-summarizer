# Google Colab Notebook Code for Loading Fine-tuned Model
# Pipeline ID: a03cefe5-689d-48b5-b43f-78cc479c1ba4
# S3 Path: s3://nba-recap-summarization-model-staging/output/artifacts/a03cefe5-689d-48b5-b43f-78cc479c1ba4/hf_model_merged/

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
    s3_prefix = 'output/artifacts/a03cefe5-689d-48b5-b43f-78cc479c1ba4/hf_model_merged/'
    local_path = './hf_model/'
    
    # Create local directory
    os.makedirs(local_path, exist_ok=True)
    
    print(f"Downloading from: s3://{bucket_name}/{s3_prefix}")
    
    # List and download all files
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
        print(f"Model download completed! Downloaded {files_downloaded} files")
        return local_path
    else:
        raise RuntimeError("No files found in S3 path")

# Download the model
model_path = download_model_from_s3()

# Load the model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Try loading with different configurations to handle various model types
try:
    # First try with standard loading
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    print("Model loaded with standard configuration")
except Exception as e1:
    print(f"Standard loading failed: {e1}")
    try:
        # Try with low_cpu_mem_usage
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("Model loaded with low_cpu_mem_usage")
    except Exception as e2:
        print(f"Low CPU memory loading failed: {e2}")
        try:
            # Try with basic loading (no device_map)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            if torch.cuda.is_available():
                model = model.cuda()
            print("Model loaded with basic configuration")
        except Exception as e3:
            print(f"All loading methods failed. Last error: {e3}")
            raise e3

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

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
    model_name = "s3://nba-recap-summarization-model-staging/output/artifacts/a03cefe5-689d-48b5-b43f-78cc479c1ba4/hf_model/"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Successfully loaded model directly from S3!")
except Exception as e:
    print(f"Direct S3 loading failed: {e}")
    print("Use Method 1 (download first) instead.")
"""

# For DPO training, you'll also need to load your preference data
import pandas as pd

def load_preference_data():
    """Load the generated summaries with scores for DPO training"""
    # Upload your CSV file to Colab first, then load it
    df = pd.read_csv('game_recaps_with_summaries_sample_for_reward_model_with_generated_full.csv')
    
    # Filter for high-quality generated summaries (narrative_style_score > 4.0)
    high_quality = df[df['narrative_style_score'] > 4.0]
    
    print(f"Total samples: {len(df)}")
    print(f"High-quality samples (score > 4.0): {len(high_quality)}")
    
    return df, high_quality

# Load your preference data
# df, high_quality_df = load_preference_data()

print("\n" + "="*50)
print("Model loading setup complete!")
print("Next steps for DPO training:")
print("1. Upload your CSV file to Colab")
print("2. Run load_preference_data() to load your data")
print("3. Prepare your data in the format required by TRL")
print("4. Run DPO training with your preference data")
print("="*50)
