#!/usr/bin/env python3
"""
Debug script to understand the padding issue locally.
This will help us see exactly what's happening with tokenization and generation.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from loguru import logger

# Add the src directory to the path
sys.path.append('src')

from nba_game_recap_summarizer.finetuning.models.llama_model import LlamaRecapSummarizationModel
from nba_game_recap_summarizer.finetuning.utils.tokenization_utils import (
    preprocess_text, 
    postprocess_text, 
    add_custom_tokens_to_tokenizer
)

def test_tokenizer_padding():
    """Test tokenizer padding behavior"""
    print("🔍 Testing tokenizer padding behavior...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=False)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test 1: Default padding side
    print(f"Default padding_side: {tokenizer.padding_side}")
    
    # Test 2: Set to left
    tokenizer.padding_side = "left"
    print(f"After setting to left: {tokenizer.padding_side}")
    
    # Test 3: Add custom tokens
    tokenizer = add_custom_tokens_to_tokenizer(tokenizer)
    print(f"After adding custom tokens: {tokenizer.padding_side}")
    
    # Test 4: Test padding behavior
    texts = [
        "Short text",
        "This is a much longer text that will require padding when batched together"
    ]
    
    print("\n📝 Testing padding behavior:")
    print("Input texts:")
    for i, text in enumerate(texts):
        print(f"  {i}: {text}")
    
    # Test with padding=True (should respect padding_side)
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    print(f"\nEncoded shape: {encoded['input_ids'].shape}")
    print(f"Attention mask shape: {encoded['attention_mask'].shape}")
    
    # Check if padding is on the left
    print("\n🔍 Checking padding positions:")
    for i in range(len(texts)):
        input_ids = encoded['input_ids'][i]
        attention_mask = encoded['attention_mask'][i]
        
        # Find where the actual text starts (first non-pad token)
        first_real_token = (attention_mask == 1).nonzero(as_tuple=True)[0][0].item()
        last_real_token = (attention_mask == 1).nonzero(as_tuple=True)[0][-1].item()
        
        print(f"Text {i}:")
        print(f"  First real token at position: {first_real_token}")
        print(f"  Last real token at position: {last_real_token}")
        print(f"  Total length: {len(input_ids)}")
        print(f"  Padding on left: {first_real_token > 0}")
        print(f"  Padding on right: {last_real_token < len(input_ids) - 1}")

def test_model_generation():
    """Test model generation with a single sample"""
    print("\n🤖 Testing model generation...")
    
    # Create a simple test case
    test_recap = """
    The Los Angeles Lakers defeated the Golden State Warriors 120-115 in a thrilling game at the Staples Center. 
    LeBron James led the Lakers with 28 points, 8 rebounds, and 7 assists. 
    Stephen Curry scored 32 points for the Warriors but it wasn't enough to secure the victory.
    The Lakers improved their record to 15-10 while the Warriors fell to 12-13.
    """
    
    print(f"Test recap: {test_recap.strip()}")
    
    try:
        # Initialize model (this will use our _initialize_model method)
        model = LlamaRecapSummarizationModel(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            use_quantization=True,
            quantization_type="4bit",
            peft_method="lora",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05
        )
        
        print(f"✅ Model initialized successfully")
        print(f"Tokenizer padding_side: {model.tokenizer.padding_side}")
        
        # Test single generation
        print("\n🎯 Testing single generation:")
        summary = model.summarize_recap(test_recap, max_length=200)
        print(f"Generated summary: {summary}")
        
        # Test batch generation (this is where the issue might be)
        print("\n🎯 Testing batch generation:")
        
        # Create a simple dataloader-like structure
        class SimpleBatch:
            def __init__(self, texts):
                self.texts = texts
                
            def __iter__(self):
                # Simulate what the dataloader does
                batch = {"game_recap": self.texts}
                yield batch
        
        # Test with our collator
        from nba_game_recap_summarizer.finetuning.data.nba_recap_dataset import CausalLMCollator
        
        # Create collator with our tokenizer
        collator = CausalLMCollator(tokenizer=model.tokenizer)
        
        # Create fake features (what the dataloader would create)
        features = [{
            "input_ids": model.tokenizer.encode(test_recap, add_special_tokens=True),
            "attention_mask": [1] * len(model.tokenizer.encode(test_recap, add_special_tokens=True)),
            "labels": model.tokenizer.encode(test_recap, add_special_tokens=True)
        }]
        
        print(f"Features before collator: {features[0]['input_ids'][:10]}...")
        
        # Apply collator
        batch = collator(features)
        print(f"Batch after collator:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  First 10 input_ids: {batch['input_ids'][0][:10].tolist()}")
        print(f"  First 10 attention_mask: {batch['attention_mask'][0][:10].tolist()}")
        
        # Check padding position
        attention_mask = batch['attention_mask'][0]
        first_real_token = (attention_mask == 1).nonzero(as_tuple=True)[0][0].item()
        last_real_token = (attention_mask == 1).nonzero(as_tuple=True)[0][-1].item()
        
        print(f"  First real token at position: {first_real_token}")
        print(f"  Last real token at position: {last_real_token}")
        print(f"  Padding on left: {first_real_token > 0}")
        print(f"  Padding on right: {last_real_token < len(attention_mask) - 1}")
        
    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting padding issue debug...")
    
    # Test 1: Tokenizer padding behavior
    test_tokenizer_padding()
    
    # Test 2: Model generation
    test_model_generation()
    
    print("\n✅ Debug complete!")
