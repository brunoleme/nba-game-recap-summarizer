#!/usr/bin/env python3
"""
Debug batch generation specifically to understand the padding issue.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from loguru import logger

# Add the src directory to the path
sys.path.append('src')

def test_batch_generation_padding():
    """Test batch generation padding behavior"""
    print("🔍 Testing batch generation padding behavior...")
    
    # Load tokenizer exactly like the model does
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=False)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    print(f"Tokenizer padding_side: {tokenizer.padding_side}")
    
    # Test texts
    texts = [
        "The Lakers won the game 120-115.",
        "LeBron James scored 28 points and had 8 rebounds."
    ]
    
    print(f"Input texts: {texts}")
    
    # Test 1: Direct tokenizer call (like in summarize_recaps)
    print("\n📝 Test 1: Direct tokenizer call")
    prompts = [
        "You are an NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n"
        "### NBA Game Recap ###\n"
        f"{text}\n\n"
        "### Recap Summary ###\n"
        for text in texts
    ]
    
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,  # This should respect padding_side="left"
        truncation=True,
        max_length=512
    )
    
    print(f"Encoded shape: {enc['input_ids'].shape}")
    print(f"Attention mask shape: {enc['attention_mask'].shape}")
    
    # Check padding positions
    for i in range(len(texts)):
        attention_mask = enc['attention_mask'][i]
        first_real_token = (attention_mask == 1).nonzero(as_tuple=True)[0][0].item()
        last_real_token = (attention_mask == 1).nonzero(as_tuple=True)[0][-1].item()
        
        print(f"Text {i}:")
        print(f"  First real token at position: {first_real_token}")
        print(f"  Last real token at position: {last_real_token}")
        print(f"  Padding on left: {first_real_token > 0}")
        print(f"  Padding on right: {last_real_token < len(attention_mask) - 1}")
    
    # Test 2: Check if this would trigger the warning
    print("\n🎯 Test 2: Checking for right-padding warning")
    
    # This is what happens in the model.generate() call
    # The warning comes from the generation process, not the tokenization
    print("This is where the 'right-padding detected' warning would appear during generation...")
    
    # Test 3: Simulate the exact issue from the logs
    print("\n🔍 Test 3: Simulating the exact issue")
    
    # The issue might be that even though we set padding_side="left",
    # the generation process still detects right-padding in the input tensors
    
    # Let's check what the actual input looks like
    print("Input IDs (first 20 tokens):")
    for i in range(len(texts)):
        print(f"  Text {i}: {enc['input_ids'][i][:20].tolist()}")
    
    print("Attention mask (first 20 tokens):")
    for i in range(len(texts)):
        print(f"  Text {i}: {enc['attention_mask'][i][:20].tolist()}")

if __name__ == "__main__":
    print("🚀 Starting batch generation padding debug...")
    test_batch_generation_padding()
    print("\n✅ Debug complete!")
