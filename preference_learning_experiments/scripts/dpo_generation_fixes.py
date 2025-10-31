#!/usr/bin/env python3
"""
Fixed DPO Training and Evaluation Script
Addresses generation issues and improves inference quality
"""

import pandas as pd
import numpy as np
import random
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
import json
import os

warnings.filterwarnings("ignore")

def preprocess_text(text: str) -> str:
    """Preprocess text to make it more tokenizer-friendly."""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    # Limit text length to prevent token overflow
    if len(text) > 1500:  # Reduced from 2000
        text = text[:1500] + "..."
    
    text = re.sub(r'(\d+)-(\d+)', r'\1 to \2', text)
    text = re.sub(r'(\d+)-pointer', r'\1 pointer', text)
    text = re.sub(r'(\d+)-point', r'\1 point', text)
    text = re.sub(r'\b(\d+)s\b', r'\1 pointers', text)
    text = re.sub(r'(\d+):(\d+)', r'\1 minutes \2 seconds', text)
    text = re.sub(r'(\d+)/(\d+)', r'\1 out of \2', text)
    text = re.sub(r'(\d+)%', r'\1 percent', text)
    return text

def postprocess_text(text: str) -> str:
    """Postprocess generated text to restore proper formatting."""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    # Clean up common generation artifacts
    text = re.sub(r'^Rewritten Summary:\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Summary:\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Evaluation:.*?(?=\n\n|\n[A-Z]|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'\|.*?\|.*?\n', '', text)  # Remove table rows
    text = re.sub(r'Example \d+:', '', text)  # Remove example references
    
    # Restore formatting
    text = re.sub(r'(\d+) to (\d+)', r'\1-\2', text)
    text = re.sub(r'three pointer', '3-pointer', text)
    text = re.sub(r'three point', '3-point', text)
    text = re.sub(r'two pointer', '2-pointer', text)
    text = re.sub(r'two point', '2-point', text)
    text = re.sub(r'three pointers', '3-pointers', text)
    text = re.sub(r'two pointers', '2-pointers', text)
    text = re.sub(r'(\d+) minutes (\d+) seconds', r'\1:\2', text)
    text = re.sub(r'(\d+) out of (\d+)', r'\1/\2', text)
    text = re.sub(r'(\d+) percent', r'\1%', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text

def build_simple_prompt(game_recap: str) -> str:
    """Build a simple, focused prompt for generation."""
    # Truncate recap to reasonable length
    if len(game_recap) > 800:
        game_recap = game_recap[:800] + "..."
    
    return f"You are an NBA Analyst. Summarize the following NBA game recap into a recap synthesis.\n\n### NBA Game Recap ###\n{game_recap}\n\n### Recap Summary ###\n"

def generate_with_model_fixed(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """
    Fixed generation function with better parameters and error handling.
    """
    try:
        # Ensure model is in eval mode
        model.eval()
        
        # Tokenize with proper truncation
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,  # Reduced from 2500
            padding=True
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Check input length
        input_length = inputs['input_ids'].shape[1]
        print(f"🔍 Input length: {input_length} tokens")
        
        if input_length > 900:  # Too long, truncate more
            print(f"⚠️ Input too long ({input_length}), truncating...")
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=800,  # More aggressive truncation
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_length = inputs['input_ids'].shape[1]
            print(f"🔍 New input length: {input_length} tokens")
        
        # Generate with better parameters
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
                use_cache=True,  # Enable cache for better performance
                early_stopping=True,
                num_beams=1
            )
        
        # Extract only the generated part
        generated_ids = outputs[0][input_length:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Postprocess
        generated = postprocess_text(generated)
        
        print(f"🔍 Generated length: {len(generated)} chars, {len(generated.split())} words")
        
        return generated.strip()
        
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return ""

def test_generation_basic(model, tokenizer):
    """Test basic generation with a simple prompt."""
    print("\n" + "="*60)
    print("TESTING BASIC GENERATION")
    print("="*60)
    
    # Simple test prompt
    test_prompt = "You are an NBA Analyst. Summarize this game: The Lakers beat the Warriors 120-115. LeBron scored 30 points.\n\nSummary:"
    
    print(f"Test prompt: {test_prompt}")
    print(f"Prompt length: {len(tokenizer.encode(test_prompt))} tokens")
    
    result = generate_with_model_fixed(model, tokenizer, test_prompt, max_new_tokens=100)
    
    print(f"Generated result: '{result}'")
    print(f"Result length: {len(result)} chars")
    
    return result

def evaluate_dpo_results_fixed(model, tokenizer, eval_pairs, num_samples=10):
    """Fixed evaluation function with better generation."""
    print("\n" + "="*60)
    print("EVALUATING DPO RESULTS (FIXED)")
    print("="*60)
    
    # Convert to list if needed
    if hasattr(eval_pairs, 'to_list'):
        eval_list = eval_pairs.to_list()
    elif hasattr(eval_pairs, '__iter__'):
        eval_list = list(eval_pairs)
    else:
        eval_list = eval_pairs
    
    # Sample evaluation pairs
    eval_samples = random.sample(eval_list, min(num_samples, len(eval_list)))
    
    results = {
        'successful_generations': 0,
        'total_samples': len(eval_samples),
        'generated_lengths': [],
        'examples': []
    }
    
    print(f"\nEvaluating {len(eval_samples)} pairs...")
    
    for i, pair in enumerate(eval_samples):
        print(f"\n--- Sample {i+1}/{len(eval_samples)} ---")
        
        # Use the prompt from the pair
        prompt = pair['prompt']
        chosen = pair['chosen']
        rejected = pair['rejected']
        
        print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")
        print(f"Chosen: {chosen[:100]}...")
        print(f"Rejected: {rejected[:100]}...")
        
        # Generate with current model
        generated = generate_with_model_fixed(model, tokenizer, prompt)
        
        if generated and len(generated) > 20:  # Valid generation
            results['successful_generations'] += 1
            results['generated_lengths'].append(len(generated))
            
            results['examples'].append({
                'prompt': prompt[:200] + "...",
                'chosen': chosen[:200] + "...",
                'rejected': rejected[:200] + "...",
                'generated': generated[:200] + "...",
                'generated_length': len(generated)
            })
            
            print(f"✅ Generated: {generated[:100]}...")
        else:
            print(f"❌ Failed to generate valid output")
    
    # Calculate success rate
    success_rate = results['successful_generations'] / results['total_samples']
    avg_length = np.mean(results['generated_lengths']) if results['generated_lengths'] else 0
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Success Rate: {success_rate:.1%}")
    print(f"  Average Generated Length: {avg_length:.0f} chars")
    print(f"  Successful Generations: {results['successful_generations']}/{results['total_samples']}")
    
    return results

def before_after_comparison_fixed(model, tokenizer, original_model, original_df, num_examples=5):
    """Fixed before/after comparison with better generation."""
    print("\n" + "="*60)
    print("BEFORE/AFTER COMPARISON (FIXED)")
    print("="*60)
    
    # Sample examples
    samples = original_df.sample(num_examples)
    
    examples = []
    
    for i, (idx, row) in enumerate(samples.iterrows()):
        print(f"\n--- Example {i+1}/{len(samples)} ---")
        
        game_recap = str(row['game_recap'])
        
        # Create simple prompt
        prompt = build_simple_prompt(game_recap)
        
        print(f"Game recap length: {len(game_recap)} chars")
        print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")
        
        # Generate with original model
        original_output = ""
        if original_model is not None:
            original_output = generate_with_model_fixed(original_model, tokenizer, prompt)
            print(f"Original model output: {original_output[:100]}...")
        
        # Generate with tuned model
        tuned_output = generate_with_model_fixed(model, tokenizer, prompt)
        print(f"Tuned model output: {tuned_output[:100]}...")
        
        examples.append({
            'index': int(idx),
            'game_recap': game_recap[:500] + "..." if len(game_recap) > 500 else game_recap,
            'original_output': original_output,
            'tuned_output': tuned_output,
            'original_length': len(original_output),
            'tuned_length': len(tuned_output)
        })
    
    return examples

def main():
    """Main function to test the fixed generation."""
    
    # This would be called after your DPO training
    # You would load your trained model and test it
    
    print("This script contains the fixed generation functions.")
    print("Use these functions in your Colab notebook to fix the generation issues.")
    
    # Example usage:
    # 1. Load your trained model
    # 2. Test basic generation: test_generation_basic(model, tokenizer)
    # 3. Evaluate DPO results: evaluate_dpo_results_fixed(model, tokenizer, dpo_test)
    # 4. Compare before/after: before_after_comparison_fixed(model, tokenizer, original_model, original_df)

if __name__ == "__main__":
    main()

