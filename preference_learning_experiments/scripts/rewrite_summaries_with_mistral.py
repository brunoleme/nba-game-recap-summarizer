#!/usr/bin/env python3
"""
Script to rewrite NBA recap summaries using Mistral-7B-Instruct-v0.1
with high-quality examples as references.

This script:
1. Loads the CSV file with generated summaries and narrative style scores
2. Identifies high-quality examples (score >= 4.0)
3. For each low-quality example (score < 4.0):
   - Samples 3 high-quality examples
   - Creates a rewriting prompt using the official Mistral format
   - Uses Mistral to rewrite the summary
4. Saves results to a new CSV file
"""

import pandas as pd
import numpy as np
import random
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import os
from typing import List, Dict, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def preprocess_text(text: str) -> str:
    """
    Preprocess text to make it more tokenizer-friendly.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
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
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
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

def build_rewriting_prompt(recap: str, recap_summary: str, example_1: str, example_2: str, example_3: str) -> str:
    """
    Build a rewriting prompt using the official Mistral format.
    
    Args:
        recap: Original game recap
        recap_summary: Current summary to be rewritten
        example_1: First high-quality example
        example_2: Second high-quality example
        example_3: Third high-quality example
        
    Returns:
        Formatted prompt string
    """
    return (
        "<s>[INST] You are an NBA chief editor. Your goal is to evaluate recap summaries made by an NBA Analyst and rewrite them if needed to improve narrative style.\n\n"
        "Your summaries are evaluated on a Narrative Style Score (1-5) based on these metrics:\n\n"
        "• Bulletiness (0-1, lower is better): Avoid bullet points (•, -, *), headings (Score:, Top Performers:), and list-like structures - prefer narrative flow.\n\n"
        "• Structure (0-1, higher is better): Aim for 3-7 sentences with 12-30 words average per sentence for natural readability.\n\n"
        "• Connectors (0-1, higher is better): Use discourse connectors (however, despite, while, as, after, because, therefore, meanwhile, although, whereas, furthermore, moreover, consequently, thus, hence, additionally, similarly, conversely) to create smooth narrative flow between sentences.\n\n"
        "• Coverage (0-1, higher is better): Maintain semantic alignment between the original recap and your summary - ensure key facts, events, and outcomes are accurately preserved without hallucination.\n\n"
        "• Readability (0-1, higher is better): Balance sentence complexity - clear, readable prose that isn't overly simple or unnecessarily complex.\n\n"
        "The overall Narrative Style Score is a weighted combination: bulletiness (30%), structure (35%), coherence (20%), and coverage (15%), scaled to 1-5.\n\n"
        "Here are the best examples from the dataset:\n\n"
        f"Example 1:\n{example_1}\n\n"
        f"Example 2:\n{example_2}\n\n"
        f"Example 3:\n{example_3}\n\n"
        f"recap: {recap}\n\n"
        f"recap_summary: {recap_summary}\n\n"
        "Please evaluate this summary and provide a rewritten version if improvements are needed. Focus on narrative style improvements. [/INST]"
    )

def rewrite_summary(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """
    Use Mistral model to rewrite a summary based on the prompt.
    
    Args:
        model: Mistral model
        tokenizer: Mistral tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Rewritten summary
    """
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, padding=True)
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=False
            )
        
        # Decode output
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Postprocess
        generated = postprocess_text(generated)
        
        return generated.strip()
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return ""

def load_and_prepare_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSV data and separate high-quality from low-quality examples.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Tuple of (high_quality_df, low_quality_df)
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total examples: {len(df)}")
    print(f"Narrative style score range: {df['narrative_style_score'].min():.2f} - {df['narrative_style_score'].max():.2f}")
    
    # Filter out rows with missing data
    df = df.dropna(subset=['game_recap', 'game_recap_summary_generated', 'narrative_style_score'])
    
    # Separate high and low quality examples
    high_quality = df[df['narrative_style_score'] >= 4.0].copy()
    low_quality = df[df['narrative_style_score'] < 4.0].copy()
    
    print(f"High quality examples (score >= 4.0): {len(high_quality)}")
    print(f"Low quality examples (score < 4.0): {len(low_quality)}")
    
    return high_quality, low_quality

def load_mistral_model():
    """
    Load Mistral model and tokenizer.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print("Loading Mistral-7B-Instruct-v0.1...")
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded on device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    return model, tokenizer

def main():
    """Main execution function."""
    
    # Configuration
    csv_path = "preference_learning_experiments/data/game_recaps_with_summaries_sample_for_reward_model_with_generated_full.csv"
    output_path = "preference_learning_experiments/data/game_recaps_with_rewritten_summaries.csv"
    max_examples_to_process = 100  # Limit for testing - remove this line for full processing
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. This will be very slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    try:
        # Load data
        high_quality_df, low_quality_df = load_and_prepare_data(csv_path)
        
        if len(high_quality_df) < 3:
            print("❌ Not enough high-quality examples (need at least 3)")
            return
        
        if len(low_quality_df) == 0:
            print("✅ No low-quality examples to rewrite!")
            return
        
        # Load model
        model, tokenizer = load_mistral_model()
        
        # Process low-quality examples
        print(f"\n🔄 Processing {min(len(low_quality_df), max_examples_to_process)} low-quality examples...")
        
        results = []
        
        for idx, row in tqdm(low_quality_df.head(max_examples_to_process).iterrows(), 
                            total=min(len(low_quality_df), max_examples_to_process),
                            desc="Rewriting summaries"):
            
            # Sample 3 high-quality examples
            sample_examples = high_quality_df.sample(n=3, random_state=idx)
            
            # Preprocess inputs
            recap = preprocess_text(row['game_recap'])
            recap_summary = preprocess_text(row['game_recap_summary_generated'])
            
            example_1 = preprocess_text(sample_examples.iloc[0]['game_recap_summary_generated'])
            example_2 = preprocess_text(sample_examples.iloc[1]['game_recap_summary_generated'])
            example_3 = preprocess_text(sample_examples.iloc[2]['game_recap_summary_generated'])
            
            # Build prompt
            prompt = build_rewriting_prompt(recap, recap_summary, example_1, example_2, example_3)
            
            # Check token limits
            prompt_tokens = len(tokenizer.encode(prompt))
            if prompt_tokens > 2000:  # Leave room for generation
                print(f"⚠️  Prompt too long ({prompt_tokens} tokens) for row {idx}, skipping...")
                continue
            
            # Generate rewritten summary
            rewritten_summary = rewrite_summary(model, tokenizer, prompt)
            
            if rewritten_summary:
                # Store result
                result = row.to_dict()
                result['rewritten_summary'] = rewritten_summary
                result['original_narrative_score'] = row['narrative_style_score']
                result['prompt_tokens'] = prompt_tokens
                results.append(result)
                
                print(f"✅ Row {idx}: Original score {row['narrative_style_score']:.2f} -> Rewritten")
            else:
                print(f"❌ Row {idx}: Failed to generate rewritten summary")
        
        # Save results
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            print(f"\n✅ Saved {len(results)} rewritten summaries to {output_path}")
            
            # Show statistics
            print(f"\n📊 Statistics:")
            print(f"Original average score: {results_df['original_narrative_score'].mean():.2f}")
            print(f"Average prompt tokens: {results_df['prompt_tokens'].mean():.0f}")
        else:
            print("❌ No results to save")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

