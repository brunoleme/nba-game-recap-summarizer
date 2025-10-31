#!/usr/bin/env python3
"""
Improved script to rewrite NBA recap summaries using Mistral-7B-Instruct-v0.1
with better prompt engineering and output handling.
"""

import pandas as pd
import numpy as np
import random
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def preprocess_text(text: str) -> str:
    """Preprocess text to make it more tokenizer-friendly."""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    # Limit text length to prevent token overflow
    if len(text) > 2000:
        text = text[:2000] + "..."
    
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

def build_rewriting_prompt(recap: str, recap_summary: str, example_1: str, example_2: str, example_3: str) -> str:
    """Build rewriting prompt with clearer instructions."""
    return (
        "<s>[INST] You are an NBA chief editor. Rewrite the following recap summary to improve its narrative style.\n\n"
        "Guidelines for high-quality summaries:\n"
        "• Write in narrative prose (no bullet points or headings)\n"
        "• Use 3-7 sentences with 12-30 words each\n"
        "• Include discourse connectors (however, while, because, therefore, etc.)\n"
        "• Preserve all key facts accurately\n"
        "• Write clear, engaging prose\n\n"
        "High-quality examples:\n\n"
        f"Example 1: {example_1}\n\n"
        f"Example 2: {example_2}\n\n"
        f"Example 3: {example_3}\n\n"
        f"Original recap: {recap}\n\n"
        f"Current summary: {recap_summary}\n\n"
        "Rewrite the current summary to match the quality and style of the examples above. "
        "Provide only the rewritten summary, nothing else. [/INST]"
    )

def rewrite_summary(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Use Mistral model to rewrite a summary with better parameters."""
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800, padding=True)
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with better parameters
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,  # Lower temperature for more focused output
                top_p=0.8,        # Lower top_p for more focused output
                top_k=50,         # Add top_k sampling
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Higher repetition penalty
                use_cache=False,
                early_stopping=True,      # Stop at EOS token
                num_beams=1              # Use greedy decoding for consistency
            )
        
        # Decode output
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Postprocess
        generated = postprocess_text(generated)
        
        # Validate output quality
        if len(generated.split()) < 20:  # Too short
            return ""
        if len(generated.split()) > 200:  # Too long
            generated = ' '.join(generated.split()[:200]) + "..."
        
        return generated.strip()
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return ""

def load_and_prepare_data(csv_path: str) -> tuple:
    """Load CSV data and separate high-quality from low-quality examples."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total examples: {len(df)}")
    print(f"Narrative style score range: {df['narrative_style_score'].min():.2f} - {df['narrative_style_score'].max():.2f}")
    
    # Filter out rows with missing data
    df = df.dropna(subset=['game_recap', 'game_recap_summary_generated', 'narrative_style_score'])
    
    # Separate high and low quality examples
    high_quality = df[df['narrative_style_score'] >= 4.0].copy()
    low_quality = df[df['narrative_style_score'] < 3.75].sort_values('narrative_style_score').copy()
    
    print(f"High quality examples (score >= 4.0): {len(high_quality)}")
    print(f"Low quality examples (score < 3.75): {len(low_quality)}")
    
    return high_quality, low_quality

def load_mistral_model():
    """Load Mistral model and tokenizer."""
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
    csv_path = "game_recaps_with_summaries_sample_for_reward_model_with_generated_full.csv"
    output_path = "game_recaps_with_rewritten_summaries_improved.csv"
    max_examples_to_process = 20  # Increased for better testing
    
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
            
            # Preprocess inputs with length limits
            recap = preprocess_text(row['game_recap'])
            recap_summary = preprocess_text(row['game_recap_summary_generated'])
            
            example_1 = preprocess_text(sample_examples.iloc[0]['game_recap_summary_generated'])
            example_2 = preprocess_text(sample_examples.iloc[1]['game_recap_summary_generated'])
            example_3 = preprocess_text(sample_examples.iloc[2]['game_recap_summary_generated'])
            
            # Build prompt
            prompt = build_rewriting_prompt(recap, recap_summary, example_1, example_2, example_3)
            
            # Check token limits
            prompt_tokens = len(tokenizer.encode(prompt))
            if prompt_tokens > 2000:
                print(f"⚠️  Prompt too long ({prompt_tokens} tokens) for row {idx}, skipping...")
                continue
            
            # Generate rewritten summary
            rewritten_summary = rewrite_summary(model, tokenizer, prompt)
            
            if rewritten_summary:
                result = row.to_dict()
                result['rewritten_summary'] = rewritten_summary
                result['original_narrative_score'] = row['narrative_style_score']
                result['prompt_tokens'] = prompt_tokens
                results.append(result)
                
                print(f"✅ Row {idx}: Original score {row['narrative_style_score']:.2f}")
                print(f"   Generated: {rewritten_summary[:100]}...")
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
            print(f"Average rewritten length: {results_df['rewritten_summary'].str.split().str.len().mean():.0f} words")
        else:
            print("❌ No results to save")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

