#!/usr/bin/env python3
"""
Simplified script using pipeline for rewriting NBA recap summaries.
"""

import pandas as pd
import numpy as np
import random
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def preprocess_text(text: str) -> str:
    """Preprocess text to make it more tokenizer-friendly."""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
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
    return text

def build_rewriting_prompt(recap: str, recap_summary: str, example_1: str, example_2: str, example_3: str) -> str:
    """Build rewriting prompt using official Mistral format."""
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

def rewrite(model, prompt: str, max_new_tokens: int = 512) -> str:
    """Use pipeline to rewrite summary."""
    try:
        # Generate using pipeline
        result = model(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=model.tokenizer.eos_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        # Extract generated text
        generated_text = result[0]['generated_text']
        
        # Postprocess
        generated_text = postprocess_text(generated_text)
        
        return generated_text.strip()
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return ""

def main():
    """Main execution function."""
    
    # Configuration
    csv_path = "preference_learning_experiments/data/game_recaps_with_summaries_sample_for_reward_model_with_generated_full.csv"
    output_path = "preference_learning_experiments/data/game_recaps_with_rewritten_summaries.csv"
    max_examples_to_process = 50  # Limit for testing
    
    print("🚀 Starting NBA Recap Summary Rewriting with Mistral-7B-Instruct-v0.1")
    
    try:
        # Load data
        print(f"📁 Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['game_recap', 'game_recap_summary_generated', 'narrative_style_score'])
        
        high_quality = df[df['narrative_style_score'] >= 4.0].copy()
        low_quality = df[df['narrative_style_score'] < 4.0].copy()
        
        print(f"📊 High quality examples (score >= 4.0): {len(high_quality)}")
        print(f"📊 Low quality examples (score < 4.0): {len(low_quality)}")
        
        if len(high_quality) < 3:
            print("❌ Not enough high-quality examples (need at least 3)")
            return
        
        if len(low_quality) == 0:
            print("✅ No low-quality examples to rewrite!")
            return
        
        # Load model using pipeline
        print("🤖 Loading Mistral-7B-Instruct-v0.1...")
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            model_kwargs={"use_cache": False}
        )
        
        print(f"✅ Model loaded on device: {next(model.parameters()).device}")
        
        # Process examples
        print(f"\n🔄 Processing {min(len(low_quality), max_examples_to_process)} low-quality examples...")
        
        results = []
        
        for idx, row in tqdm(low_quality.head(max_examples_to_process).iterrows(), 
                            total=min(len(low_quality), max_examples_to_process),
                            desc="Rewriting summaries"):
            
            # Sample 3 high-quality examples
            sample_examples = high_quality.sample(n=3, random_state=idx)
            
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
            if prompt_tokens > 2000:
                print(f"⚠️  Prompt too long ({prompt_tokens} tokens) for row {idx}, skipping...")
                continue
            
            # Generate rewritten summary
            rewritten_summary = rewrite(pipe, prompt)
            
            if rewritten_summary:
                result = row.to_dict()
                result['rewritten_summary'] = rewritten_summary
                result['original_narrative_score'] = row['narrative_style_score']
                result['prompt_tokens'] = prompt_tokens
                results.append(result)
                
                print(f"✅ Row {idx}: Original score {row['narrative_style_score']:.2f}")
            else:
                print(f"❌ Row {idx}: Failed to generate")
        
        # Save results
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            print(f"\n✅ Saved {len(results)} rewritten summaries to {output_path}")
            print(f"📊 Average original score: {results_df['original_narrative_score'].mean():.2f}")
        else:
            print("❌ No results to save")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

