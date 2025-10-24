#!/usr/bin/env python3
"""
Script to generate game recap summaries using the deployed endpoint and evaluate them
with AI-as-a-judge metrics including custom narrative style scoring.

This script will:
1. Read the CSV file with game recaps
2. Generate summaries using the deployed API endpoint
3. Evaluate summaries with multiple metrics
4. Add custom narrative style scoring
5. Save the results back to the CSV file
"""

import pandas as pd
import requests
import json
import time
import re
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SummaryGenerator:
    def __init__(self, endpoint_url: str, max_retries: int = 3, delay: float = 1.0):
        self.endpoint_url = endpoint_url
        self.max_retries = max_retries
        self.delay = delay
        
    def generate_summary(self, game_recap: str, max_length: int = 150) -> str:
        """Generate summary using the deployed endpoint"""
        payload = {
            "game_recap": game_recap,
            "max_length": max_length
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.endpoint_url}/summarize_recap",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get("summary", "")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for game recap")
                    return ""
        
        return ""

class NarrativeStyleEvaluator:
    def __init__(self):
        # Load sentence transformer for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def calculate_bulletiness_score(self, text: str) -> float:
        """Calculate bulletiness score (lower is better)"""
        lines = text.split('\n')
        bullet_lines = 0
        total_lines = len([line for line in lines if line.strip()])
        
        if total_lines == 0:
            return 1.0
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check for bullet points, dashes, or heading patterns
            if (line.startswith('-') or 
                line.startswith('•') or 
                line.startswith('*') or
                re.match(r'^(Score|Top Performers|Outcome|Key|Stats?):', line, re.IGNORECASE)):
                bullet_lines += 1
        
        return bullet_lines / total_lines
    
    def calculate_narrative_structure_score(self, text: str) -> float:
        """Calculate narrative structure score (0-1, higher is better)"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return 0.0
        if len(sentences) > 7:
            return 0.5  # Penalize too many sentences
        
        # Check sentence length distribution
        avg_length = np.mean([len(s.split()) for s in sentences])
        length_score = 1.0 if 12 <= avg_length <= 30 else max(0, 1 - abs(avg_length - 21) / 21)
        
        # Check for discourse connectors
        connectors = [
            'however', 'despite', 'while', 'as', 'after', 'because', 'therefore', 
            'meanwhile', 'although', 'though', 'whereas', 'furthermore', 'moreover',
            'consequently', 'thus', 'hence', 'additionally', 'similarly', 'conversely'
        ]
        
        text_lower = text.lower()
        connector_count = sum(1 for conn in connectors if conn in text_lower)
        connector_score = min(1.0, connector_count / 3)  # Normalize to 0-1
        
        # Combine scores
        structure_score = (length_score * 0.6 + connector_score * 0.4)
        return min(1.0, structure_score)
    
    def calculate_coherence_score(self, text: str) -> float:
        """Calculate coherence score based on sentence similarity"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        # Get sentence embeddings
        embeddings = self.model.encode(sentences)
        
        # Calculate cosine similarity between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def calculate_coverage_score(self, original: str, summary: str) -> float:
        """Calculate coverage/faithfulness score"""
        if not original or not summary:
            return 0.0
        
        # Get embeddings for both texts
        orig_embedding = self.model.encode([original])
        summ_embedding = self.model.encode([summary])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(orig_embedding, summ_embedding)[0][0]
        return similarity
    
    def evaluate_narrative_style(self, original: str, summary: str) -> Dict[str, float]:
        """Evaluate narrative style and return all scores"""
        bulletiness = self.calculate_bulletiness_score(summary)
        structure = self.calculate_narrative_structure_score(summary)
        coherence = self.calculate_coherence_score(summary)
        coverage = self.calculate_coverage_score(original, summary)
        
        # Calculate overall narrative style score (1-5)
        # Invert bulletiness (lower is better) and combine with other scores
        narrative_score = (
            (1 - bulletiness) * 0.3 +  # 30% weight for low bulletiness
            structure * 0.3 +           # 30% weight for narrative structure
            coherence * 0.2 +           # 20% weight for coherence
            coverage * 0.2              # 20% weight for coverage
        ) * 5  # Scale to 1-5
        
        return {
            'bulletiness_score': bulletiness,
            'structure_score': structure,
            'coherence_score': coherence,
            'coverage_score': coverage,
            'narrative_style_score': round(narrative_score, 2)
        }

class AIJudgeEvaluator:
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        
    def evaluate_with_ai_judge(self, original: str, generated: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate using AI-as-a-judge approach"""
        # This is a placeholder for AI judge evaluation
        # In practice, you would call an LLM API to evaluate the summaries
        # For now, we'll return placeholder scores
        
        # You can implement this by calling GPT-4 or another LLM API
        # to evaluate aspects like:
        # - Factual accuracy
        # - Completeness
        # - Clarity
        # - Engagement
        
        return {
            'factual_accuracy': 0.0,  # Placeholder
            'completeness': 0.0,      # Placeholder
            'clarity': 0.0,           # Placeholder
            'engagement': 0.0,        # Placeholder
            'overall_ai_score': 0.0   # Placeholder
        }

def process_csv_file(input_file: str, output_file: str, endpoint_url: str, batch_size: int = 10):
    """Process the CSV file and generate summaries with evaluations"""
    
    # Initialize components
    generator = SummaryGenerator(endpoint_url)
    narrative_evaluator = NarrativeStyleEvaluator()
    ai_judge = AIJudgeEvaluator(endpoint_url)
    
    # Read the CSV file
    logger.info(f"Reading CSV file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Add new columns for generated summaries and scores
    df['game_recap_summary_generated'] = ''
    df['game_recap_summary_ground_truth'] = df['game_recap_summary'].copy()
    df['bulletiness_score'] = 0.0
    df['structure_score'] = 0.0
    df['coherence_score'] = 0.0
    df['coverage_score'] = 0.0
    df['narrative_style_score'] = 0.0
    df['factual_accuracy'] = 0.0
    df['completeness'] = 0.0
    df['clarity'] = 0.0
    df['engagement'] = 0.0
    df['overall_ai_score'] = 0.0
    
    total_rows = len(df)
    logger.info(f"Processing {total_rows} rows in batches of {batch_size}")
    
    # Process in batches
    for i in range(0, total_rows, batch_size):
        batch_end = min(i + batch_size, total_rows)
        batch_df = df.iloc[i:batch_end]
        
        logger.info(f"Processing batch {i//batch_size + 1}: rows {i+1}-{batch_end}")
        
        for idx, row in batch_df.iterrows():
            game_recap = str(row['game_recap'])
            ground_truth = str(row['game_recap_summary'])
            
            if pd.isna(game_recap) or game_recap.strip() == '':
                logger.warning(f"Skipping row {idx + 1}: empty game_recap")
                continue
            
            # Generate summary
            logger.info(f"Generating summary for row {idx + 1}")
            generated_summary = generator.generate_summary(game_recap)
            
            if not generated_summary:
                logger.warning(f"Failed to generate summary for row {idx + 1}")
                continue
            
            # Evaluate narrative style
            narrative_scores = narrative_evaluator.evaluate_narrative_style(
                game_recap, generated_summary
            )
            
            # Evaluate with AI judge (placeholder for now)
            ai_scores = ai_judge.evaluate_with_ai_judge(
                game_recap, generated_summary, ground_truth
            )
            
            # Update the dataframe
            df.at[idx, 'game_recap_summary_generated'] = generated_summary
            df.at[idx, 'bulletiness_score'] = narrative_scores['bulletiness_score']
            df.at[idx, 'structure_score'] = narrative_scores['structure_score']
            df.at[idx, 'coherence_score'] = narrative_scores['coherence_score']
            df.at[idx, 'coverage_score'] = narrative_scores['coverage_score']
            df.at[idx, 'narrative_style_score'] = narrative_scores['narrative_style_score']
            df.at[idx, 'factual_accuracy'] = ai_scores['factual_accuracy']
            df.at[idx, 'completeness'] = ai_scores['completeness']
            df.at[idx, 'clarity'] = ai_scores['clarity']
            df.at[idx, 'engagement'] = ai_scores['engagement']
            df.at[idx, 'overall_ai_score'] = ai_scores['overall_ai_score']
            
            logger.info(f"Row {idx + 1} completed - Narrative score: {narrative_scores['narrative_style_score']}")
        
        # Save progress after each batch
        df.to_csv(output_file, index=False)
        logger.info(f"Progress saved after batch {i//batch_size + 1}")
    
    logger.info(f"Processing completed! Results saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Total rows processed: {len(df)}")
    print(f"Successful generations: {len(df[df['game_recap_summary_generated'] != ''])}")
    print(f"Average narrative style score: {df['narrative_style_score'].mean():.2f}")
    print(f"Average bulletiness score: {df['bulletiness_score'].mean():.2f}")
    print(f"Average structure score: {df['structure_score'].mean():.2f}")
    print(f"Average coherence score: {df['coherence_score'].mean():.2f}")
    print(f"Average coverage score: {df['coverage_score'].mean():.2f}")

def main():
    parser = argparse.ArgumentParser(description='Generate and evaluate game recap summaries')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')
    parser.add_argument('--endpoint', '-e', required=True, help='API endpoint URL')
    parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return
    
    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Process the file
    process_csv_file(args.input, args.output, args.endpoint, args.batch_size)

if __name__ == "__main__":
    main()
