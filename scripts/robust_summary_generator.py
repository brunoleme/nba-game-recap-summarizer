#!/usr/bin/env python3
"""
Robust script to generate game recap summaries and evaluate them with narrative style metrics.
This version handles CSV parsing issues and provides better error handling.
"""

import pandas as pd
import requests
import json
import time
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import csv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustSummaryGenerator:
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
                return result.get("game_recap_summary", "")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for game recap")
                    return ""
        
        return ""

class SimpleNarrativeEvaluator:
    """Simple narrative style evaluator without heavy ML dependencies"""
    
    def calculate_bulletiness_score(self, text: str) -> float:
        """Calculate bulletiness score (lower is better, 0-1)"""
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
    
    def calculate_sentence_structure_score(self, text: str) -> float:
        """Calculate sentence structure score (0-1, higher is better)"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return 0.0
        if len(sentences) > 7:
            return 0.5  # Penalize too many sentences
        
        # Check average sentence length
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if 12 <= avg_length <= 30:
            length_score = 1.0
        else:
            length_score = max(0, 1 - abs(avg_length - 21) / 21)
        
        return length_score
    
    def calculate_discourse_connectors_score(self, text: str) -> float:
        """Calculate discourse connectors score (0-1, higher is better)"""
        connectors = [
            'however', 'despite', 'while', 'as', 'after', 'because', 'therefore', 
            'meanwhile', 'although', 'though', 'whereas', 'furthermore', 'moreover',
            'consequently', 'thus', 'hence', 'additionally', 'similarly', 'conversely',
            'but', 'yet', 'still', 'instead', 'rather', 'indeed', 'in fact'
        ]
        
        text_lower = text.lower()
        connector_count = sum(1 for conn in connectors if conn in text_lower)
        
        # Normalize to 0-1 (expecting 0-5 connectors for good narrative)
        return min(1.0, connector_count / 5)
    
    def calculate_coverage_score(self, original: str, summary: str) -> float:
        """Simple coverage score based on word overlap"""
        if not original or not summary:
            return 0.0
        
        # Simple word overlap calculation
        orig_words = set(original.lower().split())
        summ_words = set(summary.lower().split())
        
        if not orig_words:
            return 0.0
        
        overlap = len(orig_words.intersection(summ_words))
        return overlap / len(orig_words)
    
    def calculate_readability_score(self, text: str) -> float:
        """Simple readability score based on sentence and word length"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Calculate average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Calculate average word length
        all_words = ' '.join(sentences).split()
        if not all_words:
            return 0.0
        
        avg_word_length = sum(len(word) for word in all_words) / len(all_words)
        
        # Simple readability score (higher is better)
        # Penalize very long sentences and very long words
        sentence_score = max(0, 1 - (avg_sentence_length - 15) / 20)
        word_score = max(0, 1 - (avg_word_length - 5) / 5)
        
        return (sentence_score + word_score) / 2
    
    def evaluate_narrative_style(self, original: str, summary: str) -> Dict[str, float]:
        """Evaluate narrative style and return all scores"""
        bulletiness = self.calculate_bulletiness_score(summary)
        structure = self.calculate_sentence_structure_score(summary)
        connectors = self.calculate_discourse_connectors_score(summary)
        coverage = self.calculate_coverage_score(original, summary)
        readability = self.calculate_readability_score(summary)
        
        # Calculate overall narrative style score (1-5)
        narrative_score = (
            (1 - bulletiness) * 0.25 +  # 25% weight for low bulletiness
            structure * 0.25 +           # 25% weight for sentence structure
            connectors * 0.2 +           # 20% weight for discourse connectors
            coverage * 0.15 +            # 15% weight for coverage
            readability * 0.15           # 15% weight for readability
        ) * 5  # Scale to 1-5
        
        return {
            'bulletiness_score': round(bulletiness, 3),
            'structure_score': round(structure, 3),
            'connectors_score': round(connectors, 3),
            'coverage_score': round(coverage, 3),
            'readability_score': round(readability, 3),
            'narrative_style_score': round(narrative_score, 2)
        }

def read_csv_robust(file_path: str) -> pd.DataFrame:
    """Read CSV file with robust error handling"""
    try:
        # Try standard pandas read first
        df = pd.read_csv(file_path)
        logger.info(f"Successfully read CSV with {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"Standard CSV read failed: {e}")
        logger.info("Trying with different parameters...")
        
        try:
            # Try with different parameters
            df = pd.read_csv(file_path, quoting=csv.QUOTE_ALL, on_bad_lines='skip')
            logger.info(f"Successfully read CSV with quoting=QUOTE_ALL: {len(df)} rows")
            return df
        except Exception as e2:
            logger.warning(f"QUOTE_ALL read failed: {e2}")
            logger.info("Trying with manual parsing...")
            
            # Manual parsing as last resort
            rows = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                for i, row in enumerate(reader):
                    if len(row) >= 7:  # Ensure we have all required columns
                        rows.append(row)
                    if i > 1000:  # Limit for testing
                        break
            
            df = pd.DataFrame(rows, columns=header)
            logger.info(f"Successfully read CSV with manual parsing: {len(df)} rows")
            return df

def process_csv_file(input_file: str, output_file: str, endpoint_url: str, batch_size: int = 5, max_rows: int = None):
    """Process the CSV file and generate summaries with evaluations"""
    
    # Initialize components
    generator = RobustSummaryGenerator(endpoint_url)
    evaluator = SimpleNarrativeEvaluator()
    
    # Read the CSV file
    logger.info(f"Reading CSV file: {input_file}")
    df = read_csv_robust(input_file)
    
    # Limit rows if specified
    if max_rows:
        df = df.head(max_rows)
        logger.info(f"Limited to {max_rows} rows for processing")
    
    # Add new columns for generated summaries and scores
    df['game_recap_summary_generated'] = ''
    df['game_recap_summary_ground_truth'] = df['game_recap_summary'].copy()
    df['bulletiness_score'] = 0.0
    df['structure_score'] = 0.0
    df['connectors_score'] = 0.0
    df['coverage_score'] = 0.0
    df['readability_score'] = 0.0
    df['narrative_style_score'] = 0.0
    
    total_rows = len(df)
    logger.info(f"Processing {total_rows} rows in batches of {batch_size}")
    
    # Process in batches
    successful_generations = 0
    
    for i in range(0, total_rows, batch_size):
        batch_end = min(i + batch_size, total_rows)
        batch_df = df.iloc[i:batch_end]
        
        logger.info(f"Processing batch {i//batch_size + 1}: rows {i+1}-{batch_end}")
        
        for row_idx, (idx, row) in enumerate(batch_df.iterrows()):
            game_recap = str(row['game_recap'])
            ground_truth = str(row['game_recap_summary'])
            
            if pd.isna(game_recap) or game_recap.strip() == '' or game_recap == 'nan':
                logger.warning(f"Skipping row {row_idx + 1}: empty game_recap")
                continue
            
            # Generate summary
            logger.info(f"Generating summary for row {row_idx + 1}")
            generated_summary = generator.generate_summary(game_recap)
            
            if not generated_summary:
                logger.warning(f"Failed to generate summary for row {row_idx + 1}")
                continue
            
            # Evaluate narrative style
            narrative_scores = evaluator.evaluate_narrative_style(
                game_recap, generated_summary
            )
            
            # Update the dataframe
            df.at[idx, 'game_recap_summary_generated'] = generated_summary
            df.at[idx, 'bulletiness_score'] = narrative_scores['bulletiness_score']
            df.at[idx, 'structure_score'] = narrative_scores['structure_score']
            df.at[idx, 'connectors_score'] = narrative_scores['connectors_score']
            df.at[idx, 'coverage_score'] = narrative_scores['coverage_score']
            df.at[idx, 'readability_score'] = narrative_scores['readability_score']
            df.at[idx, 'narrative_style_score'] = narrative_scores['narrative_style_score']
            
            successful_generations += 1
            logger.info(f"Row {row_idx + 1} completed - Narrative score: {narrative_scores['narrative_style_score']}")
        
        # Save progress after each batch
        df.to_csv(output_file, index=False)
        logger.info(f"Progress saved after batch {i//batch_size + 1}")
        
        # Add delay between batches to be respectful to the API
        if i + batch_size < total_rows:
            time.sleep(2)
    
    logger.info(f"Processing completed! Results saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total rows processed: {total_rows}")
    print(f"Successful generations: {successful_generations}")
    print(f"Success rate: {successful_generations/total_rows*100:.1f}%")
    print(f"Average narrative style score: {df['narrative_style_score'].mean():.2f}")
    print(f"Average bulletiness score: {df['bulletiness_score'].mean():.2f}")
    print(f"Average structure score: {df['structure_score'].mean():.2f}")
    print(f"Average connectors score: {df['connectors_score'].mean():.2f}")
    print(f"Average coverage score: {df['coverage_score'].mean():.2f}")
    print(f"Average readability score: {df['readability_score'].mean():.2f}")
    
    # Show distribution of narrative scores
    print(f"\nNarrative Style Score Distribution:")
    score_ranges = [(1, 2), (2, 3), (3, 4), (4, 5)]
    for low, high in score_ranges:
        count = len(df[(df['narrative_style_score'] >= low) & (df['narrative_style_score'] < high)])
        pct = count / successful_generations * 100 if successful_generations > 0 else 0
        print(f"  {low}-{high}: {count} summaries ({pct:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Generate and evaluate game recap summaries')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')
    parser.add_argument('--endpoint', '-e', required=True, help='API endpoint URL')
    parser.add_argument('--batch-size', '-b', type=int, default=5, help='Batch size for processing')
    parser.add_argument('--max-rows', '-m', type=int, help='Maximum number of rows to process (for testing)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return
    
    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Process the file
    process_csv_file(args.input, args.output, args.endpoint, args.batch_size, args.max_rows)

if __name__ == "__main__":
    main()
