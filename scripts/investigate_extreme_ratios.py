#!/usr/bin/env python3
"""
Investigate cases with extreme summary-to-recap length ratios
"""

import pandas as pd
import numpy as np
from loguru import logger

def investigate_extreme_ratios(file_path: str):
    """Investigate cases with extreme length ratios."""
    logger.info(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Calculate lengths and ratios
    df['recap_length'] = df['game_recap'].str.len()
    df['summary_length'] = df['game_recap_summary'].str.len()
    df['recap_word_count'] = df['game_recap'].str.split().str.len()
    df['summary_word_count'] = df['game_recap_summary'].str.split().str.len()
    df['length_ratio'] = df['summary_length'] / df['recap_length']
    df['word_ratio'] = df['summary_word_count'] / df['recap_word_count']
    
    # Remove NaN values
    df = df.dropna(subset=['game_recap', 'game_recap_summary'])
    
    print(f"\n📊 LENGTH RATIO STATISTICS:")
    print(f"Mean length ratio: {df['length_ratio'].mean():.4f}")
    print(f"Median length ratio: {df['length_ratio'].median():.4f}")
    print(f"Std length ratio: {df['length_ratio'].std():.4f}")
    print(f"Min length ratio: {df['length_ratio'].min():.4f}")
    print(f"Max length ratio: {df['length_ratio'].max():.4f}")
    
    # Find extreme cases
    extreme_low = df[df['length_ratio'] < 0.01].copy()
    extreme_high = df[df['length_ratio'] > 0.5].copy()
    
    print(f"\n⚠️  EXTREME LOW RATIOS (< 0.01): {len(extreme_low)} cases")
    print(f"⚠️  EXTREME HIGH RATIOS (> 0.5): {len(extreme_high)} cases")
    
    # Analyze extreme low ratio cases
    if len(extreme_low) > 0:
        print(f"\n🔍 ANALYZING {len(extreme_low)} EXTREME LOW RATIO CASES:")
        print("="*80)
        
        for idx, row in extreme_low.head(10).iterrows():
            print(f"\nCASE {idx}:")
            print(f"  Length ratio: {row['length_ratio']:.4f}")
            print(f"  Recap length: {row['recap_length']} chars, {row['recap_word_count']} words")
            print(f"  Summary length: {row['summary_length']} chars, {row['summary_word_count']} words")
            print(f"  Teams: {row['home_team']} vs {row['away_team']}")
            print(f"  Date: {row['date']}")
            print(f"  Recap preview: {row['game_recap'][:200]}...")
            print(f"  Summary: {row['game_recap_summary']}")
            print("-" * 60)
    
    # Analyze extreme high ratio cases
    if len(extreme_high) > 0:
        print(f"\n🔍 ANALYZING {len(extreme_high)} EXTREME HIGH RATIO CASES:")
        print("="*80)
        
        for idx, row in extreme_high.head(5).iterrows():
            print(f"\nCASE {idx}:")
            print(f"  Length ratio: {row['length_ratio']:.4f}")
            print(f"  Recap length: {row['recap_length']} chars, {row['recap_word_count']} words")
            print(f"  Summary length: {row['summary_length']} chars, {row['summary_word_count']} words")
            print(f"  Teams: {row['home_team']} vs {row['away_team']}")
            print(f"  Date: {row['date']}")
            print(f"  Recap preview: {row['game_recap'][:200]}...")
            print(f"  Summary: {row['game_recap_summary']}")
            print("-" * 60)
    
    # Check for patterns in extreme cases
    print(f"\n📈 PATTERNS IN EXTREME CASES:")
    
    if len(extreme_low) > 0:
        print(f"\nExtreme Low Ratio Patterns:")
        print(f"  Mean recap length: {extreme_low['recap_length'].mean():.1f} chars")
        print(f"  Mean summary length: {extreme_low['summary_length'].mean():.1f} chars")
        print(f"  Teams with most extreme low ratios:")
        team_counts = extreme_low['home_team'].value_counts().head(5)
        for team, count in team_counts.items():
            print(f"    {team}: {count} cases")
    
    if len(extreme_high) > 0:
        print(f"\nExtreme High Ratio Patterns:")
        print(f"  Mean recap length: {extreme_high['recap_length'].mean():.1f} chars")
        print(f"  Mean summary length: {extreme_high['summary_length'].mean():.1f} chars")
        print(f"  Teams with most extreme high ratios:")
        team_counts = extreme_high['home_team'].value_counts().head(5)
        for team, count in team_counts.items():
            print(f"    {team}: {count} cases")
    
    # Check for very short summaries in extreme cases
    very_short_summaries = extreme_low[extreme_low['summary_word_count'] < 20]
    print(f"\n🚨 VERY SHORT SUMMARIES in extreme low ratio cases: {len(very_short_summaries)}")
    
    if len(very_short_summaries) > 0:
        print("Examples of very short summaries:")
        for idx, row in very_short_summaries.head(5).iterrows():
            print(f"  '{row['game_recap_summary']}' ({row['summary_word_count']} words)")
    
    return extreme_low, extreme_high

def main():
    """Main function."""
    extreme_low, extreme_high = investigate_extreme_ratios('data/samples/game_recaps_with_summaries.csv')
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"1. Review {len(extreme_low)} cases with very low summary-to-recap ratios")
    print(f"2. Consider filtering out cases with summary < 20 words")
    print(f"3. Check if {len(extreme_high)} high-ratio cases are actually good quality")
    print(f"4. Investigate if extreme cases are from specific teams or time periods")

if __name__ == "__main__":
    main()
