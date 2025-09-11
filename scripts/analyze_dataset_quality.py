#!/usr/bin/env python3
"""
Dataset Quality Analysis Script
Analyzes the NBA game recap dataset for repetitive patterns, quality issues, and potential filtering opportunities.
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
from typing import List, Dict, Tuple
from loguru import logger

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the dataset from CSV file."""
    logger.info(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} samples")
    return df

def analyze_text_lengths(df: pd.DataFrame) -> Dict:
    """Analyze text length distributions."""
    logger.info("Analyzing text length distributions...")
    
    # Calculate lengths
    df['recap_length'] = df['game_recap'].str.len()
    df['summary_length'] = df['game_recap_summary'].str.len()
    df['recap_word_count'] = df['game_recap'].str.split().str.len()
    df['summary_word_count'] = df['game_recap_summary'].str.split().str.len()
    
    # Remove any NaN values
    df = df.dropna(subset=['game_recap', 'game_recap_summary'])
    
    stats = {
        'recap_length': {
            'mean': df['recap_length'].mean(),
            'std': df['recap_length'].std(),
            'min': df['recap_length'].min(),
            'max': df['recap_length'].max(),
            'median': df['recap_length'].median()
        },
        'summary_length': {
            'mean': df['summary_length'].mean(),
            'std': df['summary_length'].std(),
            'min': df['summary_length'].min(),
            'max': df['summary_length'].max(),
            'median': df['summary_length'].median()
        },
        'recap_word_count': {
            'mean': df['recap_word_count'].mean(),
            'std': df['recap_word_count'].std(),
            'min': df['recap_word_count'].min(),
            'max': df['recap_word_count'].max(),
            'median': df['recap_word_count'].median()
        },
        'summary_word_count': {
            'mean': df['summary_word_count'].mean(),
            'std': df['summary_word_count'].std(),
            'min': df['summary_word_count'].min(),
            'max': df['summary_word_count'].max(),
            'median': df['summary_word_count'].median()
        }
    }
    
    return stats, df

def find_repetitive_patterns(df: pd.DataFrame) -> Dict:
    """Find repetitive patterns in the dataset."""
    logger.info("Analyzing repetitive patterns...")
    
    patterns = {}
    
    # 1. Check for duplicate recaps
    duplicate_recaps = df[df.duplicated(subset=['game_recap'], keep=False)]
    patterns['duplicate_recaps'] = len(duplicate_recaps)
    
    # 2. Check for duplicate summaries
    duplicate_summaries = df[df.duplicated(subset=['game_recap_summary'], keep=False)]
    patterns['duplicate_summaries'] = len(duplicate_summaries)
    
    # 3. Check for very similar recaps (using simple similarity)
    patterns['very_short_recaps'] = len(df[df['recap_word_count'] < 50])
    patterns['very_short_summaries'] = len(df[df['summary_word_count'] < 10])
    patterns['very_long_recaps'] = len(df[df['recap_word_count'] > 2000])
    patterns['very_long_summaries'] = len(df[df['summary_word_count'] > 500])
    
    # 4. Check for common starting phrases
    recap_starts = df['game_recap'].str[:50].value_counts().head(10)
    summary_starts = df['game_recap_summary'].str[:30].value_counts().head(10)
    
    patterns['common_recap_starts'] = recap_starts.to_dict()
    patterns['common_summary_starts'] = summary_starts.to_dict()
    
    # 5. Check for repetitive words/phrases
    all_recap_text = ' '.join(df['game_recap'].astype(str))
    all_summary_text = ' '.join(df['game_recap_summary'].astype(str))
    
    # Find most common words
    recap_words = re.findall(r'\b\w+\b', all_recap_text.lower())
    summary_words = re.findall(r'\b\w+\b', all_summary_text.lower())
    
    patterns['top_recap_words'] = dict(Counter(recap_words).most_common(20))
    patterns['top_summary_words'] = dict(Counter(summary_words).most_common(20))
    
    return patterns

def analyze_quality_issues(df: pd.DataFrame) -> Dict:
    """Analyze potential quality issues in the dataset."""
    logger.info("Analyzing quality issues...")
    
    issues = {}
    
    # 1. Check for empty or very short content
    issues['empty_recaps'] = len(df[df['game_recap'].str.strip() == ''])
    issues['empty_summaries'] = len(df[df['game_recap_summary'].str.strip() == ''])
    
    # 2. Check for very repetitive content
    def has_repetitive_pattern(text, min_repeat=3):
        words = text.split()
        if len(words) < 10:
            return False
        # Check for repeated phrases
        for i in range(len(words) - min_repeat):
            phrase = ' '.join(words[i:i+min_repeat])
            if words.count(phrase) >= min_repeat:
                return True
        return False
    
    issues['repetitive_recaps'] = df['game_recap'].apply(lambda x: has_repetitive_pattern(str(x))).sum()
    issues['repetitive_summaries'] = df['game_recap_summary'].apply(lambda x: has_repetitive_pattern(str(x))).sum()
    
    # 3. Check for unusual characters or formatting
    issues['recaps_with_html'] = df['game_recap'].str.contains('<[^>]+>', regex=True).sum()
    issues['summaries_with_html'] = df['game_recap_summary'].str.contains('<[^>]+>', regex=True).sum()
    
    # 4. Check for very similar length ratios (potential copy-paste issues)
    df['length_ratio'] = df['summary_length'] / df['recap_length']
    issues['extreme_length_ratios'] = len(df[(df['length_ratio'] < 0.01) | (df['length_ratio'] > 0.5)])
    
    # 5. Check for common filler phrases
    filler_phrases = [
        'the game was', 'the team', 'the players', 'the coach',
        'in the first', 'in the second', 'in the third', 'in the fourth',
        'at the end', 'at the start', 'at the beginning'
    ]
    
    issues['recaps_with_filler'] = df['game_recap'].str.lower().str.contains('|'.join(filler_phrases), regex=True).sum()
    issues['summaries_with_filler'] = df['game_recap_summary'].str.lower().str.contains('|'.join(filler_phrases), regex=True).sum()
    
    return issues

def analyze_team_patterns(df: pd.DataFrame) -> Dict:
    """Analyze team-related patterns in the dataset."""
    logger.info("Analyzing team patterns...")
    
    team_stats = {}
    
    # 1. Most common teams
    all_teams = []
    for col in ['home_team', 'away_team']:
        if col in df.columns:
            all_teams.extend(df[col].dropna().tolist())
    
    team_counts = Counter(all_teams)
    team_stats['most_common_teams'] = dict(team_counts.most_common(10))
    
    # 2. Check for team name variations
    team_variations = {}
    for team in team_counts.keys():
        if pd.notna(team):
            # Find similar team names
            similar = [t for t in team_counts.keys() if pd.notna(t) and team.lower() in t.lower() or t.lower() in team.lower()]
            if len(similar) > 1:
                team_variations[team] = similar
    
    team_stats['team_variations'] = team_variations
    
    return team_stats

def generate_recommendations(patterns: Dict, issues: Dict, stats: Dict) -> List[str]:
    """Generate recommendations for dataset improvement."""
    recommendations = []
    
    # Length-based recommendations
    if stats['recap_length']['min'] < 100:
        recommendations.append(f"Remove {patterns['very_short_recaps']} very short recaps (< 50 words)")
    
    if stats['summary_length']['min'] < 20:
        recommendations.append(f"Remove {patterns['very_short_summaries']} very short summaries (< 10 words)")
    
    if patterns['very_long_recaps'] > 0:
        recommendations.append(f"Consider truncating {patterns['very_long_recaps']} very long recaps (> 2000 words)")
    
    # Duplicate recommendations
    if patterns['duplicate_recaps'] > 0:
        recommendations.append(f"Remove {patterns['duplicate_recaps']} duplicate recaps")
    
    if patterns['duplicate_summaries'] > 0:
        recommendations.append(f"Remove {patterns['duplicate_summaries']} duplicate summaries")
    
    # Quality recommendations
    if issues['repetitive_recaps'] > 0:
        recommendations.append(f"Review {issues['repetitive_recaps']} recaps with repetitive patterns")
    
    if issues['repetitive_summaries'] > 0:
        recommendations.append(f"Review {issues['repetitive_summaries']} summaries with repetitive patterns")
    
    if issues['extreme_length_ratios'] > 0:
        recommendations.append(f"Review {issues['extreme_length_ratios']} samples with extreme length ratios")
    
    # Pattern-based recommendations
    if len(patterns['common_recap_starts']) > 5:
        recommendations.append("Consider diversifying recap starting phrases")
    
    if len(patterns['common_summary_starts']) > 5:
        recommendations.append("Consider diversifying summary starting phrases")
    
    return recommendations

def main():
    """Main analysis function."""
    logger.info("Starting dataset quality analysis...")
    
    # Load dataset
    df = load_dataset('data/samples/game_recaps_with_summaries.csv')
    
    # Analyze text lengths
    stats, df = analyze_text_lengths(df)
    
    # Find repetitive patterns
    patterns = find_repetitive_patterns(df)
    
    # Analyze quality issues
    issues = analyze_quality_issues(df)
    
    # Analyze team patterns
    team_stats = analyze_team_patterns(df)
    
    # Generate recommendations
    recommendations = generate_recommendations(patterns, issues, stats)
    
    # Print results
    print("\n" + "="*80)
    print("DATASET QUALITY ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"Total samples: {len(df)}")
    print(f"Recap length - Mean: {stats['recap_length']['mean']:.1f}, Std: {stats['recap_length']['std']:.1f}")
    print(f"Summary length - Mean: {stats['summary_length']['mean']:.1f}, Std: {stats['summary_length']['std']:.1f}")
    
    print(f"\n🔄 REPETITIVE PATTERNS:")
    print(f"Duplicate recaps: {patterns['duplicate_recaps']}")
    print(f"Duplicate summaries: {patterns['duplicate_summaries']}")
    print(f"Very short recaps (< 50 words): {patterns['very_short_recaps']}")
    print(f"Very short summaries (< 10 words): {patterns['very_short_summaries']}")
    print(f"Very long recaps (> 2000 words): {patterns['very_long_recaps']}")
    print(f"Very long summaries (> 500 words): {patterns['very_long_summaries']}")
    
    print(f"\n⚠️  QUALITY ISSUES:")
    print(f"Empty recaps: {issues['empty_recaps']}")
    print(f"Empty summaries: {issues['empty_summaries']}")
    print(f"Repetitive recaps: {issues['repetitive_recaps']}")
    print(f"Repetitive summaries: {issues['repetitive_summaries']}")
    print(f"Recaps with HTML: {issues['recaps_with_html']}")
    print(f"Summaries with HTML: {issues['summaries_with_html']}")
    print(f"Extreme length ratios: {issues['extreme_length_ratios']}")
    
    print(f"\n🏀 TEAM PATTERNS:")
    print("Most common teams:")
    for team, count in list(team_stats['most_common_teams'].items())[:5]:
        print(f"  {team}: {count}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\n📈 TOP WORDS IN RECAPS:")
    for word, count in list(patterns['top_recap_words'].items())[:10]:
        print(f"  {word}: {count}")
    
    print(f"\n📈 TOP WORDS IN SUMMARIES:")
    for word, count in list(patterns['top_summary_words'].items())[:10]:
        print(f"  {word}: {count}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
