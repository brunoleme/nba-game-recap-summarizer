#!/usr/bin/env python3
"""
Test script to verify data loading with full dataset
"""
import pandas as pd
import os
from datasets import Dataset

def test_data_loading():
    # Test loading the full CSV
    csv_path = "tests/resources/source_data/game_recaps_with_summaries.csv"
    
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Original CSV shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for required columns
    required_cols = ['game_recap', 'game_recap_summary']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
    
    # Clean data
    initial_count = len(df)
    df_clean = df.dropna(subset=required_cols)
    cleaned_count = len(df_clean)
    
    print(f"After cleaning: {cleaned_count} rows (removed {initial_count - cleaned_count})")
    
    # Check data quality
    recap_lengths = df_clean['game_recap'].str.len()
    summary_lengths = df_clean['game_recap_summary'].str.len()
    
    print(f"\nData Quality Analysis:")
    print(f"Recap lengths - Min: {recap_lengths.min()}, Max: {recap_lengths.max()}, Mean: {recap_lengths.mean():.1f}")
    print(f"Summary lengths - Min: {summary_lengths.min()}, Max: {summary_lengths.max()}, Mean: {summary_lengths.mean():.1f}")
    
    # Check for error patterns
    error_patterns = [
        'No meaningful paragraphs found',
        'No summary available as input contains an error',
        'The Associated Press erroneously'
    ]
    
    error_count = 0
    for pattern in error_patterns:
        count = df_clean['game_recap'].str.contains(pattern, case=False, na=False).sum()
        error_count += count
        if count > 0:
            print(f"Found {count} samples with pattern: '{pattern}'")
    
    print(f"Total error samples: {error_count}")
    
    # Create train/val/test splits
    dataset = Dataset.from_pandas(df_clean)
    dataset = dataset.shuffle(seed=42)
    
    n = len(dataset)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)
    
    train_dataset = dataset.select(range(0, train_end))
    val_dataset = dataset.select(range(train_end, val_end))
    test_dataset = dataset.select(range(val_end, n))
    
    print(f"\nDataset Splits:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    # Sample some data
    print(f"\nSample Data:")
    sample = train_dataset[0]
    print(f"Recap: {sample['game_recap'][:200]}...")
    print(f"Summary: {sample['game_recap_summary'][:200]}...")
    
    return True

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n✅ Data loading test passed!")
    else:
        print("\n❌ Data loading test failed!")
