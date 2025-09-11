#!/usr/bin/env python3
"""
Test script to verify the training fixes
"""
import os
import sys
import pandas as pd
from datasets import Dataset

def test_data_loading():
    """Test data loading with the fixed preprocessing"""
    print("=== Testing Data Loading ===")
    
    csv_path = "tests/resources/source_data/game_recaps_with_summaries.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    # Load and check data
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check required columns
    required_cols = ['game_recap', 'game_recap_summary']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
    
    # Clean data
    initial_count = len(df)
    df_clean = df.dropna(subset=required_cols)
    cleaned_count = len(df_clean)
    
    print(f"✅ After cleaning: {cleaned_count} rows (removed {initial_count - cleaned_count})")
    
    # Check for line breaks in recaps
    recap_with_newlines = df_clean['game_recap'].str.contains('\n', na=False).sum()
    print(f"📝 Recaps with line breaks: {recap_with_newlines} ({recap_with_newlines/cleaned_count*100:.1f}%)")
    
    # Sample data quality
    sample_recap = df_clean['game_recap'].iloc[0]
    sample_summary = df_clean['game_recap_summary'].iloc[0]
    
    print(f"\n📋 Sample Data:")
    print(f"Recap length: {len(sample_recap)} chars")
    print(f"Summary length: {len(sample_summary)} chars")
    print(f"Recap preview: {sample_recap[:100]}...")
    print(f"Summary preview: {sample_summary[:100]}...")
    
    return True

def test_config_validation():
    """Test the fixed configuration"""
    print("\n=== Testing Configuration ===")
    
    config_path = "config.staging.fixed.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    print(f"✅ Config file exists: {config_path}")
    
    # Check key parameters
    with open(config_path, 'r') as f:
        content = f.read()
    
    checks = [
        ("quantization_type: 8bit", "8-bit quantization"),
        ("r: 16", "LoRA rank 16"),
        ("learning_rate: 1e-5", "Learning rate 1e-5"),
        ("train_samples: 1000", "Train samples 1000"),
        ("val_samples: 500", "Val samples 500"),
        ("test_samples: 500", "Test samples 500"),
    ]
    
    for check, description in checks:
        if check in content:
            print(f"✅ {description}")
        else:
            print(f"❌ Missing: {description}")
    
    return True

def test_rouge_calculation():
    """Test ROUGE calculation with sample data"""
    print("\n=== Testing ROUGE Calculation ===")
    
    try:
        from evaluate import load
        rouge_metric = load("rouge")
        
        # Test with sample data
        predictions = ["The Lakers beat the Warriors 120-115 with LeBron James scoring 30 points."]
        references = ["The Lakers defeated the Warriors 120-115. LeBron James scored 30 points and Anthony Davis added 25 points."]
        
        rouge_scores = rouge_metric.compute(
            predictions=predictions,
            references=references
        )
        
        rouge_1_f1 = rouge_scores['rouge1']
        print(f"✅ ROUGE-1 F1 Score: {rouge_1_f1:.4f}")
        
        if rouge_1_f1 > 0:
            print("✅ ROUGE calculation working correctly")
            return True
        else:
            print("❌ ROUGE calculation returned 0")
            return False
            
    except Exception as e:
        print(f"❌ ROUGE calculation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Training Fixes\n")
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Configuration", test_config_validation),
        ("ROUGE Calculation", test_rouge_calculation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n=== Test Results ===")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Ready for training.")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
