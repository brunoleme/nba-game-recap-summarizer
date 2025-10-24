#!/usr/bin/env python3
"""
Simple runner script for the robust summary generation.
This will process your CSV file and generate summaries with evaluation metrics.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Configuration
    input_file = "game_recaps_with_summaries_sample_for_reward_model.csv"
    output_file = "game_recaps_with_summaries_sample_for_reward_model_with_generated.csv"
    endpoint_url = "http://54.197.213.231:8000"
    batch_size = 5
    max_rows = 50  # Start with 50 rows for testing
    
    print("🚀 NBA Game Recap Summary Generation and Evaluation")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Endpoint: {endpoint_url}")
    print(f"Batch size: {batch_size}")
    print(f"Max rows (for testing): {max_rows}")
    print("")
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Error: Input file '{input_file}' not found!")
        print("Please make sure the CSV file is in the current directory.")
        return 1
    
    # Check if endpoint is reachable
    print("🔍 Checking if endpoint is reachable...")
    try:
        import requests
        response = requests.get(f"{endpoint_url}/health", timeout=10)
        response.raise_for_status()
        print("✅ Endpoint is reachable")
    except Exception as e:
        print(f"❌ Error: Cannot reach endpoint at {endpoint_url}")
        print(f"Error: {e}")
        print("Please check if the EC2 instance is running and the service is up.")
        return 1
    
    # Install required packages
    print("📦 Installing required Python packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "requests", "numpy"], 
                      check=True, capture_output=True)
        print("✅ Packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Could not install packages: {e}")
        print("Please install manually: pip install pandas requests numpy")
    
    # Run the summary generation script
    print("🔄 Starting summary generation and evaluation...")
    print("This may take a while depending on the number of rows...")
    print("")
    
    try:
        cmd = [
            sys.executable, "scripts/robust_summary_generator.py",
            "--input", input_file,
            "--output", output_file,
            "--endpoint", endpoint_url,
            "--batch-size", str(batch_size),
            "--max-rows", str(max_rows)
        ]
        
        result = subprocess.run(cmd, check=True)
        
        print("")
        print("✅ Summary generation completed successfully!")
        print(f"📊 Results saved to: {output_file}")
        print("")
        print("🔍 Next steps:")
        print("1. Review the generated summaries and scores")
        print("2. Identify high-scoring summaries (narrative_style_score >= 4.0)")
        print("3. Use these for DPO training to improve your model")
        print("4. Run with more rows by increasing --max-rows or removing the limit")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running summary generation: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())
