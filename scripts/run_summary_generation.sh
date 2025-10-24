#!/bin/bash

# Script to run the summary generation and evaluation
# Usage: ./run_summary_generation.sh

set -e

# Configuration
INPUT_FILE="game_recaps_with_summaries_sample_for_reward_model.csv"
OUTPUT_FILE="game_recaps_with_summaries_sample_for_reward_model_with_generated.csv"
ENDPOINT_URL="http://54.197.213.231:8000"
BATCH_SIZE=5

echo "🚀 Starting Game Recap Summary Generation and Evaluation"
echo "========================================================"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Endpoint: $ENDPOINT_URL"
echo "Batch size: $BATCH_SIZE"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file '$INPUT_FILE' not found!"
    echo "Please make sure the CSV file is in the current directory."
    exit 1
fi

# Check if endpoint is reachable
echo "🔍 Checking if endpoint is reachable..."
if curl -f -s "$ENDPOINT_URL/health" > /dev/null; then
    echo "✅ Endpoint is reachable"
else
    echo "❌ Error: Cannot reach endpoint at $ENDPOINT_URL"
    echo "Please check if the EC2 instance is running and the service is up."
    exit 1
fi

# Install required Python packages if not already installed
echo "📦 Installing required Python packages..."
pip install pandas requests numpy scikit-learn

# Run the summary generation script
echo "🔄 Starting summary generation and evaluation..."
python scripts/simple_summary_generator.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --endpoint "$ENDPOINT_URL" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "✅ Summary generation completed!"
echo "📊 Results saved to: $OUTPUT_FILE"
echo ""
echo "🔍 You can now analyze the results and use the best-scored summaries for DPO training."
