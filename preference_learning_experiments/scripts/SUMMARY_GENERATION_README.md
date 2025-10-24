# Game Recap Summary Generation and Evaluation

This directory contains scripts to generate game recap summaries using your deployed API endpoint and evaluate them with custom narrative style metrics.

## Overview

The scripts will:
1. Read your CSV file with game recaps
2. Generate summaries using the deployed API endpoint
3. Evaluate summaries with narrative style metrics
4. Save results back to a new CSV file with scores

## Files

- `test_endpoint.py` - Simple test to verify the API endpoint is working
- `scripts/simple_summary_generator.py` - Main script for generation and evaluation
- `scripts/run_summary_generation.sh` - Bash script to run the full process
- `scripts/requirements_evaluation.txt` - Additional Python dependencies

## Quick Start

### 1. Test the Endpoint First

```bash
python test_endpoint.py
```

This will test if your deployed API is working correctly.

### 2. Run the Full Generation Process

```bash
# Make sure your CSV file is in the current directory
./scripts/run_summary_generation.sh
```

Or run the Python script directly:

```bash
python scripts/simple_summary_generator.py \
    --input game_recaps_with_summaries_sample_for_reward_model.csv \
    --output game_recaps_with_summaries_sample_for_reward_model_with_generated.csv \
    --endpoint http://54.197.213.231:8000 \
    --batch-size 5
```

## Narrative Style Evaluation Metrics

The script evaluates generated summaries on several criteria:

### 1. Bulletiness Score (0-1, lower is better)
- Penalizes bullet points, dashes, or heading patterns
- Rewards narrative flow over list-like structures

### 2. Structure Score (0-1, higher is better)
- Rewards 3-7 sentences
- Penalizes too many or too few sentences
- Checks average sentence length (12-30 words ideal)

### 3. Connectors Score (0-1, higher is better)
- Rewards discourse connectors (however, despite, while, etc.)
- Measures narrative flow and coherence

### 4. Coverage Score (0-1, higher is better)
- Measures word overlap between original and summary
- Ensures faithfulness to source material

### 5. Readability Score (0-1, higher is better)
- Based on sentence and word length
- Rewards clear, readable prose

### 6. Overall Narrative Style Score (1-5)
- Weighted combination of all metrics
- Higher scores indicate better narrative style

## Output

The script will create a new CSV file with these additional columns:

- `game_recap_summary_generated` - AI-generated summary
- `game_recap_summary_ground_truth` - Original summary (renamed)
- `bulletiness_score` - Bulletiness metric (0-1)
- `structure_score` - Structure metric (0-1)
- `connectors_score` - Connectors metric (0-1)
- `coverage_score` - Coverage metric (0-1)
- `readability_score` - Readability metric (0-1)
- `narrative_style_score` - Overall score (1-5)

## Using Results for DPO Training

After generation, you can:

1. **Filter by high scores**: Select summaries with `narrative_style_score >= 4.0`
2. **Analyze patterns**: Look at what makes high-scoring summaries different
3. **Create preference pairs**: Use high-scoring vs low-scoring summaries for DPO
4. **Fine-tune your model**: Train on the best examples to improve narrative style

## Example Usage

```python
import pandas as pd

# Load results
df = pd.read_csv('game_recaps_with_summaries_sample_for_reward_model_with_generated.csv')

# Filter for high-quality summaries
high_quality = df[df['narrative_style_score'] >= 4.0]
print(f"Found {len(high_quality)} high-quality summaries")

# Analyze what makes them good
print("Average scores for high-quality summaries:")
print(high_quality[['bulletiness_score', 'structure_score', 'connectors_score']].mean())
```

## Troubleshooting

### Endpoint Not Responding
- Check if EC2 instance is running
- Verify the IP address is correct
- Test with `curl http://54.197.213.231:8000/health`

### Memory Issues
- Reduce batch size (e.g., `--batch-size 3`)
- Process smaller chunks of the CSV file

### Rate Limiting
- Increase delay between requests
- Reduce batch size
- Add more delay between batches

## Customization

You can modify the evaluation criteria in `scripts/simple_summary_generator.py`:

- Adjust weights in `evaluate_narrative_style()`
- Add new metrics
- Change scoring thresholds
- Modify the overall score calculation

## Next Steps

1. Run the generation script
2. Analyze the results
3. Identify patterns in high-scoring summaries
4. Use the best examples for DPO training
5. Iterate and improve your model
