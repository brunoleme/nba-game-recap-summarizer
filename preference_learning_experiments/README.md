# Preference Learning Experiments for NBA Game Recap Summarization

This directory contains experimental components for implementing preference learning (KTO/DPO) to improve the narrative style of NBA game recap summaries. This is a temporary experimental folder - the final implementation will be moved to `src/nba_game_recap_summarizer/preference_learning/`.

## Overview

This project implements **Direct Preference Optimization (DPO)** for fine-tuning language models based on narrative style preferences. We originally attempted to use Kahneman-Tversky Optimization (KTO) for single-score preference data, but encountered numerical instability issues. **DPO** provides a mature, stable alternative.

**Status**: ✅ **DPO works successfully**, ❌ **KTO shows numerical instability (NaN losses)**

## Directory Structure

```
preference_learning_experiments/
├── README.md                           # This file
├── data/                               # Dataset files
│   ├── game_recaps_with_summaries_sample_for_reward_model.csv
│   └── game_recaps_with_summaries_sample_for_reward_model_with_generated_full.csv
├── scripts/                            # Generation and evaluation scripts
│   ├── run_summary_generation.py       # Main generation runner
│   ├── simple_summary_generator.py    # Core generation logic
│   ├── robust_summary_generator.py    # Advanced generation with error handling
│   ├── test_endpoint.py               # API endpoint testing
│   └── run_summary_generation.sh       # Bash runner script
├── colab/                             # Google Colab integration
│   └── colab_model_loading.py         # Model loading for Colab
└── notebooks/                         # Jupyter notebooks (future)
```

## What is KTO vs DPO?

### Direct Preference Optimization (DPO) ✅ **RECOMMENDED**
- **Traditional approach**: Requires paired data (chosen vs rejected responses)
- **Data format**: Each example has a "chosen" and "rejected" response
- **Training**: Model learns to prefer chosen over rejected responses
- **Status**: **Successfully working** in our experiments
- **Advantage**: Numerically stable, mature implementation

### Kahneman-Tversky Optimization (KTO) ❌ **NOT RECOMMENDED**
- **Modern approach**: Works with single-score preference data
- **Data format**: Each example has a single preference score (1-5)
- **Training**: Model learns from individual preference scores
- **Status**: **Numerically unstable** (NaN losses) in our experiments
- **Issue**: Numerical instability causes training failures

## Why DPO for This Project?

1. **Proven Stability**: DPO is mature and numerically stable
2. **Works with Single Scores**: Can convert scores to chosen/rejected pairs
3. **Better for Training**: More reliable gradient flow and convergence
4. **Conversion Strategy**: High-score samples (top 25%) vs random negatives

## Narrative Style Scoring System

Our custom evaluation system scores generated summaries on multiple dimensions:

### 1. Bulletiness Score (0-1, lower is better)
- **Penalizes**: Bullet points, dashes, heading patterns
- **Rewards**: Narrative flow over list-like structures
- **Example**: "• Lakers won" vs "The Lakers secured a victory"

### 2. Structure Score (0-1, higher is better)
- **Ideal range**: 3-7 sentences
- **Sentence length**: 12-30 words average
- **Penalizes**: Too many/few sentences, awkward lengths

### 3. Connectors Score (0-1, higher is better)
- **Rewards**: Discourse connectors (however, despite, while, as, after, because, therefore, meanwhile, although)
- **Measures**: Narrative flow and coherence
- **Example**: "While the Lakers struggled early, they ultimately prevailed"

### 4. Coverage Score (0-1, higher is better)
- **Measures**: Word overlap between original recap and summary
- **Ensures**: Faithfulness to source material
- **Prevents**: Hallucination and drift

### 5. Readability Score (0-1, higher is better)
- **Based on**: Sentence and word length
- **Rewards**: Clear, readable prose
- **Penalizes**: Overly complex or simple language

### 6. Overall Narrative Style Score (1-5)
- **Weighted combination** of all metrics
- **Higher scores** indicate better narrative style
- **Used for KTO training** as preference signal

## Workflow

### 1. Data Generation
```bash
# Generate summaries and scores
python scripts/run_summary_generation.py \
    --input data/game_recaps_with_summaries_sample_for_reward_model.csv \
    --output data/game_recaps_with_summaries_sample_for_reward_model_with_generated_full.csv \
    --endpoint http://your-api-endpoint:8000 \
    --batch-size 10
```

### 2. Model Loading in Colab
```python
# Use the provided Colab script
exec(open('colab/colab_model_loading.py').read())
```

### 3. KTO Training
```python
# Filter high-quality examples
high_quality = df[df['narrative_style_score'] >= 4.0]

# Prepare data for KTO training
# (Implementation details in Colab notebook)
```

## Key Files

### Scripts
- **`scripts/run_summary_generation.py`**: Main generation pipeline
- **`scripts/simple_summary_generator.py`**: Core generation and evaluation logic
- **`scripts/robust_summary_generator.py`**: Advanced version with error handling
- **`scripts/test_endpoint.py`**: API endpoint validation

### Data
- **`data/game_recaps_with_summaries_sample_for_reward_model.csv`**: Input dataset
- **`data/game_recaps_with_summaries_sample_for_reward_model_with_generated_full.csv`**: Output with scores

### Colab Integration
- **`colab/colab_model_loading.py`**: Complete Colab setup for model loading and KTO training

## Usage Examples

### Generate Summaries and Scores
```bash
cd preference_learning_experiments
python scripts/run_summary_generation.py \
    --input data/your_dataset.csv \
    --output data/your_dataset_with_scores.csv \
    --endpoint http://your-endpoint:8000
```

### Load Model in Colab
```python
# Upload colab/colab_model_loading.py to Colab
# Run the script to download and load your model
```

### Analyze Results
```python
import pandas as pd

# Load results
df = pd.read_csv('data/game_recaps_with_summaries_sample_for_reward_model_with_generated_full.csv')

# Filter high-quality summaries
high_quality = df[df['narrative_style_score'] >= 4.0]
print(f"High-quality summaries: {len(high_quality)}")

# Analyze patterns
print("Average scores for high-quality summaries:")
print(high_quality[['bulletiness_score', 'structure_score', 'connectors_score']].mean())
```

## Next Steps

1. **Generate Data**: Run the generation script on your dataset
2. **Load Model**: Use Colab script to load your fine-tuned model
3. **Implement KTO**: Train using the preference scores
4. **Evaluate**: Test the improved model
5. **Iterate**: Refine based on results

## Technical Details

### Model Format
- **Saved as**: `hf_model_merged` (merged LoRA weights)
- **Compatible with**: KTO training frameworks
- **Format**: Standard HuggingFace model format

### API Integration
- **Endpoint**: `/summarize_recap`
- **Input**: `{"game_recap": "text", "max_length": 2048}`
- **Output**: `{"game_recap_summary": "generated text"}`

### Evaluation Metrics
- **Implementation**: Custom Python functions
- **Dependencies**: `transformers`, `torch`, `pandas`, `numpy`
- **Scoring**: Weighted combination of multiple criteria

## Troubleshooting

### Common Issues
1. **API Connection**: Check endpoint URL and health
2. **Memory Issues**: Reduce batch size
3. **Rate Limiting**: Add delays between requests
4. **Model Loading**: Verify S3 path and credentials

### Debug Tools
- **`scripts/test_endpoint.py`**: Test API connectivity
- **`scripts/robust_summary_generator.py`**: Advanced error handling
- **Logging**: Detailed logs for debugging

## Contributing

When adding new features:
1. Update this README
2. Add tests for new functionality
3. Document any new evaluation metrics
4. Ensure compatibility with existing workflow
