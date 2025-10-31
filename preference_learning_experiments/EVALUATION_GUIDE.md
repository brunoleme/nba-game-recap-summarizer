# Preference Tuning Evaluation Guide

This guide explains how to assess the results of DPO (Direct Preference Optimization) training for NBA game recap summarization.

## Overview

Preference tuning aims to align the model's outputs with desired qualities (e.g., narrative style) by training on preference pairs. Evaluation is critical to verify that alignment has improved.

## Key Metrics

### 1. Preference Accuracy
**Goal**: Does the model prefer chosen responses over rejected responses?

**How it works**:
- Generate outputs with the tuned model
- Compare semantic similarity with chosen vs. rejected responses
- Calculate percentage where generated is more similar to chosen

**Interpretation**:
- **Good (>0.7)**: Model consistently prefers chosen responses
- **Fair (0.5-0.7)**: Partial alignment
- **Poor (<0.5)**: Model not learning preferences

### 2. Alignment Score
**Goal**: How well do generated outputs match the desired (chosen) style?

**How it works**:
- Measure semantic similarity between generated and chosen responses
- Uses sentence transformers for embeddings
- Cosine similarity between embeddings

**Interpretation**:
- **High (>0.7)**: Strong semantic alignment with preferred style
- **Medium (0.5-0.7)**: Moderate alignment
- **Low (<0.5)**: Poor alignment

### 3. Semantic Preservation
**Goal**: Does the model retain factual content from the source material?

**How it works**:
- Similar to alignment score, but focuses on content fidelity
- Ensures model doesn't hallucinate or drift from source

**Interpretation**:
- **High (>0.7)**: Good content preservation
- **Medium (0.5-0.7)**: Some drift possible
- **Low (<0.5)**: Potential factual errors

## Running Evaluation

The evaluation framework is built into `dpo_training.py`. After training:

```python
# Evaluation runs automatically after DPO training
eval_results = evaluate_preference_alignment(
    model, tokenizer, dpo_test, num_samples=20
)

# Results saved to:
# - {output_dir}/evaluation_results.json
# - {output_dir}/comparison_examples.csv
```

## Expected Output

### Console Output
```
📊 Evaluation Results:
  Preference Accuracy: 75.0%
  Avg Alignment Score: 0.682
  Avg Semantic Preservation: 0.695
```

### Saved Files

**evaluation_results.json**:
```json
{
  "preference_accuracy": 0.75,
  "avg_alignment": 0.682,
  "avg_semantic_preservation": 0.695,
  "alignment_scores": [0.65, 0.71, ...],
  "semantic_preservation": [0.68, 0.69, ...]
}
```

**comparison_examples.csv**:
- `prompt`: Input game recap
- `chosen`: Preferred summary style
- `rejected`: Undesired summary style
- `generated`: Model's output after tuning

## Qualitative Assessment

### 1. Narrative Style Analysis
Review `comparison_examples.csv` to verify:

- **Low bulletiness**: No lists or bullet points
- **Narrative flow**: 3-7 sentences with discourse connectors
- **Natural language**: 12-30 words per sentence average
- **Coherence**: Smooth transitions between sentences

### 2. Content Accuracy
Compare `generated` vs. original recap:

- **Key events preserved**: Major plays, scores, players
- **No hallucination**: No fabricated events or stats
- **Temporal accuracy**: Timeline matches source

### 3. Style Consistency
Check consistency across multiple examples:

- **Uniform tone**: All examples follow similar narrative style
- **Vocabulary**: Rich, varied language (not repetitive)
- **Length**: Appropriate summary length (not too short/long)

## Manual Evaluation Checklist

After automated metrics, manually review:

### ✅ Successful Alignment Indicators
- [ ] Preference Accuracy > 70%
- [ ] Alignment Score > 0.65
- [ ] Generated summaries are more narrative (less bullet-like)
- [ ] Discourse connectors present (however, despite, while, etc.)
- [ ] Coherent flow between sentences
- [ ] Content matches source material
- [ ] No obvious hallucinations

### ⚠️ Warning Signs
- [ ] Preference Accuracy < 50% (model rejecting preferred style)
- [ ] Alignment Score < 0.5 (poor semantic match)
- [ ] Increased hallucination or factual errors
- [ ] Over-fitting to training style (too formulaic)
- [ ] Loss of factual precision

### ❌ Failed Alignment Indicators
- [ ] Preference Accuracy < 30%
- [ ] Alignment Score < 0.3
- [ ] Model outputs worse than before training
- [ ] Severe factual errors or hallucinations
- [ ] Training loss not decreasing or diverging

## Interpreting Results

### High Scores (Success)
```
Preference Accuracy: >0.75
Alignment Score: >0.7
Semantic Preservation: >0.7
```
**Meaning**: Model successfully aligned with preferred narrative style while preserving factual content.

### Medium Scores (Partial Success)
```
Preference Accuracy: 0.5-0.75
Alignment Score: 0.5-0.7
Semantic Preservation: 0.5-0.7
```
**Meaning**: Some alignment achieved. May need:
- More training data
- Longer training
- Different beta parameter
- Re-check data quality

### Low Scores (Failure)
```
Preference Accuracy: <0.5
Alignment Score: <0.5
Semantic Preservation: <0.5
```
**Meaning**: DPO not working. Possible causes:
- Data quality issues (bad preference pairs)
- Training instability (loss exploding)
- Insufficient training (too few epochs)
- Hyperparameter mismatch

## Post-Evaluation Steps

### 1. If Results Are Good
- Deploy model for production
- Run on larger test set
- Monitor real-world performance
- Gather user feedback

### 2. If Results Are Partial
- Collect more preference data
- Experiment with different beta values
- Increase training epochs
- Check data quality

### 3. If Results Are Poor
- Review training logs for issues
- Re-check data preparation
- Verify model loading correctness
- Consider alternative approaches (e.g., different reward models)

## Advanced Metrics

### Distribution Analysis
Examine the distribution of alignment scores:

```python
import matplotlib.pyplot as plt

plt.hist(eval_results['alignment_scores'], bins=20)
plt.xlabel('Alignment Score')
plt.ylabel('Frequency')
plt.title('Distribution of Alignment Scores')
plt.show()
```

**Healthy distribution**: Concentrated around 0.6-0.8
**Concerning distribution**: Bimodal or heavily skewed

### Error Analysis
Manually review low-scoring examples:

```python
# Find problematic examples
low_alignment = [i for i, score in enumerate(eval_results['alignment_scores']) 
                 if score < 0.5]

# Review these specific examples
for idx in low_alignment[:5]:
    print(f"Example {idx}:")
    print(f"  Alignment: {eval_results['alignment_scores'][idx]:.3f}")
    print(f"  Generated: {comparison_examples[idx]['generated']}")
    print()
```

## Comparison with Baseline

Always compare against the original model:

1. **Generate with baseline model** (before DPO)
2. **Generate with tuned model** (after DPO)
3. **Compare metrics** side-by-side
4. **Manual inspection** of example quality

**Expected improvement**: Tuned model should score higher on narrative style metrics while maintaining factual accuracy.

## Next Steps

After successful evaluation:

1. **Deploy**: Integrate tuned model into production API
2. **Monitor**: Track real-world performance metrics
3. **Iterate**: Collect new data for further improvement
4. **Document**: Record what worked and what didn't

## Troubleshooting

### Evaluation Fails with "TypeError: Population must be a sequence"
**Solution**: Fixed in latest version by converting HuggingFace Dataset to list

### Low preference accuracy despite good training loss
**Possible causes**:
- Test set has different distribution than training
- Model over-fitted to training data
- Need more diverse preference pairs

### High alignment but poor semantic preservation
**Possible causes**:
- Model learned to imitate style but lost factual accuracy
- Need to add factual preservation to training objective
- Adjust beta to balance style vs. accuracy

## References

- **DPO Paper**: Rafailov et al. "Direct Preference Optimization" (2023)
- **Sentence Transformers**: https://www.sbert.net/
- **Evaluation Metrics**: Standard NLP evaluation practices

---

For questions or issues, see the main project README or contact the development team.
