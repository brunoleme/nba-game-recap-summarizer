# Narrative Style Scoring System - Prompt Explanation

This document explains the Narrative Style Score for inclusion in prompt engineering.

## For Prompts to Models

### Complete Explanation (Concise)

```
### Narrative Style Scoring Criteria

Your summaries are evaluated on a Narrative Style Score (1-5) based on these metrics:

**Bulletiness (0-1, lower is better):** Avoid bullet points (•, -, *), headings (Score:, Top Performers:), and list-like structures - prefer narrative flow.

**Structure (0-1, higher is better):** Aim for 3-7 sentences with 12-30 words average per sentence for natural readability.

**Connectors (0-1, higher is better):** Use discourse connectors (however, despite, while, as, after, because, therefore, meanwhile, although, whereas, furthermore, moreover, consequently, thus, hence, additionally, similarly, conversely) to create smooth narrative flow between sentences.

**Coverage (0-1, higher is better):** Maintain semantic alignment between the original recap and your summary - ensure key facts, events, and outcomes are accurately preserved without hallucination.

**Readability (0-1, higher is better):** Balance sentence complexity - clear, readable prose that isn't overly simple or unnecessarily complex.

The overall Narrative Style Score is a weighted combination: bulletiness (30%), structure (35%), coherence (20%), and coverage (15%), scaled to 1-5.
```

### Shorter Version for Token Efficiency

```
### Narrative Style Score Guidelines

Your summary should achieve a high Narrative Style Score by:
• Avoiding bullet points and headings - write in narrative prose (Bulletiness metric)
• Using 3-7 sentences of 12-30 words each for natural structure (Structure metric)
• Employing discourse connectors (however, while, because, therefore, etc.) for smooth flow (Connectors metric)
• Preserving key facts and events from the original recap accurately (Coverage metric)
• Writing clear, balanced prose - not too simple, not overly complex (Readability metric)
```

### Ultra-Compact Version

```
Write narrative summaries (not bullet points) with 3-7 sentences (12-30 words each), 
use connectors (however, while, because, therefore), maintain factual accuracy to 
the original, and aim for clear, balanced prose.
```

### For Human Evaluators

```
### Narrative Style Score Calculation

The Narrative Style Score (1-5) evaluates summary quality across five dimensions:

**Bulletiness (0-1, lower is better, 30% weight):**
Penalizes summary formats resembling lists or structured data. Prefer narrative prose over bullet points (•, -, *), headings (Score:, Top Performers:), or enumerated items.

**Structure (0-1, higher is better, 35% weight):**
Rewards 3 represented by 7 sentences with 12-30 words average per sentence. Penalizes too few (< 3) or too many (> 7) sentences, and awkward sentence lengths.

**Connectors (0-1, higher is better, 20% weight):**
Measures presence of discourse connectors (however, despite, while, as, after, because, therefore, meanwhile, although, whereas, furthermore, moreover, consequently, thus, hence, additionally, similarly, conversely) to ensure coherent narrative flow.

**Coverage (0-1, higher is better, 15% weight):**
Measures semantic alignment between the original recap and summary using embeddings. Ensures faithfulness to source material and prevents hallucination or factual drift.

**Readability (implicit in structure, 0-1, higher is better):**
Evaluates sentence and word length balance to reward clear, readable prose that avoids being overly simplistic or unnecessarily complex.

**Final Score Calculation:**
composite = (1 - bulletiness) × 0.30 + structure × 0.35 + connectors × 0.20 + coverage × 0.15
narrative_style_score = clamp(composite × 5, 1, 5)
```

## Example Usage in Prompts

**Note:** The following examples use the official Mistral-7B-Instruct-v0.1 format with `<s>[INST]` and `[/INST]` tokens for optimal performance.

### For Mistral Model - Evaluation and Rewriting (COMPLETE FORMATTED VERSION)

```python
def build_rewriting_prompt(recap: str, recap_summary: str, example_1: str, example_2: str, example_3: str) -> str:
    return (
        "<s>[INST] You are an NBA chief editor. Your goal is to evaluate recap summaries made by an NBA Analyst and rewrite them if needed to improve narrative style.\n\n"
        "Your summaries are evaluated on a Narrative Style Score (1-5) based on these metrics:\n\n"
        "• Bulletiness (0-1, lower is better): Avoid bullet points (•, -, *), headings (Score:, Top Performers:), and list-like structures - prefer narrative flow.\n\n"
        "• Structure (0-1, higher is better): Aim for 3-7 sentences with 12-30 words average per sentence for natural readability.\n\n"
        "• Connectors (0-1, higher is better): Use discourse connectors (however, despite, while, as, after, because, therefore, meanwhile, although, whereas, furthermore, moreover, consequently, thus, hence, additionally, similarly, conversely) to create smooth narrative flow between sentences.\n\n"
        "• Coverage (0-1, higher is better): Maintain semantic alignment between the original recap and your summary - ensure key facts, events, and outcomes are accurately preserved without hallucination.\n\n"
        "• Readability (0-1, higher is better): Balance sentence complexity - clear, readable prose that isn't overly simple or unnecessarily complex.\n\n"
        "The overall Narrative Style Score is a weighted combination: bulletiness (30%), structure (35%), coherence (20%), and coverage (15%), scaled to 1-5.\n\n"
        "Here are the best examples from the dataset:\n\n"
        f"Example 1:\n{example_1}\n\n"
        f"Example 2:\n{example_2}\n\n"
        f"Example 3:\n{example_3}\n\n"
        f"recap: {recap}\n\n"
        f"recap_summary: {recap_summary}\n\n"
        "Please evaluate this summary and provide a rewritten version if improvements are needed. Focus on narrative style improvements. [/INST]"
    )
```

### Simplified Version (Without Examples)

```python
def build_rewriting_prompt_simple(recap: str, recap_summary: str) -> str:
    return (
        "<s>[INST] You are an NBA chief editor. Evaluate and rewrite NBA recap summaries to improve narrative style.\n\n"
        "Scoring criteria (1-5 scale):\n"
        "• Avoid bullet points and headings - use narrative prose\n"
        "• Write 3-7 sentences with 12-30 words each\n"
        "• Use connectors (however, while, because, therefore, etc.) for flow\n"
        "• Preserve all key facts accurately\n"
        "• Write clear, balanced prose\n\n"
        f"recap: {recap}\n\n"
        f"recap_summary: {recap_summary}\n\n"
        "Evaluate and rewrite if needed. [/INST]"
    )
```

### For Model Fine-Tuning (Original Summarization Task)

```python
def build_prompt(recap: str) -> str:
    return (
        "<s>[INST] You are an NBA Analyst tasked with summarizing game recaps.\n\n"
        "Your summaries should achieve a high Narrative Style Score by:\n"
        "• Avoiding bullet points and structured formats - write natural narrative prose\n"
        "• Using 3-7 sentences averaging 12-30 words each for optimal structure\n"
        "• Employing discourse connectors (however, while, because, therefore, etc.) for smooth transitions\n"
        "• Accurately preserving all key facts, events, and outcomes from the original recap\n"
        "• Writing clear, balanced prose that is engaging without being overly complex\n\n"
        f"Recap: {recap} [/INST]"
    )
```

### For Evaluation Instructions

```
## Summary Evaluation Guidelines

Rate each summary on a 5-point Narrative Style scale:

5 = Excellent narrative flow, no bullets, 3-7 well-structured sentences, connectors used effectively, accurate coverage
4 = Good narrative style with minor issues
3 = Adequate but some bullet points or missing connectors
2 = Too list-like or structural problems
1 = Poor formatting or significant issues

Key criteria:
- No bullet points or headings
- 3-7 sentences of natural length
- Discourse connectors for flow
- Accurate factual coverage
```

## Technical Implementation Reference

```python
# For actual scoring implementation
def calculate_narrative_score(summary, original_recap):
    bulletiness = detect_bullet_points(summary)  # 0-1, lower better
    structure = evaluate_structure(summary)      # 0-1, higher better
    connectors = count_connectors(summary)       # 0-1, higher better
    coverage = semantic_alignment(original_recap, summary)  # 0-1, higher better
    readability = evaluate_readability(summary)  # 0-1, higher better
    
    composite = (
        (1 - bulletiness) * 0.30 +
        structure * 0.35 +
        connectors * 0.20 +
        coverage * 0.15
    )
    
    return clamp(composite * 5, 1, 5)
```

## Integration Tips

1. **Token Efficiency**: Use the ultra-compact version (3 lines) for LLM prompts
2. **Clarity**: Use the concise version (5 bullet points) for human evaluators
3. **Training**: Use the complete explanation for fine-tuning instructions
4. **Evaluation**: Include specific examples when training human raters

## Variations by Use Case

### For Chain-of-Thought Prompting

```
First, read the game recap. Then generate a summary following these style rules:
1. Write in narrative prose (no bullets or headings)
2. Use 3-7 sentences, each 12-30 words long
3. Connect ideas with words like "however," "while," "because"
4. Include all key facts accurately
5. Write clearly and engagingly

Your summary:
```

This format encourages the model to explicitly consider each criterion during generation.
