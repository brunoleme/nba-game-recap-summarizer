from langchain_core.prompts import PromptTemplate

# Relevance prompt
relevance_prompt_text = """You are an evaluation assistant of a NBA game recap summary generation system.
You will be given a INSTRUCTION to summarize a complete NBA game recap, the GROUND TRUTH RECAP SUMMARY summarizes the game recap accurately, and the GENERATED RECAP SUMMARY is the model's output.
Here is the grade criteria to follow:
(1) Grade the game recap summary based ONLY on their similarity with the ground truth recap summary.
Score: A score of 5 means that the generated game recap summary meets all of the criteria. This is the highest (best) score. A score of 1 means it meets none.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset.

### INSTRUCTION
{instruction}

### GROUND TRUTH RECAP SUMMARY
{ground_truth_recap_summary}

### GENERATED RECAP SUMMARY
{generated_recap_summary}
"""

relevance_prompt = PromptTemplate.from_template(relevance_prompt_text)

# Factual Consistency prompt
factual_consistency_prompt_text = """You are an evaluation assistant of a NBA game recap summary generation system.
You will be given a INSTRUCTION to summarize a NBA game recap and the GENERATED RECAP SUMMARY with the model's output.
Grade the generated game recap summary based ONLY on their factual accuracy relative to the instruction.
Score: A score of 5 means that the generated game recap summary meets all of the criteria. This is the highest (best) score. A score of 1 means it meets none.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset.

### INSTRUCTION
{instruction}

### GENERATED RECAP SUMMARY
{generated_recap_summary}
"""

factual_consistency_prompt = PromptTemplate.from_template(factual_consistency_prompt_text)

# Completeness prompt
completeness_prompt_text = """You are an evaluation assistant of a NBA game recap summary generation system.
You will be given a INSTRUCTION to summarize a NBA game recap and the GENERATED RECAP SUMMARY with the model's output.
Grade the generated game recap summary based ONLY on their completeness relative to the instruction.
Score: A score of 5 means that the generated game recap summary meets all of the criteria. This is the highest (best) score. A score of 1 means it meets none.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset.

### INSTRUCTION
{instruction}

### GENERATED RECAP SUMMARY
{generated_recap_summary}
"""

completeness_prompt = PromptTemplate.from_template(completeness_prompt_text)

# Clarity prompt
clarity_prompt_text = """You are an evaluation assistant of a NBA game recap summary generation system.
You will be given a INSTRUCTION to summarize a NBA game recap and the GENERATED RECAP SUMMARY with the model's output.
You must evaluate GENERATED RECAP SUMMARY clarity. This includes how easy it is to read, how well-structured it is, and whether it makes sense.
Here is the grade criteria to follow:
(1) Evaluate grammar, spelling, and coherence.
(2) Check whether the game recap summary flows logically and is clearly organized.
(3) Consider how well a NBA analyst could interpret the game recap summary.

Score: A score of 5 means that the game recap summary is exceptionally clear, well-structured, and easy to interpret. A score of 1 means the game recap summary is very unclear or confusing.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset.

### INSTRUCTION
{instruction}

### GENERATED RECAP SUMMARY
{generated_recap_summary}
"""

clarity_prompt = PromptTemplate.from_template(clarity_prompt_text)

# Conciseness prompt
conciseness_prompt_text = """You are an evaluation assistant of a NBA game recap summary generation system.
You will be given a INSTRUCTION to summarize NBA game recap and the GENERATED RECAP SUMMARY with the model's output.
You must evaluate how concise the GENERATED RECAP SUMMARY is. This includes checking whether the game recap summary captures important content using as few words as needed.

Here is the grade criteria to follow:
(1) Check for redundant or repetitive information.
(2) Penalize long-winded explanations or inclusion of irrelevant data.
(3) Reward game recap summaries that convey essential information in a precise and efficient way.

Score: A score of 5 means that the game recap summary is concise and free of redundancy. A score of 1 means the game recap summary is verbose, rambling, or includes too much irrelevant information.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset.

### INSTRUCTION
{instruction}

### GENERATED RECAP SUMMARY
{generated_recap_summary}
"""

conciseness_prompt = PromptTemplate.from_template(conciseness_prompt_text)
