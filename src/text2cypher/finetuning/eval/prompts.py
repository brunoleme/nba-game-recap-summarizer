from langchain_core.prompts import PromptTemplate

# Relevance prompt
relevance_prompt_text = """You are an evaluation assistant of patient-doctor notes generation.
You will be given a INSTRUCTION to summarize a conversation between patient and doctor, the GROUND TRUTH NOTE summarizes the conversation accurately, and the GENERATED NOTE is the model's output.
Here is the grade criteria to follow:
(1) Grade the generated note based ONLY on their similarity with the ground truth note.
Score: A score of 5 means that the generated note meets all of the criteria. This is the highest (best) score. A score of 1 means it meets none.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset.

### INSTRUCTION
{instruction}

### GROUND TRUTH NOTE
{ground_truth_note}

### GENERATED NOTE
{generated_note}
"""

relevance_prompt = PromptTemplate.from_template(relevance_prompt_text)

# Factual Consistency prompt
factual_consistency_prompt_text = """You are an evaluation assistant of patient-doctor notes generation.
You will be given a INSTRUCTION to summarize a conversation between patient and doctor and the GENERATED NOTE with the model's output.
Grade the generated note based ONLY on their factual accuracy relative to the instruction.
Score: A score of 5 means that the generated note meets all of the criteria. This is the highest (best) score. A score of 1 means it meets none.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset.

### INSTRUCTION
{instruction}

### GENERATED NOTE
{generated_note}
"""

factual_consistency_prompt = PromptTemplate.from_template(factual_consistency_prompt_text)

# Completeness prompt
completeness_prompt_text = """You are an evaluation assistant of patient-doctor notes generation.
You will be given a INSTRUCTION to summarize a conversation between patient and doctor and the GENERATED NOTE with the model's output.
Grade the generated note based ONLY on their completeness relative to the instruction.
Score: A score of 5 means that the generated note meets all of the criteria. This is the highest (best) score. A score of 1 means it meets none.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset.

### INSTRUCTION
{instruction}

### GENERATED NOTE
{generated_note}
"""

completeness_prompt = PromptTemplate.from_template(completeness_prompt_text)

# Clarity prompt
clarity_prompt_text = """You are an evaluation assistant of patient-doctor notes generation.
You will be given a INSTRUCTION to summarize a conversation between patient and doctor and the GENERATED NOTE with the model's output.
You must evaluate GENERATED NOTE clarity. This includes how easy it is to read, how well-structured it is, and whether it makes sense.
Here is the grade criteria to follow:
(1) Evaluate grammar, spelling, and coherence.
(2) Check whether the note flows logically and is clearly organized.
(3) Consider how well a medical professional could interpret the note.

Score: A score of 5 means that the note is exceptionally clear, well-structured, and easy to interpret. A score of 1 means the note is very unclear or confusing.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset.

### INSTRUCTION
{instruction}

### GENERATED NOTE
{generated_note}
"""

clarity_prompt = PromptTemplate.from_template(clarity_prompt_text)

# Conciseness prompt
conciseness_prompt_text = """You are an evaluation assistant for patient-doctor notes generation.
You will be given a INSTRUCTION to summarize a conversation between patient and doctor and the GENERATED NOTE with the model's output.
You must evaluate how concise the GENERATED NOTE is. This includes checking whether the note captures important content using as few words as needed.

Here is the grade criteria to follow:
(1) Check for redundant or repetitive information.
(2) Penalize long-winded explanations or inclusion of irrelevant data.
(3) Reward notes that convey essential information in a precise and efficient way.

Score: A score of 5 means that the note is concise and free of redundancy. A score of 1 means the note is verbose, rambling, or includes too much irrelevant information.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset.

### INSTRUCTION
{instruction}

### GENERATED NOTE
{generated_note}
"""

conciseness_prompt = PromptTemplate.from_template(conciseness_prompt_text)
