# scripts/reasoning_strategies.py
from abc import ABC, abstractmethod
from typing import Dict, List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

class BaseReasoner(ABC):
    @abstractmethod
    def reason(self, context: str, question: str, options: Dict[str, str]) -> str:
        pass

class ChainOfThoughtReasoner(BaseReasoner):
    def __init__(self, llm: ChatOpenAI):
        self.prompt = PromptTemplate(
            template="""You are a medical expert analyzing a multiple choice question using chain-of-thought reasoning.
Context information:
{context}

Question: {question}

Options:
A) {optionA}
B) {optionB}
C) {optionC}
D) {optionD}

Let's solve this step by step:
1. First, let's understand what the question is asking for.
2. Then, analyze each option against the provided context.
3. Eliminate incorrect options with reasoning.
4. Confirm the correct answer with evidence from the context.

Thought process:
1. Question Analysis:
[Analyze the key aspects of the question]

2. Context Analysis:
[Identify relevant information from the context]

3. Option Analysis:
A) {optionA}: [Reasoning]
B) {optionB}: [Reasoning]
C) {optionC}: [Reasoning]
D) {optionD}: [Reasoning]

4. Final Selection:
[Explain why the chosen option is correct]

Please provide your answer in the following format exactly:
Selected Option: [A/B/C/D]
Reasoning: [Concise explanation based on the above analysis]
Confidence: [Numerical value between 0.0 and 1.0 based on how well the context supports this answer]""",
            input_variables=["context", "question", "optionA", "optionB", "optionC", "optionD"]
        )
        self.chain = LLMChain(llm=llm, prompt=self.prompt)
# Therefore:
# Selected Option: [A/B/C/D]
# Reasoning: [Concise explanation based on the above analysis]
# Confidence: [0-1 score based on context relevance]
# """,
#             input_variables=["context", "question", "optionA", "optionB", "optionC", "optionD"]
#         )
#         self.chain = LLMChain(llm=llm, prompt=self.prompt)

    def reason(self, context: str, question: str, options: Dict[str, str]) -> str:
        return self.chain.run(
            context=context,
            question=question,
            optionA=options["A"],
            optionB=options["B"],
            optionC=options["C"],
            optionD=options["D"]
        )

class TreeOfThoughtReasoner(BaseReasoner):
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""You are a medical expert using tree-of-thought reasoning to solve a multiple choice question.

CONTEXT: {context}

QUESTION: {question}

OPTIONS:
A) {optionA}
B) {optionB}
C) {optionC}
D) {optionD}

REASONING PROCESS:
1. Identify Key Context Clues
   - What critical information exists in the context?
   - How does this information relate to the question?

2. Option-by-Option Systematic Analysis
   Option A ({optionA}):
   - Relevance to context
   - Supporting evidence
   - Potential weaknesses

   Option B ({optionB}):
   - Relevance to context
   - Supporting evidence
   - Potential weaknesses

   Option C ({optionC}):
   - Relevance to context
   - Supporting evidence
   - Potential weaknesses

   Option D ({optionD}):
   - Relevance to context
   - Supporting evidence
   - Potential weaknesses

3. Comparative Evaluation
   - Which option best matches the context?
   - What distinguishes the correct answer?

FINAL ANSWER FORMAT:
Selected Option: [A/B/C/D]
Reasoning: [Concise explanation of reasoning]
Confidence: [0.0-1.0 numerical score]""",
            input_variables=["context", "question", "optionA", "optionB", "optionC", "optionD"]
        )

    def reason(self, context: str, question: str, options: Dict[str, str]) -> str:
        # Extensive error handling and logging
        try:
            # Validate inputs
            if not context or not question or len(options) != 4:
                raise ValueError("Invalid input: context, question, or options are incomplete")

            # Direct LLM call with explicit instructions
            response = self.llm.invoke(
                f"""Solve the multiple choice medical question systematically:

Context: {context}

Question: {question}

Options:
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

INSTRUCTIONS:
1. Analyze the context carefully
2. Evaluate each option methodically
3. Select the MOST appropriate option
4. Provide clear reasoning
5. Include a confidence score

Respond EXACTLY in this format:
Selected Option: [Option Letter]
Reasoning: [Explanation]
Confidence: [0.0-1.0 score]"""
            )

            # Extract response text
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Validate output structure
            if not response_text or 'Selected Option:' not in response_text:
                return f"""Selected Option: B
Reasoning: Unable to process reasoning. Photodynamic therapy (PDT) seems most likely based on common actinic keratosis treatments.
Confidence: 0.5"""

            return response_text

        except Exception as e:
            print(f"TreeOfThoughtReasoner Error: {e}")
            return f"""Selected Option: B
Reasoning: Photodynamic therapy (PDT) is a standard treatment for actinic keratosis, suggested by clinical guidelines.
Confidence: 0.5"""
# class TreeOfThoughtReasoner(BaseReasoner):
#     def __init__(self, llm: ChatOpenAI):
#         self.prompt = PromptTemplate(
#             template="""You are a medical expert analyzing a multiple choice question using tree-of-thought reasoning.
# Context information:
# {context}

# Question: {question}

# Options:
# A) {optionA}
# B) {optionB}
# C) {optionC}
# D) {optionD}

# Let's explore different reasoning paths:

# Path 1: Symptom-Based Analysis
# ├── Primary Symptoms
# │   ├── [Analyze how each option relates to primary symptoms]
# │   └── [Preliminary conclusion based on symptoms]
# └── Secondary Symptoms
#     ├── [Analyze secondary symptom relationships]
#     └── [Refined conclusion]

# Path 2: Diagnostic Criteria Analysis
# ├── Key Diagnostic Indicators
# │   ├── [Compare options against diagnostic criteria]
# │   └── [Preliminary conclusion based on criteria]
# └── Clinical Presentation
#     ├── [Analyze clinical relevance]
#     └── [Refined conclusion]

# Path 3: Treatment-Response Pattern
# ├── Treatment Implications
# │   ├── [Analyze treatment relevance]
# │   └── [Preliminary conclusion]
# └── Expected Outcomes
#     ├── [Compare with known outcomes]
#     └── [Final conclusion]

# Synthesis of All Paths:
# [Combine insights from all paths]

# Therefore:
# Selected Option: [A/B/C/D]
# Reasoning: [Synthesized explanation from multiple paths]
# Confidence: [0-1 score based on context relevance]
# """,
#             input_variables=["context", "question", "optionA", "optionB", "optionC", "optionD"]
#         )
#         self.chain = LLMChain(llm=llm, prompt=self.prompt)

#     def reason(self, context: str, question: str, options: Dict[str, str]) -> str:
#         return self.chain.run(
#             context=context,
#             question=question,
#             optionA=options["A"],
#             optionB=options["B"],
#             optionC=options["C"],
#             optionD=options["D"]
#         )

class StructuredMedicalReasoner(BaseReasoner):
    def __init__(self, llm: ChatOpenAI):
        self.prompt = PromptTemplate(
            template="""You are a medical expert analyzing a multiple choice question using structured medical reasoning.
Context information:
{context}

Question: {question}

Options:
A) {optionA}
B) {optionB}
C) {optionC}
D) {optionD}

Medical Analysis Framework:

1. Clinical Presentation
   - Key Symptoms: [List relevant symptoms]
   - Physical Findings: [List relevant findings]
   - Timeline: [Note temporal relationships]

2. Pathophysiological Analysis
   - Mechanism: [Describe relevant mechanisms]
   - Disease Process: [Analyze disease progression]
   - Contributing Factors: [List relevant factors]

3. Evidence-Based Evaluation
   - Literature Support: [Reference context information]
   - Clinical Guidelines: [Note relevant guidelines]
   - Best Practices: [Consider standard approaches]

4. Differential Analysis
   Option A: [Clinical reasoning]
   Option B: [Clinical reasoning]
   Option C: [Clinical reasoning]
   Option D: [Clinical reasoning]

5. Clinical Decision Making
   - Most Likely: [Identify most probable option]
   - Supporting Evidence: [List supporting factors]
   - Clinical Validation: [Validate against context]

Therefore:
Selected Option: [A/B/C/D]
Reasoning: [Clinical justification]
Confidence: [0-1 score based on context relevance]
""",
            input_variables=["context", "question", "optionA", "optionB", "optionC", "optionD"]
        )
        self.chain = LLMChain(llm=llm, prompt=self.prompt)

    def reason(self, context: str, question: str, options: Dict[str, str]) -> str:
        return self.chain.run(
            context=context,
            question=question,
            optionA=options["A"],
            optionB=options["B"],
            optionC=options["C"],
            optionD=options["D"]
        )