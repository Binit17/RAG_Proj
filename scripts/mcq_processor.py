# # # scripts/mcq_processor.py
from typing import List, Dict, TypedDict, Literal
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.schema import Document
from .retrievers import SingleStageRetriever, TwoStageRetriever, ThreeStageRetriever
from .reasoning_strategies import (
    ChainOfThoughtReasoner,
    TreeOfThoughtReasoner,
    StructuredMedicalReasoner
)

RetrieverType = Literal["single", "two-stage", "three-stage"]
ReasonerType = Literal["chain-of-thought", "tree-of-thought", "structured-medical"]

class MCQInput(TypedDict):
    question: str
    options: Dict[str, str]

class MCQOutput(TypedDict):
    selected_option: str
    reasoning: str
    confidence: float

class MCQOutputWithContext(TypedDict):
    selected_option: str
    reasoning: str
    confidence: float
    contexts: List[Dict[str, str]]

class MCQProcessor:
    def __init__(self, 
                 vectorstore: Chroma, 
                 retriever_type: RetrieverType = "single",
                 reasoner_type: ReasonerType = "chain-of-thought",
                 model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        
        # Initialize retriever
        if retriever_type == "single":
            self.retriever = SingleStageRetriever(vectorstore)
        elif retriever_type == "two-stage":
            self.retriever = TwoStageRetriever(vectorstore, self.llm)
        elif retriever_type == "three-stage":
            self.retriever = ThreeStageRetriever(vectorstore, self.llm)
        else:
            raise ValueError(f"Invalid retriever type: {retriever_type}")
            
        # Initialize reasoner
        if reasoner_type == "chain-of-thought":
            self.reasoner = ChainOfThoughtReasoner(self.llm)
        elif reasoner_type == "tree-of-thought":
            self.reasoner = TreeOfThoughtReasoner(self.llm)
        elif reasoner_type == "structured-medical":
            self.reasoner = StructuredMedicalReasoner(self.llm)
        else:
            raise ValueError(f"Invalid reasoner type: {reasoner_type}")

    def _parse_output(self, raw_output: str) -> MCQOutput:
        """Parse the LLM output into structured format."""
        lines = raw_output.strip().split('\n')
        selected_option = ""
        reasoning = ""
        confidence = 0.0

        for line in lines:
            line = line.strip()
            if line.startswith("Selected Option:"):
                selected_option = line.split(":", 1)[1].strip()
            elif line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("Confidence:"):
                try:
                    confidence_str = line.split(":", 1)[1].strip()
                    # Handle different potential formats
                    if confidence_str.endswith('%'):
                        confidence = float(confidence_str.rstrip('%')) / 100
                    else:
                        confidence = float(confidence_str)
                    # Ensure confidence is between 0 and 1
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse confidence from: {line}")
                    confidence = 0.0

        return MCQOutput(
            selected_option=selected_option,
            reasoning=reasoning,
            confidence=confidence
        )

    def _format_document_context(self, doc: Document) -> Dict[str, str]:
        """Format a single document into a context dictionary."""
        return {
            "content": doc.page_content,
            "metadata": str(doc.metadata) if doc.metadata else ""
        }

    def answer_mcq(self, mcq_input: MCQInput, num_chunks: int = 3) -> MCQOutput:
        """Process MCQ question and return structured answer with reasoning."""
        # Retrieve relevant context
        docs = self.retriever.retrieve(mcq_input["question"], num_chunks)
        context = "\n".join(doc.page_content for doc in docs)

        # Apply reasoning strategy
        raw_output = self.reasoner.reason(
            context=context,
            question=mcq_input["question"],
            options=mcq_input["options"]
        )

        # Parse and return structured output
        return self._parse_output(raw_output)

    def answer_mcq_with_context(self, mcq_input: MCQInput, num_chunks: int = 3) -> MCQOutputWithContext:
        """
        Process MCQ question and return structured answer with reasoning and the contexts used.
        
        Args:
            mcq_input (MCQInput): The input MCQ question and options
            num_chunks (int): Number of context chunks to retrieve
            
        Returns:
            MCQOutputWithContext: Answer, reasoning, confidence, and the contexts used
        """
        # Retrieve relevant context
        docs = self.retriever.retrieve(mcq_input["question"], num_chunks)
        context = "\n".join(doc.page_content for doc in docs)

        # Apply reasoning strategy
        raw_output = self.reasoner.reason(
            context=context,
            question=mcq_input["question"],
            options=mcq_input["options"]
        )

        # Parse the basic output
        basic_output = self._parse_output(raw_output)
        
        # Format the contexts
        formatted_contexts = [self._format_document_context(doc) for doc in docs]

        # Combine into enhanced output
        return MCQOutputWithContext(
            selected_option=basic_output["selected_option"],
            reasoning=basic_output["reasoning"],
            confidence=basic_output["confidence"],
            contexts=formatted_contexts
        )