from typing import List, Dict
import json
from tqdm import tqdm
import random
from datetime import datetime
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from .mcq_processor import MCQProcessor, MCQInput

def generate_mcq_dataset(vectordb: Chroma, 
                        mcq_processor: MCQProcessor,
                        num_questions: int = 100,
                        output_file: str = "mcq_dataset.json") -> None:
    """
    Generate MCQs from vectordb content and save to JSON file.
    
    Args:
        vectordb: Chroma vector database instance
        mcq_processor: Initialized MCQProcessor instance
        num_questions: Number of MCQs to generate
        output_file: Path to save JSON output
    """
    
    # Template for generating questions
    question_template = """
    Based on the medical content provided, generate 1 multiple choice question.
    The question should be {question_type} and {difficulty_level} difficulty.
    Focus on {topic_type} aspects.
    Ensure the question is clear, unambiguous, and has one definitively correct answer.
    Return in this exact format:
    Question: [question text]
    A) [option text]
    B) [option text]
    C) [option text]
    D) [option text]
    Correct: [A/B/C/D]
    Explanation: [explanation]
    """
    
    # Question parameters for variety
    question_types = ["explicit", "implicit"]
    difficulty_levels = ["easy", "medium", "hard"]
    topic_types = ["diagnostic", "treatment", "pathophysiology", "clinical presentation", 
                  "complications", "management", "prevention"]
    
    mcq_dataset = {"mcqs": [], "metadata": {
        "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_questions": num_questions
    }}
    
    # Get all unique documents from vectordb
    all_docs = vectordb.similarity_search("", k=1000)  # Get a large sample
    
    print(f"Generating {num_questions} MCQs...")
    for i in tqdm(range(num_questions)):
        # Randomly select parameters for variety
        question_type = random.choice(question_types)
        difficulty = random.choice(difficulty_levels)
        topic = random.choice(topic_types)
        
        # Generate question using template
        prompt = question_template.format(
            question_type=question_type,
            difficulty_level=difficulty,
            topic_type=topic
        )
        
        # Get random context chunks
        context_docs = random.sample(all_docs, min(3, len(all_docs)))
        context = "\n".join(doc.page_content for doc in context_docs)
        
        try:
            # Generate MCQ using the processor
            generated_mcq = mcq_processor.llm.predict(context + "\n" + prompt)
            
            # Parse the generated MCQ
            lines = generated_mcq.strip().split('\n')
            question = ""
            options = {}
            correct = ""
            explanation = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("Question:"):
                    question = line.split("Question:", 1)[1].strip()
                elif line.startswith("A)"):
                    options["A"] = line.split("A)", 1)[1].strip()
                elif line.startswith("B)"):
                    options["B"] = line.split("B)", 1)[1].strip()
                elif line.startswith("C)"):
                    options["C"] = line.split("C)", 1)[1].strip()
                elif line.startswith("D)"):
                    options["D"] = line.split("D)", 1)[1].strip()
                elif line.startswith("Correct:"):
                    correct = line.split("Correct:", 1)[1].strip()
                elif line.startswith("Explanation:"):
                    explanation = line.split("Explanation:", 1)[1].strip()
            
            # Process MCQ through the processor for confidence scoring
            mcq_input = MCQInput(
                question=question,
                options=options
            )
            
            result = mcq_processor.answer_mcq(mcq_input)
            
            # Add to dataset
            mcq_dataset["mcqs"].append({
                "id": i + 1,
                "question": question,
                "options": options,
                "correct_answer": correct,
                "explanation": explanation,
                "confidence": result["confidence"],
                "type": question_type,
                "difficulty": difficulty,
                "topic": topic
            })
            
        except Exception as e:
            print(f"Error generating question {i+1}: {str(e)}")
            continue
    
    # Save to JSON file
    output_path = Path(output_file)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(mcq_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated {len(mcq_dataset['mcqs'])} MCQs successfully!")
    print(f"Saved to: {output_path.absolute()}")
    
    # Print some statistics
    question_types_count = {}
    difficulty_count = {}
    topic_count = {}
    
    for mcq in mcq_dataset["mcqs"]:
        question_types_count[mcq["type"]] = question_types_count.get(mcq["type"], 0) + 1
        difficulty_count[mcq["difficulty"]] = difficulty_count.get(mcq["difficulty"], 0) + 1
        topic_count[mcq["topic"]] = topic_count.get(mcq["topic"], 0) + 1
    
    print("\nDataset Statistics:")
    print("Question Types:", question_types_count)
    print("Difficulty Levels:", difficulty_count)
    print("Topics:", topic_count)