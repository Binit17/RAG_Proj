# Medical Q&A System with Reasoning

## About the Project

This project provides an intelligent system where doctors and medical experts can interact to obtain accurate answers to medical multiple-choice questions. The system retrieves relevant context from a vector database and responds with a correct answer, accompanied by detailed reasoning. It supports reasoning strategy like Chain-of-Thought making it ideal for analyzing complex medical scenarios.

---

## Workflow

### Step 1: Prepare the Guidelines

- Download the necessary medical guidelines or data from [https://drive.google.com/drive/folders/1ZNiIDUTNjUlt1RZ4AdnaKLWi-wRkn2qJ?usp=sharing].
- Alternatively, place your own guidelines in the `data/` directory.

### Step 2: Install Dependencies

Ensure all required libraries and dependencies are installed by running the following command:

```bash
pip install -r requirements.txt
```

### Step 3: Generate Embeddings

- The embedding model (larger than 100MB) is not included in the repository.
- To create embeddings, run the `implement.ipynb` notebook. The script will automatically generate the embeddings required for the vector database.
- If you prefer a custom embedding model, configure it in `scripts/embedding_manager.py`.

### Step 4: Run the Notebook

- Execute the remaining cells in `implement.ipynb` to:
  - Process the MCQ input.
  - Retrieve contexts from the vector database.
  - Evaluate the reasoning and model outputs.

### Step 5: Explore Documentation

For a detailed explanation of the implementation, refer to `presentation.pdf`.

---

### Scripts used:

### Script Overview

| **Script Name**           | **Summary**                                                                                                                                                                  |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `document_processor.py`   | Provides functions to load PDF files or directories of PDFs, convert them into LangChain Document objects, and split them into smaller chunks for efficient processing.      |
| `embedding_manager.py`    | Implements utilities to create OpenAI embeddings and set up a Chroma vector store to store and retrieve chunked documents for embedding-based applications.                  |
| `generate_mcq_dataset.py` | Generates multiple-choice questions (MCQs) from a vector store using varied templates, saves the dataset to a JSON file, and provides metadata and statistics.               |
| `mcq_processor.py`        | Defines a class to process MCQs by retrieving relevant context, applying reasoning strategies (e.g., chain of thought), and returning answers with reasoning and confidence. |
| `reasoning_strategies.py` | Implements abstract and specific reasoning strategies (e.g., Chain of Thought) for answering questions based on provided context using a language model.                     |

## Key Features

- Retrieves context using advanced retrieval strategies.
- Supports Chain-of-Thought Reasoning strategy:
- Outputs confidence scores with answers.

This system is a robust tool designed to assist medical professionals in decision-making and education.
