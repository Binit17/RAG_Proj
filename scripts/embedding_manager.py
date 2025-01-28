# scripts/embedding_manager.py

from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.vectorstores import Chroma
from langchain.schema import Document

def create_embeddings() -> OpenAIEmbeddings:
    """
    Create OpenAI embeddings instance.
    OpenAI API key is expected to be in environment variables.
    
    Returns:
        OpenAIEmbeddings: Configured embeddings instance
    """
    try:
        return OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception as e:
        raise Exception(f"Error creating embeddings: {str(e)}")

def setup_vector_store(chunks: List[Document], 
                      embeddings: OpenAIEmbeddings,
                      persist_directory: str = "chroma_db") -> Chroma:
    """
    Create and populate a ChromaDB vector store with document chunks.
    
    Args:
        chunks (List[Document]): List of document chunks to store
        embeddings (OpenAIEmbeddings): Embeddings instance
        persist_directory (str): Directory to store the database
    
    Returns:
        Chroma: Configured vector store instance
    """
    try:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        return vectordb
    except Exception as e:
        raise Exception(f"Error setting up vector store: {str(e)}")