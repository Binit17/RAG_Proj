# scripts/document_processor.py
from typing import List
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and convert it to a list of Documents.
    
    Args:
        file_path (str): Path to the PDF file
    
    Returns:
        List[Document]: List of LangChain Document objects
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        raise Exception(f"Error loading PDF from {file_path}: {str(e)}")

def load_pdfs_from_directory(directory_path: str) -> List[Document]:
    """
    Load all PDF files from a directory and its subdirectories.
    
    Args:
        directory_path (str): Path to the directory containing PDF files
    
    Returns:
        List[Document]: Combined list of LangChain Document objects from all PDFs
    """
    pdf_files = list(Path(directory_path).rglob("*.pdf"))
    all_documents = []
    
    for pdf_path in pdf_files:
        try:
            documents = load_pdf(str(pdf_path))
            print(f"Loaded {len(documents)} pages from {pdf_path.name}")
            all_documents.extend(documents)
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {str(e)}")
            continue
    
    return all_documents

def chunk_text(documents: List[Document], 
               chunk_size: int = 1000,
               chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks for processing.
    
    Args:
        documents (List[Document]): List of documents to chunk
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
    
    Returns:
        List[Document]: List of chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks