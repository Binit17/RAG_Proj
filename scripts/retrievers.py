# scripts/retrievers.py
from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document
from langchain.retrievers import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever
from typing import Dict, Tuple
import numpy as np

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, num_chunks: int) -> List[Document]:
        pass

class SingleStageRetriever(BaseRetriever):
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, num_chunks: int) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=num_chunks)

#From Claude
# class TwoStageRetriever(BaseRetriever):
#     def __init__(self, vectorstore: Chroma, llm: ChatOpenAI):
#         self.vectorstore = vectorstore
#         self.multi_query_retriever = MultiQueryRetriever.from_llm(
#             retriever=vectorstore.as_retriever(),
#             llm=llm
#         )

#     def retrieve(self, query: str, num_chunks: int) -> List[Document]:
#         # Stage 1: Get documents using multi-query retrieval
#         docs_stage1 = self.multi_query_retriever.get_relevant_documents(query)
        
#         # Stage 2: Rerank using similarity scores
#         reranked_docs = sorted(
#             docs_stage1,
#             key=lambda x: self.vectorstore.similarity_search_with_score(
#                 query,
#                 k=1,
#                 documents=[x]
#             )[0][1],
#             reverse=True
#         )
        
#         return reranked_docs[:num_chunks]

#From GPT
class TwoStageRetriever(BaseRetriever):
    def __init__(self, vectorstore: Chroma, llm: ChatOpenAI):
        self.vectorstore = vectorstore
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(),
            llm=llm
        )

    def retrieve(self, query: str, num_chunks: int) -> List[Document]:
        # Stage 1: Get documents using multi-query retrieval
        docs_stage1 = self.multi_query_retriever.get_relevant_documents(query)
        
        # Stage 2: Rerank documents based on similarity scores
        reranked_docs = []
        for doc in docs_stage1:
            score = self.vectorstore.similarity_search_with_score(query, k=1)[0][1]  # Get score for each doc
            reranked_docs.append((doc, score))
        
        # Sort documents by score
        reranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top `num_chunks` documents
        return [doc for doc, _ in reranked_docs[:num_chunks]]

#From Claude
# class ThreeStageRetriever(BaseRetriever):
#     def __init__(self, vectorstore: Chroma, llm: ChatOpenAI):
#         self.vectorstore = vectorstore
#         self.multi_query_retriever = MultiQueryRetriever.from_llm(
#             retriever=vectorstore.as_retriever(),
#             llm=llm
#         )
#         # Initialize BM25 retriever with documents from vectorstore
#         all_docs = self.vectorstore.similarity_search("", k=1000)  # Get all documents
#         self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        
#     def _hybrid_score(self, query: str, doc: Document) -> float:
#         """
#         Combine dense and sparse retrieval scores
#         """
#         # Get dense similarity score
#         dense_score = self.vectorstore.similarity_search_with_score(
#             query,
#             k=1,
#             documents=[doc]
#         )[0][1]
        
#         # Get BM25 score
#         bm25_results = self.bm25_retriever.get_relevant_documents(query)
#         bm25_score = 1.0 if doc.page_content in [d.page_content for d in bm25_results[:5]] else 0.0
        
#         # Combine scores (you can adjust the weights)
#         return 0.7 * dense_score + 0.3 * bm25_score

#     def retrieve(self, query: str, num_chunks: int) -> List[Document]:
#         # Stage 1: Initial broad retrieval
#         initial_docs = self.vectorstore.similarity_search(
#             query, 
#             k=num_chunks * 2
#         )

#         # Stage 2: Multi-query expansion and retrieval
#         expanded_docs = self.multi_query_retriever.get_relevant_documents(query)
#         combined_docs = list({doc.page_content: doc for doc in initial_docs + expanded_docs}.values())

#         # Stage 3: Hybrid reranking (combining dense and sparse retrieval)
#         scored_docs = [
#             (doc, self._hybrid_score(query, doc))
#             for doc in combined_docs
#         ]
        
#         # Sort by hybrid score and take top k
#         ranked_results = sorted(scored_docs, key=lambda x: x[1], reverse=True)
#         return [doc for doc, _ in ranked_results[:num_chunks]]


#from GPT
class ThreeStageRetriever(BaseRetriever):
    def __init__(self, vectorstore: Chroma, llm: ChatOpenAI):
        self.vectorstore = vectorstore
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(),
            llm=llm
        )
        # Initialize BM25 retriever with documents from vectorstore
        all_docs = self.vectorstore.similarity_search("", k=1000)  # Get all documents
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        
    def _hybrid_score(self, query: str, doc: Document) -> float:
        """
        Combine dense and sparse retrieval scores.
        """
        # Get dense similarity score
        dense_score = self.vectorstore.similarity_search_with_score(query, k=1)[0][1]
        
        # Get BM25 score
        bm25_results = self.bm25_retriever.get_relevant_documents(query)
        bm25_score = 1.0 if doc.page_content in [d.page_content for d in bm25_results[:5]] else 0.0
        
        # Combine scores (you can adjust the weights)
        return 0.7 * dense_score + 0.3 * bm25_score

    def retrieve(self, query: str, num_chunks: int) -> List[Document]:
        # Stage 1: Initial broad retrieval (dense retrieval)
        initial_docs = self.vectorstore.similarity_search(query, k=num_chunks * 2)

        # Stage 2: Multi-query expansion and retrieval
        expanded_docs = self.multi_query_retriever.get_relevant_documents(query)
        combined_docs = list({doc.page_content: doc for doc in initial_docs + expanded_docs}.values())

        # Stage 3: Hybrid reranking (combining dense and sparse retrieval)
        scored_docs = [
            (doc, self._hybrid_score(query, doc))
            for doc in combined_docs
        ]
        
        # Sort by hybrid score and take top `num_chunks` documents
        ranked_results = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_results[:num_chunks]]
