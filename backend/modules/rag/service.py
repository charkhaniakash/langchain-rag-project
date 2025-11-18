# backend/modules/rag/service.py
from typing import List

from src.llm_manager import LLMManager
from src.embeddings_manager import EmbeddingsManager
from src.vectorstore_manager import VectorStoreManager
from src.retriever_manager import RetrieverManager
from src.rag_chain import RAGChain

class RAGService:
    def __init__(self):
        self.llm = LLMManager().get_llm()
        self.embeddings = EmbeddingsManager().get_embeddings()
        self.vectorstore = VectorStoreManager(self.embeddings).get_vectorstore()
        self.retriever = RetrieverManager(self.vectorstore).create_retriever()
        self.rag_chain = RAGChain(llm=self.llm, retriever=self.retriever)
        self.rag_chain.create_chain()

    def query(self, question: str, top_k: int = 3) -> dict:
        return self.rag_chain.query(question)

rag_service = RAGService()