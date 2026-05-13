"""
retriever.py
Thin wrapper around VectorStore.search – makes it easy to swap retrieval
strategies (e.g. hybrid BM25 + vector) without touching rag_chain.py.
"""

from __future__ import annotations
from src.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, top_k: int = 4):
        self.vector_store = vector_store
        self.top_k        = top_k

    def retrieve(self, query: str) -> list[dict]:
        """Return the top-k relevant chunks for *query*."""
        return self.vector_store.search(query, top_k=self.top_k)
