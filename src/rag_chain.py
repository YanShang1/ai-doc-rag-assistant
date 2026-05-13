"""
rag_chain.py
Orchestrates retrieval + LLM answer generation.
"""

from __future__ import annotations
from openai import OpenAI
from src.retriever   import Retriever
from src.vector_store import VectorStore

LLM_MODEL   = "gpt-4o-mini"
SYSTEM_PROMPT = """\
You are DocuMind, a helpful AI assistant that answers questions based strictly
on the provided document excerpts.

Rules:
1. Answer using ONLY the information in the excerpts below.
2. If the answer is not found in the excerpts, say so clearly – do not guess.
3. Keep answers concise but complete. Use bullet points where appropriate.
4. Always cite the page number(s) you relied on in your answer using [p. N] notation.
"""


class RAGChain:
    def __init__(self, api_key: str, vector_store: VectorStore, top_k: int = 4):
        self.client    = OpenAI(api_key=api_key)
        self.retriever = Retriever(vector_store, top_k=top_k)

    def ask(self, question: str) -> dict:
        """
        Parameters
        ----------
        question : str

        Returns
        -------
        dict
            {"answer": str, "sources": list[dict]}
        """
        chunks   = self.retriever.retrieve(question)
        context  = self._build_context(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Document excerpts:\n{context}\n\nQuestion: {question}"},
        ]

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=800,
        )

        answer = response.choices[0].message.content.strip()
        return {"answer": answer, "sources": chunks}

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        parts = []
        for c in chunks:
            parts.append(f"[Page {c['page_num']}]\n{c['text']}")
        return "\n\n---\n\n".join(parts)
