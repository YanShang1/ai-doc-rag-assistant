"""
vector_store.py
Builds and queries an in-memory FAISS index using OpenAI embeddings.
"""

from __future__ import annotations
import numpy as np
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE      = 100          # max texts sent to OpenAI in one call


class VectorStore:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self._chunks: list[dict] = []
        self._index  = None          # faiss.IndexFlatIP

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, chunks: list[dict]) -> None:
        """Embed all chunks and build the FAISS index."""
        import faiss

        self._chunks = chunks
        texts = [c["text"] for c in chunks]

        vectors = self._embed_batch(texts)          # (N, dim)
        dim = vectors.shape[1]

        # Inner-product index (cosine similarity after normalisation)
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        self._index = index

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """Return the top-k most relevant chunks for a query."""
        import faiss

        q_vec = self._embed_batch([query])          # (1, dim)
        faiss.normalize_L2(q_vec)
        scores, indices = self._index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def stats(self) -> dict:
        return {"num_chunks": len(self._chunks)}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Call OpenAI Embeddings API in batches; return float32 array."""
        all_vectors = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
            vecs = [item.embedding for item in response.data]
            all_vectors.extend(vecs)

        return np.array(all_vectors, dtype="float32")
