"""
text_splitter.py
Splits page-level text into overlapping chunks suitable for embedding.
Each chunk carries metadata (page number, chunk index).
"""

from __future__ import annotations
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Parameters
    ----------
    pages : list[dict]
        Output of document_loader.load_pdf  →  [{page_num, text}, ...]
    chunk_size : int
        Approximate token/character size for each chunk.
    chunk_overlap : int
        Number of characters shared between adjacent chunks.

    Returns
    -------
    list[dict]
        [{"chunk_id": int, "page_num": int, "text": str}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    chunk_id = 0

    for page in pages:
        raw_chunks = splitter.split_text(page["text"])
        for raw in raw_chunks:
            chunks.append({
                "chunk_id": chunk_id,
                "page_num": page["page_num"],
                "text":     raw.strip(),
            })
            chunk_id += 1

    return chunks
