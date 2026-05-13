"""
utils.py
Small helper functions used across the app.
"""

from __future__ import annotations


def format_sources(chunks: list[dict]) -> str:
    """
    Render source chunks as HTML chips for display in Streamlit.
    Each chip shows the page number and a short text preview.
    """
    html_parts = []
    seen_pages: set[int] = set()

    for chunk in chunks:
        page = chunk["page_num"]
        if page in seen_pages:
            continue
        seen_pages.add(page)

        preview = chunk["text"][:120].replace("\n", " ").strip()
        if len(chunk["text"]) > 120:
            preview += "…"

        html_parts.append(
            f'<span class="source-chip">📄 p.{page} – {preview}</span>'
        )

    return "\n".join(html_parts)
