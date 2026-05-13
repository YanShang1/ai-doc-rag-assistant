"""
document_loader.py
Loads a PDF (from a file path or an uploaded Streamlit file object)
and returns a list of dicts: {page_num, text}.
"""

from __future__ import annotations
import io
from typing import Union

import fitz  # PyMuPDF


def load_pdf(source: Union[str, object]) -> list[dict]:
    """
    Parameters
    ----------
    source : str | UploadedFile
        Either a file-system path or a Streamlit UploadedFile object.

    Returns
    -------
    list[dict]
        [{"page_num": int, "text": str}, ...]
    """
    if isinstance(source, str):
        doc = fitz.open(source)
    else:
        # Streamlit UploadedFile – read bytes
        data = source.read()
        doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")

    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:                          # skip blank pages
            pages.append({"page_num": i + 1, "text": text})

    doc.close()

    if not pages:
        raise ValueError("No readable text found in the PDF. "
                         "The file may be scanned (image-only).")
    return pages
