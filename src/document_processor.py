"""
Helpers for turning uploaded files (PDF or plain text) into text chunks
ready for embedding and storage in ChromaDB.
"""

from pypdf import PdfReader


def extract_text(uploaded_file):
    """Extract raw text from an uploaded PDF or text file (Streamlit UploadedFile)."""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    # Treat anything else (.txt, .md, ...) as plain text
    return uploaded_file.read().decode("utf-8", errors="ignore")


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping word-based chunks.

    chunk_size: number of words per chunk
    overlap: number of words shared between consecutive chunks
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap

    return chunks
