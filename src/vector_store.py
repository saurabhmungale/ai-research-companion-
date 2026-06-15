"""
ChromaDB-backed vector store with two collections:

- "qa_cache": embeddings of past questions. Used as a semantic cache so a
  similar question can reuse a previous answer instead of re-searching and
  re-generating.
- "documents": embeddings of chunks from user-uploaded documents, tagged
  per conversation, used for document-grounded Q&A alongside web search.
"""

import uuid

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Cosine distance below this counts as "close enough" to reuse a cached
# answer. Cosine distance = 1 - cosine similarity, so 0.08 ~= 92% similarity.
# Lower this for stricter matching, raise it for more aggressive caching.
CACHE_DISTANCE_THRESHOLD = 0.08


def load_embedding_model():
    """Load the sentence-transformer model used for all embeddings."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_chroma_client(path=CHROMA_PATH):
    return chromadb.PersistentClient(path=path)


def get_collections(client):
    """Return (qa_cache_collection, documents_collection), creating them if needed."""
    qa_cache = client.get_or_create_collection(
        name="qa_cache",
        metadata={"hnsw:space": "cosine"},
    )
    documents = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )
    return qa_cache, documents


def check_cache(qa_cache, embedding_model, query):
    """
    Look for a semantically similar past question.
    Returns a dict with 'answer' and 'sources' (JSON string) if a close
    match is found, else None.
    """
    if qa_cache.count() == 0:
        return None

    query_embedding = embedding_model.encode(query).tolist()
    results = qa_cache.query(query_embeddings=[query_embedding], n_results=1)

    distances = results.get("distances") or [[]]
    distances = distances[0] if distances else []
    if not distances or distances[0] > CACHE_DISTANCE_THRESHOLD:
        return None

    metadata = results["metadatas"][0][0]
    return {
        "answer": metadata.get("answer", ""),
        "sources": metadata.get("sources", ""),
        "matched_question": results["documents"][0][0],
    }


def add_to_cache(qa_cache, embedding_model, query, answer, sources_json):
    """Store a question/answer pair in the semantic cache."""
    embedding = embedding_model.encode(query).tolist()
    qa_cache.add(
        ids=[str(uuid.uuid4())],
        embeddings=[embedding],
        documents=[query],
        metadatas=[{"answer": answer, "sources": sources_json}],
    )


def add_document_chunks(documents_collection, embedding_model, conversation_id, doc_name, chunks):
    """Embed and store document chunks, tagged with the owning conversation and file name."""
    if not chunks:
        return

    embeddings = embedding_model.encode(chunks).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"conversation_id": conversation_id, "source": doc_name, "chunk_index": i}
        for i in range(len(chunks))
    ]
    documents_collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )


def search_documents(documents_collection, embedding_model, conversation_id, query, n_results=3):
    """Search uploaded document chunks for this conversation, return list of dicts."""
    if documents_collection.count() == 0:
        return []

    query_embedding = embedding_model.encode(query).tolist()
    try:
        results = documents_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"conversation_id": conversation_id},
        )
    except Exception:
        return []

    docs = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    docs = docs[0] if docs else []
    metadatas = metadatas[0] if metadatas else []

    return [
        {"content": doc, "source": meta.get("source", "uploaded document")}
        for doc, meta in zip(docs, metadatas)
    ]
