"""
Streamlit RAG chat app.

- Frontend: Streamlit
- LLM: Groq (LLaMA models)
- Web search: Tavily
- Embeddings: Sentence Transformers
- Vector DB: ChromaDB (semantic answer cache + uploaded document search)
- Chat history: SQLite, multiple named conversations
"""

import json
import os
import sys

# ChromaDB on Streamlit Community Cloud needs a newer sqlite3 than the
# system default. pysqlite3-binary provides one; swap it in before chromadb
# is imported anywhere. Safe to skip locally if pysqlite3 isn't installed.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

from src import database as db
from src import vector_store as vs
from src import document_processor as docproc
from src import rag_pipeline as rag


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()

st.set_page_config(page_title="Research Chat Assistant", page_icon="🔎", layout="wide")


def get_secret(key):
    """Read a secret from Streamlit secrets (cloud) or environment (local .env)."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)


@st.cache_resource(show_spinner="Loading models and connecting to APIs...")
def init_clients():
    tavily_client = TavilyClient(api_key=get_secret("TAVILY_API_KEY"))
    groq_client = Groq(api_key=get_secret("GROQ_API_KEY"))
    embedding_model = vs.load_embedding_model()
    chroma_client = vs.get_chroma_client()
    qa_cache, documents_collection = vs.get_collections(chroma_client)
    return tavily_client, groq_client, embedding_model, qa_cache, documents_collection


tavily_client, groq_client, embedding_model, qa_cache, documents_collection = init_clients()
db.init_db()


# ---------------------------------------------------------------------------
# Conversation management
# ---------------------------------------------------------------------------
if "active_conversation_id" not in st.session_state:
    existing = db.get_conversations()
    if existing:
        st.session_state.active_conversation_id = existing[0]["id"]
    else:
        st.session_state.active_conversation_id = db.create_conversation("New chat")


def new_chat():
    st.session_state.active_conversation_id = db.create_conversation("New chat")


def switch_chat(conv_id):
    st.session_state.active_conversation_id = conv_id


# ---------------------------------------------------------------------------
# Sidebar: conversation list + document upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Research Chat")

    if st.button("New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()
    st.caption("Conversations")

    conversations = db.get_conversations()
    for conv in conversations:
        is_active = conv["id"] == st.session_state.active_conversation_id
        label = ("• " if is_active else "") + conv["title"]

        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                switch_chat(conv["id"])
                st.rerun()
        with col2:
            if st.button("✕", key=f"del_{conv['id']}"):
                db.delete_conversation(conv["id"])
                if conv["id"] == st.session_state.active_conversation_id:
                    remaining = db.get_conversations()
                    st.session_state.active_conversation_id = (
                        remaining[0]["id"] if remaining else db.create_conversation("New chat")
                    )
                st.rerun()

    st.divider()
    st.caption("Documents for this conversation")
    uploaded_files = st.file_uploader(
        "Upload PDF or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("Index uploaded files", use_container_width=True):
        with st.spinner("Indexing documents..."):
            for f in uploaded_files:
                text = docproc.extract_text(f)
                chunks = docproc.chunk_text(text)
                vs.add_document_chunks(
                    documents_collection,
                    embedding_model,
                    st.session_state.active_conversation_id,
                    f.name,
                    chunks,
                )
        st.success(f"Indexed {len(uploaded_files)} file(s) for this conversation.")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
active_id = st.session_state.active_conversation_id
messages = db.get_messages(active_id)
active_conv = next((c for c in conversations if c["id"] == active_id), None)

st.title("Research Chat Assistant")
st.caption("Ask a question — answers combine live web search and any documents you've uploaded.")


def render_sources(sources, from_cache):
    cache_tag = " (from cache)" if from_cache else ""
    with st.expander(f"Sources{cache_tag}"):
        for src in sources:
            if src["type"] == "web":
                st.markdown(f"- [{src['title']}]({src['url']})")
            else:
                st.markdown(f"- {src['title']} (uploaded document)")


for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg["sources"]:
            render_sources(msg["sources"], msg["from_cache"])


query = st.chat_input("Ask a question...")

if query:
    db.add_message(active_id, "user", query)

    if active_conv and active_conv["title"] == "New chat":
        title = query[:50] + ("..." if len(query) > 50 else "")
        db.rename_conversation(active_id, title)

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            cache_hit = vs.check_cache(qa_cache, embedding_model, query)

            if cache_hit:
                answer = cache_hit["answer"]
                sources = json.loads(cache_hit["sources"]) if cache_hit["sources"] else []
                from_cache = True
            else:
                web_results = rag.search_web(tavily_client, query)
                doc_results = vs.search_documents(
                    documents_collection, embedding_model, active_id, query
                )
                context, sources = rag.build_context(web_results, doc_results)

                history = [{"role": m["role"], "content": m["content"]} for m in messages]
                answer = rag.generate_answer(groq_client, query, context, history)

                vs.add_to_cache(qa_cache, embedding_model, query, answer, json.dumps(sources))
                from_cache = False

        st.markdown(answer)
        if sources:
            render_sources(sources, from_cache)

    db.add_message(active_id, "assistant", answer, sources=sources, from_cache=from_cache)
    st.rerun()
