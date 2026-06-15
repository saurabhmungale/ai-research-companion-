# Research Chat Assistant

A multi-conversation chat app that answers questions using live web search
and your own uploaded documents, with a semantic cache to avoid repeating
expensive searches and LLM calls.

## Architecture

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq (LLaMA models) |
| Web search | Tavily |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB |
| Chat history | SQLite |
| Deployment | Streamlit Community Cloud |

### How a question gets answered

1. **Semantic cache check** — the question is embedded and compared against
   `qa_cache` in ChromaDB. If a sufficiently similar question was answered
   before, that cached answer is reused (marked "from cache" in the UI).
2. **Web search** — if no cache hit, Tavily searches the live web for the
   question (`rag_pipeline.search_web`).
3. **Document search** — any files you've uploaded to the current
   conversation are searched in the `documents` ChromaDB collection, scoped
   to that conversation only.
4. **Context assembly** — web results and document chunks are combined into
   a single numbered context (`rag_pipeline.build_context`).
5. **Answer generation** — Groq's LLaMA model answers using that context
   plus recent chat history, citing `[Source N]` for each claim.
6. **Caching** — the new question/answer pair is stored in `qa_cache` for
   future reuse.
7. **Persistence** — both the user message and the assistant's reply
   (including sources) are saved to SQLite under the active conversation.

### Project structure

```
rag_chat_app/
├── app.py                   # Streamlit UI and orchestration
├── requirements.txt
├── .env.example
├── .gitignore
└── src/
    ├── database.py          # SQLite: conversations + messages
    ├── vector_store.py       # ChromaDB: semantic cache + document search
    ├── document_processor.py # PDF/text extraction and chunking
    └── rag_pipeline.py        # Tavily search + Groq answer generation
```

## Local setup

1. **Create a virtual environment and install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add your API keys**:
   - Copy `.env.example` to `.env`
   - Get a Tavily key: https://app.tavily.com
   - Get a Groq key: https://console.groq.com
   - Fill in both keys in `.env`

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

The first run will download the embedding model (~80MB) and create
`chat_history.db` and `chroma_db/` in the project folder.

## Using the app

- **New chat** starts a fresh conversation; it's automatically titled from
  your first question.
- Click any conversation in the sidebar to switch to it; click **✕** to
  delete it.
- Upload PDF/TXT/MD files in the sidebar and click **Index uploaded files**
  to make them searchable for the *current* conversation. Documents are not
  shared between conversations.
- Each assistant reply has a **Sources** section showing which web pages
  and/or documents were used (or "from cache" if a similar question was
  answered before).

## Deploying to Streamlit Community Cloud

1. Push this project to a GitHub repository (the `.gitignore` already keeps
   `.env`, the local databases, and the vector store out of version control).
2. Go to https://share.streamlit.io, sign in, and create a new app pointing
   at your repo and `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```toml
   TAVILY_API_KEY = "tvly-xxxxxxxxxxxxxxxx"
   GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxx"
   ```
4. Deploy. `pysqlite3-binary` in `requirements.txt` provides the newer
   SQLite version ChromaDB needs on Streamlit Cloud's base image — `app.py`
   swaps it in automatically.

**Storage note**: Streamlit Community Cloud's filesystem is ephemeral —
`chat_history.db` and `chroma_db/` will reset if the app restarts or
redeploys. This is fine for a demo/portfolio project. For persistence across
restarts, you'd point `database.py` and `vector_store.py` at an external
database (e.g. a hosted Postgres or a managed vector DB) instead of local
files.

## Customization

- **Change the LLM**: edit `GROQ_MODEL` in `src/rag_pipeline.py`.
- **Cache sensitivity**: edit `CACHE_DISTANCE_THRESHOLD` in
  `src/vector_store.py` (lower = stricter matching, fewer cache hits).
- **Number of web results**: `max_results` in `rag.search_web()`.
- **Document chunk size**: `chunk_size` / `overlap` in
  `docproc.chunk_text()`.
- **Embedding model**: `EMBEDDING_MODEL_NAME` in `src/vector_store.py` (any
  Sentence Transformers model works; smaller models are faster on
  Streamlit Cloud's free tier).

## Security note

Never hardcode API keys in the source code. Locally, use `.env`
(git-ignored). On Streamlit Cloud, use the Secrets manager.
