"""
Core RAG logic: web search via Tavily, context assembly from web + document
results, and answer generation via Groq's LLaMA models.
"""

GROQ_MODEL = "llama-3.3-70b-versatile"


def search_web(tavily_client, query, max_results=5):
    """Run a Tavily search and return the list of result dicts."""
    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )
    return response.get("results", [])


def build_context(web_results, doc_results):
    """
    Combine web search results and document search results into a single
    numbered context string the LLM can cite from.

    Returns (context_string, sources) where `sources` is a list of dicts
    describing each [Source N]: {"label", "type", "title", "url"}.
    """
    parts = []
    sources = []
    n = 1

    for res in web_results:
        title = res.get("title", "Untitled")
        url = res.get("url", "")
        content = res.get("content", "")
        parts.append(f"[Source {n}] (web) {title}\nURL: {url}\nContent: {content}")
        sources.append({"label": f"Source {n}", "type": "web", "title": title, "url": url})
        n += 1

    for res in doc_results:
        source_name = res.get("source", "uploaded document")
        content = res.get("content", "")
        parts.append(f"[Source {n}] (document) {source_name}\nContent: {content}")
        sources.append({"label": f"Source {n}", "type": "document", "title": source_name, "url": ""})
        n += 1

    return "\n\n".join(parts), sources


def generate_answer(groq_client, query, context, chat_history=None, model=GROQ_MODEL):
    """
    Ask Groq's LLaMA model to answer `query` using `context`, optionally
    aware of recent `chat_history` (list of {"role", "content"} dicts).
    """
    system_prompt = (
        "You are a helpful research assistant. Answer the user's question "
        "using the information in the provided context, which may include "
        "web search results and/or excerpts from uploaded documents. If the "
        "context is insufficient, say so clearly. When you use information "
        "from a source, cite it using its [Source N] label."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        # Include recent turns for conversational continuity
        messages.extend(chat_history[-6:])

    messages.append(
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
    )

    response = groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content
