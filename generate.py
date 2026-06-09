"""
Milestone 5: Grounded generation via Groq API.

Takes a query + retrieved chunks, builds a context block, and calls
llama-3.3-70b-versatile with a strict grounding system prompt.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise ValueError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are a helpful assistant for students at George Mason University.
You answer questions about CS professors and courses ONLY using the documents provided below.

Rules you must follow without exception:
1. Answer solely from the provided documents. Do not use your general training knowledge.
2. Every factual claim must cite its source using the format: (Source: <professor or document name>).
3. If the documents do not contain enough information to answer, respond with exactly:
   "I don't have enough information in my documents to answer that question."
4. Do not invent, extrapolate, or assume anything not stated in the documents.
5. When multiple sources say different things, report each perspective and its source separately."""


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("professor") or meta.get("file_path", "unknown")
        source_type = meta.get("source_type", "")
        label = f"[Document {i} | {source_type} | {source}]"
        parts.append(f"{label}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate(query: str, chunks: list[dict], memory_context: str = "") -> dict:
    """
    Generate a grounded answer from retrieved chunks.

    Returns:
        {
          "answer": str,
          "sources": list[str],   # deduplicated source labels
          "chunks_used": int
        }
    """
    if not chunks:
        return {
            "answer": "I don't have enough information in my documents to answer that question.",
            "sources": [],
            "chunks_used": 0,
        }

    context = _build_context(chunks)
    memory_block = ""
    if memory_context:
        memory_block = f"\nConversation history (for context only — still answer from documents):\n{memory_context}\n"

    user_message = f"""Documents:
{context}
{memory_block}
Question: {query}

Answer (cite sources inline using the format: (Source: <name>)):"""

    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,   # low temperature for factual, grounded answers
        max_tokens=600,
    )

    answer = response.choices[0].message.content.strip()

    # Build a deduplicated source list from the retrieved chunks' metadata
    seen = set()
    sources = []
    for chunk in chunks:
        meta = chunk["metadata"]
        professor = meta.get("professor", "")
        source_type = meta.get("source_type", "")
        file_path = meta.get("file_path", "")
        label = professor if professor else file_path
        if source_type:
            label = f"{label} ({source_type})"
        if label and label not in seen:
            seen.add(label)
            sources.append(label)

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
    }
