"""
Milestone 4: Embed chunks and store in ChromaDB; expose retrieval functions.

Stretch features:
  - Hybrid search: BM25 keyword scores fused with semantic scores via RRF.
  - Metadata filtering: restrict results by source_type or professor.

Usage:
  python embed.py            # build the vector store (run once)
  python embed.py --test     # build then run the 3 evaluation test queries
"""

import argparse
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ingest import build_chunks

COLLECTION_NAME = "gmu_cs_guide"
CHROMA_PATH = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singletons — loaded once, reused across calls
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None

# BM25 index built lazily from the full corpus stored in ChromaDB
_bm25: BM25Okapi | None = None
_bm25_corpus: list[dict] | None = None  # parallel list of {text, metadata}


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_collection(path: str = CHROMA_PATH) -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=path)
        _collection = client.get_or_create_collection(COLLECTION_NAME)
    return _collection


# ---------------------------------------------------------------------------
# Embedding + storage
# ---------------------------------------------------------------------------

def embed_and_store(chunks: list[dict], chroma_path: str = CHROMA_PATH) -> None:
    """
    Embed every chunk with all-MiniLM-L6-v2 and upsert into ChromaDB.

    Each document is stored with metadata: source_type, professor, course, year,
    file_path, chunk_index.  These fields are used for attribution and optional
    metadata filtering at retrieval time.
    """
    model = _get_model()
    collection = _get_collection(chroma_path)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {MODEL_NAME}…", flush=True)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source_type": c.get("source_type", ""),
            "professor": c.get("professor", ""),
            "course": c.get("course", ""),
            "year": str(c.get("year", "")),
            "file_path": c.get("file_path", ""),
            "chunk_index": c.get("chunk_index", 0),
        }
        for c in chunks
    ]

    # Upsert in batches to avoid memory spikes on large corpora
    batch_size = 256
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"Stored {collection.count()} vectors in ChromaDB at '{chroma_path}'.")


# ---------------------------------------------------------------------------
# Retrieval — semantic only
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    k: int = 5,
    filters: dict | None = None,
    chroma_path: str = CHROMA_PATH,
) -> list[dict]:
    """
    Embed `query` and return the top-k most similar chunks from ChromaDB.

    Each result is a dict:
      { "text": str, "distance": float, "metadata": dict }

    `filters` is an optional ChromaDB `where` clause, e.g.:
      {"source_type": "Official"}   — restrict to catalog/faculty docs
      {"professor": "Katherine Russell"}
    """
    model = _get_model()
    collection = _get_collection(chroma_path)

    query_embedding = model.encode([query])[0].tolist()

    kwargs: dict = {"query_embeddings": [query_embedding], "n_results": k}
    if filters:
        kwargs["where"] = filters

    results = collection.query(**kwargs)

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({"text": doc, "distance": round(dist, 4), "metadata": meta})

    return output


# ---------------------------------------------------------------------------
# Stretch: Hybrid search  (BM25 + semantic, fused via Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def _get_bm25(chroma_path: str = CHROMA_PATH) -> tuple[BM25Okapi, list[dict]]:
    """Build (or return cached) BM25 index over the full ChromaDB corpus."""
    global _bm25, _bm25_corpus
    if _bm25 is not None:
        return _bm25, _bm25_corpus  # type: ignore[return-value]

    collection = _get_collection(chroma_path)
    total = collection.count()
    # Fetch all stored documents + metadata in one call
    raw = collection.get(limit=total, include=["documents", "metadatas"])
    corpus = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(raw["documents"], raw["metadatas"])
    ]
    tokenized = [_tokenize(c["text"]) for c in corpus]
    _bm25 = BM25Okapi(tokenized)
    _bm25_corpus = corpus
    return _bm25, corpus


def retrieve_hybrid(
    query: str,
    k: int = 5,
    filters: dict | None = None,
    rrf_k: int = 60,
    chroma_path: str = CHROMA_PATH,
) -> list[dict]:
    """
    Hybrid retrieval: fuse BM25 and semantic rankings with Reciprocal Rank Fusion.

    RRF score = 1/(rrf_k + rank_semantic) + 1/(rrf_k + rank_bm25)
    Higher score = better combined rank.

    Returns the top-k chunks as dicts:
      { "text": str, "distance": float, "bm25_score": float,
        "rrf_score": float, "metadata": dict }
    """
    # ---- semantic retrieval (fetch more candidates to fuse against) ----
    fetch_n = min(k * 6, 60)
    model = _get_model()
    collection = _get_collection(chroma_path)
    query_embedding = model.encode([query])[0].tolist()

    sem_kwargs: dict = {"query_embeddings": [query_embedding], "n_results": fetch_n}
    if filters:
        sem_kwargs["where"] = filters
    sem_results = collection.query(**sem_kwargs)

    semantic_chunks = []
    for doc, meta, dist in zip(
        sem_results["documents"][0],
        sem_results["metadatas"][0],
        sem_results["distances"][0],
    ):
        semantic_chunks.append({"text": doc, "distance": round(dist, 4), "metadata": meta})

    # ---- BM25 retrieval ----
    bm25, corpus = _get_bm25(chroma_path)
    query_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    # Apply same filters to BM25 candidates if requested
    if filters:
        filtered_indices = [
            i for i, c in enumerate(corpus)
            if all(c["metadata"].get(key) == val for key, val in filters.items())
        ]
    else:
        filtered_indices = list(range(len(corpus)))

    # Rank BM25 candidates by score descending
    bm25_ranked = sorted(filtered_indices, key=lambda i: bm25_scores[i], reverse=True)[:fetch_n]

    # ---- Reciprocal Rank Fusion ----
    # Map chunk text → RRF score (text is a stable identity key)
    rrf: dict[str, float] = {}
    text_to_chunk: dict[str, dict] = {}

    for rank, chunk in enumerate(semantic_chunks):
        key = chunk["text"]
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        text_to_chunk[key] = chunk

    for rank, idx in enumerate(bm25_ranked):
        key = corpus[idx]["text"]
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        if key not in text_to_chunk:
            text_to_chunk[key] = {
                "text": corpus[idx]["text"],
                "distance": 1.0,  # not ranked by semantic — no distance available
                "metadata": corpus[idx]["metadata"],
            }

    # Build final ranked list
    ranked = sorted(rrf.keys(), key=lambda t: rrf[t], reverse=True)[:k]
    output = []
    for text in ranked:
        chunk = dict(text_to_chunk[text])
        chunk["rrf_score"] = round(rrf[text], 6)
        bm25_idx = next(
            (i for i, c in enumerate(corpus) if c["text"] == text), None
        )
        chunk["bm25_score"] = round(float(bm25_scores[bm25_idx]), 4) if bm25_idx is not None else 0.0
        output.append(chunk)

    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _run_test_queries() -> None:
    test_queries = [
        "What do students say about exam difficulty in Justin Wilson's CS222?",
        "Is Wing Lam's SWE437 worth taking even though the quizzes are hard?",
        "What are the prerequisites for CS483 Analysis of Algorithms?",
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)
        results = retrieve(query)
        for rank, r in enumerate(results, 1):
            meta = r["metadata"]
            source = meta.get("professor") or meta.get("file_path", "")
            print(f"\n  [{rank}] distance={r['distance']} | {meta['source_type']} | {source}")
            print(f"  {r['text'][:300].strip()}{'…' if len(r['text']) > 300 else ''}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run evaluation queries after building")
    parser.add_argument("--chroma-path", default=CHROMA_PATH)
    args = parser.parse_args()

    chunks = build_chunks()
    embed_and_store(chunks, chroma_path=args.chroma_path)

    if args.test:
        _run_test_queries()
