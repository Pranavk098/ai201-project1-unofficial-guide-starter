"""
Milestone 4: Embed chunks and store in ChromaDB; expose a retrieval function.

Usage:
  python embed.py            # build the vector store (run once)
  python embed.py --test     # build then run the 3 evaluation test queries
"""

import argparse
import sys

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import build_chunks

COLLECTION_NAME = "gmu_cs_guide"
CHROMA_PATH = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singletons — loaded once, reused across calls
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


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
# Retrieval
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
