"""
Milestone 3: Document ingestion and chunking pipeline.

Loads .txt files from documents/, cleans them, and splits into chunks
using semantic boundaries defined in planning.md.
"""

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_documents(directory: str = "documents") -> list[tuple[str, dict]]:
    """
    Walk all subfolders under `directory`, read every .txt file, and return
    a list of (text, metadata) tuples.

    Metadata keys: professor, source_type, file_path
    """
    docs = []
    base = Path(directory)
    for path in sorted(base.rglob("*.txt")):
        if path.name == ".gitkeep":
            continue
        text = path.read_text(encoding="utf-8")
        metadata = _extract_metadata(text, path)
        docs.append((text, metadata))
    return docs


def _extract_metadata(text: str, path: Path) -> dict:
    """Pull professor name, source_type, and course from file header lines."""
    meta = {
        "professor": "",
        "source_type": _infer_source_type(path),
        "file_path": str(path),
        "course": "",
        "year": "",
    }

    for line in text.splitlines()[:20]:
        if line.startswith("Professor:"):
            meta["professor"] = line.split(":", 1)[1].strip()
        elif line.startswith("Source_Type:"):
            meta["source_type"] = line.split(":", 1)[1].strip()
        elif line.startswith("Year:"):
            meta["year"] = line.split(":", 1)[1].strip()

    return meta


def _infer_source_type(path: Path) -> str:
    parent = path.parent.name.lower()
    if parent == "rmp":
        return "RMP"
    if parent == "official":
        return "Official"
    if parent == "reddit":
        return "Reddit"
    return "Unknown"


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove boilerplate header lines and HTML artifacts; keep content."""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&nbsp;", " ").replace("&#39;", "'")
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    # Collapse excess whitespace within lines but preserve newlines
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 500       # target characters per chunk
MAX_CHUNK_SIZE = 700   # hard ceiling
OVERLAP = 50           # character overlap between adjacent chunks


def chunk_text(text: str, source_type: str, professor: str = "") -> list[str]:
    """
    Split text into chunks using semantic boundaries from planning.md:
      - RMP files: "--- Review N ---" delimiters
      - Official files: blank lines between entries
      - Reddit files: double newlines between topic blocks
    """
    if source_type == "RMP":
        segments = _split_on_review_delimiters(text)
        # Prepend professor name to every review chunk so it is searchable.
        # The first segment is the file header (already has the name); skip it.
        if professor:
            segments = [segments[0]] + [
                f"Professor: {professor}\n{s}" for s in segments[1:]
            ]
    elif source_type == "Official":
        segments = _split_on_blank_lines(text)
    else:  # Reddit and Unknown
        segments = _split_on_double_newlines(text)

    chunks = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if len(seg) <= MAX_CHUNK_SIZE:
            chunks.append(seg)
        else:
            # Segment exceeds max — split further at sentence boundaries
            chunks.extend(_split_long_segment(seg))

    # Apply small overlap by prepending tail of previous chunk.
    # Skip overlap for Official docs — each entry is fully self-contained.
    apply_overlap = OVERLAP > 0 and source_type != "Official"
    overlapped = []
    for i, chunk in enumerate(chunks):
        if apply_overlap and i > 0:
            tail = chunks[i - 1][-OVERLAP:]
            if not chunk.startswith(tail.strip()):
                chunk = tail.strip() + " " + chunk
        overlapped.append(chunk)

    return [c for c in overlapped if len(c) > 20]


def _split_on_review_delimiters(text: str) -> list[str]:
    """Split on '--- Review N ---' lines, keeping each review as one segment."""
    parts = re.split(r"(?m)^--- Review \d+ ---\s*$", text)
    # parts[0] is the file header — keep it as its own segment
    return [p for p in parts if p.strip()]


def _split_on_blank_lines(text: str) -> list[str]:
    """Split on blank lines (two or more newlines)."""
    return re.split(r"\n{2,}", text)


def _split_on_double_newlines(text: str) -> list[str]:
    """Same as blank-line split; alias for Reddit files."""
    return re.split(r"\n{2,}", text)


def _split_long_segment(text: str) -> list[str]:
    """
    Greedy character-window split at sentence boundaries for segments that
    exceed MAX_CHUNK_SIZE.
    """
    chunks = []
    while len(text) > MAX_CHUNK_SIZE:
        # Try to break at the last sentence-ending punctuation within the window
        window = text[:MAX_CHUNK_SIZE]
        cut = max(window.rfind(". "), window.rfind(".\n"), window.rfind("! "), window.rfind("? "))
        if cut < CHUNK_SIZE // 2:
            # No good sentence boundary — cut at the last space
            cut = window.rfind(" ")
        if cut <= 0:
            cut = MAX_CHUNK_SIZE
        else:
            cut += 1  # include the punctuation
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def build_chunks(directory: str = "documents") -> list[dict]:
    """
    Full pipeline: load → clean → chunk.

    Returns a list of dicts:
      { "text": str, "source_type": str, "professor": str,
        "file_path": str, "course": str, "year": str, "chunk_index": int }
    """
    all_chunks = []
    for raw_text, meta in load_documents(directory):
        text = clean_text(raw_text)
        chunks = chunk_text(text, meta["source_type"], professor=meta.get("professor", ""))
        for i, chunk in enumerate(chunks):
            record = dict(meta)
            record["text"] = chunk
            record["chunk_index"] = i
            all_chunks.append(record)
    return all_chunks


# ---------------------------------------------------------------------------
# Inspection helper (run directly to verify)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    chunks = build_chunks()
    print(f"\nTotal chunks: {len(chunks)}\n")

    print("=" * 60)
    print("First 5 chunks:")
    print("=" * 60)
    for c in chunks[:5]:
        print(f"\n[{c['source_type']} | {c['professor'] or c['file_path']} | chunk {c['chunk_index']}]")
        print(c["text"])
        print("-" * 40)

    print("\n" + "=" * 60)
    print("5 random chunks:")
    print("=" * 60)
    for c in random.sample(chunks, min(5, len(chunks))):
        print(f"\n[{c['source_type']} | {c['professor'] or c['file_path']} | chunk {c['chunk_index']}]")
        print(c["text"])
        print("-" * 40)
