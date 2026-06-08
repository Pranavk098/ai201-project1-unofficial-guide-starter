# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

Student-generated knowledge about CS professors and courses at George Mason University, collected from Rate My Professors, r/gmu on Reddit, and official GMU course catalog and faculty profiles.

GMU's official website and internal student portal only describe course schedules and catalog descriptions, but fail to provide niche information about a professor's classroom teaching style, exam difficulty, attendance policies, and whether lectures are worth attending. This system makes that scattered, informal knowledge answerable through plain-language questions.

---

## Documents

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | RMP — Justin Wilson | CS222 student reviews | https://www.ratemyprofessors.com/professor/2943034 |
| 2 | RMP — Wassim Itani | CS455/CS450 student reviews | https://www.ratemyprofessors.com/professor/2842731 |
| 3 | RMP — Wes Masri | CS310/SWE619 student reviews | https://www.ratemyprofessors.com/professor/2711222 |
| 4 | RMP — Mark Snyder | CS463/CS367 student reviews | https://www.ratemyprofessors.com/professor/1621871 |
| 5 | RMP — David Nordstrom | CS100/CS222/CS262 student reviews | https://www.ratemyprofessors.com/professor/1098501 |
| 6 | RMP — Michael Neary | CS112 student reviews | https://www.ratemyprofessors.com/professor/2426252 |
| 7 | RMP — Wing Lam | SWE437/SWE637 student reviews | https://www.ratemyprofessors.com/professor/2787175 |
| 8 | RMP — John Otten | CS110/CS222/CS306 student reviews | https://www.ratemyprofessors.com/professor/1903402 |
| 9 | RMP — Ahmed Zaman | CS330 student reviews | https://www.ratemyprofessors.com/professor/2737110 |
| 10 | RMP — Ivan Avramovic | CS330/CS483 student reviews | https://www.ratemyprofessors.com/professor/2380599 |
| 11 | RMP — Katherine Russell | CS310/CS483 student reviews | https://www.ratemyprofessors.com/professor/2038141 |
| 12 | r/gmu Reddit compilation | CS professor and course advice threads | https://www.reddit.com/r/gmu |
| 13 | GMU CS Faculty Directory | All CS faculty with titles and research areas | https://cs.gmu.edu/people/faculty |
| 14 | GMU Course Catalog | Full CS course catalog 100–600 level | https://catalog.gmu.edu/courses/cs/ |

---

## Chunking Strategy

**Chunk size:** 500 characters (max 700), split on semantic boundaries: review delimiters (--- Review N ---) for RMP files, blank lines for course catalog and faculty entries, and double newlines for Reddit threads.

**Overlap:** 50 characters, keepping it small because each review and course entry is already self-contained with its own metadata. Large overlap would be wasteful and could bleed one professor's reviews into another's context.

**Reasoning:** Fixed-size character splits would break review metadata (professor name, course number, rating) away from the review text itself. A chunk containing only "He is an amazing professor" with no attached course or professor context would fail retrieval for queries like "What do students think of Wilson in CS222?" Boundary-based chunking keeps attribution intact inside every chunk, which is essential for both accurate retrieval and source citation in the generated response.

---

## Retrieval Approach

**Embedding model:** all-MiniLM-L6-v2 via sentence-transformers.
Chosen because it runs locally with no API key, handles short casual English text well, and embeds our full corpus in seconds. Each chunk is ~100 tokens, well within the 256-token limit.

**Top-k:** 5
Enough to represent the range of student opinions on a professor without introducing off-topic chunks. At k=5, specific professor queries return most of their reviews; cross-professor comparison queries return a balanced mix from each.

**Production tradeoff reflection:** For a real GMU deployment, the main
tradeoffs would be: (1) context length will be MiniLM's 256-token limit would truncate long syllabi or full Reddit threads, requiring a switch to nomic-embed-text or text-embedding-3-large; (2) domain specificity a model fine-tuned on course review text would more accurately distinguish between "CS483 is hard" and "CS483 with  Avramovic is manageable"; (3)latency — local inference wins for live queries but OpenAI embeddings offer higher quality at ~300ms per query; (4) multilingual support MiniLM is English-only, which limits future coverage of non-English reviews.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What do students say about exam difficulty in Justin Wilson's CS222? | Exams are easy, difficulty rated ~2/5. Wilson gives lots of extra credit and genuinely cares about students passing. Hard to do badly in his class. |
| 2 | Is Wing Lam's SWE437 worth taking even though the quizzes are hard? | Mixed — quizzes are intentionally very hard and the class is considered harder than CS211 and CS310 combined, but students who finish say they genuinely learn the material and apply it on the job. |
| 3 | What are the prerequisites for CS483 Analysis of Algorithms? | CS310, CS330, and MATH125 all with a C or better. |
| 4 | Which professor is better for CS330 — Ahmed Zaman or Ivan Avramovic? | Both are highly rated. Zaman's exams closely mirror practice material. Avramovic posts all lectures on Canvas and grades entirely on exams and quizzes with no projects. |
| 5 | What do students recommend for surviving CS310 at GMU? | Study consistently, attend office hours (especially with Russell), form study groups. The material builds on itself and falling behind early is hard to recover from. |

---

## Anticipated Challenges

1.Cross-professor contamination on course-level queries. When a user asks "Is CS310 hard?", the retrieval will return chunks from multiple professors who teach CS310 (Russell, Masri, Nordstrom). Nordstrom's reviews are from 2015-2016 and reflect a very different experience than Russell's 2025 reviews. The LLM may blend these into one answer without distinguishing who the student is actually asking about, producing a misleading composite.
Mitigation: enforce source attribution so every claim is tied to a named professor and year, and consider metadata filtering by year as a stretch feature.

2.Factual queries retrieving opinion chunks instead of catalog entries. The question "What are the prerequisites for CS483?" has a precise answer in the course catalog (CS310, CS330, MATH125, C or better). However, semantic search may rank review chunks like "This class is dense make sure you're solid on algorithms" higher than the catalog entry, because the review text is semantically closer to how users phrase questions. The LLM would then answer from opinion text rather than ground truth.
Mitigation: at query time, detect factual prerequisite/credit queries and boost catalog chunks using metadata filtering by Source_Type = Official.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        QUERY INTERFACE                          │
│                    Gradio (gradio>=6.9.0)                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │ user query (plain-language string)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL                               │
│   ChromaDB semantic similarity search  ·  top-k = 5            │
│   Optional: metadata filter (source_type, course, year)        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ top-5 chunks + metadata
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        GENERATION                               │
│   Groq API  ·  llama-3.3-70b-versatile                         │
│   System prompt: answer only from retrieved context,           │
│   cite source (professor name + file) in every claim           │
└─────────────────────────┬───────────────────────────────────────┘
                          │ grounded, cited answer
                          ▼
                     User sees response

── OFFLINE (run once to build the vector store) ──────────────────

┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐
│  INGESTION   │ →  │   CHUNKING   │ →  │  EMBEDDING + STORAGE   │
│  Python /    │    │  Custom      │    │  sentence-transformers  │
│  pathlib     │    │  boundary    │    │  all-MiniLM-L6-v2      │
│  Load .txt   │    │  split on    │    │  → ChromaDB            │
│  files from  │    │  "Review N"  │    │  (local persistent     │
│  documents/  │    │  delimiter   │    │   vector store)        │
└──────────────┘    └──────────────┘    └────────────────────────┘
```

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**
- Tool: Claude
- Input: I will give Claude the Chunking Strategy section from planning.md along with a sample RMP file (rmp_justin_wilson.txt) so it can see the exact "--- Review N ---" delimiter format, and a sample course catalog entry showing the blank-line boundary structure. I will also show it the folder structure (documents/rmp/, documents/reddit/, documents/official/) so it understands the three source types.
- Expected output: A file called ingest.py containing two functions: load_documents(directory) — walks all subfolders, reads every .txt file, and returns a list of (text, metadata) tuples where metadata captures professor name, source type, and file path. chunk_text(text, source_type) — splits RMP files on "--- Review N ---" boundaries, Official files on blank lines between entries, and Reddit files on double newlines between topic blocks.
- Verification: I will run ingest.py and print the total chunk count alongside the first 5 chunks. I will manually check that each RMP chunk contains the professor name, course number, and review text together in one unit — not split apart. Target chunk count: 120–160 total across all 14 documents.

**Milestone 4 — Embedding and retrieval:**
- Tool: Claude
- Input: I will give Claude the Retrieval Approach section from this planning.md and paste in the chunk output structure from Milestone 3 (a sample chunk withits metadata dict) so it knows exactly what format it is working with.
- Expected output: A file called embed.py with two functions: embed_and_store(chunks, metadata) — uses sentence-transformers (all-MiniLM-L6-v2) to embed every chunk and loads them into a local ChromaDB collection, storing source_type, professor, course, and year as filterable metadata fields on each vector. retrieve(query, k=5, filters=None) — takes a plain-language query, embeds it with the same model, and returns the top-5 most semantically similar chunks along with their metadata.
- Verification: I will run a test query — "Is CS310 hard?" — and manually read the 5 returned chunks. All 5 should contain CS310 content. I will also verify that professor name, source URL, and year appear in the returned metadata so attribution is possible in the next stage.

**Milestone 5 — Generation and interface:**
- Tool: Claude
- Input: I will give Claude the Grounded Generation requirements from the project spec, the retrieve() function signature and a sample output from Milestone 4, and the requirement that every response must cite which document the answer came from.
- Expected output: A file called generate.py that takes a query and the retrieved chunks, formats them into a context block, and calls the Groq API (llama-3.3-70b-versatile) with a system prompt that instructs the model to answer only from the provided context and to name the source professor or document for every claim it makes. Plus app.py — a Gradio interface with a text input box and a response panel that a user can interact with without any explanation needed.
- Verification: I will run all 5 evaluation questions through the interface and check two things: (1) every answer cites a real source from my documents, and (2) when I ask something outside the corpus like "Who is the GMU president?", the system says it cannot answer from the available documents instead of making something up.
