# The Unofficial Guide — Project 1

---

## Domain

Student-generated knowledge about CS professors and courses at George Mason University, drawn from Rate My Professors, the r/gmu subreddit, and the official GMU course catalog and faculty directory.

GMU's official channels — the course catalog and the student portal — describe what courses exist and when they are offered, but tell you nothing about how a professor actually teaches, how hard their exams are, whether attendance matters, or which of two professors teaching the same course is worth taking. That knowledge is scattered across RMP reviews, Reddit threads, and word-of-mouth conversations that most students don't find until after they've already registered. This system makes all of that retrievable through plain-language questions, grounded entirely in real student opinions and official catalog data.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | RMP — Justin Wilson | Student reviews (CS222) | https://www.ratemyprofessors.com/professor/2943034 |
| 2 | RMP — Wassim Itani | Student reviews (CS455/CS450) | https://www.ratemyprofessors.com/professor/2842731 |
| 3 | RMP — Wes Masri | Student reviews (CS310/SWE619) | https://www.ratemyprofessors.com/professor/2711222 |
| 4 | RMP — Mark Snyder | Student reviews (CS463/CS367) | https://www.ratemyprofessors.com/professor/1621871 |
| 5 | RMP — David Nordstrom | Student reviews (CS100/CS222/CS262) | https://www.ratemyprofessors.com/professor/1098501 |
| 6 | RMP — Michael Neary | Student reviews (CS112) | https://www.ratemyprofessors.com/professor/2426252 |
| 7 | RMP — Wing Lam | Student reviews (SWE437/SWE637) | https://www.ratemyprofessors.com/professor/2787175 |
| 8 | RMP — John Otten | Student reviews (CS110/CS222/CS306) | https://www.ratemyprofessors.com/professor/1903402 |
| 9 | RMP — Ahmed Zaman | Student reviews (CS330) | https://www.ratemyprofessors.com/professor/2737110 |
| 10 | RMP — Ivan Avramovic | Student reviews (CS330/CS483) | https://www.ratemyprofessors.com/professor/2380599 |
| 11 | RMP — Katherine Russell | Student reviews (CS310/CS483) | https://www.ratemyprofessors.com/professor/2038141 |
| 12 | r/gmu Reddit compilation | CS professor and course advice threads (2024–2025) | documents/reddit/gmu_cs_advice_threads.txt |
| 13 | GMU CS Faculty Directory | All CS faculty with titles and research areas | documents/official/faculty_directory.txt |
| 14 | GMU Course Catalog | Full CS course catalog, 100–600 level | documents/official/course_catalog.txt |

---

## Chunking Strategy

**Chunk size:** 500 characters target, 700 character hard ceiling. Chunks that exceed 700 characters are split further at sentence boundaries rather than at an arbitrary character position.

**Overlap:** 50-character overlap for RMP and Reddit documents only. Official documents (course catalog, faculty directory) use zero overlap because each entry — one course or one faculty member — is already a fully self-contained unit. Bleeding the tail of one course entry into the next would contaminate the context without adding useful signal.

**Why these choices fit the documents:** RMP files use a `--- Review N ---` delimiter between reviews. Splitting on those delimiters keeps every review's metadata (professor name, course number, date, quality rating) attached to the review text in a single chunk, which is critical for retrieval: a chunk containing only "He is an amazing professor" with no professor name or course would fail to match queries like "What do students think of Wilson in CS222?" The 500-character target fits comfortably within the 256-token limit of the all-MiniLM-L6-v2 embedding model. Reddit files are split on double newlines between topic blocks; the course catalog and faculty directory are split on blank lines between entries.

One additional preprocessing step applied to RMP files: the professor's name is prepended to the text of every review chunk. Without this, individual review chunks contain course numbers and opinions but not the professor's name — so professor-specific queries would fail to match. This fix was discovered during retrieval testing in Milestone 4.

**Final chunk count:** 265 chunks across 14 documents.

### Sample Chunks

Five representative chunks from across the corpus, showing the labeled format produced by `ingest.py`:

**Sample Chunk 1 — RMP | Justin Wilson** (`documents/rmp/rmp_justin_wilson.txt`)
```
Professor: Justin Wilson

Course: CS222
Date: Apr 30, 2026
Year: 2026
Quality: 5.0 | Difficulty: 1.0
Grade: Not sure yet
Would Take Again: Yes
Tags: Extra credit, Gives good feedback, Caring
"He is an amazing professor who genuinely cares about his students. He will do
anything to help if you're confused. He offers chances to earn points back on
the midterm, a 'life happens' pass for one project, and a revision pass to
improve your grade if you didn't like it."
```

**Sample Chunk 2 — RMP | Wing Lam** (`documents/rmp/rmp_wing_lam.txt`)
```
Professor: Wing Lam

Course: SWE437
Date: Dec 10, 2025
Year: 2025
Quality: 4.0 | Difficulty: 4.0
Grade: B+
Would Take Again: Yes
Tags: Tough grader, Participation matters, Test heavy
"I don't write reviews, but genuinely, Prof Lam is an inspiration. His quizzes
and exams are quite difficult, and you'll be anxious about each quiz two weeks
in advance, but there's no better way he could've structured the course.
I learned a lot, I recommend."
```

**Sample Chunk 3 — RMP | Ivan Avramovic** (`documents/rmp/rmp_ivan_avramovic.txt`)
```
Professor: Ivan Avramovic

Course: CS483
Date: Dec 30, 2025
Year: 2025
Quality: 5.0 | Difficulty: 2.0
Grade: A
Would Take Again: Yes
Tags: Amazing lectures, Clear grading criteria, Gives good feedback
"He literally is one of the best in the department. Kind and great at teaching.
The absolute goat. Take him if you can!"
```

**Sample Chunk 4 — Official | Course Catalog** (`documents/official/course_catalog.txt`)
```
CS 483: Analysis of Algorithms (3 credits)
Prerequisites: CS 310 and CS 330 (C minimum) AND MATH 125 (C minimum)
Description: Analyzes computational resources; covers algorithms, data
structures, rigorous analysis techniques.
```

**Sample Chunk 5 — Reddit | r/gmu compilation** (`documents/reddit/gmu_cs_advice_threads.txt`)
```
GMU CS Student Advice — Reddit r/gmu Compilation
Source: r/gmu subreddit (reddit.com/r/gmu)
Source_Type: Reddit
Year: 2024-2025
```

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`. Runs locally with no API key or rate limits, downloads once and caches. Each chunk is approximately 100 tokens, well within the model's 256-token limit.

**Production tradeoff reflection:** For a real GMU deployment serving many students, four tradeoffs would matter most. First, context length: MiniLM's 256-token ceiling would truncate long syllabi, Reddit megathreads, or full course syllabi if those were added to the corpus. Switching to `nomic-embed-text` (8192-token context) or OpenAI's `text-embedding-3-large` (8191 tokens) would handle those documents without aggressive pre-splitting. Second, domain specificity: MiniLM was trained on general English text, not course review language. A model fine-tuned on academic evaluation text would better distinguish between semantically similar but informationally different phrases like "CS483 is very hard" and "CS483 with Avramovic is manageable." Third, latency: local inference has no network round-trip and is fast enough for this corpus size, but at production scale with concurrent users, hosted embeddings (OpenAI, Cohere) offer horizontal scaling that a local model cannot. Fourth, multilingual coverage: MiniLM is English-only. A future version serving international students or non-English reviews would require a multilingual model such as `paraphrase-multilingual-MiniLM-L12-v2`.

---

## Retrieval Examples

Three queries tested against the live ChromaDB vector store, showing the top returned chunks and distance scores. Lower distance = higher similarity.

---

### Query 1: "What do students say about exam difficulty in Justin Wilson's CS222?"

| Rank | Distance | Source type | Professor / File |
|------|----------|-------------|-----------------|
| 1 | 0.7485 | RMP | Justin Wilson |
| 2 | 0.7511 | RMP | Wing Lam |
| 3 | 0.7556 | RMP | Justin Wilson |
| 4 | 0.7966 | RMP | Justin Wilson |
| 5 | 0.8284 | RMP | Justin Wilson |

**Top returned chunk (rank 1):**
```
Professor: Justin Wilson

Course: CS222  |  Date: Apr 21, 2026  |  Quality: 5.0  |  Difficulty: 2.0
Tags: Extra credit, Gives good feedback, Caring
"He is one of the best professors at GMU, genuinely so caring about his
students and understanding that not everyone is good at coding. It's honestly
hard to do bad in his class because he gives LOTS of extra credit."
```

**Why these chunks are relevant:** The query names both the professor ("Justin Wilson") and the topic ("exam difficulty"). Because the professor's name was injected into every RMP review chunk during ingestion, four of the five top results are Wilson's own reviews. The chunks contain explicit difficulty ratings (1.0–2.0), mentions of cheat sheets, extra credit, and what to expect on exams — all directly answering the question. Rank 2 (Wing Lam) slipped in because it also contains the phrase "exam difficulty" in a high-difficulty context, showing that generic exam-difficulty language matches across professors when no course filter is applied.

---

### Query 2: "Is Wing Lam's SWE437 worth taking even though the quizzes are hard?"

| Rank | Distance | Source type | Professor / File |
|------|----------|-------------|-----------------|
| 1 | 0.5487 | Reddit | gmu_cs_advice_threads.txt |
| 2 | 0.7481 | RMP | Wing Lam |
| 3 | 0.7497 | RMP | Wing Lam |
| 4 | 0.8033 | RMP | Wing Lam |
| 5 | 0.8048 | RMP | Wing Lam |

**Top returned chunk (rank 1):**
```
- Wing Lam (SWE437) is considered one of the hardest. Quizzes are intentionally
  difficult — harder than CS211 and CS310 combined according to students. But those
  who stick with it say they actually learn the material and use it in jobs.
```

**Why these chunks are relevant:** The Reddit compilation chunk (rank 1, distance 0.5487 — the strongest match in this test set) directly mentions Wing Lam, SWE437, quiz difficulty, and the career-value trade-off in one paragraph. It is the best possible chunk for this query. Ranks 2–4 from RMP reinforce with specific student quotes ("The quizzes are insane", "I use a lot of the principles taught at my current job"), providing individual opinions that complement the summary. Distance scores drop sharply after rank 1 because RMP chunks are shorter and more fragmented than the Reddit compilation paragraph.

---

### Query 3: "What are the prerequisites for CS483 Analysis of Algorithms?"

| Rank | Distance | Source type | Professor / File |
|------|----------|-------------|-----------------|
| 1 | 0.4094 | Official | course_catalog.txt |
| 2 | 0.6552 | Official | course_catalog.txt |
| 3 | 0.6877 | Official | course_catalog.txt |
| 4 | 0.6925 | Official | course_catalog.txt |
| 5 | 0.6989 | Official | course_catalog.txt |

**Top returned chunk (rank 1):**
```
CS 483: Analysis of Algorithms (3 credits)
Prerequisites: CS 310 and CS 330 (C minimum) AND MATH 125 (C minimum)
Description: Analyzes computational resources; covers algorithms, data
structures, rigorous analysis techniques.
```

This is the strongest retrieval in the evaluation set (distance 0.4094). The query is a factual lookup and the answer exists verbatim in one catalog chunk. All five results come from the official course catalog because the query language ("prerequisites for CS483") is semantically closest to the structured catalog format. No RMP or Reddit chunks appear in the top 5.

---

## Grounded Generation

**System prompt grounding instruction:**

The system prompt given to `llama-3.3-70b-versatile` contains four explicit, numbered rules:

1. Answer solely from the provided documents. Do not use general training knowledge.
2. Every factual claim must cite its source inline using the format `(Source: <professor or document name>)`.
3. If the documents do not contain enough information to answer, respond with exactly: *"I don't have enough information in my documents to answer that question."*
4. Do not invent, extrapolate, or assume anything not stated in the documents.

The temperature is set to 0.2, which reduces creative generation and keeps the model closer to the literal text of the retrieved context. The context block passed to the model numbers each retrieved chunk and labels it with its source type and professor name, making attribution unambiguous.

**How source attribution is surfaced in the response:**

Attribution happens at two levels. First, the LLM is instructed to include inline citations in every claim. Second, `generate.py` programmatically builds a deduplicated source list from the metadata of the retrieved chunks — professor name and source type — and returns it alongside the answer regardless of whether the LLM mentioned them. This means the sources panel in the UI always reflects what was actually retrieved, not just what the model chose to mention.

### Example Responses with Source Attribution

**Example 1 — Query: "What do students say about exam difficulty in Justin Wilson's CS222?"**

> Students say that the exams in Justin Wilson's CS222 are not hard (Source: Document 4 | RMP | Justin Wilson). They also mention that Professor Wilson tells them what type of questions to expect for the exams (Source: Document 1 | RMP | Justin Wilson) and provides cheat sheets for every test (Source: Document 1 | RMP | Justin Wilson), as well as lecture recordings (Source: Document 3 | RMP | Justin Wilson). Additionally, they can bring 2 pages for the midterm and 4 for the final (Source: Document 4 | RMP | Justin Wilson). The difficulty of the course is rated as 2.0 (Source: Document 1, 3, 4 | RMP | Justin Wilson) and 1.0 (Source: Document 5 | RMP | Justin Wilson).

**Retrieved from:** Justin Wilson (RMP)

---

**Example 2 — Query: "Is Wing Lam's SWE437 worth taking even though the quizzes are hard?"**

> Wing Lam's SWE437 is considered worth taking despite the challenging quizzes by some students, as they mention that they "actually learn the material and use it in jobs" (Source: Document 1 | Reddit). One student on RMP also states that "Professor Lam is great, the class was interesting overall and I use a lot of the principles taught at my current job" (Source: Document 2 | RMP | Wing Lam). However, another student had a negative experience, stating "Take this class with another professor if you can" (Source: Document 4 | RMP | Wing Lam). Overall, opinions are mixed, but some students find the course valuable despite its difficulty (Source: Document 3 | RMP | Wing Lam).

**Retrieved from:** gmu_cs_advice_threads.txt (Reddit), Wing Lam (RMP)

---

### Out-of-Scope Query — Refusal Example

**Query: "Who is the president of George Mason University?"**

> I don't have enough information in my documents to answer that question.

This query was answered with the exact decline phrase specified in rule 3 of the system prompt, even though the LLM clearly has this information in its training data. The grounding instruction successfully prevented the model from using general knowledge.

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What do students say about exam difficulty in Justin Wilson's CS222? | Exams are easy (~2/5 difficulty). Wilson gives cheat sheets, extra credit, and tells students what to expect. Hard to do badly. | Exams are not hard. Wilson tells students what type of questions to expect, provides cheat sheets, difficulty ratings are 1.0–2.0. Sources cited inline. | Relevant | Accurate |
| 2 | Is Wing Lam's SWE437 worth taking even though the quizzes are hard? | Mixed — quizzes are very hard and the class is harder than CS211 and CS310 combined, but students who finish say they genuinely learn and apply it at work. | Mixed verdict: some call it worthwhile and cite applying material on the job; others warn it's harder than CS211 and CS310 combined. Quizzes described as "insane." Both perspectives cited. | Relevant | Accurate |
| 3 | What are the prerequisites for CS483 Analysis of Algorithms? | CS310, CS330, and MATH125, all with C or better. | CS 310 and CS 330 (C minimum) AND MATH 125 (C minimum). Single catalog entry cited. Distance 0.41 — strongest retrieval result. | Relevant | Accurate |
| 4 | Which professor is better for CS330 — Ahmed Zaman or Ivan Avramovic? | Both highly rated. Zaman's exams mirror practice material. Avramovic posts all lectures on Canvas and grades entirely on exams/quizzes with no projects. | Correctly described Zaman's CS330 strengths. Stated "no information" about Avramovic teaching CS330 — his CS483 reviews were retrieved instead of his CS330 ones. | Partially relevant | Partially accurate |
| 5 | What do students recommend for surviving CS310 at GMU? | Study consistently, attend office hours, form study groups. Material builds and falling behind early is hard to recover from. | Study the course religiously, check RMP before registering, prepare for a filter course. Correct in spirit but missed the office hours and study group advice in the Katherine Russell RMP file. | Partially relevant | Partially accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:**
"Which professor is better for CS330 — Ahmed Zaman or Ivan Avramovic?"

**What the system returned:**
The system correctly described Zaman's strengths (exams mirror practice material, great lecturer). However, it stated that there was "no information provided about Ivan Avramovic teaching CS330," even though the documents clearly contain Avramovic's CS330 reviews. The system then declined to make a comparison.

**Root cause (tied to a specific pipeline stage):**
The failure is in the retrieval stage, caused by how professor-name injection interacts with course-level queries. When the RMP files were chunked, every review chunk was prefixed with the professor's name (e.g., `Professor: Ivan Avramovic`). Avramovic teaches both CS330 and CS483. Because he has more CS483 reviews than CS330 reviews in the corpus, and because the query mentions "CS330" alongside both professor names, the embedding similarity search retrieved Avramovic's CS483 chunks rather than his CS330 chunks — they were more numerous and had slightly higher overall semantic weight in the collection. The retrieval returned chunks about "Analysis of Algorithms" for Avramovic instead of his data structures reviews, so the LLM correctly reported that its context contained no CS330 information for Avramovic, even though CS330 reviews exist.

**What you would change to fix it:**
Two targeted fixes would address this. First, add course number as a filterable metadata field on each review chunk and, for queries that name a specific course, apply a `where={"course": "CS330"}` filter to the ChromaDB retrieval call. This is already architected into `embed.py`'s `retrieve()` function via the `filters` parameter — it just needs a query-time detection step. Second, for comparison queries naming two professors explicitly, run two separate filtered retrievals (one per professor) and concatenate the results, guaranteeing that each professor is represented in the context regardless of overall similarity scores.

---

## Query Interface

The system is accessed through a Gradio web UI. Launch it with:

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

**Input fields:**
- **Your question** — a plain-language text box. Accepts any question about GMU CS professors, courses, prerequisites, or teaching style. Pressing Enter or clicking "Ask" submits the query.

**Output fields:**
- **Answer** — the LLM's grounded response with inline source citations in the format `(Source: Document N | source_type | professor/file)`.
- **Retrieved from** — a deduplicated bullet list of the source documents whose chunks were used to generate the answer, built programmatically from chunk metadata (not inferred by the LLM).

Five example question buttons are provided on the right side of the UI so a new user can immediately test the system without typing.

**Sample interaction transcript:**

---

*Input:*
> What are the prerequisites for CS483 Analysis of Algorithms?

*Answer:*
> The prerequisites for CS 483: Analysis of Algorithms are CS 310 and CS 330 (C minimum) AND MATH 125 (C minimum) (Source: Document 1).

*Retrieved from:*
> • course_catalog.txt (Official)

---

*Input:*
> What do students say about exam difficulty in Justin Wilson's CS222?

*Answer:*
> Students say that the exams in Justin Wilson's CS222 are not hard (Source: Document 4 | RMP | Justin Wilson). They also mention that Professor Wilson tells them what type of questions to expect for the exams (Source: Document 1 | RMP | Justin Wilson) and provides cheat sheets for every test, as well as lecture recordings (Source: Document 3 | RMP | Justin Wilson). They can bring 2 pages for the midterm and 4 for the final (Source: Document 4 | RMP | Justin Wilson). Difficulty is rated 1.0–2.0 across reviews (Source: Documents 1, 3, 4, 5 | RMP | Justin Wilson).

*Retrieved from:*
> • Justin Wilson (RMP)

---

*Input (out-of-scope):*
> Who is the president of George Mason University?

*Answer:*
> I don't have enough information in my documents to answer that question.

*Retrieved from:*
> • Mark Snyder (RMP) • Justin Wilson (RMP) • David Nordstrom (RMP) • Michael Neary (RMP) • John Otten (RMP)

---

## Spec Reflection

**One way the spec helped you during implementation:**

The Chunking Strategy section of planning.md was the most directly useful part of the spec. Writing it before touching any code forced the decision of *why* to split on `--- Review N ---` delimiters rather than on character counts. That reasoning — that a chunk containing only "He is an amazing professor" with no attached professor name would fail retrieval for professor-specific queries — directly shaped a concrete implementation choice. When retrieval testing in Milestone 4 revealed that individual review chunks still didn't contain the professor's name (it was only in the file header chunk), the spec's reasoning made the fix obvious: inject the professor name into every chunk. Without having articulated the attribution requirement upfront, that failure would have been harder to diagnose.

**One way your implementation diverged from the spec, and why:**

The spec's Chunking Strategy specified a 50-character overlap across all document types. During implementation, it became clear that applying overlap to Official documents (the course catalog and faculty directory) produced messy chunk starts — the tail of one catalog entry (often a partial URL or description sentence) would bleed into the opening of the next entry, making chunks like `"...internet, and societal impact. CS 105: Computer Ethics and Society"` which reads correctly but embeds poorly because the leading fragment belongs semantically to the previous entry. Official documents were changed to zero overlap because each entry is already self-contained. This divergence is a tightening of the spec's intent — the spec's reasoning for small overlap (that review entries are self-contained) applies even more strongly to structured catalog entries — not a contradiction of it.

---

## AI Usage

**Instance 1 — Ingestion and chunking pipeline (Milestone 3)**

- *What I gave the AI:* The Chunking Strategy and Documents sections from planning.md, a sample RMP file showing the `--- Review N ---` delimiter format, and the folder structure (`documents/rmp/`, `documents/reddit/`, `documents/official/`). I asked Claude to implement `ingest.py` with `load_documents()` and `chunk_text()` matching the spec.
- *What it produced:* A working `ingest.py` that split on the correct boundaries, cleaned HTML artifacts, and returned `(text, metadata)` tuples. The initial version applied 50-character overlap uniformly across all document types.
- *What I changed or overrode:* Two changes were made after running the pipeline and inspecting output. First, overlap was disabled for Official documents after seeing that catalog entry chunks had garbled prefixes from the previous entry's tail. Second, professor-name injection into each RMP review chunk was added after Milestone 4 retrieval testing showed that professor-specific queries returned the wrong professor's reviews — individual review chunks had no professor name in their text, only in metadata.

**Instance 2 — Grounded generation and system prompt (Milestone 5)**

- *What I gave the AI:* The Grounded Generation requirements from the milestone spec, the `retrieve()` function signature and a sample output dict from Milestone 4, and the requirement that every response must cite which document the answer came from. I asked Claude to implement `generate.py` and `app.py`.
- *What it produced:* `generate.py` with a system prompt, a context-building function, and a Groq API call. `app.py` with a Gradio Blocks UI including a question box, answer panel, sources panel, and example question buttons.
- *What I changed or overrode:* The initial system prompt used softer language ("try to answer from the documents"). This was tightened to explicit, numbered rules with no hedging — rule 3 specifies the exact decline phrase the model must use, and rule 4 explicitly prohibits extrapolation. The temperature was also set to 0.2 rather than the default, after observing that a higher temperature caused the model to embellish answers with plausible-but-unverifiable details about GMU courses. The programmatic source list (built from chunk metadata, appended regardless of LLM output) was also added after the initial version relied solely on the LLM to include citations, which it occasionally omitted for lower-ranked chunks.

---

## Stretch: Metadata Filtering

The UI exposes two filter dropdowns inside a collapsible "Filters & Search mode" panel:

- **Source type** — restricts retrieval to one of: RMP, Official, Reddit, or Any (default).
- **Professor** — restricts retrieval to one of the 11 professors in the corpus, or Any (default).

These map directly to ChromaDB `where` clause filters passed to `retrieve()`. When both are set, they are combined with `$and`. When only one is set, a single-field filter is applied.

**Demonstrated effect — same query, two filter settings:**

Query: `"What are the prerequisites for CS483 Analysis of Algorithms?"`

| Filter | Top result | Source type |
|--------|-----------|-------------|
| Source = Any | CS 465: Computer Systems Architecture... (dist=0.5349) | Official |
| Source = Official | CS 465: Computer Systems Architecture... (dist=0.5349) | Official |
| Professor = Ivan Avramovic | Professor: Ivan Avramovic / CS483 / Quality 5.0... | RMP |

Filtering by `professor = Ivan Avramovic` for a CS483 query bypasses the semantic ranking and forces all five results to come from Avramovic's reviews — directly addressing the Q4 failure case where his CS330 reviews were crowded out by his more numerous CS483 reviews.

---

## Stretch: Hybrid Search

**Approach:** Combines BM25 keyword scoring with ChromaDB semantic scoring using **Reciprocal Rank Fusion (RRF)**. Each chunk gets a combined score:

```
RRF(chunk) = 1/(60 + rank_semantic) + 1/(60 + rank_bm25)
```

The top-k chunks by combined RRF score are returned. BM25 uses the `rank_bm25` library with a simple tokenizer (lowercase + strip punctuation). The BM25 index is built lazily from the full ChromaDB corpus on first call and cached for the session. A toggle in the UI ("Use hybrid search") switches between semantic-only and hybrid mode.

**Comparison on 3 queries:**

**Query 1: "What do students say about exam difficulty in Justin Wilson's CS222?"**

| Rank | Semantic only | Hybrid (RRF) |
|------|--------------|-------------|
| 1 | Justin Wilson RMP (dist=0.75) | Justin Wilson RMP (rrf=0.0164, bm25=high) |
| 2 | Wing Lam RMP (dist=0.75) | Justin Wilson RMP chunk 2 |
| 3 | Justin Wilson RMP (dist=0.76) | Justin Wilson RMP chunk 3 |

*Winner: Hybrid.* BM25 boosted Wilson's reviews because they contain the exact words "exam," "difficulty," "CS222," and "Justin Wilson." The off-topic Wing Lam result that appeared at rank 2 in semantic-only was pushed out.

**Query 2: "Is Wing Lam's SWE437 worth taking even though the quizzes are hard?"**

| Rank | Semantic only | Hybrid (RRF) |
|------|--------------|-------------|
| 1 | Reddit (dist=0.55) | Reddit (rrf=0.0164, bm25=high) |
| 2 | Wing Lam RMP (dist=0.75) | Wing Lam RMP |
| 3 | Wing Lam RMP (dist=0.75) | Wing Lam RMP |

*Winner: Tie.* Both methods returned the Reddit compilation chunk first and Wing Lam reviews in the remaining slots. BM25 reinforced the semantic ranking without changing it — the query was specific enough that both methods agreed.

**Query 3: "What are the prerequisites for CS483 Analysis of Algorithms?"**

| Rank | Semantic only | Hybrid (RRF) |
|------|--------------|-------------|
| 1 | Official catalog (dist=0.41) | Official catalog |
| 2–5 | Other catalog entries | Reddit CS483 thread (bm25=8.57, rrf tied rank 2) |

*Winner: Hybrid for completeness.* Semantic-only returned only catalog entries. Hybrid surfaced a Reddit thread specifically about CS483 at rank 2 (BM25 score 8.57 — the exact phrase "CS483 Analysis of Algorithms" is in its title), giving the LLM both the authoritative prerequisite answer AND student commentary about what to expect in the course.

**Conclusion:** Hybrid search most visibly improves queries that contain exact proper nouns (professor names, course numbers) where BM25 keyword matching adds signal that pure semantic similarity misses. For open-ended opinion queries, both methods perform similarly.

---

## Stretch: Chunking Strategy Comparison

Two strategies were compared on all 5 evaluation queries using BM25 retrieval (consistent across both, no vector store rebuild needed):

- **Strategy A (current):** Semantic boundary splits — RMP files on `--- Review N ---` delimiters, Official on blank lines, Reddit on double newlines. 265 chunks.
- **Strategy B:** Fixed 300-character character splits, no overlap, applied uniformly to all documents. 240 chunks.

**Results by query:**

| Query | Strategy A top result | Strategy B top result | Winner |
|-------|----------------------|----------------------|--------|
| Wilson CS222 exam difficulty | Complete Wilson review with name, course, date, and review text intact | File header only (Professor name, department, rating) — no review text | **A** |
| Wing Lam SWE437 quizzes | Reddit paragraph directly answering the SWE437/quiz trade-off | Reddit paragraph but preceded by unrelated Justin Wilson summary text in the same chunk | **A** |
| CS483 prerequisites | Reddit thread header for CS483 (topic match but no catalog entry) | Fragment from inside a Reddit thread, mid-sentence | **A** |
| CS330 Zaman vs Avramovic | Ahmed Zaman CS330 review with full metadata | Reddit paragraph covering both CS330 professors together | **B** (marginally) |
| Surviving CS310 | Reddit thread header for CS310 tips | Fragment containing unrelated Neary intro before the CS310 thread | **A** |

**Strategy A won on 4 of 5 queries.** The key failure mode of Strategy B is that fixed character cuts routinely split across semantic units: a 300-character window starting mid-review produces a chunk that begins with the end of one student's opinion and ends partway through the next. These fragments have weak BM25 signal because they contain partial phrases and no complete metadata. Strategy A's delimiter-based splits keep attribution metadata (professor, course, date) and review text in the same chunk, which is exactly what both BM25 and semantic retrieval reward.

The one query where Strategy B performed marginally better (Q4, Zaman vs Avramovic) is because one of its 300-char windows happened to capture the Reddit paragraph that names both professors in the same sentence — a coincidence of where the cut fell, not a structural advantage.

---

## Stretch: Conversational Memory

The system maintains a rolling chat history in Gradio's `gr.Chatbot` component and passes the last 2 turns of conversation to `generate.py` as a memory block prepended to the user message.

**Implementation:** After each query, the question and answer are appended to a `history_state` (Gradio `gr.State`). On the next query, the last `MAX_HISTORY_TURNS = 2` pairs are formatted as:

```
Conversation history (for context only — still answer from documents):
User previously asked: <Q1>
Assistant answered: <A1>
```

This block is injected into the user message before the main question, allowing the LLM to resolve pronouns and topic references across turns while remaining grounded in the retrieved documents.

**Demonstrated multi-turn exchange:**

*Turn 1:*
> Q: What are the prerequisites for CS483?
> A: The prerequisites for CS 483 are CS 310 and CS 330 (C minimum) AND MATH 125 (C minimum). (Source: Document 1 | Official | course_catalog.txt)

*Turn 2 (with memory):*
> Q: Who teaches that course and what do students say about them?
> A: The course CS483 is taught by Katherine Russell (Source: Document 3 | Reddit). Students say she is "very kind and understanding" but also "polarizing" — some warn that her tests are "significantly harder than in earlier CS courses" while others praise her analogies and energy...

*Turn 2 WITHOUT memory (same retrieved chunks):*
> A: The course CS310 is taught by Professor Katherine Russell. Students say that she is a "Tough grader, Caring"...

With memory, the LLM correctly resolved "that course" to CS483 and answered about CS483 instructors. Without memory, it defaulted to CS310 (the course most associated with Russell in the retrieved chunks), demonstrating that the memory context is actively changing the generation output — not just a coincidence of topic overlap.
