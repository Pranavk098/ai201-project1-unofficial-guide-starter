# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

Student-generated knowledge about CS professors and courses at George Mason University,
collected from Rate My Professors, r/gmu on Reddit, and official GMU course catalog
and faculty profiles.

GMU's official website and internal student portal only describe course schedules and
catalog descriptions, but fail to provide niche information about a professor's classroom
teaching style, exam difficulty, attendance policies, and whether lectures are worth
attending. This system makes that scattered, informal knowledge answerable through
plain-language questions.

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

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**

**Overlap:**

**Reasoning:**

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**

**Top-k:**

**Production tradeoff reflection:**

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What do students say about exam difficulty in Justin Wilson's CS222? | Exams are easy, difficulty rated ~2/5. Wilson gives lots of extra credit and genuinely cares about students passing. Hard to do badly in his class. |
| 2 | Is Wing Lam's SWE437 worth taking even though the quizzes are hard? | Mixed — quizzes are intentionally very hard and the class is considered harder than CS211 and CS310 combined, but students who finish say they genuinely learn the material and apply it on the job. |
| 3 | What are the prerequisites for CS483 Analysis of Algorithms? | CS310, CS330, and MATH125 all with a C or better. |
| 4 | Which professor is better for CS330 — Ahmed Zaman or Ivan Avramovic? | Both are highly rated. Zaman's exams closely mirror practice material. Avramovic posts all lectures on Canvas and grades entirely on exams and quizzes with no projects. |
| 5 | What do students recommend for surviving CS310 at GMU? | Study consistently, attend office hours (especially with Russell), form study groups. The material builds on itself and falling behind early is hard to recover from. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1.

2.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
