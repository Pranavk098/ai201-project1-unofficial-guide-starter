"""
Milestone 5 + Stretch Features: Gradio web interface for the GMU CS Unofficial Guide.

Stretch features included:
  - Metadata filtering: filter results by source type or professor name.
  - Hybrid search: toggle between semantic-only and BM25+semantic (RRF) retrieval.
  - Conversational memory: chat history passed to the LLM as context.

Run with:  python app.py
Then open: http://localhost:7860
"""

import gradio as gr
from embed import retrieve, retrieve_hybrid
from generate import generate

EXAMPLE_QUESTIONS = [
    "What do students say about exam difficulty in Justin Wilson's CS222?",
    "Is Wing Lam's SWE437 worth taking even though the quizzes are hard?",
    "What are the prerequisites for CS483 Analysis of Algorithms?",
    "Which professor is better for CS330 — Ahmed Zaman or Ivan Avramovic?",
    "What do students recommend for surviving CS310 at GMU?",
]

PROFESSORS = [
    "Any",
    "Justin Wilson", "Wassim Itani", "Wes Masri", "Mark Snyder",
    "David Nordstrom", "Michael Neary", "Wing Lam", "John Otten",
    "Ahmed Zaman", "Ivan Avramovic", "Katherine Russell",
]

SOURCE_TYPES = ["Any", "RMP", "Official", "Reddit"]

MAX_HISTORY_TURNS = 2  # number of prior Q&A pairs fed to the LLM as context


# ---------------------------------------------------------------------------
# Core handler
# ---------------------------------------------------------------------------

def handle_query(
    question: str,
    source_filter: str,
    professor_filter: str,
    use_hybrid: bool,
    history: list[list[str]],
) -> tuple[list[list[str]], str, str]:
    """
    Run retrieval + generation and update the chat history.

    Returns: (updated_history, retrieval_details, sources_text)
    """
    question = question.strip()
    if not question:
        return history, "", ""

    # Build ChromaDB metadata filter from UI selections
    filters: dict | None = None
    where_clauses = []
    if source_filter != "Any":
        where_clauses.append({"source_type": source_filter})
    if professor_filter != "Any":
        where_clauses.append({"professor": professor_filter})

    if len(where_clauses) == 1:
        filters = where_clauses[0]
    elif len(where_clauses) > 1:
        filters = {"$and": where_clauses}

    # Retrieve chunks
    if use_hybrid:
        chunks = retrieve_hybrid(question, k=5, filters=filters)
    else:
        chunks = retrieve(question, k=5, filters=filters)

    # Build conversational context from recent history
    memory_context = ""
    if history:
        recent = history[-MAX_HISTORY_TURNS:]
        lines = []
        for user_msg, assistant_msg in recent:
            lines.append(f"User previously asked: {user_msg}")
            lines.append(f"Assistant answered: {assistant_msg}")
        memory_context = "\n".join(lines)

    result = generate(question, chunks, memory_context=memory_context)

    # Update chat history
    updated_history = history + [[question, result["answer"]]]

    # Retrieval details panel
    retrieval_lines = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        src = m.get("professor") or m.get("file_path", "unknown")
        dist_str = f"dist={c['distance']:.4f}"
        rrf_str = f" | rrf={c.get('rrf_score', '')}" if use_hybrid and "rrf_score" in c else ""
        bm25_str = f" | bm25={c.get('bm25_score', '')}" if use_hybrid and "bm25_score" in c else ""
        preview = c["text"][:130].replace("\n", " ")
        retrieval_lines.append(
            f"[{i}] {dist_str}{rrf_str}{bm25_str} | {m['source_type']} | {src}\n    {preview}…"
        )
    retrieval_text = "\n\n".join(retrieval_lines)

    sources_text = "\n".join(f"• {s}" for s in result["sources"]) or "No sources retrieved."

    return updated_history, retrieval_text, sources_text


def clear_history() -> tuple[list, str, str, str]:
    return [], "", "", ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="GMU CS Unofficial Guide") as demo:
    gr.Markdown(
        """# GMU CS Unofficial Guide
Ask questions about CS professors and courses at George Mason University.
Answers are grounded in student reviews (Rate My Professors, r/gmu) and the official course catalog."""
    )

    with gr.Row():
        # ---- Left column: input + filters ----
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=350)

            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What do students say about Wilson's CS222 exams?",
                lines=2,
            )

            with gr.Row():
                ask_btn = gr.Button("Ask", variant="primary", scale=3)
                clear_btn = gr.Button("Clear chat", scale=1)

            with gr.Accordion("🔍 Filters & Search mode", open=False):
                with gr.Row():
                    source_dd = gr.Dropdown(
                        choices=SOURCE_TYPES,
                        value="Any",
                        label="Source type",
                        info="Restrict retrieval to one source type",
                    )
                    professor_dd = gr.Dropdown(
                        choices=PROFESSORS,
                        value="Any",
                        label="Professor",
                        info="Restrict retrieval to one professor",
                    )
                hybrid_toggle = gr.Checkbox(
                    label="Use hybrid search (BM25 + semantic)",
                    value=False,
                    info="Combines keyword matching with semantic similarity via Reciprocal Rank Fusion",
                )

        # ---- Right column: example questions ----
        with gr.Column(scale=1):
            gr.Markdown("**Example questions**")
            for eq in EXAMPLE_QUESTIONS:
                gr.Button(eq, size="sm").click(
                    fn=lambda q=eq: q,
                    outputs=[question_box],
                )

    sources_box = gr.Textbox(label="Retrieved from", lines=3, interactive=False)
    retrieval_box = gr.Textbox(
        label="Retrieval details (rank · distance · source · preview)",
        lines=10,
        interactive=False,
    )

    # Shared state
    history_state = gr.State([])

    # Wire up Ask button and Enter key
    ask_inputs = [question_box, source_dd, professor_dd, hybrid_toggle, history_state]
    ask_outputs = [chatbot, retrieval_box, sources_box]

    def _ask_and_clear(question, source, professor, hybrid, history):
        chat, retrieval, sources = handle_query(question, source, professor, hybrid, history)
        # Return updated history into the state, clear the input box
        new_history = chat  # list of [user, assistant] pairs
        return chat, retrieval, sources, new_history, ""

    ask_btn.click(
        _ask_and_clear,
        inputs=ask_inputs,
        outputs=ask_outputs + [history_state, question_box],
    )
    question_box.submit(
        _ask_and_clear,
        inputs=ask_inputs,
        outputs=ask_outputs + [history_state, question_box],
    )
    clear_btn.click(
        clear_history,
        outputs=[chatbot, retrieval_box, sources_box, question_box],
    ).then(fn=lambda: [], outputs=[history_state])


if __name__ == "__main__":
    demo.launch()
