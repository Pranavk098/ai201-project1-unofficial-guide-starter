"""
Milestone 5: Gradio web interface for the GMU CS Unofficial Guide.

Run with:  python app.py
Then open: http://localhost:7860
"""

import gradio as gr
from query import ask

EXAMPLE_QUESTIONS = [
    "What do students say about exam difficulty in Justin Wilson's CS222?",
    "Is Wing Lam's SWE437 worth taking even though the quizzes are hard?",
    "What are the prerequisites for CS483 Analysis of Algorithms?",
    "Which professor is better for CS330 — Ahmed Zaman or Ivan Avramovic?",
    "What do students recommend for surviving CS310 at GMU?",
]


def handle_query(question: str) -> tuple[str, str]:
    question = question.strip()
    if not question:
        return "Please enter a question.", ""

    result = ask(question)
    sources_text = "\n".join(f"• {s}" for s in result["sources"]) or "No sources retrieved."
    return result["answer"], sources_text


with gr.Blocks(title="GMU CS Unofficial Guide") as demo:
    gr.Markdown(
        """# GMU CS Unofficial Guide
Ask questions about CS professors and courses at George Mason University.
Answers are grounded in student reviews (Rate My Professors, r/gmu) and the official course catalog."""
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What do students say about Wilson's CS222 exams?",
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown("**Example questions**")
            for eq in EXAMPLE_QUESTIONS:
                gr.Button(eq, size="sm").click(
                    fn=lambda q=eq: q,
                    outputs=[question_box],
                )

    answer_box = gr.Textbox(label="Answer", lines=10, interactive=False)
    sources_box = gr.Textbox(label="Retrieved from", lines=5, interactive=False)

    ask_btn.click(handle_query, inputs=question_box, outputs=[answer_box, sources_box])
    question_box.submit(handle_query, inputs=question_box, outputs=[answer_box, sources_box])

if __name__ == "__main__":
    demo.launch()
