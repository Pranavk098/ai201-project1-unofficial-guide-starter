"""
End-to-end RAG pipeline: retrieve → generate.

`ask(question)` is the single entry point used by app.py and the CLI test.
"""

from embed import retrieve
from generate import generate


def ask(question: str, k: int = 5, filters: dict | None = None) -> dict:
    """
    Retrieve relevant chunks then generate a grounded answer.

    Returns the dict from generate(): { answer, sources, chunks_used }
    """
    chunks = retrieve(question, k=k, filters=filters)
    return generate(question, chunks)


if __name__ == "__main__":
    # Quick CLI test — runs the 5 evaluation plan queries
    test_queries = [
        "What do students say about exam difficulty in Justin Wilson's CS222?",
        "Is Wing Lam's SWE437 worth taking even though the quizzes are hard?",
        "What are the prerequisites for CS483 Analysis of Algorithms?",
        "Which professor is better for CS330 — Ahmed Zaman or Ivan Avramovic?",
        "What do students recommend for surviving CS310 at GMU?",
        # Out-of-scope query to verify the system declines gracefully
        "Who is the president of George Mason University?",
    ]

    for question in test_queries:
        print("\n" + "=" * 70)
        print(f"Q: {question}")
        print("=" * 70)
        result = ask(question)
        print(f"\n{result['answer']}\n")
        print("Sources:")
        for s in result["sources"]:
            print(f"  • {s}")
