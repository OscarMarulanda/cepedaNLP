"""End-to-end RAG query pipeline.

Orchestrates: embed query -> retrieve chunks -> generate response.

Usage:
    python -m src.rag.query "¿Qué propone sobre el racismo?"
    python -m src.rag.query "¿Qué dice sobre la reforma agraria?"
"""

import logging

from src.rag.generator import generate
from src.rag.retriever import retrieve

logger = logging.getLogger(__name__)


def ask(
    query: str,
    top_k: int = 5,
    threshold: float = 0.3,
    model: str | None = None,
    conn=None,
) -> dict:
    """Ask a question and get a cited answer from the speech corpus.

    This is the main entry point for the RAG system.

    Args:
        query: User question in Spanish.
        top_k: Number of chunks to retrieve.
        threshold: Minimum similarity threshold.
        model: Claude model to use (None = dev default).
        conn: Optional DB connection.

    Returns:
        Dict with keys: query, answer, model, usage, chunks_used,
        sources (list of citation dicts with youtube_link).
    """
    # Step 1: Retrieve
    results = retrieve(query, top_k=top_k, threshold=threshold, conn=conn)

    # Step 2: Generate
    response = generate(query, results, model=model)

    # Step 3: Build structured output
    sources = [
        {
            "speech_title": r.speech_title,
            "speech_date": r.speech_date,
            "similarity": round(r.similarity, 3),
            "youtube_link": r.youtube_link,
            "chunk_preview": r.chunk_text[:200],
        }
        for r in results
    ]

    return {
        "query": query,
        "answer": response["answer"],
        "model": response["model"],
        "usage": response["usage"],
        "chunks_used": response["chunks_used"],
        "sources": sources,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "¿Qué propone sobre el racismo?"
    )

    print(f"\nPregunta: {query}\n")
    result = ask(query)

    print(f"Respuesta:\n{result['answer']}\n")
    print("Fuentes:")
    for s in result["sources"]:
        link = f" — {s['youtube_link']}" if s["youtube_link"] else ""
        print(
            f"  [{s['similarity']:.3f}] {s['speech_title']} "
            f"({s['speech_date']}){link}"
        )
    print(f"\nModelo: {result['model']}, Tokens: {result['usage']}")
