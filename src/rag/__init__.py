"""RAG (Retrieval-Augmented Generation) system for speech corpus.

Main entry point: ask() from src.rag.query (lazy import to avoid
pulling in heavy dependencies like anthropic when only embedder
is needed).
"""


def ask(*args, **kwargs):
    """Lazy wrapper — imports the real ask() on first call."""
    from src.rag.query import ask as _ask

    return _ask(*args, **kwargs)


__all__ = ["ask"]
