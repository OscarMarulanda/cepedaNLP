"""Embedding model wrapper for RAG.

Uses paraphrase-multilingual-mpnet-base-v2 (768d) for encoding
speech chunks and user queries into dense vectors.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 768

# Module-level cache (same pattern as nlp_processor.py)
_model: SentenceTransformer | None = None


def load_model() -> SentenceTransformer:
    """Load embedding model (cached after first call)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s'...", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded (dim=%d)", EMBEDDING_DIM)
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts into dense vectors.

    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for encoding.

    Returns:
        numpy array of shape (len(texts), 768).
    """
    model = load_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 20,
        normalize_embeddings=True,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string.

    Returns:
        numpy array of shape (768,).
    """
    model = load_model()
    embedding = model.encode(
        query,
        normalize_embeddings=True,
    )
    return embedding


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    test_texts = [
        "La reforma agraria es fundamental para el campo colombiano.",
        "Necesitamos justicia racial en Colombia.",
        "El sistema de salud debe ser universal.",
    ]
    embeddings = embed_texts(test_texts)
    print(f"Shape: {embeddings.shape}")

    # Cosine similarities (embeddings are normalized, so dot product = cosine sim)
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            print(f"  sim('{test_texts[i][:40]}', '{test_texts[j][:40]}') = {sim:.3f}")
