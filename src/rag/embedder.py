"""Embedding model wrapper for RAG.

Uses paraphrase-multilingual-mpnet-base-v2 (768d) for encoding
speech chunks and user queries into dense vectors.

Query embedding supports two providers via EMBEDDING_PROVIDER env var:
- "local" (default): loads SentenceTransformer model locally (~868 MB RAM)
- "hf_api": calls HuggingFace Inference API (requires HF_TOKEN)
"""

import logging
import os

import numpy as np
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
HF_MODEL_ID = f"sentence-transformers/{MODEL_NAME}"
EMBEDDING_DIM = 768

# Module-level cache (same pattern as nlp_processor.py)
_model = None  # SentenceTransformer | None


def load_model():
    """Load local embedding model (cached after first call)."""
    from sentence_transformers import SentenceTransformer

    global _model
    if _model is None:
        logger.info("Loading embedding model '%s'...", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded (dim=%d)", EMBEDDING_DIM)
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts into dense vectors.

    Always uses the local model (pipeline ingestion only).

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


def _embed_query_hf(query: str) -> np.ndarray:
    """Embed a query via HuggingFace Inference API.

    Returns:
        numpy array of shape (768,), L2-normalized.
    """
    from huggingface_hub import InferenceClient

    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN env var required when EMBEDDING_PROVIDER=hf_api")

    client = InferenceClient(token=token)
    result = client.feature_extraction(query, model=HF_MODEL_ID)

    vec = np.asarray(result, dtype=np.float32)

    # API may return (1, 768) or (num_tokens, 768) — mean-pool to (768,)
    if vec.ndim == 2:
        vec = vec.mean(axis=0)

    # L2-normalize to match local model's normalize_embeddings=True
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string.

    Provider is selected by EMBEDDING_PROVIDER env var:
    - "local" (default): local SentenceTransformer model
    - "hf_api": HuggingFace Inference API (no local model loaded)

    Returns:
        numpy array of shape (768,).
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "local")

    if provider == "hf_api":
        logger.debug("Embedding query via HuggingFace Inference API")
        return _embed_query_hf(query)

    # Default: local model
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
