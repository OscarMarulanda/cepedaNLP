"""Tests for src.rag.embedder — provider switching and HF API path."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove embedding env vars before each test."""
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)


# ── Provider selection ───────────────────────────────────────────────


@patch("src.rag.embedder.load_model")
def test_default_provider_is_local(mock_load):
    """No EMBEDDING_PROVIDER set → uses local model."""
    from src.rag.embedder import embed_query

    fake_model = MagicMock()
    fake_model.encode.return_value = np.ones(768, dtype=np.float32)
    mock_load.return_value = fake_model

    result = embed_query("test")

    mock_load.assert_called_once()
    fake_model.encode.assert_called_once()
    assert result.shape == (768,)


@patch("src.rag.embedder.load_model")
def test_local_embed_query(mock_load, monkeypatch):
    """EMBEDDING_PROVIDER=local → uses local model."""
    from src.rag.embedder import embed_query

    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")

    fake_model = MagicMock()
    fake_model.encode.return_value = np.ones(768, dtype=np.float32)
    mock_load.return_value = fake_model

    result = embed_query("reforma agraria")

    mock_load.assert_called_once()
    fake_model.encode.assert_called_once_with(
        "reforma agraria",
        normalize_embeddings=True,
    )
    assert result.shape == (768,)


# ── HuggingFace Inference API path ───────────────────────────────────


@patch("huggingface_hub.InferenceClient")
def test_hf_api_embed_query(mock_client_cls, monkeypatch):
    """EMBEDDING_PROVIDER=hf_api → calls InferenceClient.feature_extraction."""
    from src.rag.embedder import HF_MODEL_ID, embed_query

    monkeypatch.setenv("EMBEDDING_PROVIDER", "hf_api")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    # Simulate API returning (1, 768) — needs mean-pooling
    fake_embedding = np.random.randn(1, 768).astype(np.float32)
    mock_instance = MagicMock()
    mock_instance.feature_extraction.return_value = fake_embedding
    mock_client_cls.return_value = mock_instance

    result = embed_query("justicia social")

    mock_client_cls.assert_called_once_with(token="hf_test_token")
    mock_instance.feature_extraction.assert_called_once_with(
        "justicia social",
        model=HF_MODEL_ID,
    )
    assert result.shape == (768,)


@patch("huggingface_hub.InferenceClient")
def test_hf_api_normalizes_vector(mock_client_cls, monkeypatch):
    """HF API result is L2-normalized (unit vector)."""
    from src.rag.embedder import embed_query

    monkeypatch.setenv("EMBEDDING_PROVIDER", "hf_api")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    # Return an unnormalized vector
    raw = np.random.randn(768).astype(np.float32) * 5.0
    mock_instance = MagicMock()
    mock_instance.feature_extraction.return_value = raw
    mock_client_cls.return_value = mock_instance

    result = embed_query("test")

    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 1e-5, f"Expected unit vector, got norm={norm}"
