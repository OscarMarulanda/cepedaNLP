"""Tests for the retriever module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.retriever import RetrievalResult, retrieve


class TestRetrievalResult:
    def test_youtube_link_with_timestamp(self):
        r = RetrievalResult(
            chunk_id=1,
            speech_id=1,
            chunk_index=0,
            chunk_text="Test",
            similarity=0.75,
            speech_title="Test Speech",
            speech_date="2026-02-24",
            speech_location=None,
            speech_event=None,
            youtube_url="https://www.youtube.com/watch?v=abc123",
            start_time=120,
        )
        assert r.youtube_link == "https://www.youtube.com/watch?v=abc123&t=120"

    def test_youtube_link_without_timestamp(self):
        r = RetrievalResult(
            chunk_id=1,
            speech_id=1,
            chunk_index=0,
            chunk_text="Test",
            similarity=0.75,
            speech_title="Test Speech",
            speech_date=None,
            speech_location=None,
            speech_event=None,
            youtube_url="https://www.youtube.com/watch?v=abc123",
            start_time=None,
        )
        assert r.youtube_link == "https://www.youtube.com/watch?v=abc123"

    def test_youtube_link_no_url(self):
        r = RetrievalResult(
            chunk_id=1,
            speech_id=1,
            chunk_index=0,
            chunk_text="Test",
            similarity=0.75,
            speech_title="Test Speech",
            speech_date=None,
            speech_location=None,
            speech_event=None,
            youtube_url=None,
            start_time=None,
        )
        assert r.youtube_link is None

    def test_youtube_link_short_url(self):
        r = RetrievalResult(
            chunk_id=1,
            speech_id=1,
            chunk_index=0,
            chunk_text="Test",
            similarity=0.75,
            speech_title="Test Speech",
            speech_date=None,
            speech_location=None,
            speech_event=None,
            youtube_url="https://youtu.be/abc123",
            start_time=60,
        )
        assert r.youtube_link == "https://www.youtube.com/watch?v=abc123&t=60"


class TestRetrieve:
    @patch("src.rag.retriever.embed_query")
    def test_filters_below_threshold(self, mock_embed):
        """Results below the threshold are excluded."""
        mock_embed.return_value = np.zeros(768)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, 1, 0, "High relevance", 0.75, "Speech 1", "2026-02-24", None, None, "https://youtube.com/watch?v=abc", 10),
            (2, 1, 1, "Low relevance", 0.15, "Speech 1", "2026-02-24", None, None, "https://youtube.com/watch?v=abc", 20),
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(
            return_value=mock_cursor
        )
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        results = retrieve("test query", threshold=0.3, conn=mock_conn)
        assert len(results) == 1
        assert results[0].similarity == 0.75

    @patch("src.rag.retriever.embed_query")
    def test_empty_results(self, mock_embed):
        """No results when DB returns nothing."""
        mock_embed.return_value = np.zeros(768)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(
            return_value=mock_cursor
        )
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        results = retrieve("test query", conn=mock_conn)
        assert results == []

    @patch("src.rag.retriever.embed_query")
    def test_all_results_above_threshold(self, mock_embed):
        """All results pass when above threshold."""
        mock_embed.return_value = np.zeros(768)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, 1, 0, "Chunk A", 0.8, "Speech 1", "2026-02-24", "Cali", "Rally", "https://youtube.com/watch?v=a", 10),
            (2, 2, 0, "Chunk B", 0.6, "Speech 2", "2026-02-25", None, None, "https://youtube.com/watch?v=b", 20),
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(
            return_value=mock_cursor
        )
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        results = retrieve("test query", threshold=0.3, conn=mock_conn)
        assert len(results) == 2
        assert results[0].speech_location == "Cali"
        assert results[1].youtube_url == "https://youtube.com/watch?v=b"
