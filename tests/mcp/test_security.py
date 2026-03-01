"""Security tests for MCP tools — input validation and injection prevention."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.mcp.server import (
    _youtube_link,
    retrieve_chunks,
    search_entities,
    submit_opinion,
)


class TestInputValidation:
    """Test that invalid inputs are rejected or handled safely."""

    def test_search_entities_no_params(self):
        """Calling search_entities with no params returns error."""
        result = search_entities()
        assert isinstance(result, list)
        assert "error" in result[0]

    @patch("src.mcp.server.db_connection")
    @patch("src.mcp.server.embed_query")
    def test_sql_injection_in_query(self, mock_embed, mock_db_ctx):
        """SQL injection in query is harmless — it's embedded, not interpolated."""
        mock_embed.return_value = np.zeros(768)
        conn = _mock_conn([])
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        # The query string goes through embed_query, not SQL
        results = retrieve_chunks("'; DROP TABLE speeches; --")
        assert results == []

    @patch("src.mcp.server.db_connection")
    def test_sql_injection_in_entity_text(self, mock_db_ctx):
        """SQL injection in entity_text is parameterized."""
        conn = _mock_conn([])
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        results = search_entities(entity_text="'; DROP TABLE entities; --")
        assert results == []

    @patch("src.mcp.server.db_connection")
    def test_entity_label_normalized_to_upper(self, mock_db_ctx):
        """Entity labels are uppercased before querying."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        search_entities(entity_label="per")
        # Verify the SQL param was uppercased
        call_args = mock_cursor.execute.call_args
        assert "PER" in call_args[0][1]


class TestYoutubeLink:
    """Test the _youtube_link helper."""

    def test_none_url(self):
        assert _youtube_link(None, None) is None

    def test_watch_url_with_timestamp(self):
        link = _youtube_link(
            "https://www.youtube.com/watch?v=abc123", 120
        )
        assert link == "https://www.youtube.com/watch?v=abc123&t=120"

    def test_short_url(self):
        link = _youtube_link("https://youtu.be/abc123", 60)
        assert link == "https://www.youtube.com/watch?v=abc123&t=60"

    def test_no_timestamp(self):
        link = _youtube_link(
            "https://www.youtube.com/watch?v=abc123", None
        )
        assert link == "https://www.youtube.com/watch?v=abc123"


class TestOpinionSecurity:
    """Test input validation for opinion tools."""

    def test_empty_opinion_rejected(self):
        result = submit_opinion(opinion_text="", will_win=True)
        assert "error" in result

    def test_whitespace_only_rejected(self):
        result = submit_opinion(opinion_text="   \n\t  ", will_win=False)
        assert "error" in result

    @patch("src.mcp.server.db_connection")
    def test_sql_injection_in_opinion(self, mock_db_ctx):
        """SQL injection in opinion_text is parameterized."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1, "2026-02-28 12:00:00")
        conn = _mock_conn([])
        conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = submit_opinion(
            opinion_text="'; DROP TABLE user_opinions; --",
            will_win=True,
        )
        assert "opinion_id" in result


# Helper (same as test_tools.py)
def _mock_conn(rows):
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_conn
