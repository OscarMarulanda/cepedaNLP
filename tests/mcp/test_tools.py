"""Unit tests for MCP server tools."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.mcp.server import (
    get_corpus_stats,
    get_opinions,
    get_speech_detail,
    get_speech_entities,
    list_speeches,
    retrieve_chunks,
    search_entities,
    submit_opinion,
)


def _mock_conn(rows, fetchone_value=None):
    """Create a mock DB connection with cursor returning given rows."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows
    if fetchone_value is not None:
        mock_cursor.fetchone.return_value = fetchone_value
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_conn


def _mock_db_ctx(conn):
    """Wire a mock connection into the db_connection() context manager."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


class TestRetrieveChunks:
    @patch("src.mcp.server.db_connection")
    @patch("src.mcp.server.embed_query")
    def test_returns_results_with_sentences(self, mock_embed, mock_db_ctx):
        mock_embed.return_value = np.zeros(768)

        # First call: pgvector similarity search
        conn1 = _mock_conn([
            (1, 1, 0, "Fragmento sobre racismo", 0.82,
             "Discurso en Tumaco", "2026-02-24", "Tumaco", "Rally",
             "https://www.youtube.com/watch?v=abc123", 120),
        ])

        # Second call: sentence lookups (fetchone for span, fetchall for sentences)
        mock_cursor2 = MagicMock()
        mock_cursor2.fetchone.return_value = (0, 2)  # sentence_start, sentence_end
        mock_cursor2.fetchall.return_value = [
            ("Primera oración.", 120.0),
            ("Segunda oración.", 135.0),
            ("Tercera oración.", 150.0),
        ]
        conn2 = MagicMock()
        conn2.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor2)
        conn2.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_db_ctx.side_effect = [_mock_db_ctx(conn1), _mock_db_ctx(conn2)]

        results = retrieve_chunks("racismo en Colombia", top_k=3)

        assert len(results) == 1
        assert results[0]["chunk_text"] == "Fragmento sobre racismo"
        assert results[0]["similarity"] == 0.82
        assert results[0]["speech_title"] == "Discurso en Tumaco"
        assert "abc123" in results[0]["youtube_link"]
        assert "&t=120" in results[0]["youtube_link"]
        # Sentence-level data
        assert len(results[0]["sentences"]) == 3
        assert results[0]["sentences"][0]["text"] == "Primera oración."
        assert results[0]["sentences"][0]["start_time"] == 120
        assert "&t=135" in results[0]["sentences"][1]["youtube_link"]

    @patch("src.mcp.server.db_connection")
    @patch("src.mcp.server.embed_query")
    def test_filters_low_similarity(self, mock_embed, mock_db_ctx):
        mock_embed.return_value = np.zeros(768)
        conn = _mock_conn([
            (1, 1, 0, "Relevant", 0.75, "Speech", "2026-02-24",
             None, None, None, None),
            (2, 1, 1, "Irrelevant", 0.15, "Speech", "2026-02-24",
             None, None, None, None),
        ])

        # Second call for sentence lookups (1 chunk passes threshold)
        mock_cursor2 = MagicMock()
        mock_cursor2.fetchone.return_value = (0, 0)
        mock_cursor2.fetchall.return_value = [("Sentence.", 10.0)]
        conn2 = MagicMock()
        conn2.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor2)
        conn2.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_db_ctx.side_effect = [_mock_db_ctx(conn), _mock_db_ctx(conn2)]

        results = retrieve_chunks("test query")
        assert len(results) == 1

    @patch("src.mcp.server.db_connection")
    @patch("src.mcp.server.embed_query")
    def test_empty_results(self, mock_embed, mock_db_ctx):
        mock_embed.return_value = np.zeros(768)
        conn = _mock_conn([])
        mock_db_ctx.return_value = _mock_db_ctx(conn)

        results = retrieve_chunks("nonexistent topic")
        assert results == []


class TestListSpeeches:
    @patch("src.mcp.server.db_connection")
    def test_returns_all_speeches(self, mock_db_ctx):
        conn = _mock_conn([
            (1, "Discurso Tumaco", "2026-02-24", "Tumaco", "Rally", 5000,
             "https://youtube.com/watch?v=a"),
            (2, "Discurso Cali", "2026-02-25", "Cali", "Foro", 3000,
             "https://youtube.com/watch?v=b"),
        ])
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        results = list_speeches()
        assert len(results) == 2
        assert results[0]["title"] == "Discurso Tumaco"
        assert results[1]["word_count"] == 3000


class TestGetSpeechDetail:
    @patch("src.mcp.server.db_connection")
    def test_returns_speech(self, mock_db_ctx):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            1, "Discurso Tumaco", "2026-02-24", "Tumaco", "Rally",
            5000, "https://youtube.com/watch?v=a", 1200,
            "Texto limpio del discurso...", 42, 13,
        )
        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = get_speech_detail(1)
        assert result["title"] == "Discurso Tumaco"
        assert result["entity_count"] == 42
        assert result["chunk_count"] == 13

    @patch("src.mcp.server.db_connection")
    def test_not_found(self, mock_db_ctx):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = get_speech_detail(999)
        assert "error" in result


class TestSearchEntities:
    def test_requires_at_least_one_param(self):
        result = search_entities()
        assert result[0]["error"]

    @patch("src.mcp.server.db_connection")
    def test_search_by_text(self, mock_db_ctx):
        conn = _mock_conn([
            ("Colombia", "LOC", 15, ["Discurso Tumaco", "Discurso Cali"]),
        ])
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        results = search_entities(entity_text="Colombia")
        assert len(results) == 1
        assert results[0]["mention_count"] == 15

    @patch("src.mcp.server.db_connection")
    def test_search_by_label(self, mock_db_ctx):
        conn = _mock_conn([
            ("Iván Cepeda", "PER", 50, ["Discurso Tumaco"]),
            ("Gustavo Petro", "PER", 10, ["Discurso Cali"]),
        ])
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        results = search_entities(entity_label="PER")
        assert len(results) == 2


class TestGetSpeechEntities:
    @patch("src.mcp.server.db_connection")
    def test_returns_grouped_entities(self, mock_db_ctx):
        mock_cursor = MagicMock()
        # First fetchone for speech existence check
        mock_cursor.fetchone.return_value = ("Discurso Tumaco",)
        mock_cursor.fetchall.return_value = [
            ("Iván Cepeda", "PER", 5),
            ("Colombia", "LOC", 3),
            ("Tumaco", "LOC", 2),
        ]
        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = get_speech_entities(1)
        assert result["speech_title"] == "Discurso Tumaco"
        assert "PER" in result["entities"]
        assert "LOC" in result["entities"]
        assert result["total_entities"] == 3

    @patch("src.mcp.server.db_connection")
    def test_not_found(self, mock_db_ctx):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = get_speech_entities(999)
        assert "error" in result


class TestGetCorpusStats:
    @patch("src.mcp.server.db_connection")
    def test_returns_stats(self, mock_db_ctx):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            (10,),     # speeches
            (50000,),  # total_words
            (500,),    # entities
            (2000,),   # annotations
            (131,),    # chunks
            (5,),      # opinions
        ]
        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = get_corpus_stats()
        assert result["speeches"] == 10
        assert result["total_words"] == 50000
        assert result["entities"] == 500
        assert result["annotations"] == 2000
        assert result["chunks"] == 131
        assert result["opinions"] == 5


class TestSubmitOpinion:
    @patch("src.mcp.server.db_connection")
    def test_saves_opinion(self, mock_db_ctx):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1, "2026-02-28 12:00:00")
        conn = _mock_conn([])
        conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = submit_opinion(
            opinion_text="Creo que sus propuestas son interesantes",
            will_win=True,
        )
        assert result["opinion_id"] == 1
        assert "message" in result
        conn.commit.assert_called_once()

    @patch("src.mcp.server.db_connection")
    def test_strips_whitespace(self, mock_db_ctx):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (2, "2026-02-28 12:00:00")
        conn = _mock_conn([])
        conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        submit_opinion(opinion_text="  Opinión con espacios  ", will_win=False)
        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][0] == "Opinión con espacios"

    def test_rejects_empty_opinion(self):
        result = submit_opinion(opinion_text="   ", will_win=True)
        assert "error" in result


class TestGetOpinions:
    @patch("src.mcp.server.db_connection")
    def test_returns_opinions_with_stats(self, mock_db_ctx):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            (5,),   # total opinions
            (3,),   # total will_win=True
        ]
        mock_cursor.fetchall.return_value = [
            (1, "Buenas propuestas", True, "2026-02-28 10:00:00"),
            (2, "No me convence", False, "2026-02-28 11:00:00"),
        ]
        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = get_opinions()
        assert result["total_opinions"] == 5
        assert result["total_will_win"] == 3
        assert result["will_win_pct"] == 60.0
        assert len(result["opinions"]) == 2

    @patch("src.mcp.server.db_connection")
    def test_filter_by_will_win(self, mock_db_ctx):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(5,), (3,)]
        mock_cursor.fetchall.return_value = [
            (1, "Va a ganar seguro", True, "2026-02-28 10:00:00"),
        ]
        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = get_opinions(will_win=True)
        assert len(result["opinions"]) == 1
        # Third execute call should include will_win filter
        sql = mock_cursor.execute.call_args_list[2][0][0]
        assert "will_win" in sql.lower()

    @patch("src.mcp.server.db_connection")
    def test_empty_table(self, mock_db_ctx):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(0,), (0,)]
        mock_cursor.fetchall.return_value = []
        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_db_ctx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = get_opinions()
        assert result["total_opinions"] == 0
        assert result["will_win_pct"] == 0
        assert result["opinions"] == []
