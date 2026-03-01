"""Tests for inline visualization functions."""

from unittest.mock import patch

from src.frontend.visualizations import (
    _render_colombia_map,
    render_visualizations,
    viz_get_opinions,
    viz_get_speech_entities,
    viz_list_speeches,
    viz_retrieve_chunks,
    viz_search_entities,
)


class TestVizRetrieveChunks:
    """Test retrieve_chunks chart."""

    @patch("src.frontend.visualizations.st")
    def test_renders_chart_with_results(self, mock_st):
        data = [
            {
                "similarity": 0.82, "speech_title": "Discurso en Tumaco",
                "chunk_index": 0, "chunk_text": "...", "speech_id": 1,
                "speech_date": "2026-02-15", "speech_location": "Tumaco",
                "speech_event": "Mitin", "youtube_link": None,
            },
            {
                "similarity": 0.65, "speech_title": "Discurso en Cali",
                "chunk_index": 2, "chunk_text": "...", "speech_id": 2,
                "speech_date": "2026-02-20", "speech_location": "Cali",
                "speech_event": "Foro", "youtube_link": None,
            },
        ]
        viz_retrieve_chunks(data)
        mock_st.plotly_chart.assert_called_once()

    @patch("src.frontend.visualizations.st")
    def test_skips_empty_results(self, mock_st):
        viz_retrieve_chunks([])
        mock_st.plotly_chart.assert_not_called()


class TestRenderVisualizations:
    """Test the dispatch logic."""

    def test_empty_tool_calls(self):
        """No-op for empty list."""
        render_visualizations([])

    def test_skips_unknown_tools(self):
        """Unknown tool names are silently ignored."""
        render_visualizations([{
            "tool_name": "nonexistent_tool",
            "tool_input": {},
            "tool_result": {"data": 1},
        }])

    def test_skips_error_results(self):
        """Tool results with 'error' key are not visualized."""
        render_visualizations([{
            "tool_name": "retrieve_chunks",
            "tool_input": {"query": "test"},
            "tool_result": {"error": "something failed"},
        }])

    def test_skips_empty_list_results(self):
        """Empty list results are not visualized."""
        render_visualizations([{
            "tool_name": "retrieve_chunks",
            "tool_input": {"query": "test"},
            "tool_result": [],
        }])


class TestVizSearchEntities:
    """Test search_entities chart."""

    @patch("src.frontend.visualizations.st")
    def test_renders_chart_with_results(self, mock_st):
        data = [
            {
                "entity_text": "Colombia",
                "entity_label": "LOC",
                "mention_count": 42,
                "speech_titles": ["Discurso en Tumaco", "Discurso en Cali"],
            },
            {
                "entity_text": "Gustavo Petro",
                "entity_label": "PER",
                "mention_count": 15,
                "speech_titles": ["Discurso en Tumaco"],
            },
            {
                "entity_text": "Pacto Histórico",
                "entity_label": "ORG",
                "mention_count": 10,
                "speech_titles": ["Discurso en Cali"],
            },
        ]
        viz_search_entities(data)
        mock_st.plotly_chart.assert_called_once()

    @patch("src.frontend.visualizations.st")
    def test_skips_empty_results(self, mock_st):
        viz_search_entities([])
        mock_st.plotly_chart.assert_not_called()

    @patch("src.frontend.visualizations.st")
    def test_skips_error_results(self, mock_st):
        viz_search_entities([{"error": "Se requiere al menos entity_text o entity_label"}])
        mock_st.plotly_chart.assert_not_called()


class TestVizGetSpeechEntities:
    """Test get_speech_entities chart."""

    @patch("src.frontend.visualizations.st")
    def test_renders_chart_with_entities(self, mock_st):
        data = {
            "speech_id": 1,
            "speech_title": "Discurso en Tumaco",
            "entities": {
                "PER": [
                    {"entity_text": "Gustavo Petro", "mentions": 8},
                    {"entity_text": "Iván Cepeda", "mentions": 5},
                ],
                "LOC": [
                    {"entity_text": "Colombia", "mentions": 12},
                    {"entity_text": "Tumaco", "mentions": 6},
                ],
            },
            "total_entities": 4,
        }
        viz_get_speech_entities(data)
        assert mock_st.plotly_chart.call_count == 2  # bar + map (Tumaco has coords)

    @patch("src.frontend.visualizations.st")
    def test_skips_empty_entities(self, mock_st):
        data = {
            "speech_id": 1,
            "speech_title": "Discurso vacío",
            "entities": {},
            "total_entities": 0,
        }
        viz_get_speech_entities(data)
        mock_st.plotly_chart.assert_not_called()

    @patch("src.frontend.visualizations.st")
    def test_limits_to_top_10_per_label(self, mock_st):
        items = [{"entity_text": f"Entity {i}", "mentions": 20 - i} for i in range(15)]
        data = {
            "speech_id": 1,
            "speech_title": "Discurso largo",
            "entities": {"PER": items},
            "total_entities": 15,
        }
        viz_get_speech_entities(data)
        mock_st.plotly_chart.assert_called_once()
        # Verify the figure was created with at most 10 rows
        fig = mock_st.plotly_chart.call_args[0][0]
        assert len(fig.data[0].y) <= 10


class TestVizListSpeeches:
    """Test list_speeches chart."""

    @patch("src.frontend.visualizations.st")
    def test_renders_chart_with_speeches(self, mock_st):
        data = [
            {
                "id": 1, "title": "Discurso en Tumaco",
                "speech_date": "2026-02-15", "location": "Tumaco",
                "event": "Mitin", "word_count": 2500, "youtube_url": None,
            },
            {
                "id": 2, "title": "Discurso en Cali",
                "speech_date": "2026-02-20", "location": "Cali",
                "event": "Foro", "word_count": 3200, "youtube_url": None,
            },
        ]
        viz_list_speeches(data)
        mock_st.plotly_chart.assert_called_once()

    @patch("src.frontend.visualizations.st")
    def test_skips_empty_list(self, mock_st):
        viz_list_speeches([])
        mock_st.plotly_chart.assert_not_called()


class TestVizGetOpinions:
    """Test get_opinions donut chart."""

    @patch("src.frontend.visualizations.st")
    def test_renders_donut_with_opinions(self, mock_st):
        data = {
            "total_opinions": 10,
            "total_will_win": 7,
            "will_win_pct": 70.0,
            "opinions": [],
        }
        viz_get_opinions(data)
        mock_st.plotly_chart.assert_called_once()

    @patch("src.frontend.visualizations.st")
    def test_skips_zero_opinions(self, mock_st):
        data = {
            "total_opinions": 0,
            "total_will_win": 0,
            "will_win_pct": 0,
            "opinions": [],
        }
        viz_get_opinions(data)
        mock_st.plotly_chart.assert_not_called()


class TestColombiaMap:
    """Test _render_colombia_map helper."""

    @patch("src.frontend.visualizations.st")
    def test_renders_map_for_known_locations(self, mock_st):
        locations = [
            {"entity_text": "Tumaco", "mention_count": 29},
            {"entity_text": "Bogotá", "mention_count": 5},
        ]
        _render_colombia_map(locations)
        mock_st.plotly_chart.assert_called_once()

    @patch("src.frontend.visualizations.st")
    def test_skips_unknown_locations(self, mock_st):
        locations = [
            {"entity_text": "Colombia", "mention_count": 83},
            {"entity_text": "Piedremonte", "mention_count": 3},
        ]
        _render_colombia_map(locations)
        mock_st.plotly_chart.assert_not_called()

    @patch("src.frontend.visualizations.st")
    def test_accepts_mentions_field(self, mock_st):
        """The helper works with 'mentions' (get_speech_entities format)."""
        locations = [{"entity_text": "Cali", "mentions": 4}]
        _render_colombia_map(locations)
        mock_st.plotly_chart.assert_called_once()

    @patch("src.frontend.visualizations.st")
    def test_empty_list_no_chart(self, mock_st):
        _render_colombia_map([])
        mock_st.plotly_chart.assert_not_called()


class TestVizSearchEntitiesWithMap:
    """Test that viz_search_entities renders a map when LOC entities are present."""

    @patch("src.frontend.visualizations.st")
    def test_loc_results_render_bar_and_map(self, mock_st):
        data = [
            {"entity_text": "Tumaco", "entity_label": "LOC", "mention_count": 29,
             "speech_titles": ["Discurso en Tumaco"]},
            {"entity_text": "Bogotá", "entity_label": "LOC", "mention_count": 5,
             "speech_titles": ["Discurso en Bogotá"]},
        ]
        viz_search_entities(data)
        assert mock_st.plotly_chart.call_count == 2  # bar + map

    @patch("src.frontend.visualizations.st")
    def test_per_only_results_render_bar_only(self, mock_st):
        data = [
            {"entity_text": "Gustavo Petro", "entity_label": "PER",
             "mention_count": 15, "speech_titles": ["Discurso"]},
        ]
        viz_search_entities(data)
        assert mock_st.plotly_chart.call_count == 1  # bar only


class TestVizGetSpeechEntitiesWithMap:
    """Test that viz_get_speech_entities renders a map when LOC entities are present."""

    @patch("src.frontend.visualizations.st")
    def test_speech_with_loc_renders_bar_and_map(self, mock_st):
        data = {
            "speech_id": 1,
            "speech_title": "Discurso en Tumaco",
            "entities": {
                "PER": [{"entity_text": "Gustavo Petro", "mentions": 8}],
                "LOC": [
                    {"entity_text": "Tumaco", "mentions": 6},
                    {"entity_text": "Bogotá", "mentions": 3},
                ],
            },
            "total_entities": 3,
        }
        viz_get_speech_entities(data)
        assert mock_st.plotly_chart.call_count == 2  # bar + map

    @patch("src.frontend.visualizations.st")
    def test_speech_without_loc_renders_bar_only(self, mock_st):
        data = {
            "speech_id": 1,
            "speech_title": "Discurso sin lugares",
            "entities": {
                "PER": [{"entity_text": "Gustavo Petro", "mentions": 8}],
            },
            "total_entities": 1,
        }
        viz_get_speech_entities(data)
        assert mock_st.plotly_chart.call_count == 1  # bar only
