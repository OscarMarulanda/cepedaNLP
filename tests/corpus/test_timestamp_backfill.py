"""Tests for sentence-level timestamp backfill."""

from src.corpus.timestamp_backfill import (
    build_char_to_segment_map,
    find_segment_for_position,
    match_sentences_to_timestamps,
)


class TestBuildCharToSegmentMap:
    def test_single_segment(self):
        segments = [{"text": "Hello world", "start": 0.0, "end": 2.5}]
        char_map = build_char_to_segment_map(segments)
        assert len(char_map) == 1
        assert char_map[0] == (0, 11, 0.0)

    def test_multiple_segments(self):
        segments = [
            {"text": "Hola", "start": 0.0, "end": 1.0},
            {"text": "mundo", "start": 1.5, "end": 3.0},
        ]
        # full_text = "Hola mundo" (len 10, with space at position 4)
        char_map = build_char_to_segment_map(segments)
        assert len(char_map) == 2
        # "Hola" occupies 0-4, then space, "mundo" occupies 5-10
        assert char_map[0] == (0, 4, 0.0)
        assert char_map[1] == (5, 10, 1.5)

    def test_handles_none_start_time(self):
        segments = [{"text": "No time", "start": None, "end": None}]
        char_map = build_char_to_segment_map(segments)
        assert char_map[0] == (0, 7, None)


class TestFindSegmentForPosition:
    def test_finds_correct_segment(self):
        char_map = [(0, 10, 0.0), (11, 20, 5.0), (21, 30, 10.0)]
        assert find_segment_for_position(0, char_map) == 0.0
        assert find_segment_for_position(5, char_map) == 0.0
        assert find_segment_for_position(11, char_map) == 5.0
        assert find_segment_for_position(25, char_map) == 10.0

    def test_returns_none_for_gap(self):
        char_map = [(0, 5, 0.0), (10, 15, 5.0)]
        assert find_segment_for_position(7, char_map) is None

    def test_returns_none_for_out_of_range(self):
        char_map = [(0, 10, 0.0)]
        assert find_segment_for_position(15, char_map) is None


class TestMatchSentencesToTimestamps:
    def test_matches_sequential_sentences(self):
        segments = [
            {"text": "Primera oración.", "start": 0.0, "end": 3.0},
            {"text": "Segunda oración.", "start": 3.5, "end": 6.0},
        ]
        full_text = "Primera oración. Segunda oración."
        char_map = build_char_to_segment_map(segments)

        sentences = [
            (0, "Primera oración."),
            (1, "Segunda oración."),
        ]
        results = match_sentences_to_timestamps(sentences, full_text, char_map)

        assert results == [(0, 0.0), (1, 3.5)]

    def test_sentence_spanning_multiple_segments(self):
        # spaCy may merge segments into one sentence
        segments = [
            {"text": "Parte uno.", "start": 0.0, "end": 2.0},
            {"text": "Parte dos.", "start": 2.5, "end": 5.0},
        ]
        full_text = "Parte uno. Parte dos."
        char_map = build_char_to_segment_map(segments)

        # spaCy sees the whole thing as one sentence
        sentences = [(0, "Parte uno. Parte dos.")]
        results = match_sentences_to_timestamps(sentences, full_text, char_map)

        # Maps to the first segment's start_time
        assert results == [(0, 0.0)]

    def test_unmatched_sentence_returns_none(self):
        segments = [{"text": "Original text.", "start": 5.0, "end": 8.0}]
        full_text = "Original text."
        char_map = build_char_to_segment_map(segments)

        sentences = [(0, "This text does not exist.")]
        results = match_sentences_to_timestamps(sentences, full_text, char_map)

        assert results == [(0, None)]

    def test_empty_sentences_list(self):
        segments = [{"text": "Algo.", "start": 0.0, "end": 1.0}]
        full_text = "Algo."
        char_map = build_char_to_segment_map(segments)

        results = match_sentences_to_timestamps([], full_text, char_map)
        assert results == []
