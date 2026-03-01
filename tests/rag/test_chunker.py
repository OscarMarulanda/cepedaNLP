"""Tests for the sentence-grouping chunker."""

import pytest

from src.rag.chunker import Chunk, chunk_sentences, map_chunk_timestamps


class TestChunkSentences:
    """Test the core chunking algorithm."""

    @pytest.fixture
    def short_sentences(self) -> list[dict]:
        """10 sentences of ~10 words each (100 total words)."""
        return [
            {
                "sentence_index": i,
                "sentence_text": (
                    f"Esta es la oración número {i} del discurso de prueba."
                ),
            }
            for i in range(10)
        ]

    @pytest.fixture
    def varied_sentences(self) -> list[dict]:
        """Mix of short and long sentences."""
        return [
            {"sentence_index": 0, "sentence_text": "Gracias."},
            {
                "sentence_index": 1,
                "sentence_text": "Compañeras y compañeros, "
                + "palabra " * 48,
            },
            {"sentence_index": 2, "sentence_text": "Breve frase aquí."},
            {
                "sentence_index": 3,
                "sentence_text": " ".join(["contenido"] * 200),
            },
            {"sentence_index": 4, "sentence_text": "Final."},
        ]

    def test_single_chunk_for_short_text(self, short_sentences):
        """Text under target_words should produce a single chunk."""
        chunks = chunk_sentences(short_sentences, target_words=200)
        assert len(chunks) == 1
        assert chunks[0].sentence_start == 0
        assert chunks[0].sentence_end == 9

    def test_multiple_chunks_produced(self, short_sentences):
        """Low target_words forces multiple chunks."""
        chunks = chunk_sentences(
            short_sentences, target_words=30, min_words=15
        )
        assert len(chunks) > 1

    def test_overlap_sentences(self, short_sentences):
        """With overlap=1, consecutive chunks share a sentence."""
        chunks = chunk_sentences(
            short_sentences,
            target_words=30,
            min_words=15,
            overlap_sentences=1,
        )
        if len(chunks) >= 2:
            assert chunks[0].sentence_end == chunks[1].sentence_start

    def test_empty_input(self):
        assert chunk_sentences([]) == []

    def test_single_sentence(self):
        sentences = [
            {"sentence_index": 0, "sentence_text": "Una sola oración."}
        ]
        chunks = chunk_sentences(sentences)
        assert len(chunks) == 1
        assert chunks[0].sentence_start == 0
        assert chunks[0].sentence_end == 0

    def test_chunk_indices_sequential(self, short_sentences):
        chunks = chunk_sentences(
            short_sentences, target_words=30, min_words=15
        )
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_all_sentences_covered(self, short_sentences):
        """Every sentence_index should appear in at least one chunk."""
        chunks = chunk_sentences(
            short_sentences, target_words=30, min_words=15
        )
        all_indices = set()
        for chunk in chunks:
            for idx in range(chunk.sentence_start, chunk.sentence_end + 1):
                all_indices.add(idx)
        expected = set(range(len(short_sentences)))
        assert expected.issubset(all_indices)

    def test_word_count_accuracy(self):
        sentences = [
            {
                "sentence_index": 0,
                "sentence_text": "uno dos tres cuatro cinco",
            },
        ]
        chunks = chunk_sentences(sentences)
        assert chunks[0].word_count == 5

    def test_runt_chunk_merged(self):
        """A very short final group should merge into the previous chunk."""
        sentences = [
            {
                "sentence_index": 0,
                "sentence_text": " ".join(["palabra"] * 200),
            },
            {
                "sentence_index": 1,
                "sentence_text": " ".join(["palabra"] * 200),
            },
            {"sentence_index": 2, "sentence_text": "Gracias."},
        ]
        chunks = chunk_sentences(
            sentences, target_words=200, overlap_sentences=0
        )
        # "Gracias." alone is <30 words, so it should merge into chunk 1
        assert chunks[-1].sentence_end == 2
        assert len(chunks) == 2

    def test_no_overlap(self, short_sentences):
        """With overlap=0, chunks should not share sentences."""
        chunks = chunk_sentences(
            short_sentences,
            target_words=30,
            min_words=15,
            overlap_sentences=0,
        )
        if len(chunks) >= 2:
            assert chunks[0].sentence_end < chunks[1].sentence_start


class TestMapChunkTimestamps:
    """Test timestamp mapping from raw segments."""

    def test_maps_matching_segment(self):
        chunks = [
            Chunk(
                chunk_index=0,
                text="Queridas compañeras y queridos compañeros, anoche se presentó",
                sentence_start=0,
                sentence_end=2,
                start_char=0,
                end_char=60,
                word_count=8,
            ),
        ]
        raw_segments = [
            {
                "start": 0.45,
                "end": 11.99,
                "text": "Queridas compañeras y queridos compañeros, anoche se presentó",
            },
            {
                "start": 11.99,
                "end": 22.29,
                "text": "algo completamente diferente",
            },
        ]
        result = map_chunk_timestamps(chunks, raw_segments)
        assert result[0].metadata["start_time"] == 0

    def test_no_match_no_timestamp(self):
        chunks = [
            Chunk(
                chunk_index=0,
                text="Texto completamente diferente al original",
                sentence_start=0,
                sentence_end=0,
                start_char=0,
                end_char=40,
                word_count=5,
            ),
        ]
        raw_segments = [
            {"start": 10.0, "end": 20.0, "text": "Otro contenido aquí"},
        ]
        result = map_chunk_timestamps(chunks, raw_segments)
        assert "start_time" not in result[0].metadata

    def test_empty_segments(self):
        chunks = [
            Chunk(
                chunk_index=0,
                text="Algo",
                sentence_start=0,
                sentence_end=0,
                start_char=0,
                end_char=4,
                word_count=1,
            ),
        ]
        result = map_chunk_timestamps(chunks, [])
        assert "start_time" not in result[0].metadata
