"""Tests for speaker diarization module.

Tests pure logic functions that don't require loading heavy models.
Model-dependent functions are tested via the CLI (manual integration tests).
"""

import numpy as np
import pytest
import torch
import torchaudio
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.corpus.diarizer import (
    SpeakerSegment,
    OffsetMapping,
    DiarizationResult,
    _remap_time,
    remap_timestamps,
    extract_speaker_audio,
    load_reference_embedding,
)


# ---------------------------------------------------------------------------
# Test _remap_time
# ---------------------------------------------------------------------------

class TestRemapTime:
    """Test single timestamp remapping from concatenated to original time."""

    @pytest.fixture
    def mappings(self) -> list[OffsetMapping]:
        """Two original segments: [10-20] and [30-45], concatenated to [0-10] and [10-25]."""
        return [
            OffsetMapping(concat_start=0.0, concat_end=10.0, original_start=10.0, original_end=20.0),
            OffsetMapping(concat_start=10.0, concat_end=25.0, original_start=30.0, original_end=45.0),
        ]

    def test_start_of_first_segment(self, mappings):
        assert _remap_time(0.0, mappings) == 10.0

    def test_middle_of_first_segment(self, mappings):
        assert _remap_time(5.0, mappings) == 15.0

    def test_end_of_first_segment(self, mappings):
        # At the exact boundary, the first mapping claims it (both are valid)
        assert _remap_time(10.0, mappings) == 20.0

    def test_middle_of_second_segment(self, mappings):
        assert _remap_time(17.5, mappings) == 37.5

    def test_end_of_second_segment(self, mappings):
        assert _remap_time(25.0, mappings) == 45.0

    def test_before_first_segment(self, mappings):
        """Time before all mappings clamps to first segment start."""
        assert _remap_time(-1.0, mappings) == 10.0

    def test_after_last_segment(self, mappings):
        """Time after all mappings clamps to last segment end."""
        assert _remap_time(30.0, mappings) == 45.0


# ---------------------------------------------------------------------------
# Test remap_timestamps
# ---------------------------------------------------------------------------

class TestRemapTimestamps:
    """Test full transcript timestamp remapping."""

    def test_remaps_all_segments(self):
        mappings = [
            OffsetMapping(concat_start=0.0, concat_end=10.0, original_start=50.0, original_end=60.0),
        ]
        transcript = {
            "speech_id": "test",
            "language": "es",
            "segments": [
                {"start": 0.0, "end": 3.5, "text": "Hola"},
                {"start": 3.5, "end": 8.0, "text": "Mundo"},
            ],
            "full_text": "Hola Mundo",
        }

        result = remap_timestamps(transcript, mappings)

        assert result["segments"][0]["start"] == 50.0
        assert result["segments"][0]["end"] == 53.5
        assert result["segments"][1]["start"] == 53.5
        assert result["segments"][1]["end"] == 58.0

    def test_preserves_text(self):
        mappings = [
            OffsetMapping(concat_start=0.0, concat_end=5.0, original_start=0.0, original_end=5.0),
        ]
        transcript = {
            "speech_id": "test",
            "language": "es",
            "segments": [{"start": 0.0, "end": 5.0, "text": "Texto original"}],
            "full_text": "Texto original",
        }

        result = remap_timestamps(transcript, mappings)

        assert result["segments"][0]["text"] == "Texto original"
        assert result["full_text"] == "Texto original"

    def test_adds_diarized_flag(self):
        mappings = [
            OffsetMapping(concat_start=0.0, concat_end=5.0, original_start=0.0, original_end=5.0),
        ]
        transcript = {
            "speech_id": "test",
            "language": "es",
            "segments": [],
            "full_text": "",
        }

        result = remap_timestamps(transcript, mappings)

        assert result["diarized"] is True

    def test_empty_mappings_returns_original(self):
        transcript = {
            "speech_id": "test",
            "language": "es",
            "segments": [{"start": 0.0, "end": 5.0, "text": "Hello"}],
            "full_text": "Hello",
        }

        result = remap_timestamps(transcript, [])

        assert result is transcript  # same object, no transformation

    def test_multi_mapping_segments(self):
        """Transcript segments spanning multiple offset mappings."""
        mappings = [
            OffsetMapping(concat_start=0.0, concat_end=10.0, original_start=100.0, original_end=110.0),
            OffsetMapping(concat_start=10.0, concat_end=20.0, original_start=200.0, original_end=210.0),
        ]
        transcript = {
            "speech_id": "test",
            "language": "es",
            "segments": [
                {"start": 5.0, "end": 15.0, "text": "Spans two mappings"},
            ],
            "full_text": "Spans two mappings",
        }

        result = remap_timestamps(transcript, mappings)

        # start=5.0 -> in first mapping -> 105.0
        assert result["segments"][0]["start"] == 105.0
        # end=15.0 -> in second mapping -> 205.0
        assert result["segments"][0]["end"] == 205.0


# ---------------------------------------------------------------------------
# Test extract_speaker_audio
# ---------------------------------------------------------------------------

class TestExtractSpeakerAudio:
    """Test audio extraction and offset mapping generation."""

    def test_correct_segments_selected(self, tmp_path):
        """Only target speaker's segments are extracted."""
        # Create a 10-second mono audio at 16kHz
        sample_rate = 16000
        duration = 10.0
        waveform = torch.randn(1, int(duration * sample_rate))
        audio_path = tmp_path / "test.wav"
        torchaudio.save(str(audio_path), waveform, sample_rate)

        segments = [
            SpeakerSegment(speaker="A", start=0.0, end=3.0),
            SpeakerSegment(speaker="B", start=3.0, end=6.0),
            SpeakerSegment(speaker="A", start=6.0, end=10.0),
        ]

        output_path = tmp_path / "extracted.wav"
        result_path, mappings = extract_speaker_audio(
            audio_path, segments, "A", output_path,
        )

        assert result_path == output_path
        assert len(mappings) == 2
        # First mapping: concat [0, 3] -> original [0, 3]
        assert mappings[0].concat_start == 0.0
        assert mappings[0].original_start == 0.0
        assert mappings[0].original_end == 3.0
        # Second mapping: concat [3, 7] -> original [6, 10]
        assert mappings[1].concat_start == 3.0
        assert mappings[1].original_start == 6.0
        assert mappings[1].original_end == 10.0

        # Verify output audio duration (~7 seconds: 3 + 4)
        extracted, sr = torchaudio.load(str(output_path))
        extracted_duration = extracted.shape[1] / sr
        assert abs(extracted_duration - 7.0) < 0.01

    def test_no_segments_for_speaker(self, tmp_path):
        """Returns original path if no segments match target speaker."""
        sample_rate = 16000
        waveform = torch.randn(1, int(5.0 * sample_rate))
        audio_path = tmp_path / "test.wav"
        torchaudio.save(str(audio_path), waveform, sample_rate)

        segments = [
            SpeakerSegment(speaker="B", start=0.0, end=5.0),
        ]

        output_path = tmp_path / "extracted.wav"
        result_path, mappings = extract_speaker_audio(
            audio_path, segments, "A", output_path,
        )

        assert result_path == audio_path  # falls back to original
        assert mappings == []

    def test_single_segment(self, tmp_path):
        """Single segment extraction works correctly."""
        sample_rate = 16000
        waveform = torch.randn(1, int(10.0 * sample_rate))
        audio_path = tmp_path / "test.wav"
        torchaudio.save(str(audio_path), waveform, sample_rate)

        segments = [
            SpeakerSegment(speaker="A", start=2.0, end=7.0),
        ]

        output_path = tmp_path / "extracted.wav"
        result_path, mappings = extract_speaker_audio(
            audio_path, segments, "A", output_path,
        )

        assert len(mappings) == 1
        assert mappings[0].concat_start == 0.0
        assert mappings[0].original_start == 2.0
        assert mappings[0].original_end == 7.0

        extracted, sr = torchaudio.load(str(output_path))
        extracted_duration = extracted.shape[1] / sr
        assert abs(extracted_duration - 5.0) < 0.01


# ---------------------------------------------------------------------------
# Test load_reference_embedding
# ---------------------------------------------------------------------------

class TestLoadReferenceEmbedding:
    """Test reference embedding loading."""

    def test_missing_file_raises_clear_error(self, tmp_path):
        missing_path = tmp_path / "nonexistent.npy"

        with pytest.raises(FileNotFoundError, match="Reference embedding not found"):
            load_reference_embedding(missing_path)

    def test_error_message_includes_create_command(self, tmp_path):
        missing_path = tmp_path / "nonexistent.npy"

        with pytest.raises(FileNotFoundError, match="create-reference"):
            load_reference_embedding(missing_path)

    def test_loads_valid_npy(self, tmp_path):
        embedding = np.random.randn(512).astype(np.float32)
        npy_path = tmp_path / "test_embedding.npy"
        np.save(npy_path, embedding)

        loaded = load_reference_embedding(npy_path)

        np.testing.assert_array_equal(loaded, embedding)


# ---------------------------------------------------------------------------
# Test DiarizationResult
# ---------------------------------------------------------------------------

class TestDiarizationResult:
    """Test DiarizationResult dataclass."""

    def test_creation(self):
        segments = [
            SpeakerSegment(speaker="A", start=0.0, end=30.0),
            SpeakerSegment(speaker="B", start=30.0, end=60.0),
        ]
        result = DiarizationResult(
            speech_id="test",
            speaker_segments=segments,
            num_speakers=2,
            target_speaker="A",
            confidence=0.85,
            duration_seconds=60.0,
            target_duration_seconds=30.0,
        )

        assert result.num_speakers == 2
        assert result.target_speaker == "A"
        assert result.confidence == 0.85
        assert result.target_duration_seconds == 30.0


# ---------------------------------------------------------------------------
# Test SpeakerSegment
# ---------------------------------------------------------------------------

class TestSpeakerSegment:
    """Test SpeakerSegment dataclass."""

    def test_duration_property(self):
        seg = SpeakerSegment(speaker="A", start=5.0, end=15.0)
        assert seg.duration == 10.0

    def test_zero_duration(self):
        seg = SpeakerSegment(speaker="A", start=5.0, end=5.0)
        assert seg.duration == 0.0
