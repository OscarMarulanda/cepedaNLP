"""Speaker diarization pipeline using pyannote-audio.

Identifies and extracts the target speaker (Iván Cepeda) from multi-speaker
audio recordings. Integrates into the corpus pipeline before transcription
to save Whisper processing time and ensure data quality.
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

REFERENCE_EMBEDDING_PATH = Path("data/reference_embedding.npy")
DIARIZATION_DIR = Path("data/diarization")

# Global model caches
_diarization_pipeline = None
_embedding_model = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpeakerSegment:
    """A contiguous time range attributed to a single speaker."""
    speaker: str   # cluster label, e.g. "SPEAKER_00"
    start: float   # seconds (original audio time)
    end: float     # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class OffsetMapping:
    """Maps concatenated audio time back to original audio time."""
    concat_start: float
    concat_end: float
    original_start: float
    original_end: float


@dataclass
class DiarizationResult:
    """Full diarization output for a single audio file."""
    speech_id: str
    speaker_segments: list[SpeakerSegment]
    num_speakers: int
    target_speaker: str | None    # Cepeda's cluster label, or None
    confidence: float             # cosine similarity of best match
    duration_seconds: float       # total audio duration
    target_duration_seconds: float  # how much Cepeda spoke


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

def _load_diarization_pipeline():
    """Load pyannote diarization pipeline (cached after first call).

    Uses pyannote/speaker-diarization-community-1 on CPU.
    """
    global _diarization_pipeline
    if _diarization_pipeline is None:
        from pyannote.audio import Pipeline

        logger.info("Loading pyannote diarization pipeline...")
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
        )
        _diarization_pipeline.to(torch.device("cpu"))
        logger.info("Diarization pipeline loaded (CPU)")
    return _diarization_pipeline


def _load_embedding_model():
    """Load pyannote embedding model (cached after first call).

    Used for speaker verification via cosine similarity.
    """
    global _embedding_model
    if _embedding_model is None:
        from pyannote.audio import Inference, Model

        logger.info("Loading pyannote embedding model...")
        model = Model.from_pretrained("pyannote/embedding")
        model = model.to(torch.device("cpu"))
        _embedding_model = Inference(model, window="whole")
        logger.info("Embedding model loaded (CPU)")
    return _embedding_model


# ---------------------------------------------------------------------------
# Audio format helper
# ---------------------------------------------------------------------------

def _ensure_wav(audio_path: Path) -> tuple[Path, bool]:
    """Convert audio to WAV if needed. MP3 causes sample count mismatches in pyannote.

    Returns (wav_path, is_temp) where is_temp indicates the file should be
    cleaned up after use.
    """
    if audio_path.suffix.lower() == ".wav":
        return audio_path, False

    wav_path = audio_path.with_suffix(".wav")
    if wav_path.exists():
        return wav_path, False

    logger.info("Converting %s to WAV for pyannote...", audio_path.name)
    waveform, sample_rate = torchaudio.load(str(audio_path))
    torchaudio.save(str(wav_path), waveform, sample_rate)
    return wav_path, True


# ---------------------------------------------------------------------------
# Reference embedding management
# ---------------------------------------------------------------------------

def create_reference_embedding(
    audio_path: Path,
    output_path: Path = REFERENCE_EMBEDDING_PATH,
    start: float | None = None,
    end: float | None = None,
) -> Path:
    """Extract speaker embedding from a reference audio clip.

    Uses pyannote/embedding to create a voice fingerprint from a clip
    where only Cepeda is speaking (~30-60 seconds ideal).

    Args:
        audio_path: Path to audio file (MP3, WAV, etc.)
        output_path: Where to save the .npy embedding.
        start: Optional start time in seconds to crop.
        end: Optional end time in seconds to crop.

    Returns:
        Path to the saved embedding file.
    """
    inference = _load_embedding_model()
    wav_path, is_temp = _ensure_wav(audio_path)

    try:
        if start is not None or end is not None:
            from pyannote.core import Segment
            excerpt = Segment(start or 0.0, end)
            embedding = inference.crop(str(wav_path), excerpt)
        else:
            embedding = inference(str(wav_path))
    finally:
        if is_temp and wav_path.exists():
            wav_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embedding)
    logger.info(
        "Reference embedding saved to %s (shape=%s)",
        output_path, embedding.shape,
    )
    return output_path


def load_reference_embedding(
    path: Path = REFERENCE_EMBEDDING_PATH,
) -> np.ndarray:
    """Load the pre-computed reference embedding from disk.

    Raises:
        FileNotFoundError: If the reference embedding file does not exist,
            with a message explaining how to create one.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Reference embedding not found at {path}. "
            "Create one with: python -m src.corpus.diarizer create-reference "
            "--audio <path-to-clean-cepeda-clip> [--start S] [--end E]"
        )
    embedding = np.load(path)
    logger.info("Loaded reference embedding from %s (shape=%s)", path, embedding.shape)
    return embedding


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------

def diarize_audio(
    audio_path: Path,
    speech_id: str,
    min_speakers: int = 1,
    max_speakers: int = 10,
) -> tuple[list[SpeakerSegment], dict[str, np.ndarray]]:
    """Run speaker diarization on an audio file.

    Returns:
        Tuple of (segments, speaker_embeddings_dict).
        speaker_embeddings_dict maps speaker labels to their embedding vectors,
        pre-computed by pyannote (avoids re-extracting embeddings later).
    """
    pipeline = _load_diarization_pipeline()
    wav_path, is_temp_wav = _ensure_wav(audio_path)

    logger.info("Diarizing %s...", audio_path.name)
    try:
        diarize_output = pipeline(
            str(wav_path),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    finally:
        if is_temp_wav and wav_path.exists():
            wav_path.unlink()
            logger.info("Cleaned up temp WAV: %s", wav_path.name)

    # pyannote 4.0+: pipeline returns DiarizeOutput dataclass
    annotation = diarize_output.speaker_diarization

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(SpeakerSegment(
            speaker=speaker,
            start=round(turn.start, 2),
            end=round(turn.end, 2),
        ))

    # Build speaker -> embedding dict from pre-computed embeddings
    speaker_embeddings: dict[str, np.ndarray] = {}
    labels = annotation.labels()
    if diarize_output.speaker_embeddings is not None and len(labels) > 0:
        for i, label in enumerate(labels):
            speaker_embeddings[label] = diarize_output.speaker_embeddings[i]

    speakers = {s.speaker for s in segments}
    total_duration = max(s.end for s in segments) if segments else 0.0
    logger.info(
        "Diarization complete: %d segments, %d speakers, %.1fs total",
        len(segments), len(speakers), total_duration,
    )
    return segments, speaker_embeddings


# ---------------------------------------------------------------------------
# Speaker identification
# ---------------------------------------------------------------------------

def identify_target_speaker(
    audio_path: Path,
    speaker_segments: list[SpeakerSegment],
    reference_embedding: np.ndarray,
    similarity_threshold: float = 0.25,
) -> tuple[str | None, float]:
    """Identify which speaker cluster matches the reference embedding.

    Extracts a 512-D embedding from each speaker's longest segment using
    the standalone pyannote/embedding model, then compares via cosine
    similarity against the reference.

    Args:
        audio_path: Path to the original audio file.
        speaker_segments: Diarization output.
        reference_embedding: Pre-computed Cepeda voice embedding (512-D).
        similarity_threshold: Minimum cosine similarity to accept a match.

    Returns:
        Tuple of (speaker_label, similarity_score). speaker_label is None
        if no speaker exceeds the threshold.
    """
    from pyannote.core import Segment
    from scipy.spatial.distance import cosine

    inference = _load_embedding_model()
    wav_path, is_temp_wav = _ensure_wav(audio_path)

    # Group segments by speaker and find longest segment for each
    speaker_longest: dict[str, SpeakerSegment] = {}
    for seg in speaker_segments:
        if seg.speaker not in speaker_longest or seg.duration > speaker_longest[seg.speaker].duration:
            speaker_longest[seg.speaker] = seg

    best_speaker = None
    best_similarity = -1.0

    try:
        for speaker, longest_seg in speaker_longest.items():
            excerpt = Segment(longest_seg.start, longest_seg.end)
            try:
                speaker_embedding = inference.crop(str(wav_path), excerpt)
            except Exception:
                logger.warning("Failed to extract embedding for %s, skipping", speaker)
                continue

            # Cosine similarity = 1 - cosine distance
            similarity = 1.0 - cosine(reference_embedding.flatten(), speaker_embedding.flatten())

            logger.info(
                "Speaker %s: similarity=%.3f (longest segment: %.1fs-%.1fs, %.1fs)",
                speaker, similarity,
                longest_seg.start, longest_seg.end, longest_seg.duration,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker
    finally:
        if is_temp_wav and wav_path.exists():
            wav_path.unlink()

    if best_similarity >= similarity_threshold:
        logger.info(
            "Target speaker identified: %s (similarity=%.3f)",
            best_speaker, best_similarity,
        )
        return best_speaker, best_similarity

    logger.warning(
        "No speaker exceeded threshold %.2f (best: %s at %.3f)",
        similarity_threshold, best_speaker, best_similarity,
    )
    return None, best_similarity


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_speaker_audio(
    audio_path: Path,
    speaker_segments: list[SpeakerSegment],
    target_speaker: str,
    output_path: Path,
) -> tuple[Path, list[OffsetMapping]]:
    """Extract target speaker's audio segments into a concatenated file.

    Uses torchaudio to read the original audio, extract only the segments
    attributed to the target speaker, concatenate them, and save to a
    new WAV file.

    Args:
        audio_path: Path to the original audio file.
        speaker_segments: Full diarization output.
        target_speaker: Which speaker cluster label to extract.
        output_path: Where to save the extracted audio.

    Returns:
        Tuple of (output_path, offset_mappings) where offset_mappings
        allow remapping Whisper timestamps back to original audio time.
    """
    waveform, sample_rate = torchaudio.load(str(audio_path))

    target_segs = [s for s in speaker_segments if s.speaker == target_speaker]
    target_segs.sort(key=lambda s: s.start)

    chunks = []
    mappings = []
    concat_offset = 0.0

    for seg in target_segs:
        start_sample = int(seg.start * sample_rate)
        end_sample = int(seg.end * sample_rate)
        # Clamp to waveform bounds
        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[1], end_sample)

        if end_sample <= start_sample:
            continue

        chunk = waveform[:, start_sample:end_sample]
        chunk_duration = chunk.shape[1] / sample_rate

        chunks.append(chunk)
        mappings.append(OffsetMapping(
            concat_start=round(concat_offset, 2),
            concat_end=round(concat_offset + chunk_duration, 2),
            original_start=seg.start,
            original_end=seg.end,
        ))
        concat_offset += chunk_duration

    if not chunks:
        logger.warning("No audio chunks extracted for speaker %s", target_speaker)
        return audio_path, []

    concatenated = torch.cat(chunks, dim=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), concatenated, sample_rate)

    total_duration = concatenated.shape[1] / sample_rate
    logger.info(
        "Extracted %d segments (%.1fs) for %s -> %s",
        len(chunks), total_duration, target_speaker, output_path.name,
    )
    return output_path, mappings


def remap_timestamps(
    transcript: dict,
    offset_mappings: list[OffsetMapping],
) -> dict:
    """Remap Whisper segment timestamps from concatenated to original audio time.

    For each Whisper segment, finds the offset mapping that contains its
    start time and adjusts both start and end to original audio time.

    Args:
        transcript: Whisper transcript dict with 'segments' list
            (each having 'start', 'end', 'text' keys).
        offset_mappings: Mappings from extract_speaker_audio().

    Returns:
        New transcript dict with remapped timestamps. Adds 'diarized' flag.
    """
    if not offset_mappings:
        return transcript

    remapped_segments = []
    for seg in transcript.get("segments", []):
        seg_start = seg["start"]
        seg_end = seg["end"]

        new_start = _remap_time(seg_start, offset_mappings)
        new_end = _remap_time(seg_end, offset_mappings)

        remapped_segments.append({
            "start": round(new_start, 2),
            "end": round(new_end, 2),
            "text": seg["text"],
        })

    result = {
        "speech_id": transcript["speech_id"],
        "language": transcript.get("language", "es"),
        "segments": remapped_segments,
        "full_text": transcript["full_text"],
        "diarized": True,
    }
    return result


def _remap_time(t: float, mappings: list[OffsetMapping]) -> float:
    """Remap a single timestamp from concatenated to original audio time.

    Finds the mapping interval containing t and applies the offset.
    If t falls between mappings (shouldn't happen for well-formed data),
    clamps to the nearest mapping boundary.
    """
    for m in mappings:
        if m.concat_start <= t <= m.concat_end:
            offset_within = t - m.concat_start
            return m.original_start + offset_within

    # Fallback: find the closest mapping
    if t <= mappings[0].concat_start:
        return mappings[0].original_start
    if t >= mappings[-1].concat_end:
        return mappings[-1].original_end

    # Between two mappings — snap to the end of the preceding one
    for i in range(len(mappings) - 1):
        if mappings[i].concat_end < t < mappings[i + 1].concat_start:
            return mappings[i].original_end

    return mappings[-1].original_end


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_diarization(
    audio_path: Path,
    speech_id: str,
    reference_embedding_path: Path = REFERENCE_EMBEDDING_PATH,
) -> tuple[Path, list[OffsetMapping], DiarizationResult]:
    """Full diarization pipeline: diarize -> identify -> extract.

    This is the main entry point called from pipeline_runner.

    Args:
        audio_path: Path to the audio file.
        speech_id: Unique speech identifier.
        reference_embedding_path: Path to the reference .npy embedding.

    Returns:
        Tuple of (audio_to_transcribe, offset_mappings, diarization_result).
        If the target speaker is identified, audio_to_transcribe is a temp
        WAV file with only their segments. If not identified, it's the
        original audio_path (fallback) with empty offset_mappings.
    """
    reference = load_reference_embedding(reference_embedding_path)

    # Step 1: Diarize
    segments, speaker_embeddings = diarize_audio(audio_path, speech_id)

    if not segments:
        logger.warning("No speaker segments found for %s", speech_id)
        return audio_path, [], DiarizationResult(
            speech_id=speech_id,
            speaker_segments=[],
            num_speakers=0,
            target_speaker=None,
            confidence=0.0,
            duration_seconds=0.0,
            target_duration_seconds=0.0,
        )

    speakers = {s.speaker for s in segments}
    total_duration = max(s.end for s in segments)

    # Step 2: Identify target speaker
    target_speaker, confidence = identify_target_speaker(
        audio_path, segments, reference,
    )

    # Step 3: Extract target speaker's audio
    if target_speaker is not None:
        target_segs = [s for s in segments if s.speaker == target_speaker]
        target_duration = sum(s.duration for s in target_segs)

        output_path = audio_path.parent / f"{speech_id}_speaker.wav"
        extracted_path, mappings = extract_speaker_audio(
            audio_path, segments, target_speaker, output_path,
        )

        result = DiarizationResult(
            speech_id=speech_id,
            speaker_segments=segments,
            num_speakers=len(speakers),
            target_speaker=target_speaker,
            confidence=confidence,
            duration_seconds=total_duration,
            target_duration_seconds=target_duration,
        )
        return extracted_path, mappings, result

    # Fallback: could not identify target speaker
    logger.warning(
        "Could not identify Cepeda in %s (best confidence=%.3f). "
        "Transcribing full audio.",
        speech_id, confidence,
    )
    return audio_path, [], DiarizationResult(
        speech_id=speech_id,
        speaker_segments=segments,
        num_speakers=len(speakers),
        target_speaker=None,
        confidence=confidence,
        duration_seconds=total_duration,
        target_duration_seconds=0.0,
    )


# ---------------------------------------------------------------------------
# Save diarization metadata
# ---------------------------------------------------------------------------

def save_diarization_result(
    speech_id: str,
    result: DiarizationResult,
    output_dir: Path = DIARIZATION_DIR,
) -> Path:
    """Save diarization result to JSON for debugging and auditing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{speech_id}.json"

    data = {
        "speech_id": result.speech_id,
        "num_speakers": result.num_speakers,
        "target_speaker": result.target_speaker,
        "confidence": float(round(result.confidence, 4)),
        "duration_seconds": float(round(result.duration_seconds, 2)),
        "target_duration_seconds": float(round(result.target_duration_seconds, 2)),
        "speaker_segments": [
            {
                "speaker": s.speaker,
                "start": float(s.start),
                "end": float(s.end),
            }
            for s in result.speaker_segments
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Diarization result saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python -m src.corpus.diarizer create-reference --audio PATH [--start S] [--end E]")
        print("  python -m src.corpus.diarizer test --audio PATH")
        print()
        print("Examples:")
        print("  python -m src.corpus.diarizer create-reference --audio data/audio/clip.mp3 --start 10 --end 50")
        print("  python -m src.corpus.diarizer test --audio data/audio/bGeWx5YWoro.mp3")
        sys.exit(0)

    command = args[0]

    if command == "create-reference":
        audio_path = None
        start = None
        end = None

        i = 1
        while i < len(args):
            if args[i] == "--audio" and i + 1 < len(args):
                audio_path = Path(args[i + 1])
                i += 2
            elif args[i] == "--start" and i + 1 < len(args):
                start = float(args[i + 1])
                i += 2
            elif args[i] == "--end" and i + 1 < len(args):
                end = float(args[i + 1])
                i += 2
            else:
                i += 1

        if audio_path is None:
            print("Error: --audio is required")
            sys.exit(1)
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)

        result_path = create_reference_embedding(audio_path, start=start, end=end)
        print(f"\nReference embedding saved to: {result_path}")

    elif command == "test":
        audio_path = None

        i = 1
        while i < len(args):
            if args[i] == "--audio" and i + 1 < len(args):
                audio_path = Path(args[i + 1])
                i += 2
            else:
                i += 1

        if audio_path is None:
            print("Error: --audio is required")
            sys.exit(1)
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)

        speech_id = audio_path.stem
        audio_to_transcribe, offsets, result = run_diarization(
            audio_path, speech_id,
        )

        print(f"\n{'=' * 60}")
        print(f"Speech: {speech_id}")
        print(f"Speakers: {result.num_speakers}")
        print(f"Target: {result.target_speaker} (confidence={result.confidence:.3f})")
        print(f"Duration: {result.duration_seconds:.1f}s total, "
              f"{result.target_duration_seconds:.1f}s target")

        if result.target_speaker:
            pct = result.target_duration_seconds / result.duration_seconds * 100
            print(f"Target speaks {pct:.1f}% of audio")
            print(f"Audio to transcribe: {audio_to_transcribe}")

        save_diarization_result(speech_id, result)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
