"""Whisper-based speech transcription pipeline."""

import json
import logging
from pathlib import Path

import whisper

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

# Global model cache to avoid reloading
_model = None


def load_model(model_name: str = "large-v3") -> whisper.Whisper:
    """Load Whisper model (cached after first call)."""
    global _model
    if _model is None:
        logger.info("Loading Whisper model '%s'...", model_name)
        _model = whisper.load_model(model_name)
        logger.info("Whisper model loaded")
    return _model


def transcribe_audio(
    audio_path: Path,
    speech_id: str,
    output_dir: Path = RAW_DIR,
    model_name: str = "large-v3",
    language: str = "es",
) -> dict:
    """Transcribe an audio file using Whisper.

    Returns transcript dict with segments and full text.
    Saves to data/raw/{speech_id}.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{speech_id}.json"

    # Skip if already transcribed
    if output_path.exists():
        logger.info("Transcript already exists: %s", output_path)
        with open(output_path) as f:
            return json.load(f)

    model = load_model(model_name)

    logger.info("Transcribing: %s", audio_path.name)
    result = model.transcribe(
        str(audio_path),
        language=language,
        verbose=False,
    )

    # Build structured output
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    transcript = {
        "speech_id": speech_id,
        "language": result.get("language", language),
        "segments": segments,
        "full_text": result["text"].strip(),
    }

    # Save to file
    with open(output_path, "w") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    word_count = len(transcript["full_text"].split())
    logger.info(
        "Transcribed %s: %d segments, %d words",
        speech_id, len(segments), word_count,
    )
    return transcript


def load_text_speech(
    speech_id: str,
    text_file: Path,
    output_dir: Path = RAW_DIR,
) -> dict:
    """Load a pre-existing text speech into the same JSON format.

    For the ~6 text speeches from the website.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{speech_id}.json"

    if output_path.exists():
        logger.info("Text speech already loaded: %s", output_path)
        with open(output_path) as f:
            return json.load(f)

    text = text_file.read_text(encoding="utf-8").strip()

    # Split into paragraphs as "segments" (no timestamps for text speeches)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    segments = []
    for i, para in enumerate(paragraphs):
        segments.append({
            "start": None,
            "end": None,
            "text": para,
        })

    transcript = {
        "speech_id": speech_id,
        "language": "es",
        "segments": segments,
        "full_text": text,
    }

    with open(output_path, "w") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    word_count = len(text.split())
    logger.info("Loaded text speech %s: %d paragraphs, %d words", speech_id, len(segments), word_count)
    return transcript


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Test transcription on the downloaded test file
    test_audio = Path("data/audio/bGeWx5YWoro.mp3")
    if test_audio.exists():
        logger.info("Testing transcription on %s", test_audio)
        result = transcribe_audio(test_audio, "bGeWx5YWoro")
        print(f"\nSegments: {len(result['segments'])}")
        print(f"Words: {len(result['full_text'].split())}")
        print(f"\nFirst 500 chars:\n{result['full_text'][:500]}")
    else:
        logger.error("Test audio not found: %s", test_audio)
