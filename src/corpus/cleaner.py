"""Spoken-language cleaning pipeline for Colombian Spanish transcripts."""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Colombian Spanish filler words/phrases
# Note: Some of these (e.g., "bueno", "pues") can be legitimate words in context.
# We use patterns that target filler usage (e.g., sentence-initial, isolated, repeated).
FILLER_PATTERNS = [
    # Standalone fillers (surrounded by commas, periods, or start/end of segment)
    r"(?<=[,.\s])\s*eh\s*(?=[,.\s])",
    r"(?<=[,.\s])\s*eeh\s*(?=[,.\s])",
    r"(?<=[,.\s])\s*ehh\s*(?=[,.\s])",
    # "o sea" as filler (not as "that is to say" in formal context)
    r"\bo sea\b(?=\s*,)",
    # "digamos" as filler
    r"(?<=[,.\s])\s*digamos\s*(?=[,.\s])",
    # "pues" as sentence-initial filler (not "pues" as "because/since")
    r"^pues\s*,\s*",
    r"(?<=\.\s)pues\s*,\s*",
    # "bueno" as sentence-initial filler
    r"^bueno\s*,\s*",
    r"(?<=\.\s)bueno\s*,\s*",
    # "entonces" as filler when followed by comma
    r"(?<=[,.\s])\s*entonces\s*,\s*",
    # "verdad" / "cierto" as tag questions
    r",\s*¿?verdad\??\s*(?=[,.])",
    r",\s*¿?cierto\??\s*(?=[,.])",
    # "mire" / "miren" as discourse markers
    r"(?<=[,.\s])\s*mire(?:n)?\s*(?=[,.\s])",
    # "este" as filler (isolated between commas)
    r",\s*este\s*,",
]

COMPILED_FILLERS = [re.compile(p, re.IGNORECASE) for p in FILLER_PATTERNS]


@dataclass
class CleaningReport:
    """Report of what was changed during cleaning."""

    speech_id: str
    original_word_count: int = 0
    cleaned_word_count: int = 0
    fillers_removed: int = 0
    repetitions_removed: int = 0
    normalizations_applied: int = 0
    changes: list[dict] = field(default_factory=list)

    @property
    def summary(self) -> str:
        removed = self.original_word_count - self.cleaned_word_count
        pct = (removed / self.original_word_count * 100) if self.original_word_count else 0
        return (
            f"[{self.speech_id}] {self.original_word_count} → {self.cleaned_word_count} words "
            f"({removed} removed, {pct:.1f}%). "
            f"Fillers: {self.fillers_removed}, Repetitions: {self.repetitions_removed}, "
            f"Normalizations: {self.normalizations_applied}"
        )


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form."""
    return unicodedata.normalize("NFC", text)


def normalize_punctuation(text: str) -> str:
    """Normalize quotes, dashes, and other punctuation."""
    # Smart quotes to straight quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # Em/en dashes to standard dash
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Ellipsis character to three dots
    text = text.replace("\u2026", "...")
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces, strip trailing whitespace."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def remove_fillers(text: str, report: CleaningReport) -> str:
    """Remove filler words/phrases based on context-aware patterns."""
    for pattern in COMPILED_FILLERS:
        matches = pattern.findall(text)
        if matches:
            report.fillers_removed += len(matches)
            for match in matches:
                report.changes.append({
                    "type": "filler_removed",
                    "original": match.strip(),
                })
            text = pattern.sub(" ", text)
    return text


def remove_repetitions(text: str, report: CleaningReport) -> str:
    """Remove immediate word/phrase repetitions (false starts).

    Catches patterns like "vamos a vamos a hacer" → "vamos a hacer"
    """
    # Single word repetition: "la la casa" → "la casa"
    pattern = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        report.repetitions_removed += len(matches)
        for match in matches:
            report.changes.append({
                "type": "repetition_removed",
                "original": f"{match} {match}",
            })
        text = pattern.sub(r"\1", text)

    # Two-word phrase repetition: "vamos a vamos a" → "vamos a"
    pattern2 = re.compile(r"\b(\w+\s+\w+)\s+\1\b", re.IGNORECASE)
    matches2 = pattern2.findall(text)
    if matches2:
        report.repetitions_removed += len(matches2)
        for match in matches2:
            report.changes.append({
                "type": "phrase_repetition_removed",
                "original": f"{match} {match}",
            })
        text = pattern2.sub(r"\1", text)

    return text


def clean_text(text: str, report: CleaningReport) -> str:
    """Apply full cleaning pipeline to a text string."""
    report.original_word_count += len(text.split())

    text = normalize_unicode(text)
    text = normalize_punctuation(text)
    report.normalizations_applied += 1

    text = remove_fillers(text, report)
    text = remove_repetitions(text, report)
    text = normalize_whitespace(text)

    report.cleaned_word_count += len(text.split())
    return text


def clean_transcript(
    speech_id: str,
    input_dir: Path = RAW_DIR,
    output_dir: Path = PROCESSED_DIR,
) -> tuple[dict, CleaningReport]:
    """Clean a raw transcript and save the result.

    Returns the cleaned transcript dict and a cleaning report.
    """
    input_path = input_dir / f"{speech_id}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{speech_id}.json"

    # Check if already cleaned
    if output_path.exists():
        logger.info("Cleaned transcript already exists: %s", output_path)
        with open(output_path) as f:
            cleaned = json.load(f)
        report = CleaningReport(speech_id=speech_id)
        return cleaned, report

    with open(input_path) as f:
        raw = json.load(f)

    report = CleaningReport(speech_id=speech_id)

    # Clean each segment
    cleaned_segments = []
    for seg in raw["segments"]:
        cleaned_text = clean_text(seg["text"], report)
        if cleaned_text:  # Skip empty segments after cleaning
            cleaned_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": cleaned_text,
            })

    # Rebuild full text from cleaned segments
    full_text = " ".join(seg["text"] for seg in cleaned_segments)

    cleaned = {
        "speech_id": speech_id,
        "language": raw["language"],
        "segments": cleaned_segments,
        "full_text": full_text,
    }

    with open(output_path, "w") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    logger.info("Cleaned %s: %s", speech_id, report.summary)
    return cleaned, report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Test on the transcribed speech
    test_id = "bGeWx5YWoro"
    test_path = RAW_DIR / f"{test_id}.json"

    if test_path.exists():
        cleaned, report = clean_transcript(test_id)
        print(f"\n{report.summary}")
        print(f"\nChanges made:")
        for change in report.changes:
            print(f"  [{change['type']}] \"{change['original']}\"")
        print(f"\nFirst 500 chars of cleaned text:")
        print(cleaned["full_text"][:500])
    else:
        logger.error("Raw transcript not found: %s", test_path)
