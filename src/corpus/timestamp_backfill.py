"""Backfill sentence-level timestamps in the annotations table.

Maps each annotation sentence to its Whisper segment timestamp by
reconstructing the char-offset-to-segment index from the cleaned transcript.
Since full_text = " ".join(seg["text"] for seg in cleaned_segments),
the mapping is deterministic.
"""

import json
import logging
from pathlib import Path

from src.corpus.db_loader import get_connection

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def build_char_to_segment_map(
    segments: list[dict],
) -> list[tuple[int, int, float | None]]:
    """Build a list of (start_char, end_char, start_time) for each segment.

    Reconstructs the character offsets by simulating the join:
        full_text = " ".join(seg["text"] for seg in segments)
    Each segment's text occupies a contiguous range in full_text,
    with single-space separators between segments.
    """
    char_map = []
    offset = 0
    for seg in segments:
        text = seg["text"]
        start_char = offset
        end_char = offset + len(text)
        start_time = seg.get("start")
        char_map.append((start_char, end_char, start_time))
        offset = end_char + 1  # +1 for the space separator
    return char_map


def find_segment_for_position(
    char_pos: int,
    char_map: list[tuple[int, int, float | None]],
) -> float | None:
    """Find the segment start_time that covers a given character position."""
    for start_char, end_char, start_time in char_map:
        if start_char <= char_pos < end_char:
            return start_time
    return None


def match_sentences_to_timestamps(
    sentences: list[tuple[int, str]],
    full_text: str,
    char_map: list[tuple[int, int, float | None]],
) -> list[tuple[int, float | None]]:
    """Match annotation sentences to their Whisper segment timestamps.

    Args:
        sentences: list of (sentence_index, sentence_text)
        full_text: the cleaned transcript full_text
        char_map: output of build_char_to_segment_map

    Returns:
        list of (sentence_index, start_time) pairs
    """
    results = []
    search_start = 0

    for sent_idx, sent_text in sentences:
        # Find sentence position in full_text, searching forward from last match
        pos = full_text.find(sent_text, search_start)

        if pos == -1:
            # Fallback: search from beginning (in case of reordering)
            pos = full_text.find(sent_text)

        if pos == -1:
            # Sentence not found — try trimmed match
            trimmed = sent_text.strip()
            pos = full_text.find(trimmed, search_start)
            if pos == -1:
                pos = full_text.find(trimmed)

        if pos >= 0:
            start_time = find_segment_for_position(pos, char_map)
            results.append((sent_idx, start_time))
            search_start = pos + len(sent_text)
        else:
            logger.warning(
                "Could not find sentence %d in full_text: %.80s...",
                sent_idx, sent_text,
            )
            results.append((sent_idx, None))

    return results


def backfill_speech(conn, speech_id: int, youtube_id: str) -> dict:
    """Backfill timestamps for all sentences of a single speech.

    Returns stats dict with matched/unmatched/skipped counts.
    """
    processed_path = PROCESSED_DIR / f"{youtube_id}.json"
    if not processed_path.exists():
        logger.warning("No processed file for %s, skipping", youtube_id)
        return {"matched": 0, "unmatched": 0, "skipped": 0, "error": "no_file"}

    with open(processed_path) as f:
        processed = json.load(f)

    segments = processed.get("segments", [])
    full_text = processed.get("full_text", "")

    if not segments or not full_text:
        logger.warning("Empty segments/full_text for %s", youtube_id)
        return {"matched": 0, "unmatched": 0, "skipped": 0, "error": "empty"}

    # Build char-offset → segment map
    char_map = build_char_to_segment_map(segments)

    # Fetch annotations without timestamps
    cur = conn.cursor()
    cur.execute(
        """SELECT sentence_index, sentence_text
           FROM annotations
           WHERE speech_id = %s AND start_time IS NULL
           ORDER BY sentence_index""",
        (speech_id,),
    )
    sentences = cur.fetchall()

    if not sentences:
        return {"matched": 0, "unmatched": 0, "skipped": len(sentences)}

    # Match sentences to timestamps
    matches = match_sentences_to_timestamps(sentences, full_text, char_map)

    # Update DB
    matched = 0
    unmatched = 0
    for sent_idx, start_time in matches:
        if start_time is not None:
            cur.execute(
                """UPDATE annotations
                   SET start_time = %s
                   WHERE speech_id = %s AND sentence_index = %s""",
                (round(float(start_time)), speech_id, sent_idx),
            )
            matched += 1
        else:
            unmatched += 1

    conn.commit()
    return {"matched": matched, "unmatched": unmatched, "skipped": 0}


def backfill_all(conn) -> dict:
    """Backfill timestamps for all speeches in the DB.

    Returns aggregate stats.
    """
    cur = conn.cursor()
    cur.execute(
        """SELECT s.id, s.youtube_url, s.title
           FROM speeches s
           WHERE EXISTS (
               SELECT 1 FROM annotations a
               WHERE a.speech_id = s.id AND a.start_time IS NULL
           )
           ORDER BY s.id""",
    )
    speeches = cur.fetchall()

    if not speeches:
        logger.info("All annotations already have timestamps")
        return {"total_speeches": 0, "total_matched": 0, "total_unmatched": 0}

    logger.info("Backfilling timestamps for %d speeches", len(speeches))

    total_matched = 0
    total_unmatched = 0

    for speech_id, youtube_url, title in speeches:
        # Extract youtube_id from URL
        if youtube_url and "v=" in youtube_url:
            youtube_id = youtube_url.split("v=")[-1].split("&")[0]
        else:
            logger.warning(
                "No youtube_url for speech %d (%s), skipping", speech_id, title,
            )
            continue

        stats = backfill_speech(conn, speech_id, youtube_id)
        total_matched += stats["matched"]
        total_unmatched += stats["unmatched"]

        logger.info(
            "  [%d] %s: matched=%d, unmatched=%d",
            speech_id, title[:50], stats["matched"], stats["unmatched"],
        )

    logger.info(
        "Backfill complete: %d matched, %d unmatched across %d speeches",
        total_matched, total_unmatched, len(speeches),
    )
    return {
        "total_speeches": len(speeches),
        "total_matched": total_matched,
        "total_unmatched": total_unmatched,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    conn = get_connection()
    try:
        result = backfill_all(conn)
        print(f"\nResult: {result}")
    finally:
        conn.close()
