"""Semantic chunker for speech transcripts.

Groups consecutive sentences into chunks of ~150-250 words,
respecting sentence boundaries from the NLP pipeline.
Includes timestamp mapping from raw Whisper segments.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


@dataclass
class Chunk:
    """A text chunk with metadata for RAG storage."""

    chunk_index: int
    text: str
    sentence_start: int  # first sentence_index in this chunk
    sentence_end: int  # last sentence_index in this chunk (inclusive)
    start_char: int  # character offset in full transcript
    end_char: int  # character offset in full transcript
    word_count: int
    metadata: dict = field(default_factory=dict)


def chunk_sentences(
    sentences: list[dict],
    target_words: int = 200,
    min_words: int = 100,
    max_words: int = 300,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Group sentences into chunks targeting ~200 words.

    Args:
        sentences: List of dicts with keys: sentence_index, sentence_text.
            Must be sorted by sentence_index.
        target_words: Target word count per chunk.
        min_words: Minimum words before considering a split.
        max_words: Hard maximum — forces a split even mid-accumulation.
        overlap_sentences: Number of trailing sentences to repeat in next chunk.

    Returns:
        List of Chunk objects ready for embedding and DB storage.
    """
    if not sentences:
        return []

    chunks: list[Chunk] = []
    buffer: list[dict] = []
    buffer_words = 0

    def _finalize_chunk() -> Chunk:
        text = " ".join(s["sentence_text"] for s in buffer)
        return Chunk(
            chunk_index=len(chunks),
            text=text,
            sentence_start=buffer[0]["sentence_index"],
            sentence_end=buffer[-1]["sentence_index"],
            start_char=0,  # computed later by compute_char_offsets
            end_char=0,
            word_count=len(text.split()),
        )

    for sent in sentences:
        words = len(sent["sentence_text"].split())
        buffer.append(sent)
        buffer_words += words

        if buffer_words >= target_words or buffer_words >= max_words:
            chunk = _finalize_chunk()
            chunks.append(chunk)

            # Seed next buffer with overlap sentences
            if overlap_sentences > 0 and len(buffer) > overlap_sentences:
                overlap = buffer[-overlap_sentences:]
                buffer = list(overlap)
                buffer_words = sum(
                    len(s["sentence_text"].split()) for s in buffer
                )
            else:
                buffer = []
                buffer_words = 0

    # Handle remaining sentences in buffer
    if buffer:
        # If runt chunk (<30 words) and we have previous chunks, merge
        if buffer_words < 30 and chunks:
            prev = chunks[-1]
            merged_text = prev.text + " " + " ".join(
                s["sentence_text"] for s in buffer
            )
            chunks[-1] = Chunk(
                chunk_index=prev.chunk_index,
                text=merged_text,
                sentence_start=prev.sentence_start,
                sentence_end=buffer[-1]["sentence_index"],
                start_char=0,
                end_char=0,
                word_count=len(merged_text.split()),
            )
        else:
            chunks.append(_finalize_chunk())

    return chunks


def compute_char_offsets(chunks: list[Chunk], full_text: str) -> list[Chunk]:
    """Compute start_char/end_char by finding chunk text in the full transcript.

    Uses sequential search starting from the end of the previous match
    to handle overlapping text correctly.
    """
    search_start = 0

    for chunk in chunks:
        # Find the first sentence's text to locate the chunk start
        # (more reliable than matching the full chunk which has joined spaces)
        idx = full_text.find(chunk.text[:80], search_start)
        if idx == -1:
            # Fallback: try finding just the first 40 chars
            idx = full_text.find(chunk.text[:40], search_start)
        if idx == -1:
            # Last resort: search from beginning
            idx = full_text.find(chunk.text[:40])

        if idx >= 0:
            chunk.start_char = idx
            chunk.end_char = idx + len(chunk.text)
            search_start = idx + 1
        else:
            logger.warning(
                "Could not find char offset for chunk %d", chunk.chunk_index
            )
            chunk.start_char = 0
            chunk.end_char = 0

    return chunks


def map_chunk_timestamps(
    chunks: list[Chunk], raw_segments: list[dict]
) -> list[Chunk]:
    """Map chunks to their earliest Whisper segment timestamp.

    For each chunk, finds the first raw segment whose text overlaps
    with the chunk text and stores its start time in chunk.metadata.

    Args:
        chunks: List of Chunk objects.
        raw_segments: List of dicts with keys: start, end, text
            (from data/raw/{id}.json "segments" array).

    Returns:
        The same chunks with metadata["start_time"] populated.
    """
    for chunk in chunks:
        chunk_lower = chunk.text[:100].lower()

        for seg in raw_segments:
            seg_text = seg.get("text", "").lower().strip()
            if not seg_text:
                continue

            # Check if the segment text appears in the chunk
            # or if the chunk starts with part of the segment
            if seg_text[:30] in chunk_lower or chunk_lower[:30] in seg_text:
                start_time = seg.get("start")
                if start_time is not None:
                    chunk.metadata["start_time"] = round(float(start_time))
                break

    return chunks


def chunk_speech_from_db(
    conn,
    speech_id: int,
    youtube_id: str | None = None,
    **kwargs,
) -> list[Chunk]:
    """Fetch sentences from annotations table and chunk them.

    Also maps timestamps from raw transcript if available.

    Args:
        conn: Database connection.
        speech_id: Database speech ID.
        youtube_id: YouTube video ID for loading raw transcript.
            If None, attempts to extract from youtube_url in speeches table.
        **kwargs: Passed to chunk_sentences().

    Returns:
        List of Chunk objects with char offsets and timestamps mapped.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT sentence_index, sentence_text
            FROM annotations
            WHERE speech_id = %s
            ORDER BY sentence_index
            """,
            (speech_id,),
        )
        rows = cur.fetchall()

    if not rows:
        logger.warning("No sentences found for speech_id=%d", speech_id)
        return []

    sentences = [
        {"sentence_index": row[0], "sentence_text": row[1]} for row in rows
    ]

    # Chunk the sentences
    chunks = chunk_sentences(sentences, **kwargs)

    # Compute character offsets using cleaned transcript from DB
    with conn.cursor() as cur:
        cur.execute(
            "SELECT cleaned_transcript, youtube_url FROM speeches WHERE id = %s",
            (speech_id,),
        )
        row = cur.fetchone()

    if row and row[0]:
        chunks = compute_char_offsets(chunks, row[0])

    # Map timestamps from raw transcript
    yt_url = row[1] if row else None
    if youtube_id is None and yt_url:
        # Extract video ID from URL
        if "v=" in yt_url:
            youtube_id = yt_url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in yt_url:
            youtube_id = yt_url.split("youtu.be/")[-1].split("?")[0]

    if youtube_id:
        raw_path = RAW_DIR / f"{youtube_id}.json"
        if raw_path.exists():
            with open(raw_path) as f:
                raw_data = json.load(f)
            raw_segments = raw_data.get("segments", [])
            if raw_segments:
                chunks = map_chunk_timestamps(chunks, raw_segments)
                logger.info(
                    "Mapped timestamps for %d chunks from %s",
                    len(chunks),
                    raw_path.name,
                )

    logger.info(
        "Chunked speech_id=%d: %d sentences -> %d chunks",
        speech_id,
        len(sentences),
        len(chunks),
    )

    return chunks


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from src.corpus.db_loader import get_connection

    conn = get_connection()
    try:
        chunks = chunk_speech_from_db(conn, speech_id=1)
        for c in chunks:
            ts = c.metadata.get("start_time", "?")
            print(
                f"Chunk {c.chunk_index}: sentences {c.sentence_start}-{c.sentence_end}, "
                f"{c.word_count} words, t={ts}s, "
                f"preview: {c.text[:80]}..."
            )
        print(f"\nTotal: {len(chunks)} chunks")
    finally:
        conn.close()
