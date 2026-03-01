"""Database loader for speech data into PostgreSQL."""

import json
import logging
import os
from pathlib import Path

import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

try:
    from pgvector.psycopg2 import register_vector
    _HAS_PGVECTOR = True
except ImportError:
    _HAS_PGVECTOR = False

from src.pipeline.nlp_processor import SpeechAnalysis
from src.corpus.diarizer import DiarizationResult

logger = logging.getLogger(__name__)

load_dotenv()


def get_connection():
    """Create a PostgreSQL connection from .env config."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "cepeda_nlp"),
        user=os.getenv("DB_USER", "oscarm"),
        password=os.getenv("DB_PASSWORD", ""),
    )
    if _HAS_PGVECTOR:
        register_vector(conn)
    return conn


def speech_exists(conn, youtube_url: str | None, title: str) -> int | None:
    """Check if a speech is already loaded. Returns speech ID or None."""
    with conn.cursor() as cur:
        if youtube_url:
            cur.execute(
                "SELECT id FROM speeches WHERE youtube_url = %s",
                (youtube_url,),
            )
        else:
            cur.execute(
                "SELECT id FROM speeches WHERE title = %s",
                (title,),
            )
        row = cur.fetchone()
        return row[0] if row else None


def load_diarization(
    conn,
    speech_db_id: int,
    diarization_result: DiarizationResult,
) -> None:
    """Store diarization results in the speaker_segments table.

    Also updates the speeches table with diarization metadata.
    """
    with conn.cursor() as cur:
        # Insert speaker segments
        segment_rows = []
        for seg in diarization_result.speaker_segments:
            is_target = seg.speaker == diarization_result.target_speaker
            segment_rows.append((
                speech_db_id,
                seg.speaker,
                is_target,
                float(seg.start),
                float(seg.end),
                float(diarization_result.confidence) if is_target else None,
            ))

        if segment_rows:
            cur.executemany(
                """
                INSERT INTO speaker_segments (
                    speech_id, speaker_label, is_target,
                    start_time, end_time, confidence
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                segment_rows,
            )

        # Update speeches table with diarization metadata
        cur.execute(
            """
            UPDATE speeches
            SET diarized = TRUE, cepeda_duration_seconds = %s
            WHERE id = %s
            """,
            (float(diarization_result.target_duration_seconds), speech_db_id),
        )

    logger.info(
        "Loaded diarization for speech %d: %d segments, target=%s",
        speech_db_id,
        len(segment_rows),
        diarization_result.target_speaker,
    )


def load_speech(
    conn,
    manifest_entry: dict,
    raw_transcript: dict,
    cleaned_transcript: dict,
    nlp_analysis: SpeechAnalysis,
    diarization_result: DiarizationResult | None = None,
) -> int:
    """Load a fully processed speech into the database.

    Returns the speech ID (existing or newly created).
    """
    title = manifest_entry["title"]
    youtube_url = manifest_entry.get("url")

    # Check idempotency
    existing_id = speech_exists(conn, youtube_url, title)
    if existing_id:
        logger.info("Speech already loaded (id=%d): %s", existing_id, title)
        return existing_id

    try:
        with conn.cursor() as cur:
            # Insert speech
            cur.execute(
                """
                INSERT INTO speeches (
                    title, candidate, speech_date, location, event,
                    duration_seconds, youtube_url, raw_transcript,
                    cleaned_transcript, word_count, language
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    title,
                    "Iván Cepeda",
                    manifest_entry.get("upload_date"),
                    manifest_entry.get("location"),
                    manifest_entry.get("event"),
                    manifest_entry.get("duration_seconds"),
                    youtube_url,
                    raw_transcript.get("full_text"),
                    cleaned_transcript.get("full_text"),
                    len(cleaned_transcript.get("full_text", "").split()),
                    "es",
                ),
            )
            speech_id = cur.fetchone()[0]

            # Insert entities
            entity_rows = []
            for sent in nlp_analysis.sentences:
                for entity in sent.entities:
                    entity_rows.append((
                        speech_id,
                        entity.text,
                        entity.label,
                        entity.start_char,
                        entity.end_char,
                        sent.sentence_index,
                    ))

            if entity_rows:
                cur.executemany(
                    """
                    INSERT INTO entities (
                        speech_id, entity_text, entity_label,
                        start_char, end_char, sentence_index
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    entity_rows,
                )

            # Insert annotations (per sentence)
            annotation_rows = []
            for sent in nlp_analysis.sentences:
                sent_dict = sent.to_dict()
                annotation_rows.append((
                    speech_id,
                    sent.sentence_index,
                    sent.text,
                    Json(sent_dict["tokens"]),
                    Json(sent_dict["pos_tags"]),
                    Json(sent_dict["tokens"]),  # dep_parse uses same token structure
                    None,  # sentiment_score — added in Phase 2
                ))

            if annotation_rows:
                cur.executemany(
                    """
                    INSERT INTO annotations (
                        speech_id, sentence_index, sentence_text,
                        tokens, pos_tags, dep_parse, sentiment_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    annotation_rows,
                )

        # Store diarization results if available
        if diarization_result is not None:
            load_diarization(conn, speech_id, diarization_result)

        conn.commit()
        logger.info(
            "Loaded speech (id=%d): %s — %d entities, %d annotations",
            speech_id, title, len(entity_rows), len(annotation_rows),
        )
        return speech_id

    except Exception:
        conn.rollback()
        logger.exception("Failed to load speech: %s", title)
        raise


def get_corpus_stats(conn) -> dict:
    """Get summary statistics for the loaded corpus."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM speeches")
        num_speeches = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM entities")
        num_entities = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM annotations")
        num_annotations = cur.fetchone()[0]

        cur.execute("SELECT SUM(word_count) FROM speeches")
        total_words = cur.fetchone()[0] or 0

        cur.execute("SELECT COUNT(*) FROM speeches WHERE diarized = TRUE")
        num_diarized = cur.fetchone()[0]

    return {
        "speeches": num_speeches,
        "entities": num_entities,
        "annotations": num_annotations,
        "total_words": total_words,
        "diarized": num_diarized,
    }


def chunks_exist(conn, speech_id: int) -> bool:
    """Check if chunks have already been generated for a speech."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM speech_chunks WHERE speech_id = %s",
            (speech_id,),
        )
        return cur.fetchone()[0] > 0


def load_chunks(
    conn,
    speech_id: int,
    chunks: list,
    embeddings,
) -> int:
    """Store speech chunks with embeddings in the database.

    Args:
        conn: Database connection.
        speech_id: Database speech ID.
        chunks: List of Chunk objects from chunker.
        embeddings: numpy array of shape (n_chunks, embedding_dim).

    Returns:
        The number of chunks inserted.
    """
    if chunks_exist(conn, speech_id):
        logger.info("Chunks already exist for speech_id=%d, skipping", speech_id)
        return 0

    try:
        with conn.cursor() as cur:
            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO speech_chunks (
                        speech_id, chunk_index, chunk_text,
                        start_char, end_char, embedding,
                        sentence_start, sentence_end, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        speech_id,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.start_char,
                        chunk.end_char,
                        embedding.tolist(),
                        chunk.sentence_start,
                        chunk.sentence_end,
                        Json(chunk.metadata),
                    ),
                )
        conn.commit()
        logger.info(
            "Loaded %d chunks for speech_id=%d", len(chunks), speech_id
        )
        return len(chunks)

    except Exception:
        conn.rollback()
        logger.exception("Failed to load chunks for speech_id=%d", speech_id)
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from src.pipeline.nlp_processor import analyze_from_file

    test_id = "bGeWx5YWoro"
    raw_path = Path("data/raw") / f"{test_id}.json"
    cleaned_path = Path("data/processed") / f"{test_id}.json"

    if not raw_path.exists() or not cleaned_path.exists():
        logger.error("Test data not found")
        exit(1)

    # Load test data
    with open(raw_path) as f:
        raw = json.load(f)
    with open(cleaned_path) as f:
        cleaned = json.load(f)

    # Run NLP analysis
    logger.info("Running NLP analysis...")
    analysis = analyze_from_file(test_id)

    # Create manifest entry
    manifest_entry = {
        "title": "LA REBELIÓN ANTIRACISTA Y EL DESARROLLO DE TUMACO",
        "url": "https://www.youtube.com/watch?v=bGeWx5YWoro",
        "upload_date": "2026-02-24",
        "duration_seconds": 1209,
    }

    # Load into DB
    logger.info("Loading into database...")
    conn = get_connection()
    try:
        speech_id = load_speech(conn, manifest_entry, raw, cleaned, analysis)
        stats = get_corpus_stats(conn)
        print(f"\nCorpus stats: {stats}")
    finally:
        conn.close()
