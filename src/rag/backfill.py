"""Backfill script: chunk and embed all existing speeches.

Run this once after setting up the RAG system to process
speeches that were loaded before the chunk+embed step
was added to the pipeline.

Usage:
    python -m src.rag.backfill
"""

import logging

from src.corpus.db_loader import chunks_exist, get_connection, load_chunks
from src.rag.chunker import chunk_speech_from_db
from src.rag.embedder import embed_texts

logger = logging.getLogger(__name__)


def backfill_all(conn=None) -> dict:
    """Chunk and embed all speeches that don't have chunks yet.

    Returns summary statistics.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, title FROM speeches ORDER BY id")
            speeches = cur.fetchall()

        processed = 0
        skipped = 0
        total_chunks = 0

        for speech_id, title in speeches:
            if chunks_exist(conn, speech_id):
                logger.info(
                    "Skipping (already has chunks): [%d] %s", speech_id, title
                )
                skipped += 1
                continue

            logger.info("Processing: [%d] %s", speech_id, title)
            chunks = chunk_speech_from_db(conn, speech_id)

            if not chunks:
                logger.warning("No sentences found for speech_id=%d", speech_id)
                continue

            chunk_texts = [c.text for c in chunks]
            embeddings = embed_texts(chunk_texts)
            n = load_chunks(conn, speech_id, chunks, embeddings)

            total_chunks += n
            processed += 1
            logger.info("  -> %d chunks created", n)

        return {
            "speeches_processed": processed,
            "speeches_skipped": skipped,
            "total_chunks_created": total_chunks,
        }

    finally:
        if close_conn:
            conn.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    result = backfill_all()
    print(f"\nBackfill complete: {result}")
