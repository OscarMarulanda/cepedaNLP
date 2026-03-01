"""Semantic retriever using pgvector cosine similarity.

Retrieves the most relevant speech chunks for a user query,
joining with the speeches table for citation metadata including
YouTube URL and timestamp.
"""

import logging
import re
from dataclasses import dataclass

from src.corpus.db_loader import get_connection
from src.rag.embedder import embed_query

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.3


@dataclass
class RetrievalResult:
    """A single retrieved chunk with citation metadata."""

    chunk_id: int
    speech_id: int
    chunk_index: int
    chunk_text: str
    similarity: float
    # Citation metadata from speeches table
    speech_title: str
    speech_date: str | None
    speech_location: str | None
    speech_event: str | None
    youtube_url: str | None
    start_time: int | None  # seconds into the video

    @property
    def youtube_link(self) -> str | None:
        """Build a timestamped YouTube link."""
        if not self.youtube_url:
            return None
        url = self.youtube_url
        # Normalize to watch URL format
        match = re.search(r"(?:v=|youtu\.be/)([\w-]+)", url)
        if not match:
            return url
        video_id = match.group(1)
        base = f"https://www.youtube.com/watch?v={video_id}"
        if self.start_time is not None:
            return f"{base}&t={self.start_time}"
        return base


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    conn=None,
) -> list[RetrievalResult]:
    """Retrieve top-k relevant chunks for a query.

    Args:
        query: User question in Spanish.
        top_k: Number of results to return.
        threshold: Minimum cosine similarity (0-1).
        conn: Optional DB connection. If None, creates a new one.

    Returns:
        List of RetrievalResult sorted by similarity (descending).
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        query_embedding = embed_query(query)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    sc.id,
                    sc.speech_id,
                    sc.chunk_index,
                    sc.chunk_text,
                    1 - (sc.embedding <=> %s::vector) AS similarity,
                    s.title,
                    s.speech_date,
                    s.location,
                    s.event,
                    s.youtube_url,
                    (sc.metadata->>'start_time')::int
                FROM speech_chunks sc
                JOIN speeches s ON s.id = sc.speech_id
                WHERE sc.embedding IS NOT NULL
                ORDER BY sc.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding.tolist(), query_embedding.tolist(), top_k),
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            sim = float(row[4])
            if sim < threshold:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=row[0],
                    speech_id=row[1],
                    chunk_index=row[2],
                    chunk_text=row[3],
                    similarity=sim,
                    speech_title=row[5],
                    speech_date=str(row[6]) if row[6] else None,
                    speech_location=row[7],
                    speech_event=row[8],
                    youtube_url=row[9],
                    start_time=row[10],
                )
            )

        logger.info(
            "Retrieved %d chunks for query: '%s' (top sim=%.3f)",
            len(results),
            query[:50],
            results[0].similarity if results else 0.0,
        )
        return results

    finally:
        if close_conn:
            conn.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    query = "reforma agraria y campesinos"
    results = retrieve(query, top_k=5)
    for r in results:
        print(
            f"\n[{r.similarity:.3f}] {r.speech_title} ({r.speech_date})"
        )
        print(f"  Link: {r.youtube_link}")
        print(f"  {r.chunk_text[:150]}...")
