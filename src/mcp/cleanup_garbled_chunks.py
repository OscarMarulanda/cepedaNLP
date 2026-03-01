"""One-time cleanup of garbled Whisper chunks in the database.

Fixes 5 chunks with Whisper hallucinations or duplicated sentences.
Cleans the text, re-embeds, and updates the DB.

Run:
    python -m src.mcp.cleanup_garbled_chunks
"""

import logging

from src.mcp.db import db_connection
from src.rag.embedder import embed_query

logger = logging.getLogger(__name__)


# Each entry: (chunk_id, description, cleaning function)
# Cleaning functions take the original text and return cleaned text.

def _clean_84(text: str) -> str:
    """Remove Whisper hallucination prefix (nonsense about 'estudiantes')."""
    marker = "Para en primer lugar tener claro a qué nos enfrentamos."
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[idx:]


def _clean_76(text: str) -> str:
    """Remove stuttered 'Como ustedes lo saben' and tail repetition."""
    # Remove the false start
    text = text.replace(
        "Como ustedes lo saben, la lucha de nuestro gobierno por los "
        "derechos sociales... Como ustedes lo saben, nuestra lucha y la ",
        "Como ustedes lo saben, nuestra lucha y la ",
    )
    # Remove the repeated tail (duplicated from chunk 77)
    tail = (
        " Y en el caso de los sectores de la economía, en el caso de "
        "los sectores de la economía, en el caso de los sectores "
        "populares, encabezados por mujeres y jóvenes de la primera "
        "línea, y luego vino la elección del gobierno progresista."
    )
    if text.endswith(tail):
        text = text[: -len(tail)]
    return text


def _clean_77(text: str) -> str:
    """Remove stuttered prefix 'en el caso de los sectores'."""
    text = text.replace(
        "Y en el caso de los sectores de la economía, en el caso de "
        "los sectores de la economía, en el caso de los sectores "
        "populares,",
        "Y en el caso de los sectores populares,",
    )
    return text


def _clean_108(text: str) -> str:
    """Remove duplicated IDEAM sentence and repeated 'Es por esto'."""
    # Remove one of the duplicated IDEAM sentences
    dup_sentence = (
        "La causa, según lo dice el Instituto de Hidrología, "
        "Meteorología y Estudios Ambientales, IDEAM, es un frente frío "
        "que ha producido precipitaciones. "
    )
    if text.startswith(dup_sentence + dup_sentence[:-1]):
        text = text[len(dup_sentence):]

    # Remove two of the three repeated "Es por esto" sentences
    es_por_esto = (
        "Es por esto que respaldamos con firmeza la decisión de nuestro "
        "compañero presidente Gustavo Petro de declarar el estado de "
        "emergencia para atender las graves consecuencias de las "
        "inundaciones. "
    )
    # Keep one, remove duplicates
    count = text.count(es_por_esto)
    if count > 1:
        text = text.replace(es_por_esto, "", count - 1)
    return text


def _clean_131(text: str) -> str:
    """Remove garbled tail after rally chant."""
    marker = "Cuidado con premieres"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[:idx].rstrip()


CLEANUPS = [
    (84, "Whisper hallucination prefix (estudiantes nonsense)", _clean_84),
    (76, "Stuttered intro + duplicated tail", _clean_76),
    (77, "Stuttered prefix (sectores de la economía)", _clean_77),
    (108, "Duplicated IDEAM sentence + repeated Es por esto", _clean_108),
    (131, "Garbled tail after rally chant", _clean_131),
]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with db_connection() as conn:
        for chunk_id, description, clean_fn in CLEANUPS:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_text FROM speech_chunks WHERE id = %s",
                    (chunk_id,),
                )
                row = cur.fetchone()
                if not row:
                    logger.warning("Chunk %d not found, skipping", chunk_id)
                    continue

                original = row[0]
                cleaned = clean_fn(original)

                if cleaned == original:
                    logger.info(
                        "Chunk %d: no changes needed (%s)", chunk_id, description
                    )
                    continue

                removed = len(original) - len(cleaned)
                logger.info(
                    "Chunk %d: cleaned (%s) — removed %d chars (%.0f%%)",
                    chunk_id,
                    description,
                    removed,
                    removed / len(original) * 100,
                )
                logger.info("  BEFORE: %s...", original[:80])
                logger.info("  AFTER:  %s...", cleaned[:80])

                # Re-embed
                new_embedding = embed_query(cleaned)

                # Update DB
                cur.execute(
                    """
                    UPDATE speech_chunks
                    SET chunk_text = %s, embedding = %s
                    WHERE id = %s
                    """,
                    (cleaned, new_embedding.tolist(), chunk_id),
                )

        conn.commit()
        logger.info("All chunks cleaned and re-embedded.")


if __name__ == "__main__":
    main()
