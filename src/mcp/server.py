"""FastMCP server with 8 data-fetcher tools (6 read-only + 2 opinion tools).

Exposes the RAG corpus (speeches, entities, chunks) via MCP tools.
All tools use parameterized SQL and return plain dicts — no LLM calls.

Run standalone:
    python -m src.mcp.server          # STDIO (for Claude Desktop)
    fastmcp run src/mcp/server.py --transport sse --port 8000  # HTTP/SSE
"""

import logging
import re
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from src.mcp.db import db_connection
from src.rag.embedder import embed_query

logger = logging.getLogger(__name__)

mcp = FastMCP("CepedaNLP")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _youtube_link(youtube_url: str | None, start_time: int | None) -> str | None:
    """Build a timestamped YouTube link (same logic as retriever.py)."""
    if not youtube_url:
        return None
    match = re.search(r"(?:v=|youtu\.be/)([\w-]+)", youtube_url)
    if not match:
        return youtube_url
    video_id = match.group(1)
    base = f"https://www.youtube.com/watch?v={video_id}"
    if start_time is not None:
        return f"{base}&t={start_time}"
    return base


# ---------------------------------------------------------------------------
# Tool 1: retrieve_chunks
# ---------------------------------------------------------------------------

@mcp.tool
def retrieve_chunks(
    query: Annotated[str, Field(description="Pregunta en español", max_length=500)],
    top_k: Annotated[int, Field(description="Número de fragmentos", ge=1, le=20)] = 5,
) -> list[dict]:
    """Busca fragmentos de discursos por similitud semántica.

    Devuelve los fragmentos más relevantes con metadatos de citación
    (título, fecha, enlace YouTube con timestamp, puntaje de similitud).
    """
    query_embedding = embed_query(query)

    with db_connection() as conn:
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
        similarity = float(row[4])
        if similarity < 0.3:
            continue
        results.append({
            "chunk_id": row[0],
            "speech_id": row[1],
            "chunk_index": row[2],
            "chunk_text": row[3],
            "similarity": round(similarity, 3),
            "speech_title": row[5],
            "speech_date": str(row[6]) if row[6] else None,
            "speech_location": row[7],
            "speech_event": row[8],
            "youtube_link": _youtube_link(row[9], row[10]),
        })

    logger.info(
        "retrieve_chunks: %d results for '%s'",
        len(results),
        query[:50],
    )
    return results


# ---------------------------------------------------------------------------
# Tool 2: list_speeches
# ---------------------------------------------------------------------------

@mcp.tool
def list_speeches() -> list[dict]:
    """Lista todos los discursos del corpus con metadatos.

    Devuelve id, título, fecha, ubicación, evento, conteo de palabras
    y URL de YouTube para cada discurso.
    """
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, speech_date, location, event,
                       word_count, youtube_url
                FROM speeches
                ORDER BY speech_date DESC NULLS LAST
                """
            )
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "title": row[1],
            "speech_date": str(row[2]) if row[2] else None,
            "location": row[3],
            "event": row[4],
            "word_count": row[5],
            "youtube_url": row[6],
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Tool 3: get_speech_detail
# ---------------------------------------------------------------------------

@mcp.tool
def get_speech_detail(
    speech_id: Annotated[int, Field(description="ID del discurso", ge=1)],
) -> dict:
    """Obtiene los detalles completos de un discurso específico.

    Incluye el texto limpio, conteo de entidades, conteo de fragmentos
    y duración.
    """
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    s.id, s.title, s.speech_date, s.location, s.event,
                    s.word_count, s.youtube_url, s.duration_seconds,
                    s.cleaned_transcript,
                    (SELECT COUNT(*) FROM entities e WHERE e.speech_id = s.id),
                    (SELECT COUNT(*) FROM speech_chunks sc WHERE sc.speech_id = s.id)
                FROM speeches s
                WHERE s.id = %s
                """,
                (speech_id,),
            )
            row = cur.fetchone()

    if not row:
        return {"error": f"No se encontró el discurso con id={speech_id}"}

    return {
        "id": row[0],
        "title": row[1],
        "speech_date": str(row[2]) if row[2] else None,
        "location": row[3],
        "event": row[4],
        "word_count": row[5],
        "youtube_url": row[6],
        "duration_seconds": row[7],
        "cleaned_transcript": row[8],
        "entity_count": row[9],
        "chunk_count": row[10],
    }


# ---------------------------------------------------------------------------
# Tool 4: search_entities
# ---------------------------------------------------------------------------

@mcp.tool
def search_entities(
    entity_text: Annotated[
        str | None,
        Field(description="Texto de la entidad (búsqueda parcial)", max_length=200),
    ] = None,
    entity_label: Annotated[
        str | None,
        Field(description="Etiqueta NER: PER, ORG, LOC, MISC", max_length=20),
    ] = None,
    limit: Annotated[
        int,
        Field(description="Máximo número de resultados (1-50)", ge=1, le=50),
    ] = 10,
) -> list[dict]:
    """Busca entidades nombradas en todo el corpus.

    Requiere al menos un parámetro (entity_text o entity_label).
    Devuelve las entidades con su frecuencia y los discursos donde aparecen.
    """
    if not entity_text and not entity_label:
        return [{"error": "Se requiere al menos entity_text o entity_label"}]

    conditions = []
    params: list = []

    if entity_text:
        conditions.append("e.entity_text ILIKE %s")
        params.append(f"%{entity_text}%")
    if entity_label:
        conditions.append("e.entity_label = %s")
        params.append(entity_label.upper())

    where_clause = " AND ".join(conditions)

    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    e.entity_text,
                    e.entity_label,
                    COUNT(*) AS mention_count,
                    ARRAY_AGG(DISTINCT s.title) AS speech_titles
                FROM entities e
                JOIN speeches s ON s.id = e.speech_id
                WHERE {where_clause}
                GROUP BY e.entity_text, e.entity_label
                ORDER BY mention_count DESC
                LIMIT %s
                """,
                [*params, limit],
            )
            rows = cur.fetchall()

    return [
        {
            "entity_text": row[0],
            "entity_label": row[1],
            "mention_count": row[2],
            "speech_titles": row[3],
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Tool 5: get_speech_entities
# ---------------------------------------------------------------------------

@mcp.tool
def get_speech_entities(
    speech_id: Annotated[int, Field(description="ID del discurso", ge=1)],
) -> dict:
    """Obtiene todas las entidades nombradas de un discurso, agrupadas por etiqueta.

    Devuelve las entidades organizadas por tipo (PER, ORG, LOC, MISC)
    con su frecuencia en el discurso.
    """
    with db_connection() as conn:
        with conn.cursor() as cur:
            # Verify speech exists
            cur.execute("SELECT title FROM speeches WHERE id = %s", (speech_id,))
            speech = cur.fetchone()
            if not speech:
                return {"error": f"No se encontró el discurso con id={speech_id}"}

            cur.execute(
                """
                SELECT entity_text, entity_label, COUNT(*) AS mentions
                FROM entities
                WHERE speech_id = %s
                GROUP BY entity_text, entity_label
                ORDER BY entity_label, mentions DESC
                """,
                (speech_id,),
            )
            rows = cur.fetchall()

    entities_by_label: dict[str, list[dict]] = {}
    for row in rows:
        label = row[1]
        if label not in entities_by_label:
            entities_by_label[label] = []
        entities_by_label[label].append({
            "entity_text": row[0],
            "mentions": row[2],
        })

    return {
        "speech_id": speech_id,
        "speech_title": speech[0],
        "entities": entities_by_label,
        "total_entities": sum(len(v) for v in entities_by_label.values()),
    }


# ---------------------------------------------------------------------------
# Tool 6: get_corpus_stats
# ---------------------------------------------------------------------------

@mcp.tool
def get_corpus_stats() -> dict:
    """Obtiene estadísticas generales del corpus de discursos.

    Devuelve totales de discursos, palabras, entidades, anotaciones
    y fragmentos indexados.
    """
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM speeches")
            num_speeches = cur.fetchone()[0]

            cur.execute("SELECT COALESCE(SUM(word_count), 0) FROM speeches")
            total_words = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM entities")
            num_entities = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM annotations")
            num_annotations = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM speech_chunks")
            num_chunks = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM user_opinions")
            num_opinions = cur.fetchone()[0]

    return {
        "speeches": num_speeches,
        "total_words": total_words,
        "entities": num_entities,
        "annotations": num_annotations,
        "chunks": num_chunks,
        "opinions": num_opinions,
    }


# ---------------------------------------------------------------------------
# Tool 7: submit_opinion
# ---------------------------------------------------------------------------

@mcp.tool
def submit_opinion(
    opinion_text: Annotated[
        str,
        Field(
            description="Opinión del usuario sobre las propuestas de Iván Cepeda",
            max_length=2000,
        ),
    ],
    will_win: Annotated[
        bool,
        Field(description="¿Cree el usuario que Cepeda va a ganar las elecciones?"),
    ],
) -> dict:
    """Guarda la opinión de un usuario sobre el candidato.

    Registra el texto de la opinión y si el usuario cree que el candidato
    ganará las elecciones. Devuelve confirmación con el ID de la opinión.
    """
    cleaned_text = opinion_text.strip()
    if not cleaned_text:
        return {"error": "La opinión no puede estar vacía"}

    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_opinions (opinion_text, will_win)
                VALUES (%s, %s)
                RETURNING id, created_at
                """,
                (cleaned_text, will_win),
            )
            row = cur.fetchone()
            conn.commit()

    logger.info("submit_opinion: saved opinion id=%d", row[0])
    return {
        "opinion_id": row[0],
        "created_at": str(row[1]),
        "message": "Opinión registrada exitosamente",
    }


# ---------------------------------------------------------------------------
# Tool 8: get_opinions
# ---------------------------------------------------------------------------

@mcp.tool
def get_opinions(
    will_win: Annotated[
        bool | None,
        Field(
            description="Filtrar por respuesta electoral: true/false, o null para todas",
        ),
    ] = None,
    limit: Annotated[
        int,
        Field(description="Número máximo de opiniones a devolver", ge=1, le=100),
    ] = 20,
) -> dict:
    """Consulta las opiniones de los usuarios sobre el candidato.

    Devuelve opiniones guardadas con estadísticas de resumen
    (total, porcentaje que cree que ganará).
    """
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM user_opinions")
            total = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM user_opinions WHERE will_win = TRUE"
            )
            total_yes = cur.fetchone()[0]

            if will_win is not None:
                cur.execute(
                    """
                    SELECT id, opinion_text, will_win, created_at
                    FROM user_opinions
                    WHERE will_win = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (will_win, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, opinion_text, will_win, created_at
                    FROM user_opinions
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            rows = cur.fetchall()

    opinions = [
        {
            "id": row[0],
            "opinion_text": row[1],
            "will_win": row[2],
            "created_at": str(row[3]),
        }
        for row in rows
    ]

    logger.info(
        "get_opinions: %d results (filter=%s, limit=%d)",
        len(opinions),
        will_win,
        limit,
    )
    return {
        "total_opinions": total,
        "total_will_win": total_yes,
        "will_win_pct": round(total_yes / total * 100, 1) if total > 0 else 0,
        "opinions": opinions,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    mcp.run()
