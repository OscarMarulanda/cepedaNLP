"""Response generator using Anthropic Claude API.

Takes retrieved speech chunks and generates a cited, neutral response
constrained by the project's style guide. Citations include YouTube
links with timestamps.
"""

import logging
import os

import anthropic
from dotenv import load_dotenv

from src.rag.retriever import RetrievalResult

logger = logging.getLogger(__name__)

load_dotenv()

# Model selection per CLAUDE.md:
# Dev/testing: claude-haiku-4-5-20251001
# Production: claude-sonnet-4-6 or claude-opus-4-6
DEV_MODEL = "claude-haiku-4-5-20251001"
PROD_MODEL = "claude-sonnet-4-6"


SYSTEM_PROMPT = """\
Eres un asistente informativo especializado en los discursos \
del candidato presidencial colombiano Iván Cepeda. Tu rol es responder preguntas \
sobre las propuestas, ideas y posiciones del candidato basándote EXCLUSIVAMENTE en \
los fragmentos de discursos que se te proporcionan como contexto.

REGLAS ESTRICTAS:
1. Responde SOLO con información presente en los fragmentos proporcionados.
2. SIEMPRE incluye citas con el título del discurso, la fecha, y el enlace al video \
entre paréntesis cuando esté disponible.
3. Si los fragmentos no contienen información relevante para la pregunta, di claramente: \
"No encontré referencias a ese tema en los discursos analizados."
4. NO inventes, NO especules, NO editoriales. Presenta lo que el candidato dijo, tal como lo dijo.
5. Mantén un tono neutral e informativo. No promuevas ni critiques al candidato.
6. Responde en español.
7. Si hay información relevante en múltiples discursos, sintetiza y cita cada fuente.

Formato de cita: (Discurso: "TÍTULO", fecha — [ver video](URL))
Si no hay URL disponible, usa solo: (Discurso: "TÍTULO", fecha)
"""


def _build_context_block(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks as context for the LLM."""
    if not results:
        return "No se encontraron fragmentos relevantes."

    blocks = []
    for i, r in enumerate(results, 1):
        date_str = r.speech_date or "fecha desconocida"
        location_str = f", {r.speech_location}" if r.speech_location else ""
        event_str = f", {r.speech_event}" if r.speech_event else ""
        link_str = f"\nEnlace: {r.youtube_link}" if r.youtube_link else ""

        header = (
            f"[Fragmento {i}] Discurso: \"{r.speech_title}\" "
            f"({date_str}{location_str}{event_str}) "
            f"[Relevancia: {r.similarity:.2f}]"
            f"{link_str}"
        )
        blocks.append(f"{header}\n{r.chunk_text}")

    return "\n\n---\n\n".join(blocks)


def generate(
    query: str,
    results: list[RetrievalResult],
    model: str | None = None,
    max_tokens: int = 1024,
) -> dict:
    """Generate a cited response from retrieved chunks.

    Args:
        query: The user's question.
        results: Retrieved chunks with citation metadata.
        model: Claude model to use. Defaults to DEV_MODEL.
        max_tokens: Maximum response length.

    Returns:
        Dict with keys: answer, model, usage, chunks_used.
    """
    if model is None:
        model = DEV_MODEL

    client = anthropic.Anthropic()

    context = _build_context_block(results)

    user_message = (
        f"CONTEXTO (fragmentos de discursos del candidato):\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"PREGUNTA DEL USUARIO: {query}"
    )

    logger.info(
        "Generating response with %s (%d chunks, %d context chars)",
        model,
        len(results),
        len(context),
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message},
        ],
    )

    answer = response.content[0].text

    logger.info(
        "Generated response: %d chars, usage: %d in / %d out tokens",
        len(answer),
        response.usage.input_tokens,
        response.usage.output_tokens,
    )

    return {
        "answer": answer,
        "model": model,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "chunks_used": len(results),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from src.rag.retriever import retrieve

    query = "¿Qué propone sobre el racismo?"
    results = retrieve(query, top_k=5)
    response = generate(query, results)
    print(f"\nRespuesta:\n{response['answer']}")
    print(f"\nModelo: {response['model']}")
    print(f"Tokens: {response['usage']}")
