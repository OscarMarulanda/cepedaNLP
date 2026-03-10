"""Streamlit chat UI with Claude (Haiku) orchestration.

Claude acts as the orchestrator — it receives user messages, decides which
MCP tools to call, and generates cited responses from the results.

Run:
    streamlit run src/frontend/app.py
"""

import json
import logging
import sys
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
# regardless of the working directory Streamlit uses.
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import anthropic
import streamlit as st
import streamlit.components.v1 as components

from src.frontend.abuse_detector import (
    MATRIX_RAIN_HEIGHT,
    MATRIX_RAIN_HTML,
    detect_abuse,
)
from src.frontend.prompts import SYSTEM_PROMPT, TOOLS
from src.frontend.visualizations import render_visualizations
from src.mcp.server import (
    get_corpus_stats,
    get_opinions,
    get_speech_detail,
    get_speech_entities,
    list_speeches,
    retrieve_chunks,
    search_entities,
    submit_opinion,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """Record of a single tool call made during a Claude interaction."""

    tool_name: str
    tool_input: dict
    tool_result: dict


@dataclass
class ToolRoundResult:
    """Result of tool-use rounds before the final streamed response."""

    tool_calls: list[ToolCallRecord]
    api_messages: list[dict]
    direct_text: str | None = None
    is_easter_egg: bool = False


MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4096
MAX_MESSAGES_PER_SESSION = 30
MAX_TOOL_ROUNDS = 5

# Tool dispatch — maps tool names to their Python implementations
TOOL_DISPATCH = {
    "retrieve_chunks": retrieve_chunks,
    "list_speeches": list_speeches,
    "get_speech_detail": get_speech_detail,
    "search_entities": search_entities,
    "get_speech_entities": get_speech_entities,
    "get_corpus_stats": get_corpus_stats,
    "submit_opinion": submit_opinion,
    "get_opinions": get_opinions,
}


def _init_session_state():
    """Initialize Streamlit session state on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0


def _dump_content_block(block) -> dict:
    """Serialize an API content block, keeping only API-accepted fields.

    The SDK's ``model_dump()`` can include internal fields (e.g.
    ``parsed_output`` on ``TextBlock``) that the API rejects with
    ``Extra inputs are not permitted``.
    """
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    # Fallback for any other block type
    return block.model_dump()


def _execute_tool(name: str, input_args: dict) -> str:
    """Execute a tool by name and return JSON result."""
    func = TOOL_DISPATCH.get(name)
    if not func:
        return json.dumps({"error": f"Herramienta desconocida: {name}"})
    try:
        result = func(**input_args)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return json.dumps({"error": str(e)})


def _run_tool_rounds(
    client: anthropic.Anthropic, messages: list[dict],
) -> ToolRoundResult:
    """Run tool-use rounds, returning early after the last tool execution.

    When tools are called, returns with ``direct_text=None`` so the caller
    can stream the final text generation.  When no tools are needed (e.g.
    greetings), returns the text directly in ``direct_text``.
    """
    api_messages = [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]
    tool_call_records: list[ToolCallRecord] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=api_messages,
        )

        # No tool use — return text directly (fast, short response)
        if response.stop_reason == "end_turn":
            text_parts = [
                block.text
                for block in response.content
                if block.type == "text"
            ]
            return ToolRoundResult(
                tool_calls=tool_call_records,
                api_messages=api_messages,
                direct_text="\n".join(text_parts),
            )

        # Handle tool use
        if response.stop_reason == "tool_use":
            api_messages.append({
                "role": "assistant",
                "content": [
                    _dump_content_block(block) for block in response.content
                ],
            })

            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "matrix_rain_easter_egg":
                    tool_call_records.append(ToolCallRecord(
                        tool_name=block.name,
                        tool_input=block.input,
                        tool_result={},
                    ))
                    return ToolRoundResult(
                        tool_calls=tool_call_records,
                        api_messages=api_messages,
                        direct_text="xD",
                        is_easter_egg=True,
                    )
                if block.type == "tool_use":
                    logger.info(
                        "Tool call: %s(%s)",
                        block.name,
                        json.dumps(block.input, ensure_ascii=False)[:100],
                    )
                    result_str = _execute_tool(block.name, block.input)

                    try:
                        parsed = json.loads(result_str)
                    except json.JSONDecodeError:
                        parsed = {}
                    tool_call_records.append(ToolCallRecord(
                        tool_name=block.name,
                        tool_input=block.input,
                        tool_result=parsed,
                    ))

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            api_messages.append({
                "role": "user",
                "content": tool_results,
            })
            # Return for streaming — don't call create() for the next round
            return ToolRoundResult(
                tool_calls=tool_call_records,
                api_messages=api_messages,
                direct_text=None,
            )

        # Unexpected stop reason — return whatever text we have
        text_parts = [
            block.text
            for block in response.content
            if block.type == "text"
        ]
        return ToolRoundResult(
            tool_calls=tool_call_records,
            api_messages=api_messages,
            direct_text="\n".join(text_parts) if text_parts else "",
        )

    return ToolRoundResult(
        tool_calls=tool_call_records,
        api_messages=api_messages,
        direct_text="Se alcanzó el límite de llamadas a herramientas. Intenta reformular tu pregunta.",
    )


def _stream_response(
    client: anthropic.Anthropic, api_messages: list[dict],
) -> Generator[str, None, None]:
    """Stream the final text response after tool rounds complete.

    Handles the rare case where the streaming round unexpectedly triggers
    another tool call by executing the tool and continuing to stream.
    """
    for _ in range(MAX_TOOL_ROUNDS):
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=api_messages,
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()

        if final.stop_reason == "end_turn":
            return

        if final.stop_reason == "tool_use":
            # Rare: another tool round during streaming — handle silently
            api_messages.append({
                "role": "assistant",
                "content": [_dump_content_block(b) for b in final.content],
            })
            tool_results = []
            for block in final.content:
                if block.type == "tool_use":
                    result_str = _execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
            api_messages.append({"role": "user", "content": tool_results})
            continue


def _render_sidebar():
    """Render sidebar with corpus stats and info."""
    with st.sidebar:
        st.markdown("**IvanCepedaNLP**  \nAsistente de discursos")
        st.divider()

        try:
            stats = get_corpus_stats()
            c1, c2 = st.columns(2)
            c1.metric("Discursos", stats["speeches"])
            c2.metric("Palabras", f"{stats['total_words']:,}")
            c3, c4 = st.columns(2)
            c3.metric("Fragmentos", stats["chunks"])
            c4.metric("Entidades", stats["entities"])
            st.metric("Opiniones", stats["opinions"])
        except Exception:
            st.warning("No se pudo cargar las estadísticas del corpus.")

        st.divider()
        st.caption(
            f"Modelo: {MODEL} · "
            f"Mensajes: {st.session_state.message_count}/{MAX_MESSAGES_PER_SESSION}"
        )

        if st.button("Limpiar conversación"):
            st.session_state.messages = []
            st.session_state.message_count = 0
            st.rerun()

        st.divider()
        st.caption(
            "**Aviso:** Este asistente es un proyecto académico de "
            "procesamiento de lenguaje natural. No es una herramienta "
            "oficial de campaña ni representa la posición del candidato. "
            "Las respuestas se generan exclusivamente a partir de "
            "transcripciones de discursos públicos y pueden contener "
            "errores de transcripción o interpretación."
        )


def main():
    st.set_page_config(
        page_title="Asistente Cepeda",
        page_icon="🎤",
    )

    _init_session_state()

    client = anthropic.Anthropic()

    _render_sidebar()

    st.markdown(
        "<h3 style='margin:0 0 0.1rem 0'>Asistente de Discursos de Iván Cepeda</h3>"
        "<p style='margin:0 0 0.5rem 0;color:gray;font-size:0.85rem'>"
        "Pregunta sobre las propuestas, ideas y posiciones del candidato. "
        "Las respuestas se basan exclusivamente en discursos reales.</p>",
        unsafe_allow_html=True,
    )

    # Conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                render_visualizations(msg["tool_calls"])
            st.markdown(msg["content"])

    # Chat input
    prompt = st.chat_input("Escribe tu pregunta...")

    if prompt and st.session_state.message_count >= MAX_MESSAGES_PER_SESSION:
        st.error(
            f"Se alcanzó el límite de {MAX_MESSAGES_PER_SESSION} mensajes "
            "por sesión. Limpia la conversación para continuar."
        )
    elif prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.message_count += 1

        if detect_abuse(prompt):
            with st.chat_message("assistant"):
                components.html(MATRIX_RAIN_HTML, height=MATRIX_RAIN_HEIGHT)
                st.markdown("xD")
            st.session_state.messages.append({
                "role": "assistant", "content": "xD", "tool_calls": [],
            })
        else:
            with st.chat_message("assistant"):
                with st.spinner("Buscando en los discursos..."):
                    result = _run_tool_rounds(
                        client, st.session_state.messages,
                    )

                tool_call_dicts = [
                    {
                        "tool_name": tc.tool_name,
                        "tool_input": tc.tool_input,
                        "tool_result": tc.tool_result,
                    }
                    for tc in result.tool_calls
                ]

                if result.is_easter_egg:
                    components.html(
                        MATRIX_RAIN_HTML, height=MATRIX_RAIN_HEIGHT,
                    )
                    full_text = "xD"
                elif result.direct_text is not None:
                    if tool_call_dicts:
                        render_visualizations(tool_call_dicts)
                    st.markdown(result.direct_text)
                    full_text = result.direct_text
                else:
                    if tool_call_dicts:
                        render_visualizations(tool_call_dicts)
                    full_text = st.write_stream(
                        _stream_response(client, result.api_messages)
                    )

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text,
                "tool_calls": tool_call_dicts,
            })


if __name__ == "__main__":
    main()
