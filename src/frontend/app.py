"""Streamlit chat UI with Claude (Haiku) orchestration.

Claude acts as the orchestrator — it receives user messages, decides which
MCP tools to call, and generates cited responses from the results.

Run:
    streamlit run src/frontend/app.py
"""

import json
import logging
import sys
import time
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
class AssistantResponse:
    """Complete response from a Claude interaction."""

    text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


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


def _call_claude(
    client: anthropic.Anthropic, messages: list[dict],
) -> AssistantResponse:
    """Call Claude with tool-use loop, return text + tool call records."""
    # Strip non-API keys (e.g. tool_calls for visualization) before sending
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

        # If no tool use, extract text and return
        if response.stop_reason == "end_turn":
            text_parts = [
                block.text
                for block in response.content
                if block.type == "text"
            ]
            return AssistantResponse(
                text="\n".join(text_parts),
                tool_calls=tool_call_records,
            )

        # Handle tool use
        if response.stop_reason == "tool_use":
            # Add assistant message with all content blocks
            api_messages.append({
                "role": "assistant",
                "content": [
                    block.model_dump() for block in response.content
                ],
            })

            # Execute each tool call and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "matrix_rain_easter_egg":
                    tool_call_records.append(ToolCallRecord(
                        tool_name=block.name,
                        tool_input=block.input,
                        tool_result={},
                    ))
                    return AssistantResponse(
                        text="xD", tool_calls=tool_call_records,
                    )
                if block.type == "tool_use":
                    logger.info(
                        "Tool call: %s(%s)",
                        block.name,
                        json.dumps(block.input, ensure_ascii=False)[:100],
                    )
                    result_str = _execute_tool(block.name, block.input)

                    # Capture parsed result for visualization
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
            continue

        # Unexpected stop reason — return whatever text we have
        text_parts = [
            block.text
            for block in response.content
            if block.type == "text"
        ]
        return AssistantResponse(
            text="\n".join(text_parts) if text_parts else "",
            tool_calls=tool_call_records,
        )

    return AssistantResponse(
        text="Se alcanzó el límite de llamadas a herramientas. Intenta reformular tu pregunta.",
        tool_calls=tool_call_records,
    )


def _render_sidebar():
    """Render sidebar with corpus stats and info."""
    with st.sidebar:
        st.title("Iván Cepeda")
        st.caption("Asistente de discursos")
        st.divider()

        try:
            stats = get_corpus_stats()
            st.metric("Discursos", stats["speeches"])
            st.metric("Palabras", f"{stats['total_words']:,}")
            st.metric("Fragmentos indexados", stats["chunks"])
            st.metric("Entidades", stats["entities"])
            st.metric("Opiniones", stats["opinions"])
        except Exception:
            st.warning("No se pudo cargar las estadísticas del corpus.")

        st.divider()
        st.caption(
            f"Modelo: {MODEL}\n\n"
            f"Mensajes: {st.session_state.message_count}/{MAX_MESSAGES_PER_SESSION}"
        )

        if st.button("Limpiar conversación"):
            st.session_state.messages = []
            st.session_state.message_count = 0
            st.rerun()


def main():
    st.set_page_config(
        page_title="Asistente Cepeda",
        page_icon="🎤",
        layout="centered",
    )

    _init_session_state()

    client = anthropic.Anthropic()

    _render_sidebar()

    st.title("Asistente de Discursos de Iván Cepeda")
    st.caption(
        "Pregunta sobre las propuestas, ideas y posiciones del candidato. "
        "Las respuestas se basan exclusivamente en discursos reales."
    )

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                render_visualizations(msg["tool_calls"])
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Escribe tu pregunta..."):
        # Rate limiting
        if st.session_state.message_count >= MAX_MESSAGES_PER_SESSION:
            st.error(
                f"Se alcanzó el límite de {MAX_MESSAGES_PER_SESSION} mensajes "
                "por sesión. Limpia la conversación para continuar."
            )
            return

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.message_count += 1

        # Easter egg: intercept obvious tech attacks
        if detect_abuse(prompt):
            with st.chat_message("assistant"):
                components.html(MATRIX_RAIN_HTML, height=MATRIX_RAIN_HEIGHT)
                st.markdown("xD")
            st.session_state.messages.append({
                "role": "assistant", "content": "xD", "tool_calls": [],
            })
            return

        # Call Claude with tool loop
        with st.chat_message("assistant"):
            with st.spinner("Buscando en los discursos..."):
                response = _call_claude(
                    client,
                    st.session_state.messages,
                )
            easter_egg = any(
                tc.tool_name == "matrix_rain_easter_egg"
                for tc in response.tool_calls
            )
            if easter_egg:
                components.html(MATRIX_RAIN_HTML, height=MATRIX_RAIN_HEIGHT)
            elif response.tool_calls:
                render_visualizations([
                    {
                        "tool_name": tc.tool_name,
                        "tool_input": tc.tool_input,
                        "tool_result": tc.tool_result,
                    }
                    for tc in response.tool_calls
                ])
            st.markdown(response.text)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.text,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "tool_input": tc.tool_input,
                    "tool_result": tc.tool_result,
                }
                for tc in response.tool_calls
            ],
        })


if __name__ == "__main__":
    main()
