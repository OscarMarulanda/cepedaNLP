"""Inline visualizations for tool results in the Streamlit chat UI.

Each viz function receives the parsed tool result (dict or list[dict])
and renders Plotly charts inside the current st.chat_message block.
"""

from collections.abc import Callable
from uuid import uuid4

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# NER label color palette (consistent across all entity charts)
LABEL_COLORS = {
    "PER": "#636EFA",
    "LOC": "#00CC96",
    "ORG": "#EF553B",
    "MISC": "#AB63FA",
}

# Hardcoded coordinates for Colombian locations found in the corpus.
# Departments use their capital-city coordinates; regions use approximate center.
# "Colombia" excluded (country-level, not a point on the map).
COLOMBIAN_COORDS: dict[str, tuple[float, float]] = {
    "Bogotá": (4.7110, -74.0721),
    "Medellín": (6.2518, -75.5636),
    "Cali": (3.4516, -76.5320),
    "Cartagena": (10.3910, -75.5144),
    "Buenaventura": (3.8801, -77.0192),
    "Tumaco": (1.7986, -78.7647),
    "Puerto Tejada": (3.2358, -76.4172),
    "Puerto Gaitán": (4.3137, -72.0829),
    "Puerto Asís": (0.5044, -76.5008),
    "Soacha": (4.5862, -74.2173),
    "Quibdó": (5.6919, -76.6583),
    "Mocoa": (1.1494, -76.6519),
    "Antioquia": (6.2518, -75.5636),
    "Cauca": (2.4448, -76.6147),
    "Meta": (4.1420, -73.6266),
    "Putumayo": (1.1494, -76.6519),
    "Chocó": (5.6919, -76.6583),
    "Nariño": (1.2136, -77.2811),
    "Urabá": (7.8830, -76.6350),
    "Catatumbo": (8.5000, -73.0000),
    "Pasacaballos": (10.2905, -75.5082),
}


def _render_colombia_map(locations: list[dict]) -> None:
    """Render a bubble map of Colombia for LOC entities with known coordinates.

    Each item in *locations* must have ``entity_text`` and a count field
    (either ``mention_count`` or ``mentions``).  Locations without hardcoded
    coordinates are silently skipped.
    """
    rows = []
    for loc in locations:
        name = loc["entity_text"]
        coords = COLOMBIAN_COORDS.get(name)
        if coords is None:
            continue
        count = loc.get("mention_count") or loc.get("mentions", 0)
        rows.append({"name": name, "lat": coords[0], "lon": coords[1], "count": count})

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        size="count",
        hover_name="name",
        text="name",
        size_max=30,
        scope="south america",
        labels={"count": "Menciones"},
    )
    fig.update_traces(
        marker_color=LABEL_COLORS["LOC"],
        textposition="top center",
        textfont_size=10,
    )
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        geo=dict(
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showocean=True,
            oceancolor="rgb(204, 229, 255)",
            showcountries=True,
            projection_scale=4.5,
            center=dict(lat=4.5, lon=-74.0),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=str(uuid4()))


# ---------------------------------------------------------------------------
# Registry: tool_name -> viz function (None = no chart for this tool)
# ---------------------------------------------------------------------------

VIZ_REGISTRY: dict[str, Callable | None] = {
    "retrieve_chunks": lambda r: viz_retrieve_chunks(r),
    "list_speeches": lambda r: viz_list_speeches(r),
    "search_entities": lambda r: viz_search_entities(r),
    "get_speech_entities": lambda r: viz_get_speech_entities(r),
    "get_opinions": lambda r: viz_get_opinions(r),
    "get_corpus_stats": None,
    "get_speech_detail": None,
    "submit_opinion": None,
}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Chart implementations
# ---------------------------------------------------------------------------

def viz_retrieve_chunks(result: list[dict]) -> None:
    """Horizontal bar chart of chunk similarity scores."""
    if not result:
        return
    df = pd.DataFrame(result)
    df["label"] = df.apply(
        lambda r: f"{r['speech_title'][:30]}... (#{r['chunk_index']})", axis=1,
    )
    df = df.sort_values("similarity", ascending=True)

    fig = px.bar(
        df, x="similarity", y="label", orientation="h",
        labels={"similarity": "Similitud", "label": "Fragmento"},
        color="similarity",
        color_continuous_scale="Blues",
        range_x=[0, 1],
    )
    fig.update_layout(
        height=max(200, len(df) * 40),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        yaxis_title=None,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=str(uuid4()))


def viz_search_entities(result: list[dict]) -> None:
    """Horizontal bar chart of entity mentions, color-coded by NER label."""
    if not result:
        return
    # Skip if the result is an error list (single dict with "error" key)
    if len(result) == 1 and "error" in result[0]:
        return
    df = pd.DataFrame(result)
    df = df.sort_values("mention_count", ascending=True)

    fig = px.bar(
        df, x="mention_count", y="entity_text", orientation="h",
        color="entity_label",
        color_discrete_map=LABEL_COLORS,
        labels={
            "mention_count": "Menciones",
            "entity_text": "Entidad",
            "entity_label": "Tipo",
        },
    )
    fig.update_layout(
        height=max(250, len(df) * 30),
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title=None,
        legend_title_text="Tipo NER",
    )
    st.plotly_chart(fig, use_container_width=True, key=str(uuid4()))

    # Map for LOC entities
    loc_items = [r for r in result if r.get("entity_label") == "LOC"]
    if loc_items:
        _render_colombia_map(loc_items)


def viz_get_speech_entities(result: dict) -> None:
    """Horizontal bar chart of entities for a single speech, grouped by NER label."""
    entities = result.get("entities", {})
    if not entities:
        return

    rows = []
    for label, items in entities.items():
        for item in items[:10]:  # top 10 per type
            rows.append({
                "entity_text": item["entity_text"],
                "mentions": item["mentions"],
                "entity_label": label,
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("mentions", ascending=True)

    fig = px.bar(
        df, x="mentions", y="entity_text", orientation="h",
        color="entity_label",
        color_discrete_map=LABEL_COLORS,
        labels={
            "mentions": "Menciones",
            "entity_text": "Entidad",
            "entity_label": "Tipo",
        },
    )
    fig.update_layout(
        height=max(250, len(df) * 28),
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title=None,
        legend_title_text="Tipo NER",
    )
    st.plotly_chart(fig, use_container_width=True, key=str(uuid4()))

    # Map for LOC entities
    loc_items = entities.get("LOC", [])
    if loc_items:
        _render_colombia_map(loc_items)


def viz_list_speeches(result: list[dict]) -> None:
    """Vertical bar chart of word counts per speech, ordered by date."""
    if not result:
        return
    df = pd.DataFrame(result)
    # Sort by date (earliest first)
    df = df.sort_values("speech_date", ascending=True, na_position="last")
    # Shorten titles for display
    df["short_title"] = df["title"].str[:35]

    fig = px.bar(
        df, x="short_title", y="word_count",
        labels={"short_title": "Discurso", "word_count": "Palabras"},
        color="word_count",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        height=max(300, 400),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_tickangle=-45,
        xaxis_title=None,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=str(uuid4()))


def viz_get_opinions(result: dict) -> None:
    """Donut chart showing will_win yes/no split."""
    total = result.get("total_opinions", 0)
    if total == 0:
        return
    yes = result.get("total_will_win", 0)
    no = total - yes

    fig = go.Figure(data=[go.Pie(
        labels=["Sí ganará", "No ganará"],
        values=[yes, no],
        hole=0.5,
        marker_colors=["#00CC96", "#EF553B"],
        textinfo="label+percent",
    )])
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=str(uuid4()))


# ---------------------------------------------------------------------------
# Source chunk expanders (citation verification)
# ---------------------------------------------------------------------------

def render_source_chunks(tool_calls: list[dict]) -> None:
    """Render collapsed expanders showing the raw chunks Claude based its answer on.

    Only processes ``retrieve_chunks`` tool calls — all others are skipped.
    Each chunk is shown in a collapsed ``st.expander`` with metadata, full text,
    and a YouTube link so users can verify citations against the source material.
    """
    chunks: list[dict] = []
    for tc in tool_calls:
        if tc["tool_name"] != "retrieve_chunks":
            continue
        result = tc["tool_result"]
        if isinstance(result, dict) and "error" in result:
            continue
        if isinstance(result, list):
            chunks.extend(result)

    if not chunks:
        return

    st.caption("Fragmentos fuente")
    for chunk in chunks:
        title = chunk.get("speech_title", "Sin título")
        date = chunk.get("speech_date", "")
        score = chunk.get("similarity", 0)
        label = f"{title} — {date}  (similitud: {score:.0%})"

        with st.expander(label, expanded=False):
            event = chunk.get("speech_event", "")
            location = chunk.get("speech_location", "")
            meta_parts = []
            if event:
                meta_parts.append(f"**Evento:** {event}")
            if location:
                meta_parts.append(f"**Lugar:** {location}")
            if meta_parts:
                st.markdown(" | ".join(meta_parts))

            st.markdown(chunk.get("chunk_text", ""))

            yt_link = chunk.get("youtube_link")
            if yt_link:
                st.markdown(f"[Ver en YouTube]({yt_link})")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def render_visualizations(tool_calls: list[dict]) -> None:
    """Render inline charts for all tool calls that have a viz function."""
    for tc in tool_calls:
        viz_fn = VIZ_REGISTRY.get(tc["tool_name"])
        if viz_fn is None:
            continue
        result = tc["tool_result"]
        # Skip error results
        if isinstance(result, dict) and "error" in result:
            continue
        # Skip empty list results
        if isinstance(result, list) and len(result) == 0:
            continue
        viz_fn(result)
