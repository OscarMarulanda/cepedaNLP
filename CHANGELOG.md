# Changelog

All notable changes to the cepedaNLP project.

## 2026-03-01

### Added
- **Colombia bubble map visualization** — geographic map showing where LOC entities are mentioned:
  - `COLOMBIAN_COORDS` constant with 21 hardcoded (lat, lon) for corpus locations
  - `_render_colombia_map()` helper using Plotly `scatter_geo`, scoped/zoomed to Colombia
  - Auto-renders below entity bar charts in `viz_search_entities()` and `viz_get_speech_entities()` when LOC entities are present
  - 8 new visualization tests (24 total viz tests, 94 total project tests)
- **`limit` parameter for `search_entities`** — MCP tool + Anthropic tool schema now accept `limit` (1-50, default 10) so chart count matches user request
- **`REGLA DE LÍMITES` in system prompt** — instructs Haiku to pass user's requested count as `limit`
- **User opinions feature** — 2 new MCP tools (first write operations in the system):
  - `submit_opinion` — saves user opinion (free text + will_win boolean) to DB with `conn.commit()`
  - `get_opinions` — retrieves opinions with summary stats (total, yes/no breakdown, percentage)
  - `user_opinions` table + indexes in `schema.sql`
  - `get_corpus_stats` now includes opinion count
  - Sidebar shows opinion count
  - 9 new tests (6 in test_tools.py, 3 in test_security.py) — 29 total, all passing

### Fixed
- **Streamlit duplicate element ID error** — added `key=str(uuid4())` to all `st.plotly_chart()` calls to prevent crashes when multiple charts render in the same session
- **Haiku tool hallucination** — Claude Haiku fabricated a `submit_opinion` call (returned fake "Opinion ID: 2") without actually invoking the tool. DB confirmed only 1 row, sequence at 1. Strengthened system prompt: "NUNCA finjas haberla guardado sin haber ejecutado la herramienta. NO inventes IDs ni resultados de herramientas."

## 2026-02-28 (evening)

### Added
- **Phase 6: MCP Server + Streamlit Frontend — COMPLETE**
  - MCP server (`src/mcp/server.py`) — 6 read-only tools via FastMCP 3.0:
    - `retrieve_chunks` — pgvector semantic search with citations
    - `list_speeches` — corpus listing with metadata
    - `get_speech_detail` — single speech with entity/chunk counts
    - `search_entities` — ILIKE entity search across corpus
    - `get_speech_entities` — entities grouped by NER label
    - `get_corpus_stats` — corpus-wide statistics
  - Lightweight DB module (`src/mcp/db.py`) — no heavy pipeline imports
  - Streamlit chat UI (`src/frontend/app.py`) — Claude Haiku orchestrator with tool dispatch, multi-turn, rate limiting (30 msg/session), sidebar stats
  - System prompt + tool definitions (`src/frontend/prompts.py`)
  - 20 tests (`tests/mcp/test_tools.py`, `tests/mcp/test_security.py`) — all passing
  - `fastmcp>=3.0` added to `requirements.txt`

### Fixed
- **Whisper hallucination cleanup** — 5 garbled chunks cleaned and re-embedded:
  - Chunk 84: removed nonsense prefix (48% of text was Whisper hallucination)
  - Chunks 76, 77: deduplicated stuttered sentences
  - Chunk 108: removed triplicated sentences
  - Chunk 131: removed garbled tail after rally chant
- **Citation accuracy** — system prompt tightened to prevent fabricated quotes. Quotes must be verbatim from chunk_text; each citation must use the youtube_link from its specific chunk.

## 2026-02-28 (afternoon)

### Added
- RAG backfill for speeches 11 and 12 (25 new chunks, 131 total)
- API cost analysis document (`docs/API_COST_ANALYSIS.md`)
- Deployment architecture document (`docs/DEPLOYMENT_ARCHITECTURE.md`)
- Phase 6 implementation plan (`docs/PHASE6_API_PLAN.md`)
- Architecture Decision Records (`docs/decisions/001-007`)
- CHANGELOG.md, session notes (`memory/sessions.md`)

### Changed
- Corpus grew from 8 to 10 speeches (~25,600 words)
- Pipeline batch `--new=4` completed (speeches 9, 10, 11, 12)
- **Architecture pivot:** Dropped FastAPI REST + MCP hybrid in favor of MCP-only (ADR 007). Claude (Haiku) on Streamlit orchestrates MCP tools via tool_use. No REST API needed.

## 2026-02-27

### Added
- RAG system — complete end-to-end pipeline (Phase 4)
  - Semantic chunker (`src/rag/chunker.py`) — sentence-grouping, ~200 words/chunk
  - Embedder (`src/rag/embedder.py`) — `paraphrase-multilingual-mpnet-base-v2` (768d)
  - Retriever (`src/rag/retriever.py`) — pgvector cosine similarity + citation joins
  - Generator (`src/rag/generator.py`) — Claude API with Spanish system prompt
  - Query orchestrator (`src/rag/query.py`) — `ask()` entry point
  - Backfill script (`src/rag/backfill.py`)
- RAG tests (20/20 passing)
- Auto chunk+embed in pipeline_runner (Step 5)
- `speech_chunks` table: `sentence_start`, `sentence_end`, `metadata` JSONB columns
- HNSW index on `speech_chunks.embedding`
- Speeches 7-10 ingested via pipeline

### Changed
- `db_loader.get_connection()` now calls `register_vector(conn)` for pgvector support
- RAG design decisions documented (`docs/RAG_DESIGN_DECISIONS.md`)

## 2026-02-26

### Added
- BETO NER integration (`mrm8488/bert-spanish-cased-finetuned-ner`)
- DANE DIVIPOLA gazetteer (1,099 Colombian locations)
- NER model evaluation document (`docs/NER_MODEL_EVALUATION.md`)
- Speaker diarization pipeline (`src/corpus/diarizer.py`)
- Speeches 1, 4, 5, 6 ingested

## 2026-02-24

### Added
- Initial project setup (Phase 0)
- YouTube downloader, Whisper transcriber, text cleaner
- spaCy NLP pipeline (tokenizer, POS tagger, dependency parser)
- PostgreSQL schema with pgvector extension
- DB loader and pipeline runner
