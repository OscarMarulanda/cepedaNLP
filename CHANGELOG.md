# Changelog

All notable changes to the cepedaNLP project.

## 2026-03-03

### Added
- **Security middleware for MCP SSE endpoint** (`src/mcp/middleware.py`):
  - `APIKeyMiddleware` — Bearer token auth, optional via `MCP_API_KEY` env var, skips `/health`
  - `RateLimitMiddleware` — per-IP sliding window (default 30 req/min via `MCP_RATE_LIMIT`), in-memory, periodic stale-IP cleanup
  - `SSEConnectionMiddleware` — per-IP concurrent SSE connection limiter (default 5, configurable via `MCP_MAX_SSE_CONNS`)
  - All skip `/health` for Render health checks
  - No new dependencies (Starlette `BaseHTTPMiddleware`)
- `MCP_API_KEY` and `MCP_RATE_LIMIT` env vars in `render.yaml`
- Auth header instructions in `docs/MCP_CLIENT_SETUP.md` (all client configs updated)
- 401/429 troubleshooting entries in `docs/MCP_CLIENT_SETUP.md`

### Changed
- `run_mcp.py` — wires middleware stack via `mcp.http_app(middleware=[...])`

### Added (later)
- **Production sync script** (`src/corpus/sync_to_production.py`) — pushes new speeches from local DB to Supabase without re-processing. Connects to both DBs simultaneously, finds missing speeches by `youtube_url`, copies speech + entities + annotations + speaker_segments + chunks with embeddings. Remaps foreign keys automatically. Supports `--dry-run`.
- Sync instructions document (`docs/SYNC_TO_PRODUCTION.md`)
- 2 new speeches processed locally (IDs 17-18, corpus now 16 speeches / 35,287 words / 198 chunks)

### Security
- Hardened API key validation against side-channel attacks
- Hardened client IP extraction for proxy-aware rate limiting
- Added SSE connection limiting to prevent resource exhaustion
- Upgraded Render DB transport to `verify-full` with bundled CA certificate
- Hardened ILIKE query input handling in `search_entities`

## 2026-03-02 (session 3)

### Added
- **Render deployment for MCP server** — public SSE endpoint at `https://cepeda-nlp-mcp.onrender.com/sse`:
  - `run_mcp.py` — programmatic entry point (uvicorn, `/health` route)
  - `render.yaml` — Render IaC (env vars, build/start commands, health check)
  - `requirements-mcp.txt` — lean 6-package dependency file (no Streamlit, no Anthropic)
  - `.python-version` for Render's Python version detection
- Production (Supabase) connection instructions in `docs/MCP_CLIENT_SETUP.md`

### Fixed
- **Lazy-import `ask()` in `src/rag/__init__.py`** — top-level import pulled `query → generator → anthropic`, making MCP server depend on the `anthropic` package unnecessarily. Wrapped in lazy function.

## 2026-03-02 (session 2)

### Added
- **Deployed to Streamlit Community Cloud** — app is live at public URL
- **Supabase PostgreSQL** — provisioned free-tier project (us-west-2), migrated all data:
  - 14 speeches, 825 entities, 1,594 annotations, 174 chunks (768d embeddings), 2,533 speaker segments, 4 opinions
  - Uses Session Pooler (`aws-0-us-west-2.pooler.supabase.com`) — free tier is IPv6-only for direct connections
- **SSL `verify-full`** — added `sslmode` + `sslrootcert` params to `psycopg2.connect()` in both `src/mcp/db.py` and `src/corpus/db_loader.py`
- **Supabase CA certificate** — full chain (leaf + intermediate + root) bundled at `certs/supabase-ca.crt`
- **`requirements-deploy.txt` → `requirements.txt`** — slim 13-package file for Streamlit Cloud (no PyTorch/Whisper/spaCy). Full pipeline deps moved to `requirements-full.txt`
- **`.env.example`** — template with all 8 required env vars
- **`.streamlit/config.toml`** — headless mode, usage stats disabled
- **`runtime.txt`** — pins Python 3.13 for Streamlit Cloud
- **Ethical disclaimer** — added to sidebar in `src/frontend/app.py`
- **Deployment checklist** (`docs/DEPLOYMENT_CHECKLIST.md`) — blockers, SSL analysis, RAM/concurrency estimates, scalability plan

### Fixed
- **HuggingFace Inference API 403** — original HF token lacked "Inference Providers" permission. New fine-grained token required.

## 2026-03-02 (session 1)

### Added
- **Three-panel layout** — source chunks in dedicated right column for side-by-side citation comparison
- **Resizable panels** — draggable JS splitter between chat and chunks columns, ratio persisted in sessionStorage
- **Independent scrolling** — each panel scrolls independently, page-level scrolling disabled
- **MCP client setup documentation** (`docs/MCP_CLIENT_SETUP.md`) — connection instructions for Claude Desktop, Claude Code, Kiro, Cursor, SSE transport
- **Workflow documentation** (`docs/workflow/`) — business case docs (idea, problem, initiative, solution)

### Changed
- Title replaced with compact `h3` + small caption, reduced top padding to minimize wasted space
- Sidebar compacted — metrics paired in 2-column rows, model/message count on single line
- Layout switched from `centered` to `wide` for three-panel support

## 2026-03-01 (session 3)

### Added
- **Source chunk expanders** for citation verification (`render_source_chunks()` in `visualizations.py`):
  - Collapsed `st.expander` per retrieved chunk — speech title, date, similarity score, full text, YouTube link
  - Lets users verify citations against raw source material
  - 4 new tests (28 viz tests, 177 total)

### Changed
- **Streamed Claude's final response** for better perceived latency:
  - Split `_call_claude()` into `_run_tool_rounds()` (non-streaming tool loop) + `_stream_response()` (generator using `client.messages.stream()`)
  - Tool rounds run via `create()`, final text streams token-by-token via `st.write_stream()`
  - No-tool responses (greetings) render instantly via `st.markdown()`
  - Replaced `AssistantResponse` dataclass with `ToolRoundResult`

### Fixed
- **Anthropic SDK serialization error** — `TextBlock.model_dump()` includes SDK-internal `parsed_output` field rejected by the API ("Extra inputs are not permitted"). Added `_dump_content_block()` helper that serializes only API-accepted fields.

## 2026-03-01 (sessions 1-2)

### Added
- **HuggingFace Inference API for query embeddings** (`feature/hf-embedding-api` branch):
  - `EMBEDDING_PROVIDER` env var switches `embed_query()` between local SentenceTransformer and HF API
  - Lazy-imports SentenceTransformer to avoid loading ~868 MB when using API path
  - 4 new tests in `tests/rag/test_embedder.py` (98 total project tests)
- **DB connection security analysis** (`docs/DB_CONNECTION_SECURITY.md`):
  - Documents the 3 network hops (browser→Streamlit, Streamlit→DB, Streamlit→APIs)
  - Identifies MITM downgrade risk with default `sslmode=prefer` on remote DB connections
  - Recommends `DB_SSLMODE=require` for all remote deployments
  - ADR 008: `docs/decisions/008-db-ssl-for-remote-connections.md`

### Changed
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
