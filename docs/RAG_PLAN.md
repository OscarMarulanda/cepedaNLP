# RAG System Implementation Plan

## Context
10 speeches are being ingested into PostgreSQL (6 done, 4 in progress). The goal is to build the RAG pipeline so that once built, adding new speeches is automatic: the pipeline chunks, embeds, and stores them — retrieval and generation just work against whatever's in the DB.

## Architecture Overview
```
INGESTION (automatic per speech):
  annotations table (sentences) → chunker → embedder → speech_chunks table
  raw transcript segments (timestamps) → stored in chunk metadata for video links

QUERY TIME:
  user question → embed query → pgvector cosine search → top-k chunks → Claude API → cited answer
  citations include: speech title, date, YouTube URL with timestamp (e.g. youtu.be/abc?t=120)
```

## Implementation Steps

### Step 0: Install missing dependencies + schema migration
- Verify/install: `sentence-transformers`, `anthropic`, `pgvector` (Python package), `fastapi`, `uvicorn`
- Add columns to `speech_chunks`: `sentence_start INTEGER`, `sentence_end INTEGER`, `metadata JSONB`
- Add HNSW vector index on `speech_chunks.embedding`
- Update `schema.sql` to match

### Step 1: `src/rag/chunker.py` — Sentence-grouping chunker
- Group consecutive sentences (from `annotations` table) into ~150-250 word chunks
- 1-sentence overlap between consecutive chunks for context continuity
- Runt chunks (<30 words) merged into previous chunk
- Track `sentence_start`/`sentence_end` indices + `start_char`/`end_char` offsets
- **Timestamp mapping:** Cross-reference chunk text against raw transcript segments (`data/raw/{id}.json`) to find the earliest `start` time for each chunk. Store as `start_time` in the chunk's `metadata` JSONB field.
- Key functions:
  - `chunk_sentences(sentences, target_words=200, overlap_sentences=1) -> list[Chunk]`
  - `chunk_speech_from_db(conn, speech_id) -> list[Chunk]`
  - `map_chunk_timestamps(chunks, raw_segments) -> list[Chunk]` — finds earliest matching segment timestamp per chunk
- **Why sentence-grouping over ROOT-boundary splitting:** spaCy already gives us sentence boundaries. Individual sentences are too short (median 14 words) for embedding quality. Grouping to ~200 words hits the sweet spot.

### Step 2: `src/rag/embedder.py` — Sentence-transformers wrapper
- Model: `paraphrase-multilingual-mpnet-base-v2` (768d, matches schema)
- Module-level model cache (same pattern as `nlp_processor.py`)
- `normalize_embeddings=True` for efficient cosine similarity
- Key functions:
  - `embed_texts(texts: list[str]) -> np.ndarray` (batch)
  - `embed_query(query: str) -> np.ndarray` (single)

### Step 3: Modify `src/corpus/db_loader.py` — Add chunk storage
- Add `register_vector(conn)` in `get_connection()` (with try/except guard for backward compat)
- New functions:
  - `chunks_exist(conn, speech_id) -> bool`
  - `load_chunks(conn, speech_id, chunks, embeddings) -> int`
- `load_chunks` stores `metadata` JSONB (which includes `start_time` from the timestamp mapping)

### Step 4: `src/rag/backfill.py` — Backfill existing speeches
- One-time script: iterates all speeches in DB, chunks + embeds any without chunks
- Also loads raw transcript JSONs to map timestamps into chunk metadata
- Idempotent (skips speeches that already have chunks)
- Run: `python -m src.rag.backfill`
- **Run this early** to populate data for testing retrieval

### Step 5: `src/rag/retriever.py` — pgvector semantic search
- Embed query → cosine similarity search on `speech_chunks` → JOIN `speeches` for citation metadata
- Returns `list[RetrievalResult]` with:
  - chunk text + similarity score
  - speech title, date, location, event
  - `youtube_url` from `speeches` table
  - `start_time` from chunk `metadata` JSONB
  - Computed `youtube_link`: `{youtube_url}&t={start_time_seconds}` for direct timestamp linking
- Default: top_k=5, threshold=0.3
- Key function: `retrieve(query, top_k=5, threshold=0.3, conn=None) -> list[RetrievalResult]`

### Step 6: `src/rag/generator.py` — Claude API with citations
- System prompt in Spanish enforcing: cite sources with YouTube links, only use provided context, neutral tone, say "no encontré referencias" when topic not covered
- Context block formats each chunk with speech title, date, YouTube timestamp link, relevance score
- Citation format: `(Discurso: "TÍTULO", fecha — ver video)`  where "ver video" is the timestamped YouTube link
- Model defaults to `claude-haiku-4-5-20251001` (dev), caller can pass `claude-sonnet-4-6` (prod)
- Key function: `generate(query, results, model=None) -> dict`

### Step 7: `src/rag/query.py` — End-to-end orchestrator + CLI
- `ask(query, top_k=5, model=None) -> dict` — single entry point: retrieve → generate → return answer + sources
- Response includes `sources` list with `youtube_link` per source for frontend use
- CLI via `__main__`: `python -m src.rag.query "¿Qué propone sobre el racismo?"`
- This is the main interface; FastAPI wraps this later in Phase 6

### Step 8: Modify `src/corpus/pipeline_runner.py` — Hook chunk+embed into ingestion
- Add Step 5 after DB load: `chunk_speech_from_db()` → `map_chunk_timestamps()` → `embed_texts()` → `load_chunks()`
- Raw transcript is already available in the processing function, so timestamp mapping happens naturally
- Import guard so pipeline still works if sentence-transformers not installed
- **This is what makes expansion automatic** — new speeches get chunked during ingestion

### Step 9: Tests
- `tests/rag/test_chunker.py` — chunk algorithm (grouping, overlap, edge cases, coverage)
- `tests/rag/test_retriever.py` — result filtering, threshold, mocked DB
- Update `src/rag/__init__.py` to expose `ask()`

## Timestamp Mapping Strategy

**Updated (2026-03-09):** Timestamps are now mapped at the **sentence level** instead of chunk level.

Raw transcripts (`data/raw/{id}.json`) have Whisper segments with `start`/`end` times:
```json
{"start": 43.26, "end": 55.68, "text": "de seguir explicando en los territorios..."}
```

The original approach mapped timestamps per chunk via fuzzy text matching. This was replaced by deterministic character-offset matching at the sentence level:
1. `full_text = " ".join(seg["text"] for seg in cleaned_segments)` — character offsets are deterministic
2. For each annotation sentence, find its position in `full_text`
3. Look up which Whisper segment covers that character position
4. Store the segment's `start` time in `annotations.start_time`

**Result:** 7,193/7,193 sentences matched, 0 failures. `retrieve_chunks` returns a `sentences` array per chunk with `{text, start_time, youtube_link}`.

**Script:** `src/corpus/timestamp_backfill.py`

## Files Summary

| Action | File |
|--------|------|
| Create | `src/rag/chunker.py` |
| Create | `src/rag/embedder.py` |
| Create | `src/rag/retriever.py` |
| Create | `src/rag/generator.py` |
| Create | `src/rag/query.py` |
| Create | `src/rag/backfill.py` |
| Create | `tests/rag/test_chunker.py` |
| Create | `tests/rag/test_retriever.py` |
| Modify | `src/corpus/db_loader.py` — add `load_chunks()`, `chunks_exist()`, pgvector registration |
| Modify | `src/corpus/pipeline_runner.py` — add chunk+embed step |
| Modify | `src/rag/__init__.py` — expose `ask()` |
| Modify | `schema.sql` — add columns + HNSW index |

## Build Order
```
0. Install deps + schema migration
1. chunker.py          → test: python -m src.rag.chunker
2. embedder.py         → test: python -m src.rag.embedder
3. db_loader.py mods   → tested via backfill
4. backfill.py         → test: python -m src.rag.backfill (populates chunks for retrieval testing)
5. retriever.py        → test: python -m src.rag.retriever
6. generator.py        → test: python -m src.rag.generator (uses API key)
7. query.py            → test: python -m src.rag.query "¿Qué dice sobre la paz?"
8. pipeline_runner mod → tested when next speech is processed
9. tests + __init__    → pytest tests/rag/ -v
```

## Verification
End-to-end test after all steps:
```bash
# 1. Backfill existing speeches
python -m src.rag.backfill

# 2. Query the system
python -m src.rag.query "¿Qué propone Cepeda sobre el racismo?"
python -m src.rag.query "¿Qué dice sobre la reforma agraria?"
python -m src.rag.query "¿Cuál es su posición sobre el cambio climático?"  # should say "no encontré"

# 3. Verify DB has chunks with timestamps
psql -d cepeda_nlp -c "SELECT speech_id, COUNT(*), MIN((metadata->>'start_time')::float) FROM speech_chunks GROUP BY speech_id;"

# 4. Run tests
pytest tests/rag/ -v
```
