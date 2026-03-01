# RAG System Design Decisions

This document explains the architectural decisions made while building the RAG (Retrieval-Augmented Generation) system for the Political Speech NLP Analyzer.

## 1. Chunking Strategy: Sentence Grouping

### The Decision
We group consecutive sentences into chunks of ~150-250 words with 1-sentence overlap between adjacent chunks.

### Why Not Character-Count Splitting?
Fixed character-count splitting (e.g., 500 characters) is the simplest approach but breaks text mid-sentence or mid-word. For political speeches, this destroys the rhetorical structure — a policy proposal split across two chunks loses its coherence and becomes harder to retrieve.

### Why Not Dependency Parse ROOT Boundaries?
The original plan (CHECKLIST.md) called for "dependency parse ROOT boundaries for clause-level splitting." After examining the actual data, this approach was abandoned because:

1. **spaCy already segments sentences.** The NLP pipeline stores sentence boundaries in the `annotations` table. ROOT-boundary splitting would split *within* sentences at clause boundaries, producing fragments too small for meaningful retrieval.

2. **Individual sentences are too short.** The median sentence length in our corpus is 14 words. Embedding models perform poorly on very short texts — they need enough context (~100-300 words) to capture semantic meaning. A single clause of 5-8 words like "Compañeras y compañeros" produces a nearly useless embedding.

3. **Sentence grouping respects natural discourse.** Political speeches have a natural flow: a speaker introduces a topic across 3-5 sentences, develops it, then transitions. Grouping to ~200 words captures these thematic units.

### Implementation Details
- **Target:** 200 words per chunk (configurable via `target_words`)
- **Range:** 100-300 words (soft min, hard max)
- **Overlap:** 1 sentence repeated between consecutive chunks, so no information falls into a gap
- **Runt handling:** If the final chunk is <30 words, it merges into the previous chunk instead of creating a fragment
- **Source:** `src/rag/chunker.py :: chunk_sentences()`

### Results
- 8 speeches → 106 chunks
- Average ~13 chunks per speech (~200 words each)
- Average ~120 sentences per speech → ~10:1 compression ratio

## 2. Embedding Model: paraphrase-multilingual-mpnet-base-v2

### The Decision
We use `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, a 768-dimensional multilingual model.

### Why This Model?
1. **768 dimensions** matches the `vector(768)` column already defined in the PostgreSQL schema.
2. **Multilingual quality.** Trained on 50+ languages including Spanish. Produces high-quality sentence embeddings for semantic similarity tasks.
3. **Proven for search.** This model is specifically designed for paraphrase detection and semantic search — exactly our use case.
4. **Reasonable size.** ~420MB, loads in ~5 seconds on M4 Mac Mini. Acceptable for a CPU-only deployment.
5. **Normalized embeddings.** We encode with `normalize_embeddings=True` so all vectors are unit-length. This means cosine similarity = dot product, which is computationally cheaper.

### Alternatives Considered
- **E5 models** (`multilingual-e5-base`): Designed specifically for retrieval with query/passage prefixes. Better recall in some benchmarks but requires prepending "query:" and "passage:" to texts, adding complexity.
- **MiniLM** (`all-MiniLM-L12-v2`): Only 384 dimensions and English-only. Disqualified.
- **distiluse-base-multilingual**: Only 512 dimensions. Would require schema changes.

## 3. Timestamp Mapping via Fuzzy Text Matching

### The Decision
Each chunk stores a `start_time` in its metadata, derived from the original Whisper transcript segments. This enables citations with YouTube timestamp links (e.g., `youtube.com/watch?v=abc&t=120`).

### The Challenge
Whisper segments and spaCy sentences don't align 1:1. Whisper segments are based on audio silence boundaries (~10-30 second chunks), while spaCy splits on syntactic sentence boundaries. A single spaCy sentence might span two Whisper segments, or vice versa.

### The Solution
For each chunk, we search for the first Whisper segment whose text overlaps with the chunk's first ~100 characters. The match is case-insensitive and uses substring matching (first 30 characters of each). The matched segment's `start` time becomes the chunk's `start_time`.

This gives approximate but useful timestamps — typically within a few seconds of the actual quote. For a YouTube viewer, being dropped within 5-10 seconds of the relevant passage is good enough.

### Source
`src/rag/chunker.py :: map_chunk_timestamps()`

## 4. Vector Index: HNSW

### The Decision
We use an HNSW (Hierarchical Navigable Small World) index on the `speech_chunks.embedding` column with pgvector's `vector_cosine_ops` operator.

### Why HNSW Over IVFFlat?
- **Better recall.** HNSW provides higher recall at the same search speed, especially for small datasets.
- **No training step.** IVFFlat requires a separate `CREATE INDEX` with cluster training. HNSW builds incrementally.
- **Auto-tuning.** Good defaults (`m=16, ef_construction=128`) work well for datasets from hundreds to millions of vectors.

### Does It Even Matter?
With ~100 chunks across 8 speeches, sequential scan is sub-millisecond. The index is for future-proofing: when the corpus grows to 50 speeches (~600+ chunks), HNSW will keep retrieval fast.

## 5. Citation System: Title + Date + YouTube Timestamp Link

### The Decision
Every RAG response includes citations in the format:
```
(Discurso: "TÍTULO", fecha — [ver video](https://youtube.com/watch?v=id&t=seconds))
```

### Design Choices
- **YouTube URL stored per speech** in the `speeches` table. The retriever JOINs on this.
- **Timestamp stored per chunk** in the `metadata` JSONB column. The retriever reads `metadata->>'start_time'`.
- **The generator formats the links.** Claude receives the chunk text, speech title, date, and YouTube link in its context block and is instructed to cite sources using the provided links.
- **`RetrievalResult.youtube_link` property** constructs the timestamped URL, normalizing both `youtube.com/watch?v=` and `youtu.be/` formats.

### Why JSONB for Timestamps Instead of a Column?
`start_time` is approximate metadata, not a first-class data field. Storing it in the `metadata` JSONB column keeps the schema clean and allows adding more metadata later (e.g., `end_time`, `speaker`, `topic_tags`) without schema migrations.

## 6. System Prompt: Neutral, Cited, Spanish

### The Decision
The Claude system prompt is written in Spanish and enforces strict constraints:
1. Only answer from provided context — never fabricate
2. Always include citations with speech title, date, and video link
3. Say "No encontré referencias a ese tema" when the topic isn't covered
4. Maintain neutral, informational tone — no advocacy or editorializing
5. Synthesize across multiple sources when relevant

### Why Spanish?
The data, user queries, and expected responses are all in Spanish. A Spanish system prompt produces more natural Spanish output with fewer English artifacts.

### Model Selection
- **Development/testing:** `claude-haiku-4-5-20251001` (fast, cheap, ~2700 input tokens per query)
- **Production:** `claude-sonnet-4-6` or `claude-opus-4-6` (higher quality synthesis)
- The model is passed as a parameter to `generate()`, defaulting to haiku for dev.

## 7. Pipeline Integration: Auto-Chunk on Ingest

### The Decision
The ingestion pipeline (`pipeline_runner.py`) automatically chunks and embeds each speech as Step 5, right after the DB load:

```
download → diarize → transcribe → clean → NLP → DB load → chunk + embed
```

### Why Not a Separate Batch Job?
Integrating into the pipeline means every new speech is automatically RAG-ready. No manual step, no separate script to remember. The `backfill.py` script exists only for speeches that were loaded *before* this integration was added.

### Import Guard
The RAG imports are wrapped in `try/except ImportError` so the pipeline still works if `sentence-transformers` isn't installed. This avoids breaking the corpus pipeline for environments that don't need RAG.

## 8. Retrieval: Top-5, Threshold 0.3

### The Decision
Default retrieval returns the top 5 most similar chunks with a minimum cosine similarity of 0.3.

### Why 5?
- Enough context for Claude to synthesize across multiple speeches
- Not so many that irrelevant chunks dilute the context
- Keeps input token count manageable (~2500-3000 tokens per query)

### Why Threshold 0.3?
With normalized multilingual embeddings, cosine similarities for same-topic Spanish text typically range 0.3-0.7. A threshold of 0.3 is intentionally permissive — it's better to retrieve a marginally relevant chunk than to miss a real match. Claude handles relevance filtering in its response: if the chunks don't actually answer the question, it says so.

### Observed Similarity Ranges
- **Highly relevant chunks:** 0.55-0.70 (e.g., "racismo" query → anti-racism speech chunks)
- **Moderately relevant:** 0.40-0.55 (e.g., related social justice topics)
- **Low relevance / off-topic:** 0.30-0.40 (still included but Claude typically ignores)
- **Below threshold:** <0.30 (filtered out)
