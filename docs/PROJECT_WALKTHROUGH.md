# Project Walkthrough: Political Speech NLP Analyzer & RAG Chatbot

> A step-by-step explanation of how the cepedaNLP system works, from scraping YouTube videos to answering user questions with cited, grounded responses.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Step 1: Scraping the YouTube Channel](#2-step-1-scraping-the-youtube-channel)
3. [Step 2: Downloading Audio](#3-step-2-downloading-audio)
4. [Step 3: Speaker Diarization](#4-step-3-speaker-diarization)
5. [Step 4: Transcription](#5-step-4-transcription)
6. [Step 5: Text Cleaning](#6-step-5-text-cleaning)
7. [Step 6: NLP Pipeline](#7-step-6-nlp-pipeline)
8. [Step 7: Database Storage](#8-step-7-database-storage)
9. [Step 8: Semantic Chunking](#9-step-8-semantic-chunking)
10. [Step 9: Embedding](#10-step-9-embedding)
11. [Step 10: Retrieval (Semantic Search)](#11-step-10-retrieval-semantic-search)
12. [Step 11: Answer Generation (Claude API)](#12-step-11-answer-generation-claude-api)
13. [Step 12: MCP Server (Tool Layer)](#13-step-12-mcp-server-tool-layer)
14. [Step 13: Streamlit Frontend](#14-step-13-streamlit-frontend)
15. [Pipeline Orchestration](#15-pipeline-orchestration)
16. [Database Schema](#16-database-schema)
17. [Test Suite](#17-test-suite)
18. [Deployment Architecture](#18-deployment-architecture)
19. [Design Decisions & Trade-offs](#19-design-decisions--trade-offs)
20. [API Cost Analysis](#20-api-cost-analysis)

---

## 1. Project Overview

This system analyzes ~50 campaign speeches by **Ivan Cepeda** (2026 Colombian presidential candidate) and exposes them through a conversational RAG chatbot. Users ask questions in Spanish and get cited, grounded answers synthesized from actual speech fragments, with YouTube video links pointing to the exact timestamp where the candidate said it.

**Architecture principle:** "Claude is the mouth, BERT is the brain." The LLM (Claude) only generates responses from pre-retrieved context — it never fabricates answers. All NLP analysis (NER, POS tagging, dependency parsing) is done by dedicated models (spaCy, BETO), and all retrieval is done by pgvector semantic search. Claude's role is strictly last-mile generation.

**End-to-end data flow:**

```
YouTube Channel
    |
    v
[1] Scrape channel metadata (yt-dlp)
    |
    v
[2] Download audio as MP3 (yt-dlp + FFmpeg)
    |
    v
[3] Speaker diarization — identify Cepeda's voice (pyannote.audio)
    |
    v
[4] Transcribe speech-to-text (OpenAI Whisper large-v3)
    |
    v
[5] Clean transcription — remove fillers, repetitions, normalize (regex)
    |
    v
[6] NLP pipeline — tokenize, POS tag, NER, dependency parse (spaCy + BETO)
    |
    v
[7] Store everything in PostgreSQL (speeches, entities, annotations)
    |
    v
[8] Chunk text into ~200-word semantic units (sentence grouping)
    |
    v
[9] Embed chunks into 768-dim vectors (sentence-transformers)
    |
    v
[10] Store chunks + embeddings in pgvector (HNSW indexed)
    |
    ===== INGESTION COMPLETE — QUERY TIME BELOW =====
    |
    v
[11] User asks a question → embed query → pgvector cosine search → top-5 chunks
    |
    v
[12] Claude API generates cited answer from retrieved chunks
    |
    v
[13] MCP tools provide data access layer (8 tools, no REST API)
    |
    v
[14] Streamlit chat UI displays conversation with citations
```

---

## 2. Step 1: Scraping the YouTube Channel

**What happens:** The system scrapes the YouTube channel `@IvanCepedaCastro` to discover campaign speech videos and build a manifest (index) of all available speeches.

**How it works:**
- `yt-dlp` is used in `extract_flat=True` mode, which scrapes channel metadata (title, duration, view count, URL) without downloading any video content.
- The results are saved to a JSON manifest file (`data/speech_manifest.json`).
- For each video, a second call (`get_video_full_metadata`) fetches the upload date and other detailed metadata.
- The manifest is **idempotent** — running it again merges new videos with existing entries, skipping duplicates by video ID.

**Tools/libraries:** `yt-dlp` (YouTube metadata extraction)

**Key file:** `src/corpus/downloader.py`

**Key functions:**
| Function | Purpose |
|----------|---------|
| `scrape_channel_metadata()` | Discovers videos on the channel (flat extraction, no download) |
| `get_video_full_metadata()` | Fetches upload date and full metadata for a single video |
| `build_manifest()` | Builds/updates the speech manifest JSON — idempotent |
| `register_text_speech()` | Registers non-YouTube speeches (e.g., from websites) into the manifest |

**Design note:** Only videos 1–44 on the channel are campaign speeches. Older videos are different content (senate sessions, interviews). The manifest builder accepts a `max_videos` parameter to limit scraping scope.

---

## 3. Step 2: Downloading Audio

**What happens:** For each speech in the manifest, the audio track is downloaded as an MP3 file.

**How it works:**
- `yt-dlp` downloads the best available audio stream.
- FFmpeg (as a postprocessor) converts it to MP3 at 192kbps.
- The output is saved to `data/audio/{video_id}.mp3`.
- If the file already exists, the download is skipped (idempotent).

**Tools/libraries:** `yt-dlp`, `FFmpeg`

**Key file:** `src/corpus/downloader.py`

**Key function:** `download_audio(video_url, output_dir)` — downloads and converts to MP3, returns the file path.

**Design note:** Audio is deleted after transcription (step 4) to minimize disk usage. The entire pipeline processes one speech at a time ("stream processing"), keeping peak disk usage at ~60MB. The `--keep-audio` flag can override this behavior for debugging.

---

## 4. Step 3: Speaker Diarization

**What happens:** Many campaign videos contain multiple speakers (panel discussions, Q&A sessions, debates). Diarization identifies **which segments** of the audio belong to Ivan Cepeda versus other speakers, so only his words are transcribed.

**How it works:**
1. A **reference voice embedding** was created once from a ~60-second clip of clean Cepeda speech (video `KrRrL13tqpw`, 0–60s). This 512-dimensional speaker fingerprint is saved at `data/reference_embedding.npy`.
2. For each new audio file:
   - **pyannote.audio** runs speaker diarization, producing a list of labeled segments (e.g., `SPEAKER_00: 0.0–15.3s`, `SPEAKER_01: 15.3–42.7s`, etc.).
   - pyannote also extracts a speaker embedding for each detected speaker.
   - The system compares each speaker's embedding to the reference using **cosine similarity** and identifies the best match as Cepeda.
3. Only Cepeda's segments are concatenated into a new WAV file for transcription.
4. **Offset mappings** are generated to remap timestamps from the concatenated audio back to the original video timeline (critical for YouTube timestamp citations later).

**Tools/libraries:**
- `pyannote.audio` 4.0.4 (`pyannote/speaker-diarization-community-1` model) — speaker diarization
- `torch`, `torchaudio` — audio I/O and speaker embedding extraction
- `scipy.spatial.distance` — cosine similarity for speaker matching

**Key file:** `src/corpus/diarizer.py`

**Key functions:**
| Function | Purpose |
|----------|---------|
| `create_reference_embedding()` | One-time setup: creates the voice fingerprint from a clean audio clip |
| `diarize_audio()` | Runs pyannote diarization, returns speaker segments + embeddings |
| `identify_target_speaker()` | Compares each speaker to the reference, returns best match |
| `extract_speaker_audio()` | Concatenates target speaker's segments into a single WAV |
| `remap_timestamps()` | Maps timestamps from concatenated audio back to original video timeline |
| `run_diarization()` | Orchestrates the full diarization flow |

**Design decisions and changes:**
- **CPU only:** Testing on M4 Mac Mini showed CPU is **1.9x faster** than MPS (Metal Performance Shaders) for both Whisper and pyannote. So both run on CPU.
- **MP3 bug workaround:** pyannote has a sample-count validation bug with MP3 files. The `_ensure_wav()` helper converts to WAV before processing, then cleans up the temp file.
- **Similarity threshold = 0.25:** Cross-recording voice similarity is lower than expected (~0.3–0.4 for the same speaker with different mics/environments). The threshold was lowered from the typical 0.5+ to avoid false negatives. Better to transcribe full audio (fallback) than miss the target speaker.
- **Fallback behavior:** If the target speaker isn't identified (confidence below threshold), the pipeline falls back to transcribing the full audio and logs a warning.

---

## 5. Step 4: Transcription

**What happens:** Audio is converted to text with segment-level timestamps using OpenAI Whisper.

**How it works:**
- Whisper `large-v3` runs locally (not the API) on the extracted speaker audio (or full audio if diarization failed).
- Language is explicitly set to Spanish (`language="es"`) to avoid misidentification of regional variants.
- Output is a JSON file at `data/raw/{speech_id}.json` with a list of segments (each with `start`, `end`, `text`) and a concatenated `full_text`.
- If the transcript already exists, the step is skipped (idempotent).

**Tools/libraries:** `openai-whisper` (local model, not the API)

**Key file:** `src/corpus/transcriber.py`

**Key functions:**
| Function | Purpose |
|----------|---------|
| `load_model()` | Lazy-loads Whisper large-v3, cached globally |
| `transcribe_audio()` | Transcribes an audio file, saves segments + full text to JSON |
| `load_text_speech()` | For website-sourced speeches (no audio): loads .txt, splits by paragraph |

**Design decisions:**
- **Model choice:** `large-v3` is the largest and most accurate Whisper model for Spanish. It takes ~7.5 minutes to transcribe a 20-minute speech on CPU.
- **CPU over MPS:** As noted in step 3, CPU is 1.9x faster than MPS on the M4 Mac Mini.
- **Whisper hallucinations:** ~8% of transcribed chunks contained garbled text (repetitive nonsense, typically at rally chanting sections). The 5 worst chunks were cleaned manually via `src/mcp/cleanup_garbled_chunks.py`. The remaining garbled sections were determined to be legitimate rally chants.
- **Text-only path:** ~6 speeches were sourced from websites (not YouTube). `load_text_speech()` handles these by treating paragraphs as segments with null timestamps, producing the same JSON format for pipeline compatibility.

**After transcription:** If diarization was used, `remap_timestamps()` is called to convert the timestamps from the concatenated audio timeline back to the original video timeline. The original audio is then deleted (unless `--keep-audio` was specified).

---

## 6. Step 5: Text Cleaning

**What happens:** Raw Whisper transcriptions contain spoken-language artifacts — fillers, false starts, repetitions, encoding inconsistencies. The cleaner removes these while preserving the meaning.

**How it works:**
- Each segment from the raw transcript is cleaned independently.
- **Unicode normalization** (NFC) handles accented characters.
- **Punctuation normalization** converts smart quotes, em-dashes, and ellipsis variants to standard forms.
- **Filler removal** uses context-aware regex patterns for Colombian Spanish fillers: `eh/eeh/ehh`, `o sea`, `digamos`, `pues` (sentence-initial), `bueno` (sentence-initial), `entonces` (before comma), `verdad/cierto` (tag questions), `mire/miren`, `este` (between commas).
- **Repetition removal** handles single-word repetitions (`"la la casa"` → `"la casa"`) and two-word phrase repetitions.
- An **audit trail** tracks every removal (type, original text) for manual review.
- Output is saved to `data/processed/{speech_id}.json`.

**Tools/libraries:** Python `re` (regex), `unicodedata`

**Key file:** `src/corpus/cleaner.py`

**Key functions:**
| Function | Purpose |
|----------|---------|
| `normalize_unicode()` | NFC normalization for accented characters |
| `normalize_punctuation()` | Standardizes quotes, dashes, ellipsis |
| `remove_fillers()` | Context-aware filler pattern removal |
| `remove_repetitions()` | Single-word and two-word phrase repetition removal |
| `clean_text()` | Orchestrates the full cleaning pipeline |
| `clean_transcript()` | Loads raw JSON, cleans each segment, saves processed JSON |

**Design decisions:**
- **Pattern-based, not blacklist:** Fillers like `"pues"` are only removed when they match filler usage patterns (e.g., sentence-initial + comma). The word `"pues"` in non-filler contexts ("pues" meaning "since" or "because") is preserved.
- **~5–10% word reduction** typical per speech.
- **Empty segments** after cleaning are discarded.

---

## 7. Step 6: NLP Pipeline

**What happens:** The cleaned text is analyzed with a multi-layer NLP pipeline that extracts tokens, lemmas, POS tags, named entities, and dependency parses for every sentence.

**How it works:**
1. **spaCy** (`es_core_news_lg`) processes the full text: sentence segmentation, tokenization, lemmatization, POS tagging, and dependency parsing.
2. For each sentence, **BETO NER** (`mrm8488/bert-spanish-cased-finetuned-ner`) runs separately to extract named entities (PER, ORG, LOC, MISC).
3. **Post-processing:** Entity text is cleaned (trailing punctuation stripped — a BETO tokenizer artifact). Entities are checked against a blacklist (`{"Pela"}` — known noise). Entities matching the **DANE gazetteer** (1,099 Colombian locations) are corrected to LOC if misclassified.
4. Character offsets are adjusted from sentence-relative to speech-relative.
5. Results are packaged into `SentenceAnalysis` objects (per sentence) and aggregated into a `SpeechAnalysis` object (per speech).

**Tools/libraries:**
- `spaCy` with `es_core_news_lg` — sentence segmentation, tokenization, POS tagging, dependency parsing
- HuggingFace `transformers` — BETO NER pipeline (`mrm8488/bert-spanish-cased-finetuned-ner`)
- DANE DIVIPOLA gazetteer — 1,099 Colombian municipality/department/region names (`data/gazetteer/colombian_locations.txt`)

**Key file:** `src/pipeline/nlp_processor.py`

**Supporting wrappers** (thin layers over the shared processor):
- `src/pipeline/tokenizer.py` — exposes `tokenize()` (text, lemma, is_stop, is_punct)
- `src/pipeline/pos_tagger.py` — exposes `pos_tag()` (token, POS tag)
- `src/pipeline/ner_extractor.py` — exposes `extract_entities()` (raw BETO output)
- `src/pipeline/parser.py` — exposes `parse_dependencies()` (dep relation, head, children)

**Key functions in `nlp_processor.py`:**
| Function | Purpose |
|----------|---------|
| `load_model()` | Cached spaCy model loader |
| `_get_ner_pipeline()` | Cached BETO NER transformer pipeline |
| `_load_gazetteer()` | Loads Colombian locations set from file |
| `analyze_sentence()` | Full NLP analysis for one sentence (spaCy + BETO + gazetteer) |
| `analyze_speech()` | Processes all sentences in a speech, aggregates entities |

**Design decisions and changes — NER migration (spaCy → BETO):**

This was a major change. Originally the pipeline used spaCy's built-in NER. Testing revealed that spaCy's CNN-based NER (trained on the AnCora corpus) produced ~60 garbage entities per speech — a **45% noise rate**, primarily from the MISC category tagging random words and full sentences as entities.

Three models were evaluated:
| Model | Result |
|-------|--------|
| spaCy `es_core_news_lg` | 133 entities, 59 MISC (55 garbage), 45% noise |
| BETO NER (`mrm8488/bert-spanish-cased-finetuned-ner`) | 75 entities, 3 MISC (all valid), **0% noise** |
| XLM-RoBERTa Large | Good accuracy but punctuation leaking issues, 3x larger model |

BETO was selected for:
- Zero garbage entities
- Smaller size (~420MB vs. XLM's ~1.2GB)
- Same architecture family as the planned intent classifier (BETO base)
- Clean subword merging with `aggregation_strategy="first"`

The **gazetteer** handles a remaining issue: Colombian municipalities named after people (e.g., "Roberto Payan", "Mosquera") are misclassified as PER by BETO. The gazetteer deterministically corrects these to LOC. This is a three-level domain adaptation strategy:
1. **Model selection** (spaCy → BETO)
2. **Post-processing** (gazetteer for known locations)
3. **Fine-tuning** (planned: after full corpus provides 4,000+ training sentences)

Full evaluation documented in `docs/NER_MODEL_EVALUATION.md`.

---

## 8. Step 7: Database Storage

**What happens:** All processed data — speech metadata, NLP annotations, entities, diarization segments — is persisted to PostgreSQL.

**How it works:**
- `load_speech()` inserts a row into the `speeches` table with metadata (title, date, location, event, YouTube URL, word count, full transcript).
- Entities from the NLP analysis are decomposed into individual rows in the `entities` table (entity_text, entity_label, character offsets, sentence_index).
- Each sentence's full annotation (tokens, POS tags, dependency parse) is stored as JSONB in the `annotations` table.
- If diarization was performed, speaker segments are stored in `speaker_segments` with the `is_target` flag indicating Cepeda's segments.
- All operations are **idempotent**: `speech_exists()` checks by YouTube URL (or title for text speeches) before inserting.

**Tools/libraries:** `psycopg2` (PostgreSQL driver), `pgvector.psycopg2` (vector type registration)

**Key file:** `src/corpus/db_loader.py`

**Key functions:**
| Function | Purpose |
|----------|---------|
| `get_connection()` | Creates psycopg2 connection from .env vars |
| `speech_exists()` | Idempotency check — prevents duplicate loading |
| `load_speech()` | Inserts speech + entities + annotations in a transaction |
| `load_diarization()` | Stores speaker segments, updates diarized flag |
| `load_chunks()` | Stores RAG chunks with embeddings (called in step 9) |
| `chunks_exist()` | Checks if chunks already exist for a speech |

**Database:** PostgreSQL 17.8 with pgvector 0.8.1 extension. DB name: `cepeda_nlp`.

**Design decisions:**
- **Decomposed storage:** NLP analysis is NOT stored as a single blob. Entities and annotations are normalized into separate tables for efficient querying (e.g., "find all speeches mentioning Bogota" is a simple SQL query).
- **JSONB for flexibility:** Tokens, POS tags, and dependency parses are stored as JSONB, allowing schema evolution without migrations.
- **ON DELETE CASCADE:** All foreign keys cascade from `speeches`, ensuring data consistency.
- **Transport security:** Remote DB connections use `DB_SSLMODE=verify-full` with bundled CA certificate to enforce TLS encryption and server identity verification. See `docs/DB_CONNECTION_SECURITY.md`.

---

## 9. Step 8: Semantic Chunking

**What happens:** The annotated sentences from the database are grouped into semantic chunks of ~200 words each, suitable for embedding and retrieval.

**How it works:**
1. Sentences are fetched from the `annotations` table for a given speech.
2. Consecutive sentences are grouped until the chunk reaches ~200 words (configurable range: 100–300 words).
3. A **1-sentence overlap** between consecutive chunks prevents information loss at chunk boundaries.
4. "Runt" chunks (final chunks under 30 words, like "Gracias.") are merged into the previous chunk.
5. **Character offsets** are computed by matching chunk text against the full transcript.
6. **Timestamps** are mapped at the sentence level: each annotation has a `start_time` column backfilled via deterministic character-offset matching against Whisper segments. At query time, `retrieve_chunks` returns a `sentences` array per chunk with per-sentence YouTube timestamp links.

**Tools/libraries:** Python standard library only (`dataclasses`, `logging`, `json`)

**Key file:** `src/rag/chunker.py`

**Key functions:**
| Function | Purpose |
|----------|---------|
| `chunk_sentences()` | Groups sentences into ~200-word chunks with overlap |
| `compute_char_offsets()` | Finds character positions of chunks in the full transcript |
| `map_chunk_timestamps()` | Maps chunks to Whisper segment timestamps via fuzzy matching |
| `chunk_speech_from_db()` | High-level: fetch sentences → chunk → compute offsets → map timestamps |

**Design decisions and changes — why sentence grouping:**

The original plan called for "dependency parse ROOT boundaries" — splitting text at clause-level boundaries. This was abandoned because:

- spaCy already segments sentences, so ROOT-splitting would fragment *within* sentences
- Individual sentences are too short (median 14 words) for meaningful embeddings — embedding models need 100–300 words for good semantic representation
- Sentence grouping respects the natural rhetorical flow of political speeches (a speaker develops a topic across 3–5 sentences)

The chunking strategy is documented in detail in `docs/RAG_DESIGN_DECISIONS.md`.

**Current stats:** 51 speeches → 174 chunks (avg ~12 per speech, ~10:1 compression ratio from sentences). 7,193 annotated sentences, all with sentence-level timestamps.

---

## 10. Step 9: Embedding

**What happens:** Each text chunk (and later, each user query) is converted into a 768-dimensional dense vector for semantic similarity search.

**How it works:**
- The model `paraphrase-multilingual-mpnet-base-v2` from the `sentence-transformers` library encodes text into 768-dimensional vectors.
- Embeddings are **normalized** to unit length, so cosine similarity equals dot product (faster computation).
- Batch processing with configurable batch size (default 32) for efficiency.
- Chunk embeddings are stored in the `speech_chunks.embedding` column (pgvector `vector(768)` type).

**Tools/libraries:** `sentence-transformers` (`paraphrase-multilingual-mpnet-base-v2`), `numpy`

**Key file:** `src/rag/embedder.py`

**Key functions:**
| Function | Purpose |
|----------|---------|
| `load_model()` | Cached model loader (~420MB, loads in ~5s on M4 Mac Mini) |
| `embed_texts()` | Batch-encodes multiple texts into embeddings (for chunk ingestion) |
| `embed_query()` | Encodes a single query string (for retrieval at query time) |

**Design decisions — model selection:**

| Model Considered | Dimensions | Why Rejected |
|-----------------|------------|--------------|
| `multilingual-e5-base` | 768 | Requires "query:" and "passage:" prefixes — more complex |
| `all-MiniLM-L12-v2` | 384 | English-only, lower dimensionality |
| `distiluse-base-multilingual` | 512 | Lower dimensionality, would require schema changes |
| **`paraphrase-multilingual-mpnet-base-v2`** | **768** | **Selected:** multilingual, proven for semantic search, normalized embeddings |

Embedding runs locally at zero API cost. Only the Claude generation step (step 11) incurs API charges.

---

## 11. Step 10: Retrieval (Semantic Search)

**What happens:** When a user asks a question, the query is embedded and compared against all chunk embeddings using pgvector cosine similarity search. The top-5 most relevant chunks are returned with full citation metadata.

**How it works:**
1. The query text is embedded using the same model (`embed_query()`).
2. A SQL query uses pgvector's `<=>` operator (cosine distance) with the HNSW index to find nearest neighbors.
3. The query JOINs with the `speeches` table to fetch citation metadata (title, date, location, event, YouTube URL).
4. Sentence-level `start_time` values from the `annotations` table provide per-sentence YouTube timestamp links.
5. Results below the similarity threshold (0.3) are filtered out.
6. Results are returned as `RetrievalResult` objects sorted by similarity (descending).

**Tools/libraries:** `psycopg2` (SQL execution), `pgvector` (cosine distance operator), `sentence-transformers` (query embedding)

**Key file:** `src/rag/retriever.py`

**Key class:** `RetrievalResult` — contains chunk text, similarity score, speech metadata, and a `youtube_link` property that constructs timestamped URLs (e.g., `https://www.youtube.com/watch?v=abc&t=120`).

**Key function:** `retrieve(query, top_k=5, threshold=0.3)` — orchestrates embed → search → filter → return.

**Design decisions:**
- **Top-5:** Enough chunks for synthesis without overwhelming Claude's context or inflating token costs.
- **Threshold 0.3:** Intentionally permissive. Observed similarity ranges for Spanish: 0.55–0.70 (highly relevant), 0.40–0.55 (moderately relevant), 0.30–0.40 (low). Better to retrieve marginally relevant chunks and let Claude filter than to miss real matches.
- **HNSW index:** `m=16, ef_construction=128`. Better recall than IVFFlat, no training step required. With ~131 chunks, sequential scan is fast anyway, but the index future-proofs for 50+ speeches (~700+ chunks).

---

## 12. Step 11: Answer Generation (Claude API)

**What happens:** The retrieved chunks are formatted into a context block and sent to Claude with a strict Spanish system prompt. Claude synthesizes a cited answer.

**How it works:**
1. `_build_context_block()` formats each chunk with a header:
   ```
   [Fragmento 1] Discurso: "TITLE" (date, location, event) [Relevancia: 0.65]
   YouTube: https://youtube.com/watch?v=abc&t=120

   chunk text here...
   ```
2. The user message is constructed: `"CONTEXTO:\n{context}\n\n---\n\nPREGUNTA: {query}"`
3. The Anthropic API is called with the system prompt + user message.

**System prompt (Spanish) enforces 7 rules:**
1. Answer ONLY from provided context — no fabrication
2. Always include citations (title, date, video link)
3. If the topic wasn't addressed: "No encontre referencias a ese tema en los discursos analizados"
4. NO invention, speculation, or editorializing
5. Maintain neutral, informational tone
6. Respond in Spanish
7. Synthesize across multiple sources when relevant

**Citation format:** `(Discurso: "TITLE", fecha — [ver video](URL))`

**Tools/libraries:** `anthropic` (Anthropic Python SDK)

**Key file:** `src/rag/generator.py`

**Key functions:**
| Function | Purpose |
|----------|---------|
| `_build_context_block()` | Formats retrieved chunks with headers and YouTube links |
| `generate()` | Calls Claude API with system prompt + context + query |

**Model configuration:**
- Dev/testing: `claude-haiku-4-5-20251001` (fast, ~$0.005/query)
- Production: `claude-sonnet-4-6` or `claude-opus-4-6` (higher quality)
- Model is specified per API call, not per key

**End-to-end orchestrator:** `src/rag/query.py` exposes `ask(query)` which calls retrieve → generate → format sources. This can be used standalone from the CLI:
```bash
python -m src.rag.query "¿Que propone sobre el racismo?"
```

---

## 13. Step 12: MCP Server (Tool Layer)

**What happens:** Instead of a traditional REST API, the system uses the **Model Context Protocol (MCP)** — Claude calls Python functions directly as tools. The MCP server provides 8 tools that give Claude structured access to the database.

**How it works:**
- The MCP server is built with `fastmcp>=3.0` (Anthropic's MCP SDK).
- Each tool is a pure Python function that queries PostgreSQL and returns structured data.
- **No LLM calls happen inside the tools.** They are pure data fetchers. Claude (running in the Streamlit frontend) decides which tools to call and when.
- The tools can also be connected to Claude Desktop for direct interaction.
- The public SSE endpoint (`run_mcp.py`) is protected by three middleware layers: API key authentication (constant-time), proxy-aware per-IP rate limiting, and SSE connection limiting (`src/mcp/middleware.py`).

**Tools/libraries:** `fastmcp>=3.0`, `psycopg2`, `pgvector`, `sentence-transformers`

**Key files:**
- `src/mcp/server.py` — 9 tool definitions
- `src/mcp/db.py` — lightweight DB connection (avoids importing heavy NLP modules)
- `src/mcp/middleware.py` — API key auth + rate limiting + SSE connection limiting middleware

**The 9 MCP tools:**

| Tool | Type | Purpose |
|------|------|---------|
| `retrieve_chunks` | Read | Semantic search — embed query, pgvector search, return cited chunks with per-sentence timestamps |
| `list_speeches` | Read | List all speeches with metadata (title, date, location, word count) |
| `get_speech_detail` | Read | Full details for one speech (transcript, entity count, chunk count) |
| `search_entities` | Read | Search named entities across all speeches by text or label |
| `get_speech_entities` | Read | All entities from one speech, grouped by NER label |
| `get_corpus_stats` | Read | Aggregate statistics (speech count, word count, entity count, etc.) |
| `submit_opinion` | **Write** | Save a user's opinion about the candidate (text + will_win boolean) |
| `get_opinions` | Read | Retrieve user opinions with summary statistics |
| `matrix_rain_easter_egg` | Read | Easter egg — triggers Matrix rain animation for creative abuse attempts |

**Design decisions and changes — MCP-only architecture:**

This was a significant architectural pivot. The original plan (Phase 6) called for a FastAPI REST API with endpoints. This was replaced with MCP-only (documented in ADR 007: `docs/decisions/007-mcp-only-no-rest-api.md`). Reasons:

1. **Simpler:** No REST routes, no HTTP server, no API layer to maintain
2. **Natural fit:** Claude already uses tool_use — MCP tools map directly to tool_use calls
3. **Reusable:** The same MCP server works in Streamlit AND Claude Desktop
4. **Cost-effective:** Every query costs ~$0.005 on Haiku — acceptable at demo/MVP scale

**Separate DB module:** `src/mcp/db.py` provides a lightweight `get_connection()` / `db_connection()` context manager that only imports `psycopg2` and `pgvector` — it does NOT import spaCy, BETO, or other heavy pipeline modules. This keeps the MCP server startup fast.

**Data cleanup:** `src/mcp/cleanup_garbled_chunks.py` was a one-time script that fixed 5 chunks with Whisper transcription errors (hallucinated or duplicated text). Each chunk had a custom cleaning function, re-embedded the cleaned text, and updated the database.

---

## 14. Step 13: Streamlit Frontend

**What happens:** A Streamlit chat interface lets users ask questions in Spanish. Claude Haiku acts as the orchestrator — it reads the user's message, decides which MCP tools to call, executes them, and generates a cited response.

**How it works:**

1. User types a message in the Streamlit chat input.
2. Streamlit sends the message (along with conversation history) to the Claude API with the system prompt and 8 tool definitions.
3. Claude analyzes the intent and returns either:
   - `stop_reason: "end_turn"` — a direct text response (for greetings, clarifications)
   - `stop_reason: "tool_use"` — one or more tool calls to execute
4. If tool_use: Streamlit executes the tool(s), sends results back to Claude as `tool_result` blocks, and loops (up to 5 rounds of tool use).
5. Claude generates the final response with citations and sources.
6. The response is rendered in the Streamlit chat interface.

**Tools/libraries:** `streamlit`, `anthropic`

**Key files:**
- `src/frontend/app.py` — Streamlit chat UI + Claude orchestration loop
- `src/frontend/prompts.py` — system prompt and tool definitions

**Key components:**

| Component | Purpose |
|-----------|---------|
| `_call_claude()` | Core orchestration loop — handles tool_use → execute → loop pattern |
| `_execute_tool()` | Dispatcher: maps tool name strings to Python functions |
| `_render_sidebar()` | Shows live corpus stats, model info, message counter |
| `main()` | Streamlit page setup, session state, chat loop |

**Constants:**
```python
MODEL = "claude-haiku-4-5-20251001"  # Fast, cheap, ~$0.005/query
MAX_TOKENS = 1024                     # Max response length
MAX_MESSAGES_PER_SESSION = 30         # Rate limiting
MAX_TOOL_ROUNDS = 5                   # Max tool-use loop iterations
```

**Example interaction flow:**
```
User: "¿Que dice sobre la educacion?"
  ↓
Streamlit → Claude API (with 8 tool definitions)
  ↓
Claude thinks: "This is a policy question → call retrieve_chunks"
  ↓
Claude returns: tool_use("retrieve_chunks", {"query": "educacion", "top_k": 5})
  ↓
Streamlit executes: retrieve_chunks("educacion", 5)
  → Embeds "educacion" → pgvector search → returns 5 chunks with metadata
  ↓
Streamlit sends tool_result back to Claude
  ↓
Claude generates: "Segun Ivan Cepeda, 'la educacion es un derecho fundamental...'
  (Discurso: 'Mitin en Bogota', 2026-02-15 — [ver video](https://youtube.com/watch?v=xyz&t=120))"
  ↓
Streamlit renders the response with clickable YouTube links
```

**Design decisions:**
- **Streaming responses:** The final Claude response is streamed token-by-token for a typewriter effect. `_run_tool_rounds()` handles tool execution loops (non-streaming `create()` calls), then `_stream_response()` streams the final text via `client.messages.stream()`. `st.write_stream()` renders the tokens as they arrive. No-tool responses (greetings) render instantly via `st.markdown()`.
- **Session-based rate limiting:** 30 messages per session. No authentication — sufficient for demo/MVP scale.
- **API key safety:** Because Streamlit runs Python server-side, the Anthropic API key is never exposed to the browser.
- **Tool dispatch is direct:** MCP tools are imported from `src/mcp/server` and called as local Python functions — no HTTP, no IPC, no serialization overhead. This means the MCP layer is **not a network attack surface** — there is nothing to intercept between Streamlit and the tools.
- **Abuse detection:** Two-layer defense against prompt injection and abuse. Layer 1: regex pre-LLM filter (`src/frontend/abuse_detector.py`) catches structural attacks (SQLi, XSS, prompt injection keywords) at zero API cost. Layer 2: `matrix_rain_easter_egg` MCP tool lets Claude trigger a Matrix rain animation for creative social-engineering attacks that bypass regex.

**Known issue — Haiku tool hallucination:** In rare cases, Haiku fabricated tool results without actually calling the tool (e.g., returning a fake `opinion_id`). Fixed by adding explicit rules to the system prompt: `"NUNCA finjas haberla guardado"` (never pretend you saved it without calling the tool). If the issue recurs, the orchestrator can be upgraded to Sonnet.

---

## 15. Pipeline Orchestration

**What happens:** All the steps above (download → diarize → transcribe → clean → NLP → DB → chunk → embed) are coordinated by a single orchestrator that processes speeches one at a time.

**Key file:** `src/corpus/pipeline_runner.py`

**Pipeline steps per speech:**
```
1. Get transcript  →  download + diarize + transcribe (YouTube)
                      OR load text file (website source)
2. Clean transcript → remove fillers, repetitions, normalize
3. NLP analysis     → spaCy tokenize/POS/parse + BETO NER + gazetteer
4. Load to DB       → speeches + entities + annotations tables
5. Chunk + embed    → sentence grouping + sentence-transformers + pgvector
```

**CLI usage:**
```bash
# Process 5 new speeches
python -m src.corpus.pipeline_runner --new=5

# Process all remaining speeches
python -m src.corpus.pipeline_runner --all

# Must use nohup for long-running jobs (survives session end)
nohup python -m src.corpus.pipeline_runner --new=5 > data/pipeline_run.log 2>&1 &
```

**Design decisions:**
- **Stream processing:** One speech at a time, audio deleted after transcription. Peak disk usage ~60MB.
- **Idempotent:** Every step checks for existing output before processing. The pipeline can be killed and restarted without re-downloading or re-processing.
- **Auto-RAG:** Step 5 (chunk + embed) is hooked into the pipeline via an import guard. If `sentence-transformers` is installed, every new speech gets RAG-ready automatically. If not installed, the pipeline still works but skips chunking.
- **Backfill script:** `src/rag/backfill.py` was a one-time script to chunk and embed speeches that were loaded before the RAG system existed. Now unnecessary since auto-RAG is integrated.
- **nohup required:** The pipeline processes are long-running (7.5 min per speech for transcription alone). They must be launched with `nohup` to survive session disconnection.

---

## 16. Database Schema

**Database:** PostgreSQL 17.8 + pgvector 0.8.1
**Schema file:** `schema.sql`

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `speeches` | Core speech metadata | title, candidate, speech_date, location, event, youtube_url, cleaned_transcript, word_count |
| `entities` | NER-extracted named entities | speech_id, entity_text, entity_label (PER/ORG/LOC/MISC), start_char, end_char |
| `annotations` | Sentence-level NLP output | speech_id, sentence_index, sentence_text, tokens (JSONB), pos_tags (JSONB), dep_parse (JSONB), start_time (FLOAT — Whisper segment timestamp) |
| `speech_chunks` | RAG chunks + embeddings | speech_id, chunk_text, embedding vector(768), metadata JSONB (start_time), sentence_start/end |
| `speech_topics` | Topic modeling results | speech_id, topic, confidence, source (bertopic/manual) — reserved for Phase 2 |
| `speaker_segments` | Diarization results | speech_id, speaker_label, is_target, start_time, end_time, confidence |
| `user_opinions` | User-submitted opinions | opinion_text, will_win (boolean), created_at |

**Key index:** `idx_chunks_embedding_hnsw` — HNSW vector index on `speech_chunks.embedding` with `vector_cosine_ops`, `m=16`, `ef_construction=128`. This is what makes semantic search fast.

---

## 17. Test Suite

**183 tests** across multiple test files, all passing.

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `tests/corpus/test_diarizer.py` | 8 | Timestamp remapping, audio extraction, reference embedding loading |
| `tests/rag/test_chunker.py` | 11 | Sentence grouping, overlap, runt merging, word count, timestamp mapping |
| `tests/rag/test_retriever.py` | 4 | YouTube link construction, similarity threshold filtering, empty results |
| `tests/rag/test_embedder.py` | 4 | Embedding provider switching (local vs HF API) |
| `tests/mcp/test_tools.py` | 15 | All 9 MCP tools (data access, opinion writes, `conn.commit()` verification) |
| `tests/mcp/test_security.py` | 9 | SQL injection prevention, input validation, opinion security |
| `tests/frontend/test_abuse_detector.py` | 75 | Regex-based abuse detection patterns (SQLi, XSS, prompt injection) |
| `tests/frontend/test_visualizations.py` | 24 | Chart rendering, Colombia bubble map |
| `tests/corpus/test_timestamp_backfill.py` | 7 | Sentence-level timestamp backfill logic |
| Other test files | 23 | Additional coverage across modules |

**Testing approach:**
- Unit tests with mocked DB connections (no live database required)
- Real audio file creation with `torchaudio` for diarization tests
- Parameterized SQL verification for injection safety

---

## 18. Deployment Architecture

**Principle: Process locally, serve from cloud (all free tiers).**

```
Mac Mini (local)                    Cloud
├── yt-dlp (download)               ├── Supabase PostgreSQL 17 + pgvector (free tier)
├── pyannote (diarize)               ├── Streamlit Community Cloud (frontend + Claude orchestration)
├── Whisper (transcribe)             ├── Render (MCP SSE server, API key + rate limiting)
├── spaCy + BETO (NLP)              ├── Claude API (Haiku 4.5)
├── sentence-transformers (embed)   └── HuggingFace Inference API (query embedding)
└── pipeline_runner ──writes to──→ Supabase
```

**Sync method:** Point the pipeline at the Supabase endpoint by changing `DB_HOST` in `.env`. No export/import needed — the pipeline writes directly to production DB.

**Connection security:** Remote DB connections use `DB_SSLMODE=verify-full` with bundled Supabase CA cert. MCP tool calls from Streamlit stay in-process (no network hop). The public MCP SSE endpoint on Render is protected by three middleware layers: API key authentication, per-IP rate limiting, and SSE connection limiting. Full analysis in `docs/DB_CONNECTION_SECURITY.md`.

**Cost:** $0/month (all free tiers) + ~$0.0047/query Claude API. Interview demo (50 queries) costs ~$0.23.

**Why not process on cloud?** The M4 Mac Mini handles Whisper + pyannote + spaCy for free. Cloud GPU instances would add cost and operational complexity. Full corpus processing: ~8 hours local (free).

Documented in `docs/DEPLOYMENT_ARCHITECTURE.md`.

---

## 19. Design Decisions & Trade-offs

| Decision | Chosen | Alternative Rejected | Rationale |
|----------|--------|---------------------|-----------|
| **Chunking strategy** | Sentence grouping (~200 words) | ROOT-boundary splitting, fixed character count | Individual sentences too short (~14 words); character splitting breaks rhetoric |
| **Embedding model** | paraphrase-multilingual-mpnet-base-v2 (768d) | E5, MiniLM, distiluse | Multilingual, proven for semantic search, matches schema |
| **NER model** | BETO + DANE gazetteer | spaCy NER, XLM-RoBERTa | spaCy had 45% noise; XLM had punctuation leaking |
| **Vector index** | HNSW (m=16, ef_construction=128) | IVFFlat | Better recall, no training step, future-proof |
| **Architecture** | MCP-only (no REST API) | FastAPI REST API | Simpler, natural fit for Claude tool_use, reusable in Claude Desktop |
| **Transcription** | Whisper large-v3 on CPU | Whisper on MPS, Whisper API | CPU 1.9x faster than MPS on M4; local = free |
| **Diarization threshold** | 0.25 cosine similarity | 0.5+ (standard) | Cross-recording voice variation ~0.3–0.4; permissive avoids false negatives |
| **Orchestrator model** | Claude Haiku 4.5 | Sonnet, Opus | $0.005/query vs. $0.014 or $0.023; good enough for tool selection |
| **Retrieval threshold** | 0.3 similarity | Higher thresholds | Permissive — let Claude filter irrelevance in generation |
| **Processing model** | Stream (one speech at a time) | Batch (all at once) | ~60MB peak disk vs. 100GB+ intermediate files |
| **Deployment** | Process locally, serve from AWS | Full cloud | Free local compute; only pay for serving |

All major decisions are documented as Architecture Decision Records (ADRs) in `docs/decisions/001-009.md`.

---

## 20. API Cost Analysis

**Per-query costs (measured on real queries, 10-speech corpus):**

| Model | Input (~2,700 tokens) | Output (~393 tokens avg) | Total per Query |
|-------|-----------------------|--------------------------|-----------------|
| Haiku 4.5 | $0.0027 | $0.0020 | **$0.0047** |
| Sonnet 4.6 | $0.0081 | $0.0059 | **$0.014** |
| Opus 4.6 | $0.0135 | $0.0098 | **$0.023** |

**Projections:**
| Scenario | Queries | Haiku Cost |
|----------|---------|------------|
| MVP demo | 50 | $0.23 |
| Dev & testing | 500 | $2.33 |
| Light production (100/day) | 3,000/month | $14/month |

**Key insight:** Embedding and retrieval run locally at zero cost. Only the Claude generation step costs money. Full analysis in `docs/API_COST_ANALYSIS.md`.

---

## File Reference

| Module | Key Files |
|--------|-----------|
| **Corpus Building** | `src/corpus/downloader.py`, `src/corpus/diarizer.py`, `src/corpus/transcriber.py`, `src/corpus/cleaner.py`, `src/corpus/db_loader.py`, `src/corpus/pipeline_runner.py` |
| **NLP Pipeline** | `src/pipeline/nlp_processor.py`, `src/pipeline/tokenizer.py`, `src/pipeline/pos_tagger.py`, `src/pipeline/ner_extractor.py`, `src/pipeline/parser.py` |
| **RAG System** | `src/rag/chunker.py`, `src/rag/embedder.py`, `src/rag/retriever.py`, `src/rag/generator.py`, `src/rag/query.py`, `src/rag/backfill.py` |
| **MCP Server** | `src/mcp/server.py`, `src/mcp/db.py`, `src/mcp/middleware.py`, `run_mcp.py` |
| **Frontend** | `src/frontend/app.py`, `src/frontend/prompts.py`, `src/frontend/abuse_detector.py`, `src/frontend/visualizations.py` |
| **Data** | `data/gazetteer/colombian_locations.txt`, `data/reference_embedding.npy`, `data/speech_manifest.json` |
| **Schema** | `schema.sql` |
| **Tests** | `tests/corpus/test_diarizer.py`, `tests/rag/test_chunker.py`, `tests/rag/test_retriever.py`, `tests/mcp/test_tools.py`, `tests/mcp/test_security.py` |
| **Documentation** | `docs/RAG_DESIGN_DECISIONS.md`, `docs/NER_MODEL_EVALUATION.md`, `docs/API_COST_ANALYSIS.md`, `docs/DEPLOYMENT_ARCHITECTURE.md`, `docs/DB_CONNECTION_SECURITY.md`, `docs/PHASE6_API_PLAN.md`, `docs/decisions/001-008.md` |
