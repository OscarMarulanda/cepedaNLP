# CLAUDE.md — Political Speech NLP Analyzer

## Project
Spanish-language political speech analyzer + RAG-powered conversational assistant. Analyzes ~50 speeches by **Iván Cepeda** (2026 Colombian presidential candidate). Users ask questions and get cited, grounded answers synthesized from actual speech fragments.

## Language & Conventions
- **Python 3.13** (venv) — primary language for everything
- **Spanish** is the language of all speech data, UI text, and bot responses
- **English** for code, comments, variable names, docstrings, and docs
- Follow PEP 8. Use type hints on public functions.
- Use `snake_case` for functions/variables, `PascalCase` for classes
- Use `pathlib.Path` over `os.path`
- Use `logging` module, not `print()`, for operational output
- f-strings for string formatting

## Project Structure
```
src/corpus/       — download, transcribe, clean, load to DB, embed
src/pipeline/     — spaCy NLP pipeline (tokenize, POS, NER, parse)
src/analysis/     — n-grams, topics, sentiment, rhetorical analysis
src/rag/          — semantic chunking, retrieval, Claude generation
src/chatbot/      — intent classification, dialogue management, personality
src/linguistics/  — formal grammar (CFG), domain ontology
data/             — audio/, raw/, processed/, intents/, ontology/
notebooks/        — numbered Jupyter notebooks (01_ through 06_)
docs/             — PERSONALITY_GUIDE.md, CONVERSATION_FLOWS.md, etc.
models/           — saved fine-tuned models (gitignored)
```

## Key Tech
- **NLP:** spaCy `es_core_news_lg`, NLTK
- **ML:** HuggingFace transformers, BETO (`dccuchile/bert-base-spanish-wwm-cased`)
- **Embeddings:** sentence-transformers (multilingual)
- **Topics:** BERTopic
- **DB:** PostgreSQL + pgvector
- **LLM:** Anthropic Claude API (constrained by style guide — last mile only)
  - Dev/testing: `claude-haiku-4-5-20251001` (fast, cheap)
  - Production generation: `claude-sonnet-4-6` or `claude-opus-4-6` (higher quality)
  - Model is specified per API call, not per key
- **Transcription:** OpenAI Whisper (local model)
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Audio:** yt-dlp

## Design Principles
- Claude is the mouth, BERT is the brain. The LLM only generates from pre-retrieved context.
- Every answer must include citations (speech title, date, event).
- Neutral and informational — no advocacy, no editorializing, no fabrication.
- If the candidate never addressed a topic, say so transparently.
- Semantic chunking uses linguistic units, not character counts.

## Database
- PostgreSQL 17.8 with pgvector 0.8.1 extension
- DB: `cepeda_nlp`, user: `oscarm`, host: localhost:5432
- Tables: speeches, entities, speech_topics, annotations, speech_chunks
- All processed data lives in the DB, not flat files
- Schema defined in `schema.sql`

## Testing
- Tests live in `tests/` mirroring `src/` structure
- Use `pytest`
- Test NLP pipelines with known input/output pairs

## Environment
- Virtual environment via `venv` (not conda)
- API keys in `.env` (never committed)
- Dependencies in `requirements.txt`

## Build Order (RAG-first)
Phase 0 (done) → 1 (in progress) → **4 (RAG — next)** → 6 (frontend) → 2, 3, 5 (linguistics — deferred) → 7 (polish)
See CHECKLIST.md for details. NLP pipeline runs during ingestion so annotations accumulate for later analysis.
