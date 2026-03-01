# Project Checklist: Political Speech NLP Analyzer + RAG Assistant

## Phase 0: Project Setup ✓
- [x] Initialize git repository
- [x] Create folder structure (src/, data/, notebooks/, docs/, models/)
- [x] Set up Python virtual environment (Python 3.13)
- [x] Create requirements.txt with initial dependencies
- [x] Set up PostgreSQL database + schema
- [x] Create CLAUDE.md with project conventions
- [x] Create .gitignore (audio files, model weights, .env, __pycache__, etc.)
- [x] Create .env file for API keys (Anthropic, etc.)

## Phase 1: Corpus Building & Preprocessing (Module 1) — IN PROGRESS
- [x] Build YouTube audio downloader (yt-dlp)
- [x] Collect ~50 speech URLs for the candidate
- [x] Build speaker diarization pipeline (pyannote-audio)
- [x] Build Whisper transcription pipeline
- [ ] Download, diarize, and transcribe all speeches (10/44 done, pipeline idle)
- [x] Build spoken-language cleaning pipeline:
  - [x] Filler word removal ("eh", "este", "digamos", "o sea")
  - [x] Handle false starts and repetitions
  - [x] Normalize encoding and punctuation
  - [x] Regex-based text normalization
- [ ] Manual review of cleaned transcripts (spot-check quality)
- [x] Build NLP pipeline (spaCy + BETO NER + gazetteer):
  - [x] Tokenization
  - [x] Lemmatization / Stemming
  - [x] POS tagging
  - [x] Named Entity Recognition (NER) — BETO NER + DANE gazetteer
  - [x] Dependency parsing
- [x] Design PostgreSQL schema (speeches, entities, metadata, annotations)
- [x] Build DB loader script
- [ ] Load all processed data into PostgreSQL (10/44 speeches loaded, 131 chunks)
- [ ] Create notebook: 01_transcription_exploration.ipynb (DEFERRED — after RAG)
- [ ] Create notebook: 02_nlp_pipeline_demo.ipynb (DEFERRED — after RAG)

## Phase 2: Linguistic Analysis (Module 2) — DEFERRED (after RAG)
- [ ] N-gram analysis (bigrams, trigrams, frequencies)
- [ ] Build basic statistical language model from n-grams
- [ ] Topic modeling (BERTopic or LDA) across all speeches
- [ ] Sentiment analysis per speech and per topic
- [ ] Keyword/phrase frequency over time (rhetorical evolution)
- [ ] Register and style analysis:
  - [ ] Formal vs. informal markers
  - [ ] Rhetorical devices (anaphora, tricolon, rhetorical questions)
  - [ ] Average sentence length, vocabulary richness
- [ ] Build quantified syntactic style profile from dependency parses:
  - [ ] Avg sentence length, infinitive rate, first-person plural rate
  - [ ] Evaluative adjective density
  - [ ] Top recurring dep patterns (e.g., "gracias por + INF", "tenemos que + INF")
  - [ ] Export as structured data to feed into personality guide (Phase 5)
- [ ] N-gram vs. BERT comparison demo
- [ ] Create notebook: 03_corpus_analysis.ipynb
- [ ] Create notebook: 04_topic_modeling.ipynb
- [ ] Create notebook: 05_ngram_vs_bert.ipynb

## Phase 3: Formal Linguistics Layer (Module 5) — DEFERRED (after RAG)
- [ ] Implement Context-Free Grammar (CFG) for Spanish subset (NLTK)
- [ ] Parse sample sentences from speeches, visualize parse trees
- [ ] Build political discourse domain ontology:
  - [ ] Define concepts (Proposal, Topic, Speech, Event, Person, etc.)
  - [ ] Define topic taxonomy (economy, security, education, health, etc.)
  - [ ] Define relationships (proposal_belongs_to_topic, etc.)
  - [ ] Store as JSON-LD or Python dict
- [ ] Create notebook: 06_formal_grammar.ipynb

## Phase 4: RAG System (Module 3) — COMPLETE ✓
- [x] Build semantic chunker (sentence-grouping strategy, ~200 words/chunk)
  - [x] Sentence-level grouping with 1-sentence overlap (see docs/RAG_DESIGN_DECISIONS.md)
- [x] Review chunks manually for coherence
- [x] Choose embedding model: `paraphrase-multilingual-mpnet-base-v2` (768d)
- [x] Embed all speech chunks (131 chunks across 10 speeches)
- [x] Set up vector storage: pgvector with HNSW index
- [x] Build retrieval pipeline: cosine similarity + JOIN for citations
- [x] Integrate Anthropic Claude API for response generation
- [x] Implement citation system (speech title, date, YouTube URL + timestamp)
- [x] Build backfill script for existing speeches
- [x] Hook chunk+embed into pipeline_runner (auto-indexes new speeches)
- [x] Test end-to-end: question -> retrieval -> cited answer (20/20 tests passing)
- [ ] Build FastAPI backend endpoint for RAG queries (Phase 6)
- [ ] Compare baseline prompt vs. linguistically-conditioned prompt (Phase 7):
  - [ ] Baseline: raw RAG (retrieve + generate)
  - [ ] Enhanced: RAG + syntactic style profile injected into prompt
  - [ ] Measure lexical/syntactic similarity to candidate's actual speech

## Phase 5: Personality, Style Guide & Dialogue (Module 4) — DEFERRED (after RAG)
- [ ] Write personality/style guide document (PERSONALITY_GUIDE.md):
  - [ ] Register, tone, vocabulary constraints
  - [ ] Ambiguity handling strategies
  - [ ] Out-of-scope handling
  - [ ] Biased question handling
  - [ ] Forbidden patterns
- [ ] Label 200-400 intent examples manually
- [ ] Fine-tune BERT intent classifier (BETO)
- [ ] Evaluate classifier accuracy
- [ ] Build dialogue manager (state machine or Rasa):
  - [ ] Greeting flow
  - [ ] Query -> retrieval -> response flow
  - [ ] Follow-up handling (multi-turn state)
  - [ ] Topic change detection
  - [ ] Fallback / out-of-scope handling
- [ ] Implement pragmatic flow handling:
  - [ ] Vague question narrowing
  - [ ] Repeated question variation
  - [ ] Opinion request deflection
  - [ ] User frustration handling
- [ ] Create conversation flow diagrams (CONVERSATION_FLOWS.md)

## Phase 6: MCP Server + Streamlit Frontend (Module 6) ✓
- [x] Build MCP server with 8 tools (`src/mcp/server.py`):
  - [x] `retrieve_chunks` — pgvector search, returns chunks + citations (no LLM call)
  - [x] `list_speeches` — all speeches with metadata
  - [x] `get_speech_detail` — single speech + entity/chunk counts
  - [x] `search_entities` — entity search across speeches
  - [x] `get_speech_entities` — entities for one speech
  - [x] `get_corpus_stats` — corpus-wide statistics
  - [x] `submit_opinion` — save user opinion (text + will_win) to DB
  - [x] `get_opinions` — retrieve opinions with summary stats
- [x] Add `user_opinions` table to schema + live DB
- [x] Add input validation + prompt injection filter to MCP tools
- [x] Write MCP tool tests (`tests/mcp/`) — 29 tests (tools + security)
- [x] Build Streamlit chat interface (`src/frontend/app.py`):
  - [x] Claude (Haiku) as orchestrator with tool_use
  - [x] System prompt with personality, citation format, constraints
  - [x] Multi-turn conversation with `st.chat_message`
  - [x] Session-based rate limiting (30 msg/session)
- [x] Add source citations with YouTube timestamp links
- [x] Add inline visualizations (5 chart types + Colombia bubble map, 24 viz tests)
- [x] Add `limit` parameter to `search_entities` tool (chart count matches user request)
- [ ] Add ethical disclaimer
- [ ] Test with Claude Desktop as bonus MCP client

## Phase 7: Final Polish
- [ ] Write README.md with project overview, setup instructions, architecture diagram
- [ ] Record short demo video (optional)
- [ ] Prepare MVP talking points
- [ ] Test full system end-to-end with diverse questions
- [ ] Add ethical disclaimer to the interface
- [ ] Clean up code, add type hints where needed
- [ ] Final spot-check of all notebooks
