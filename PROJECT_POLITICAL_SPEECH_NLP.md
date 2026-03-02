# Proyecto: Analizador de Discursos Políticos + Asistente RAG

## Context & Purpose

This project involves training conversational assistants (personality, style guide, linguistic traits), NLP processing, conversation flow design, and working with language models from n-grams to transformers.

**Oscar's background:**
- BA in English Philology (Universidad Nacional de Colombia, 2018)
- 6 years teaching English (deep pragmatics/grammar intuition)
- Software Engineer: Python, Java, Go, React Native, Laravel, Node.js
- Databases: SQL, PostgreSQL, MongoDB, MariaDB
- Infrastructure: AWS, Terraform, Docker
- Bilingual: Spanish (native), English (C2)


**What he needs to demonstrate:**
- NLP pipeline skills (tokenization, lemmatization, POS tagging, NER, parsing)
- Chatbot personality/style guide design
- Pragmatic conversation flow handling
- Language models: statistical (n-grams) AND neural (BERT/transformers)
- Corpus management, annotation, regex, scraping, SQL/NoSQL
- Formal grammars and ontologies

---

## Project Overview

A **Spanish-language political speech analyzer + RAG-powered conversational assistant** built around a corpus of **~50 speeches by a Colombian presidential candidate** (~25 hours of audio, ~200k+ words after transcription).

Users can ask the assistant questions about the candidate's proposals, ideas, and positions. The system **retrieves the actual speech fragments** where the candidate addressed the topic, then uses the Anthropic Claude API to synthesize a coherent answer **in the candidate's linguistic style** — with citations back to specific speeches and dates.

**The LLM (Claude) is the last mile, not the whole system.** Oscar builds everything upstream: the transcription pipeline, corpus cleaning, full NLP analysis, embedding index, semantic retrieval, intent classification, dialogue management, personality/style guide, and pragmatic flow handling. The LLM receives pre-retrieved context and strict style constraints — it generates within boundaries Oscar defines.

**Important: This is an analytical/informational tool, not advocacy.** The assistant presents what the candidate said, with sources. It does not editorialize or promote.

### How BERT and Claude divide the work

BERT and Claude are not redundant — they do completely different jobs:

| Component | Role | When it runs |
|---|---|---|
| **Fine-tuned BERT (intent classifier)** | Classifies what the user wants: `consultar_propuesta`, `pedir_cita_textual`, etc. Determines dialogue flow. | First — before anything else |
| **BERT embeddings (sentence-transformers)** | Creates the vector index over speech chunks. Embeds user queries. **Finds** the relevant speech fragments via similarity search. | Indexing: once, offline. Retrieval: every user query |
| **BERTopic** | Automatically discovers topics across all 50 speeches. | Offline corpus analysis only |
| **N-gram vs BERT comparison** | Analytical demo — compares statistical vs. neural language models on the same corpus. | Notebook demo only |
| **Claude API** | Takes the already-retrieved fragments + style guide constraints and writes a coherent, cited answer. | Last step only — never sees the full corpus |

**BERT is the brain** (understands user intent, finds relevant content). **Claude is the mouth** (writes the final response from pre-selected material). Without BERT, Claude wouldn't know which fragments to talk about.

---

## Architecture: 6 Modules

### Module 1: Corpus Building & Preprocessing
**Purpose:** Show ability to build and manage a real linguistic corpus from raw audio.

**What it does:**
- Download speech videos/audio from YouTube
- **Speaker diarization** using **pyannote-audio** — identifies *who* is speaking and *when* in multi-speaker recordings (panels, debates, interviews). Extracts only the candidate's audio segments before transcription, saving processing time and ensuring corpus purity.
  - Reference voice embedding for speaker identification across videos
  - Pre-transcription filtering: diarize → identify → extract → transcribe only the candidate
  - Timestamp remapping to preserve original video timing
- Transcribe using **OpenAI Whisper** (audio → text) — demonstrates speech-to-text processing
- Cleaning pipeline for spoken language:
  - Remove filler words ("eh", "este", "digamos", "o sea")
  - Handle false starts, repetitions, audience interruptions
  - Normalize encoding, punctuation, speaker turns
  - Regex-based text normalization
- Full NLP analysis on the cleaned corpus:
  - Tokenization, lemmatization, stemming (spaCy `es_core_news_lg` + NLTK)
  - POS tagging (grammatical categories for every word)
  - Named Entity Recognition (people, places, organizations, dates mentioned)
  - Dependency parsing (syntactic tree visualization)
- Store everything in **PostgreSQL**:
  - Raw transcripts (with timestamps)
  - Cleaned text
  - Extracted entities
  - Metadata per speech (date, location, event, duration, topic tags)

**Tech:**
- Python, OpenAI Whisper (local model, not API)
- pyannote-audio (speaker diarization + speaker embedding verification)
- spaCy (Spanish model), NLTK
- Regex for spoken-language cleaning
- PostgreSQL
- yt-dlp for downloading audio
- Jupyter notebooks for exploration/visualization

**Deliverable:** A populated PostgreSQL database with 50 transcribed, cleaned, and linguistically annotated speeches (with speaker diarization metadata) + a notebook demonstrating the full NLP pipeline with visualizations (dependency trees, entity highlighting, POS distribution charts).

---

### Module 2: Linguistic Analysis
**Purpose:** Show ability to extract meaningful patterns from a corpus — this is where the philology degree shines.

**What it does:**
- **N-gram analysis:** What phrases does the candidate repeat most? Bigrams, trigrams, and their frequencies. Build a basic statistical language model.
- **Topic modeling** (BERTopic or LDA): Automatically discover the main themes across all 50 speeches (economy, security, education, health, agrarian reform, etc.)
- **Sentiment analysis** per speech and per topic: How does the candidate's emotional tone shift depending on the subject?
- **Keyword/phrase frequency over time:** Does the candidate talk more about economy early in the campaign and security later? Track rhetorical evolution.
- **Register and style analysis:** Formal vs. informal markers, use of "usted" vs. "tú", rhetorical devices (anaphora, tricolon, rhetorical questions), average sentence length.
- **Compare n-gram model predictions vs. BERT predictions** — statistical vs. neural language modeling.

**Tech:**
- NLTK for n-gram models
- BERTopic or Gensim (LDA) for topic modeling
- Hugging Face transformers for sentiment analysis and BERT comparisons
- Matplotlib, Plotly for visualizations
- Jupyter notebooks

**Deliverable:** Jupyter notebooks with full corpus analysis: topic distribution, sentiment trends, rhetorical patterns, n-gram vs. BERT comparison, and visualizations showing linguistic evolution across the campaign.

---

### Module 3: RAG System
**Purpose:** Show ability to build retrieval-augmented generation — the core architecture companies use for grounded conversational AI.

**What it does:**
- **Semantic chunking** of speeches: Not by character count, but by paragraph or semantic unit. This requires linguistic judgment — knowing where one idea ends and another begins. This is where the philology background directly applies.
- **Embed chunks** using a Spanish-capable embedding model (e.g., `sentence-transformers` with a multilingual model, or BETO-based embeddings)
- **Store embeddings** in a vector database (pgvector extension in PostgreSQL, or ChromaDB)
- **Retrieval pipeline:**
  1. User asks a question (e.g., "¿Qué propone sobre reforma agraria?")
  2. Query is embedded
  3. Top-k most relevant speech chunks are retrieved via similarity search
  4. Retrieved chunks are passed to Claude API as context
  5. Claude generates a coherent answer **constrained by the style guide** (Module 4)
  6. Response includes **citations**: which speech, what date, what event
- Example output: *"En su discurso del 15 de marzo en Cali, el candidato propuso [...]. Más tarde, en el debate del 2 de abril, amplió esta idea diciendo [...]."*

**Tech:**
- sentence-transformers (multilingual embeddings)
- pgvector (PostgreSQL extension) or ChromaDB
- Anthropic Claude API (response generation with retrieved context)
- FastAPI (API backend)
- Python

**Deliverable:** Working RAG pipeline that answers questions about the candidate's positions with cited sources from specific speeches.

---

### Module 4: Personality, Style Guide & Dialogue Management
**Purpose:** Show ability to design conversational assistants with personality and pragmatic awareness — the core of the job.

**Personality & Style Guide Document (critical deliverable):**
- Written document defining the assistant's linguistic personality:
  - Register: formal but accessible, uses "usted" (mirrors the candidate's public register)
  - Tone: informative, neutral, respectful — presents facts without editorializing
  - Vocabulary: political/policy vocabulary, avoids colloquialisms unless quoting the candidate directly
  - How it handles ambiguity: asks clarifying questions ("¿Se refiere a la reforma tributaria o a la reforma agraria?")
  - How it handles topics the candidate never addressed: transparently says "No encontré referencias a ese tema en los discursos analizados"
  - How it handles leading/biased questions: redirects to factual content ("Lo que el candidato dijo específicamente fue...")
  - How it handles requests for comparison with other candidates: politely declines, stays in scope
  - How it handles controversial topics: presents the candidate's stated position with direct quotes, no interpretation
  - Forbidden patterns: no editorializing, no speculation, no invented positions
- This document is a DELIVERABLE on its own — it maps directly to the job requirement "personalidad y guía de estilo"

**Intent Classification:**
- ~200-400 manually labeled examples (2-3 hours of work)
- Fine-tuned BERT model (Hugging Face `transformers`) for intent classification
- Intents: `consultar_propuesta`, `pedir_cita_textual`, `comparar_temas`, `consultar_biografia`, `preguntar_cronologia`, `fuera_de_tema`, `saludar`, `despedirse`

**Dialogue Management:**
- State machine or Rasa-based flow control
- Defined flows: greeting → user asks question → system retrieves + responds with citations → follow-up questions → new topic or farewell
- Multi-turn state: remembers what topic is being discussed, handles follow-ups ("¿Y qué más dijo sobre eso?")
- Fallback handling for out-of-scope inputs

**Pragmatic Flow Handling (directly maps to "tratamiento pragmático de flujos"):**
- Topic changes mid-conversation
- Vague or incomplete questions ("¿Qué piensa de la economía?" — too broad, needs narrowing)
- User frustration or disagreement with the candidate's positions
- Repeated questions (vary the response, don't repeat verbatim)
- Requests for opinions (the bot has none — it reports what was said)

**Tech:**
- Python, Rasa (open source) OR FastAPI + custom state machine
- Hugging Face transformers for intent classification (fine-tuned BERT)
- spaCy for entity extraction from user input
- Anthropic Claude API (constrained by style guide for response generation)

**Deliverable:** Working dialogue system + personality/style guide document + conversation flow diagrams.

---

### Module 5: Formal Linguistics Layer
**Purpose:** Show knowledge of formal grammars and ontologies.

**What it does:**
- Implement a Context-Free Grammar (CFG) for a subset of Spanish using NLTK
- Parse sample sentences from the speeches and visualize parse trees
- Build a small **domain ontology for political discourse**:
  - Concepts: Proposal, Topic, Speech, Event, Person, Organization, Date, Region
  - Topics: economy, security, education, health, agrarian_reform, infrastructure, corruption, peace_process
  - Relationships: proposal_belongs_to_topic, speech_covers_topic, candidate_mentioned_person, proposal_targets_region
  - Store as JSON-LD or simple Python dict structure
- The ontology **helps the RAG system understand relationships** between topics and proposals — e.g., knowing that "reforma agraria" and "tierras" and "campesinos" belong to the same topic cluster

**Tech:**
- NLTK (CFG parser)
- Python (ontology representation)
- Optional: OWL/RDF if feeling ambitious

**Deliverable:** CFG grammar file + ontology definition + notebook demonstrating parsing and ontology queries.

---

### Module 6: Frontend & Demo
**Purpose:** Make the project presentable and interactive.

**What it does:**
- **Chat interface:** Users type questions, get cited answers in the candidate's linguistic style
- **Analytics dashboard:** Visualizations from Module 2 — topic distribution across speeches, entity frequency, sentiment timeline, rhetorical patterns
- **Speech explorer:** Browse individual speeches, see their NLP annotations, entities, topics
- **Transparency layer:** Every response shows its sources (which speech, what date, direct quotes)

**Tech:**
- Streamlit (fast to build, good for demos)
- Plotly for interactive charts
- Alternative: Vue.js if you want to flex your frontend skills

**Deliverable:** Working web application with chat + analytics dashboard.

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| Language | Python (primary) |
| Transcription | OpenAI Whisper (local) |
| Speaker Diarization | pyannote-audio (`speaker-diarization-community-1`) |
| NLP Libraries | spaCy (es_core_news_lg), NLTK, Hugging Face transformers |
| ML/Classification | Fine-tuned BERT (e.g., `dccuchile/bert-base-spanish-wwm-cased`) |
| Topic Modeling | BERTopic or Gensim (LDA) |
| Embeddings | sentence-transformers (multilingual) |
| Vector Search | pgvector (PostgreSQL) or ChromaDB |
| Database | PostgreSQL (corpus, metadata, vectors) |
| LLM | Anthropic Claude API (response generation, constrained by style guide) |
| Backend/API | FastAPI |
| Chatbot Framework | Rasa OR custom state machine |
| Frontend/Demo | Streamlit |
| Notebooks | Jupyter |
| Visualization | Matplotlib, Plotly, displaCy (spaCy's visualizer) |
| Audio Download | yt-dlp |
| Version Control | Git + GitHub |

---

## Suggested Folder Structure

```
political-speech-nlp/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_transcription_exploration.ipynb
│   ├── 02_nlp_pipeline_demo.ipynb
│   ├── 03_corpus_analysis.ipynb
│   ├── 04_topic_modeling.ipynb
│   ├── 05_ngram_vs_bert.ipynb
│   └── 06_formal_grammar.ipynb
├── src/
│   ├── corpus/
│   │   ├── downloader.py          # yt-dlp audio download
│   │   ├── diarizer.py            # pyannote speaker diarization + identification
│   │   ├── transcriber.py         # Whisper transcription
│   │   ├── cleaner.py             # spoken-language cleaning (regex, rules)
│   │   ├── db_loader.py           # PostgreSQL loading
│   │   └── embedder.py            # builds embedding index for RAG
│   ├── pipeline/
│   │   ├── tokenizer.py
│   │   ├── pos_tagger.py
│   │   ├── ner_extractor.py
│   │   └── parser.py
│   ├── analysis/
│   │   ├── ngram_model.py
│   │   ├── topic_modeler.py
│   │   ├── sentiment_analyzer.py
│   │   └── rhetorical_analyzer.py
│   ├── rag/
│   │   ├── chunker.py             # semantic chunking (linguistic units)
│   │   ├── retriever.py           # vector similarity search
│   │   └── generator.py           # Claude API call with style constraints
│   ├── chatbot/
│   │   ├── intent_classifier.py
│   │   ├── entity_extractor.py
│   │   ├── dialogue_manager.py
│   │   └── personality.py         # style guide enforcement
│   └── linguistics/
│       ├── grammar.py
│       └── ontology.py
├── data/
│   ├── audio/             # downloaded speech audio files
│   ├── raw/               # raw Whisper transcripts
│   ├── processed/         # cleaned text
│   ├── intents/           # labeled intent examples (JSON/CSV)
│   └── ontology/          # ontology definition files
├── docs/
│   ├── PERSONALITY_GUIDE.md    # bot personality & style guide
│   ├── CONVERSATION_FLOWS.md   # dialogue flow diagrams
│   └── LINGUISTIC_ANALYSIS.md  # corpus analysis findings
├── models/
│   └── intent_classifier/  # saved fine-tuned BERT model
├── app.py                  # Streamlit demo app (chat + dashboard)
└── api.py                  # FastAPI backend
```

---

## Build Order (revised — RAG-first)

**Strategy:** Get the RAG chatbot working end-to-end first, then layer in computational linguistics and dialogue sophistication. The NLP pipeline runs during corpus ingestion so annotations accumulate for free — they'll be ready when we return to linguistic analysis.

1. **Phase 0 (DONE):** Set up repo, venv, PostgreSQL, schema, project docs
2. **Phase 1 (IN PROGRESS):** Download audio, diarize speakers (pyannote), transcribe (Whisper), clean, NLP annotate (spaCy + BETO NER), load to DB
3. **Phase 4 (NEXT):** Build RAG system — semantic chunking, sentence-transformers embeddings, pgvector, retrieval pipeline, Claude API generation with citations, FastAPI backend
4. **Phase 6:** Build Streamlit frontend — chat interface + speech explorer + transparency layer
5. **Phase 2 (DEFERRED):** Linguistic analysis — n-grams, BERTopic, sentiment, syntactic style profile (uses NLP annotations already in DB)
6. **Phase 3 (DEFERRED):** Formal linguistics — CFG, domain ontology
7. **Phase 5 (DEFERRED):** Intent classifier (BETO fine-tune), dialogue manager, personality/style guide, pragmatic flows
8. **Phase 7:** Final polish, README, MVP demo prep

---

## What's Manual vs. Automated

| Task | Effort |
|---|---|
| Downloading speech audio | Automated (yt-dlp script) |
| Speaker diarization | Automated (pyannote-audio, one-time reference embedding setup) |
| Transcription | Automated (Whisper) |
| Cleaning spoken language | Semi-auto (regex + rules, but needs manual review of edge cases) |
| POS tagging, NER, parsing | Automated (spaCy pre-trained model) |
| N-gram analysis | Automated (NLTK) |
| Topic modeling | Automated (BERTopic/LDA) |
| Sentiment analysis | Automated (pre-trained model) |
| Building embedding index | Automated (run embeddings over corpus, store vectors) |
| RAG retrieval at runtime | Automated (similarity search) |
| **Labeling 200-400 intent examples** | **Manual, ~2-3 hours** |
| **Fine-tuning BERT** | Semi-auto (~20 lines of code, GPU trains it) |
| **Semantic chunking decisions** | **Semi-manual: automated splitting + manual review to ensure chunks are coherent units** |
| **Writing personality/style guide** | **Manual, creative writing + linguistic analysis** |
| **Designing dialogue flows** | **Manual, design work** |
| **Designing pragmatic edge cases** | **Manual, requires pragmatics knowledge** |
| **Writing formal grammar rules** | **Manual, linguistics knowledge** |
| **Building ontology** | **Manual, domain modeling (political topics, proposals, relationships)** |
| **Ethical/transparency layer design** | **Manual, policy decisions** |

---

## Key MVP Talking Points

1. **"I combined my philology training with engineering"** — you understand both WHY language works (pragmatics, syntax theory) and HOW to process it (Python, spaCy, BERT). The semantic chunking is a perfect example: you need linguistic intuition to know where ideas begin and end.
2. **"I designed the personality from a linguistic perspective"** — not just a system prompt, but a formal style guide with register, tone, vocabulary constraints, and pragmatic strategies for handling edge cases (biased questions, out-of-scope requests, vague queries).
3. **"I compared statistical and neural approaches"** — n-grams vs. BERT on the same corpus, showing understanding of the evolution of language models.
4. **"I built the full pipeline, the LLM is just the last step"** — transcription, cleaning, NLP analysis, topic modeling, embedding, retrieval, intent classification, dialogue management — all built by hand. Claude only generates the final response within strict constraints.
5. **"My teaching experience informed the pragmatic flow design"** — 6 years of handling real-time conversation (classroom) translates to designing bot conversation flows: how to handle vague questions, redirect off-topic users, and vary responses.
6. **"Every answer is grounded and cited"** — the system doesn't hallucinate. It retrieves actual speech fragments and cites them. This demonstrates understanding of RAG, which is how production conversational AI systems work.
7. **"I built speaker diarization to isolate the candidate's voice"** — many videos have multiple speakers (panels, debates, Q&A). Used pyannote-audio with a reference voice embedding to automatically identify and extract only the candidate's speech segments before transcription — solving a real-world data quality problem.
8. **"I analyzed a real corpus of Colombian political speech"** — not toy data. Real spoken Spanish with all its messiness: fillers, regional expressions, rhetorical devices. The cleaning and analysis required genuine linguistic expertise.

---

## Ethical Considerations

- **Neutrality:** The system presents what the candidate said, with sources. It does not promote, endorse, or editorialize.
- **Transparency:** Every response includes citations (speech title, date, event). A disclaimer states: "Este es un asistente de IA entrenado con discursos públicos de [Nombre]. Las respuestas se basan en fragmentos reales de sus intervenciones."
- **No fabrication:** If the candidate never addressed a topic, the system says so instead of inventing a position.
- **Source limitation:** The system only draws from the 50 analyzed speeches. It does not claim to represent the candidate's complete views.

---

## Resources to Get Started

- OpenAI Whisper: https://github.com/openai/whisper
- yt-dlp: https://github.com/yt-dlp/yt-dlp
- spaCy Spanish model: `python -m spacy download es_core_news_lg`
- BETO (Spanish BERT): `dccuchile/bert-base-spanish-wwm-cased` on Hugging Face
- sentence-transformers: https://www.sbert.net/
- BERTopic: https://maartengr.github.io/BERTopic/
- pgvector: https://github.com/pgvector/pgvector
- ChromaDB: https://www.trychroma.com/
- Rasa docs: https://rasa.com/docs/
- NLTK book (free): https://www.nltk.org/book/
- Anthropic Claude API: https://docs.anthropic.com/
