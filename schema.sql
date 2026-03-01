-- Political Speech NLP Analyzer — Initial Schema
-- Database: cepeda_nlp

-- Enable pgvector extension (for embedding storage)
CREATE EXTENSION IF NOT EXISTS vector;

-- Speeches table: one row per speech
CREATE TABLE IF NOT EXISTS speeches (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    candidate VARCHAR(200) NOT NULL DEFAULT 'Iván Cepeda',
    speech_date DATE,
    location VARCHAR(300),
    event VARCHAR(500),
    duration_seconds INTEGER,
    youtube_url TEXT,
    audio_file_path TEXT,
    raw_transcript TEXT,
    cleaned_transcript TEXT,
    word_count INTEGER,
    language VARCHAR(10) DEFAULT 'es',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Entities extracted via NER
CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    speech_id INTEGER REFERENCES speeches(id) ON DELETE CASCADE,
    entity_text VARCHAR(500) NOT NULL,
    entity_label VARCHAR(50) NOT NULL,  -- PER, ORG, LOC, DATE, etc.
    start_char INTEGER,
    end_char INTEGER,
    sentence_index INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Speech metadata / topic tags
CREATE TABLE IF NOT EXISTS speech_topics (
    id SERIAL PRIMARY KEY,
    speech_id INTEGER REFERENCES speeches(id) ON DELETE CASCADE,
    topic VARCHAR(200) NOT NULL,  -- economy, security, education, etc.
    confidence FLOAT,
    source VARCHAR(50) DEFAULT 'bertopic',  -- bertopic, manual, lda
    created_at TIMESTAMP DEFAULT NOW()
);

-- Annotations: POS tags, dependency parses, etc. per sentence
CREATE TABLE IF NOT EXISTS annotations (
    id SERIAL PRIMARY KEY,
    speech_id INTEGER REFERENCES speeches(id) ON DELETE CASCADE,
    sentence_index INTEGER NOT NULL,
    sentence_text TEXT NOT NULL,
    tokens JSONB,          -- [{text, lemma, pos, dep, head}, ...]
    pos_tags JSONB,        -- ["NOUN", "VERB", ...]
    dep_parse JSONB,       -- dependency tree structure
    sentiment_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Embeddings for RAG (speech chunks)
CREATE TABLE IF NOT EXISTS speech_chunks (
    id SERIAL PRIMARY KEY,
    speech_id INTEGER REFERENCES speeches(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    sentence_start INTEGER,           -- first annotation sentence_index
    sentence_end INTEGER,             -- last annotation sentence_index (inclusive)
    metadata JSONB DEFAULT '{}',      -- flexible: start_time, etc.
    embedding vector(768),            -- sentence-transformers dimension
    created_at TIMESTAMP DEFAULT NOW()
);

-- Speaker diarization results
CREATE TABLE IF NOT EXISTS speaker_segments (
    id SERIAL PRIMARY KEY,
    speech_id INTEGER REFERENCES speeches(id) ON DELETE CASCADE,
    speaker_label VARCHAR(50) NOT NULL,   -- "SPEAKER_00", "SPEAKER_01", etc.
    is_target BOOLEAN DEFAULT FALSE,      -- TRUE if identified as Cepeda
    start_time FLOAT NOT NULL,            -- seconds
    end_time FLOAT NOT NULL,              -- seconds
    confidence FLOAT,                     -- cosine similarity (for target speaker)
    created_at TIMESTAMP DEFAULT NOW()
);

-- User opinions about the candidate
CREATE TABLE IF NOT EXISTS user_opinions (
    id SERIAL PRIMARY KEY,
    opinion_text TEXT NOT NULL,
    will_win BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_entities_speech ON entities(speech_id);
CREATE INDEX IF NOT EXISTS idx_entities_label ON entities(entity_label);
CREATE INDEX IF NOT EXISTS idx_topics_speech ON speech_topics(speech_id);
CREATE INDEX IF NOT EXISTS idx_topics_topic ON speech_topics(topic);
CREATE INDEX IF NOT EXISTS idx_annotations_speech ON annotations(speech_id);
CREATE INDEX IF NOT EXISTS idx_chunks_speech ON speech_chunks(speech_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON speech_chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_speaker_segments_speech ON speaker_segments(speech_id);
CREATE INDEX IF NOT EXISTS idx_speaker_segments_target ON speaker_segments(is_target);
CREATE INDEX IF NOT EXISTS idx_opinions_created_at ON user_opinions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_opinions_will_win ON user_opinions(will_win);
