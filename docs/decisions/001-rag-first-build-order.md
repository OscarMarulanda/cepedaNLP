# ADR 001: RAG-First Build Order

**Date:** 2026-02-27
**Status:** Accepted

## Context
The project has 7 phases. The computational linguistics phases (2, 3, 5) are academically interesting but not required for the core deliverable — a working chatbot that answers questions about Cepeda's speeches with citations.

## Decision
Skip phases 2, 3, 5 for now. Build in order: Phase 1 (corpus) → 4 (RAG) → 6 (frontend) → then return to linguistics.

## Rationale
- The NLP pipeline runs during corpus ingestion anyway — annotations accumulate for free.
- RAG is the core deliverable for the MVP (working chatbot with citations).
- Linguistic analysis phases consume stored annotations — data will be waiting when we return.
- Faster path to a demo-ready product.

## Consequences
- Phases 2, 3, 5 are deferred but not blocked — all their input data is being generated.
- `speech_topics` table remains empty until BERTopic runs (Phase 2).
- Intent classifier and dialogue manager are not built yet (Phase 5).
