# ADR 002: Sentence-Grouping Chunking Strategy

**Date:** 2026-02-27
**Status:** Accepted

## Context
RAG systems need to split documents into chunks for embedding and retrieval. Two approaches were considered for Spanish political speeches.

## Options Considered
1. **ROOT-boundary splitting** — split at spaCy dependency parse ROOT boundaries (clause-level).
2. **Sentence-grouping** — group consecutive sentences to ~200 words per chunk.

## Decision
Sentence-grouping with ~200-word target, 1-sentence overlap, runt merging (<30 words).

## Rationale
- Median sentence length in the corpus is ~14 words — too short for meaningful retrieval individually.
- ROOT-boundary splitting produced even smaller fragments.
- ~200 words gives enough context for the LLM to generate a coherent, cited answer.
- 1-sentence overlap maintains continuity between chunks.

## Consequences
- Chunks are semantically coherent (complete thoughts, not mid-sentence cuts).
- ~13 chunks per speech on average.
- See `docs/RAG_DESIGN_DECISIONS.md` for full analysis.
