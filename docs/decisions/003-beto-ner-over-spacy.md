# ADR 003: BETO NER Over spaCy for Named Entity Recognition

**Date:** 2026-02-26
**Status:** Accepted

## Context
spaCy's `es_core_news_lg` NER was missing Colombian-specific entities (municipalities, departments, local political figures). Needed a more accurate NER for Spanish political speech.

## Decision
Replace spaCy NER with `mrm8488/bert-spanish-cased-finetuned-ner` (BETO-based) + DANE DIVIPOLA gazetteer (1,099 Colombian locations).

## Rationale
- BETO NER significantly outperforms spaCy on Spanish NER benchmarks.
- DANE gazetteer catches Colombian municipalities that no pre-trained model knows.
- spaCy still used for tokenization, POS tagging, and dependency parsing — only NER is replaced.

## Consequences
- NER pipeline is slower (BERT inference vs. spaCy's CNN) but accuracy matters more for this use case.
- Two-pass NER: BETO first, then gazetteer overlay for LOC entities.
- See `docs/NER_MODEL_EVALUATION.md` for the evaluation (MVP talking point).
