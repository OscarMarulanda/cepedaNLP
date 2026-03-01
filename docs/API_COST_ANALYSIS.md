# API Cost Analysis

Cost estimates for the RAG chatbot's Anthropic Claude API usage, based on real query data collected 2026-02-28.

## Anthropic Pricing (per million tokens)

| Model | Input | Output |
|-------|------:|-------:|
| Haiku 4.5 (dev/testing) | $1 | $5 |
| Sonnet 4.6 (production) | $3 | $15 |
| Opus 4.6 (production) | $5 | $25 |

Source: https://platform.claude.com/docs/en/about-claude/pricing

## Observed Token Usage

Measured from 4 real queries against the 10-speech corpus (131 chunks):

| Query | Input Tokens | Output Tokens |
|-------|-------------:|--------------:|
| "...sobre el racismo?" | 2,630 | 251 |
| "...sobre Uribe?" | 2,691 | 745 |
| "...sobre la educacion bilingue?" | 2,685 | 87 |
| "...sobre Petro?" | 2,800 | 487 |
| **Average** | **2,702** | **393** |

Input tokens are stable (~2,700) because the retriever always sends 5 chunks of similar size plus the system prompt. Output tokens vary by topic relevance (87 when the topic isn't found vs 745 for a rich answer).

## Per-Query Cost

| Model | Input Cost | Output Cost | Total |
|-------|----------:|----------:|------:|
| Haiku 4.5 | $0.0027 | $0.0020 | **$0.0047** |
| Sonnet 4.6 | $0.0081 | $0.0059 | **$0.014** |
| Opus 4.6 | $0.0135 | $0.0098 | **$0.023** |

## Projected Monthly Costs

| Scenario | Queries | Haiku 4.5 | Sonnet 4.6 | Opus 4.6 |
|----------|--------:|----------:|----------:|----------:|
| MVP demo | 50 | $0.23 | $0.70 | $1.17 |
| Dev & testing | 500 | $2.33 | $7.00 | $11.66 |
| Light production (100/day) | 3,000 | $14 | $42 | $70 |
| Heavy production (1k/day) | 30,000 | $140 | $420 | $700 |

## Cost Optimization Options

- **Prompt caching:** Cache the system prompt across queries for a 90% reduction on cached input tokens (cache hits cost $0.10/MTok on Haiku). Minimal impact at demo scale, meaningful at production scale.
- **Batch API:** 50% discount on input and output tokens for non-real-time workloads.
- **Model tiering:** Use Haiku for simple factual lookups, Sonnet for nuanced multi-source synthesis.

## Notes

- Embedding (sentence-transformers) and retrieval (pgvector) run locally at zero API cost.
- All costs above are Claude API only. No other paid APIs are used in the RAG pipeline.
- As the corpus grows (target: ~50 speeches), input tokens per query may increase slightly if chunk context gets richer, but the retriever's `top_k=5` cap keeps it bounded.
