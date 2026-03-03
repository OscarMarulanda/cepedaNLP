# Deployment Architecture

How the app runs in production, and how new speeches get processed.

## Principle: Process Locally, Serve from Cloud

The heavy NLP pipeline (Whisper, pyannote, spaCy, BETO, sentence-transformers) runs on the local Mac Mini for free. Cloud only handles serving the app and storing data.

## Architecture

```
Local Mac Mini                          Cloud
┌──────────────────────────┐            ┌──────────────────────────┐
│                          │            │                          │
│  yt-dlp (download)       │            │  Supabase PostgreSQL 17  │
│  pyannote (diarization)  │            │   + pgvector extension   │
│  Whisper (transcription) │            │   (Session Pooler)       │
│  cleaner                 │   sync     │                          │
│  spaCy + BETO NER        │  ───────>  │  Streamlit Community     │
│  chunker + embedder      │            │   Cloud (frontend +      │
│                          │            │   Claude orchestration)  │
│  pipeline_runner.py      │            │                          │
│                          │            │  Anthropic Claude API    │
│                          │            │  HuggingFace Inf. API    │
└──────────────────────────┘            └──────────────────────────┘
```

## Current Deployment (2026-03-02)

| Component | Service | Cost |
|-----------|---------|-----:|
| Frontend + orchestration | Streamlit Community Cloud (free, 1 GiB RAM) | $0 |
| Database | Supabase PostgreSQL free tier (500 MB, us-west-2) | $0 |
| pgvector | Supabase extension (included) | $0 |
| LLM orchestration | Anthropic Claude API (Haiku 4.5) | ~$0.0047/query |
| Query embedding | HuggingFace Inference API (free tier) | $0 |
| **Total** | | **~$0/month** |

At demo/MVP scale (< 100 queries/month), the Claude API cost is negligible (~$0.50).

## Supabase Connection Details

- **Host:** `aws-0-us-west-2.pooler.supabase.com` (Session Pooler)
- **Port:** `5432`
- **Database:** `postgres`
- **User:** `postgres.airqmqvntfdvhivoenlj`
- **SSL:** `verify-full` with CA cert at `certs/supabase-ca.crt`

> **IPv4 note:** Supabase free tier uses IPv6-only for direct connections (`db.xxx.supabase.co`). The Session Pooler provides IPv4 access at no extra cost. No functional limitations for this app.

## Sync Strategy: Point Pipeline at Remote DB

The simplest approach — no export/import scripts needed.

1. Pipeline runs locally on the Mac Mini as usual
2. Set `DB_HOST` in `.env` to the Supabase pooler endpoint
3. Pipeline writes speeches, entities, annotations, and chunks directly to the production DB
4. Done — no sync step, no data migration

### .env for local development
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=cepeda_nlp
DB_USER=oscarm
DB_SSLMODE=prefer
```

### .env for production sync
```
DB_HOST=aws-0-us-west-2.pooler.supabase.com
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres.airqmqvntfdvhivoenlj
DB_PASSWORD=<password>
DB_SSLMODE=verify-full
DB_SSLROOTCERT=certs/supabase-ca.crt
```

## Connection Security

All remote connections are encrypted:

```
Streamlit Cloud ──psycopg2 (TLS, verify-full)──▶ Supabase PostgreSQL
               ──HTTPS──▶ Anthropic Claude API
               ──HTTPS──▶ HuggingFace Inference API
```

`DB_SSLMODE=verify-full` verifies both encryption and server identity against the bundled Supabase CA certificate (`certs/supabase-ca.crt`). This prevents both passive eavesdropping and active MITM attacks.

MCP tool calls are **in-process Python function calls** — they never leave the Streamlit process, so there is no network attack surface between the app and the tool layer.

Full analysis: `docs/DB_CONNECTION_SECURITY.md`, ADR: `docs/decisions/008-db-ssl-for-remote-connections.md`.

## Processing New Speeches

When new speeches appear on the YouTube channel:

```bash
# On the Mac Mini
# 1. Set .env to point at Supabase
# 2. Run the pipeline
source venv/bin/activate
nohup python -m src.corpus.pipeline_runner --new=5 > data/pipeline_run.log 2>&1 &

# Pipeline automatically: downloads → diarizes → transcribes → cleans →
# NLP → loads to DB → chunks + embeds
# All written directly to the production database
```

No redeployment of the app needed — Streamlit reads from the same DB, so new speeches are immediately available for RAG queries.

## Why Not Process on Cloud?

| Step | Local (M4 Mac Mini) | Cloud GPU (g4dn.xlarge) |
|------|--------------------:|------------------------:|
| Whisper (per speech) | ~7.5 min, **free** | ~3 min, ~$0.03 |
| Diarization (per speech) | ~5 min, **free** | ~3 min, ~$0.03 |
| Full corpus (44 speeches) | ~8 hours, **free** | ~4 hours, **~$2.10** |
| Monthly (2-3 new speeches) | ~30 min, **free** | ~15 min, ~$0.15 |

The Mac Mini has an M4 chip with 16GB RAM — more than enough for Whisper large-v3 and pyannote. No reason to pay for cloud GPU when the local machine handles it well.

## Requirements Files

- **`requirements.txt`** — slim deployment dependencies (13 packages, no PyTorch/Whisper/spaCy). Used by Streamlit Cloud.
- **`requirements-full.txt`** — full pipeline + development dependencies. Used for local development.

## Scalability

See `docs/DEPLOYMENT_CHECKLIST.md` for the full scalability plan. In brief:

- **Current (Streamlit Cloud):** ~20 concurrent users comfortably on 1 GiB RAM with `hf_api` embedding
- **Tier 2 ($5-15/mo):** Containerized deployment (Fly.io, Railway, GCP Cloud Run) with 2-4 GiB RAM
- **Tier 3 ($20-50/mo):** Multiple instances + Redis + PgBouncer

## Related Documents

- `docs/DEPLOYMENT_CHECKLIST.md` — deployment tasks, SSL analysis, RAM estimates, scalability tiers
- `docs/DB_CONNECTION_SECURITY.md` — transport security analysis
- `docs/decisions/008-db-ssl-for-remote-connections.md` — ADR for SSL requirement
- `docs/API_COST_ANALYSIS.md` — per-query and monthly API cost projections
