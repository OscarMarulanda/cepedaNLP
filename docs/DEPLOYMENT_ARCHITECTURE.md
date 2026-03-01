# Deployment Architecture

How the app runs in production on AWS, and how new speeches get processed.

## Principle: Process Locally, Serve from AWS

The heavy NLP pipeline (Whisper, pyannote, spaCy, BETO, sentence-transformers) runs on the local Mac Mini for free. AWS only handles serving the app and storing data.

## Architecture

```
Local Mac Mini                          AWS
┌──────────────────────────┐            ┌──────────────────────────┐
│                          │            │                          │
│  yt-dlp (download)       │            │  RDS PostgreSQL 17       │
│  pyannote (diarization)  │            │   + pgvector extension   │
│  Whisper (transcription) │            │                          │
│  cleaner                 │   sync     │  EC2 / ECS               │
│  spaCy + BETO NER        │  ───────>  │   FastAPI backend        │
│  chunker + embedder      │            │   Streamlit frontend     │
│                          │            │                          │
│  pipeline_runner.py      │            │  Anthropic Claude API    │
│                          │            │   (called from FastAPI)  │
└──────────────────────────┘            └──────────────────────────┘
```

## Sync Strategy: Point Pipeline at Remote DB

The simplest approach — no export/import scripts needed.

1. Pipeline runs locally on the Mac Mini as usual
2. Set `DB_HOST` in `.env` to the RDS endpoint instead of `localhost`
3. Pipeline writes speeches, entities, annotations, and chunks directly to the production DB
4. Done — no sync step, no data migration

### .env for local development
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=cepeda_nlp
DB_USER=oscarm
```

### .env for production sync
```
DB_HOST=cepeda-nlp.xxxxxxxxxxxx.us-east-1.rds.amazonaws.com
DB_PORT=5432
DB_NAME=cepeda_nlp
DB_USER=oscarm
DB_PASSWORD=<secure password>
DB_SSLMODE=require
```

> **Security note:** Remote DB connections must use `DB_SSLMODE=require` to enforce TLS encryption. Without it, `psycopg2` defaults to `prefer`, which silently falls back to plaintext and is vulnerable to MITM downgrade attacks. See `docs/DB_CONNECTION_SECURITY.md` for full analysis.

### RDS Access
- Whitelist the Mac Mini's public IP in the RDS security group, or
- Use an SSH tunnel through the EC2 instance, or
- Use AWS Systems Manager Session Manager for secure access

## AWS Components & Estimated Costs

| Component | Service | Est. Monthly Cost |
|-----------|---------|------------------:|
| Database | RDS PostgreSQL db.t3.micro (free tier eligible) | $0 – $15 |
| pgvector | RDS extension (no extra cost) | $0 |
| App server | EC2 t3.small or ECS Fargate | $5 – $20 |
| Claude API | Anthropic API (Haiku 4.5) | ~$0.0047/query |
| Domain/SSL | Route 53 + ACM (optional) | ~$1 |
| **Total** | | **~$10 – $35/month** |

At demo/MVP scale (< 100 queries/month), the Claude API cost is negligible (~$0.50).

## Processing New Speeches

When new speeches appear on the YouTube channel:

```bash
# On the Mac Mini
# 1. Set .env to point at the production RDS
# 2. Run the pipeline
source venv/bin/activate
nohup python -m src.corpus.pipeline_runner --new=5 > data/pipeline_run.log 2>&1 &

# Pipeline automatically: downloads → diarizes → transcribes → cleans →
# NLP → loads to DB → chunks + embeds
# All written directly to the production database
```

No redeployment of the app needed — FastAPI/Streamlit read from the same DB, so new speeches are immediately available for RAG queries.

## Why Not Process on AWS?

| Step | Local (M4 Mac Mini) | AWS GPU (g4dn.xlarge) |
|------|--------------------:|----------------------:|
| Whisper (per speech) | ~7.5 min, **free** | ~3 min, ~$0.03 |
| Diarization (per speech) | ~5 min, **free** | ~3 min, ~$0.03 |
| Full corpus (44 speeches) | ~8 hours, **free** | ~4 hours, **~$2.10** |
| Monthly (2-3 new speeches) | ~30 min, **free** | ~15 min, ~$0.15 |

The Mac Mini has an M4 chip with 16GB RAM — more than enough for Whisper large-v3 and pyannote. No reason to pay for cloud GPU when the local machine handles it well.

## Connection Security

When the app connects to a remote database over the internet, the connection must be encrypted:

```
Streamlit Cloud ──psycopg2 (TLS)──▶ Remote PostgreSQL
```

Set `DB_SSLMODE=require` in the production environment. This ensures `psycopg2` uses TLS and refuses plaintext fallback. For maximum security, use `DB_SSLMODE=verify-full` with the provider's CA certificate.

MCP tool calls are **in-process Python function calls** — they never leave the Streamlit process, so there is no network attack surface between the app and the tool layer.

Full analysis: `docs/DB_CONNECTION_SECURITY.md`, ADR: `docs/decisions/008-db-ssl-for-remote-connections.md`.

## Future Considerations

- **CI/CD:** If the project grows, could use GitHub Actions to deploy FastAPI/Streamlit to ECS on push
- **Auto-scaling:** Not needed at demo scale; ECS Fargate handles it if it becomes relevant
- **Monitoring:** CloudWatch for RDS + API error rates
- **Backup:** RDS automated snapshots (free tier includes 20GB)
