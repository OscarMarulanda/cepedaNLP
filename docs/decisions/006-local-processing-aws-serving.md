# ADR 006: Process Locally, Serve from AWS

**Date:** 2026-02-28
**Status:** Accepted

## Context
Need to deploy the app for public access. The NLP pipeline (Whisper, pyannote, spaCy, BETO, sentence-transformers) is computationally heavy.

## Decision
Run the entire NLP pipeline on the local Mac Mini. Deploy only the serving layer (FastAPI, Streamlit, RDS) to AWS. Sync by pointing the pipeline at the remote DB.

## Rationale
- Mac Mini M4 processes speeches for free; AWS GPU instances cost ~$0.53/hr.
- Pipeline writes directly to the DB — changing `DB_HOST` in `.env` is the only sync step needed.
- AWS only needs a small instance (t3.small ~$15/mo) + RDS (t3.micro, free tier eligible).

## Consequences
- New speeches require the Mac Mini to be online and run the pipeline.
- No automated ingestion — manual `pipeline_runner --new=N` trigger.
- Total AWS cost: ~$10-35/month. See `docs/DEPLOYMENT_ARCHITECTURE.md`.
