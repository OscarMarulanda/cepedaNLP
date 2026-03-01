# ADR 008: Require SSL for Remote Database Connections

**Date:** 2026-03-01
**Status:** Accepted

## Context
The app is deployed to Streamlit Community Cloud and connects to a remote PostgreSQL instance (Supabase or AWS RDS) over the public internet. The `psycopg2` default `sslmode=prefer` silently falls back to plaintext if TLS negotiation fails, making the connection vulnerable to MITM downgrade attacks.

## Decision
Add a `DB_SSLMODE` environment variable to both `src/mcp/db.py` and `src/corpus/db_loader.py`. Default to `prefer` (preserves local dev behavior). Set to `require` in all remote/production environments.

## Rationale
- `sslmode=prefer` is unsafe over the internet — an attacker can force a plaintext downgrade.
- `sslmode=require` guarantees TLS encryption with zero configuration beyond the env var.
- `sslmode=verify-full` is the gold standard but requires distributing CA certificates — overkill for an MVP.
- The env var approach keeps local dev unchanged while securing remote deployments.

## Consequences
- Remote DB connections are encrypted, preventing credential theft and data interception.
- Local development is unaffected (default remains `prefer`, localhost doesn't need SSL).
- Full certificate verification (`verify-full`) is available but optional.
- Both `db.py` and `db_loader.py` must be updated (two connection factories in the codebase).
