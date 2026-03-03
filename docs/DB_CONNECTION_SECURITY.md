# Database Connection Security

How data travels between the Streamlit app and the PostgreSQL database, and how to protect it.

## The Problem

When the app is deployed to Streamlit Community Cloud and the database lives on a remote host (Supabase, AWS RDS, etc.), the `psycopg2` connection travels over the public internet. By default, `psycopg2` uses `sslmode=prefer`, which **tries** TLS but silently falls back to plaintext if the server doesn't enforce it. This means a man-in-the-middle attacker could:

1. **Downgrade the connection** to plaintext by intercepting the TLS negotiation
2. **Read** all SQL queries and results in transit (speech data, embeddings, user opinions)
3. **Modify** queries or results before they reach either end

## Data Flow Diagram

```
Browser ──HTTPS──▶ Streamlit Community Cloud (Python server-side)
                        │
                        ├── psycopg2 ──TCP/TLS──▶ Remote PostgreSQL (Supabase / RDS)
                        │
                        ├── HTTPS ──────────────▶ Anthropic Claude API
                        │
                        └── HTTPS ──────────────▶ HuggingFace Inference API
```

### Hop-by-hop analysis

| Hop | Protocol | Risk | Status |
|-----|----------|------|--------|
| Browser → Streamlit Cloud | HTTPS (TLS, managed by Streamlit) | None — browser only sees rendered HTML, no credentials | Secure |
| Streamlit → PostgreSQL | TCP, `sslmode=verify-full` | Encrypted + server identity verified via Supabase CA cert | **Secure** |
| Streamlit → Anthropic API | HTTPS (TLS, verified by `anthropic` SDK) | None | Secure |
| Streamlit → HuggingFace API | HTTPS (TLS, verified by `huggingface_hub`) | None | Secure |

### Why the MCP layer is not an attack surface

MCP tools (`src/mcp/server.py`) are called as **local Python function calls** within the same Streamlit process. There is no HTTP, no IPC, no socket, and no serialization between the Streamlit orchestrator and the MCP tools. Nothing to intercept.

## SSL Modes in psycopg2

| `sslmode` | Behavior | Internet-safe? |
|-----------|----------|---------------|
| `disable` | Never use SSL | No |
| `allow` | Try plaintext first, then SSL | No |
| `prefer` (default) | Try SSL first, fall back to plaintext | **No** — MITM can force downgrade |
| `require` | Require SSL, but don't verify the server certificate | **Yes** — encrypted, prevents passive eavesdropping |
| `verify-ca` | Require SSL + verify server cert against a CA | Yes — prevents MITM with forged certs |
| `verify-full` | Require SSL + verify cert + verify hostname | **Best** — full MITM protection |

## The Fix (applied 2026-03-02)

### In `src/mcp/db.py` and `src/corpus/db_loader.py`

Both files now include `sslmode` and conditional `sslrootcert`:

```python
ssl_kwargs = {}
sslrootcert = os.getenv("DB_SSLROOTCERT", "")
if sslrootcert:
    ssl_kwargs["sslrootcert"] = sslrootcert

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432"),
    dbname=os.getenv("DB_NAME", "cepeda_nlp"),
    user=os.getenv("DB_USER", "oscarm"),
    password=os.getenv("DB_PASSWORD", ""),
    sslmode=os.getenv("DB_SSLMODE", "prefer"),
    **ssl_kwargs,
)
```

### Environment configuration

**Local development** (`.env`):
```
DB_HOST=localhost
DB_SSLMODE=prefer
```
Localhost connections don't need SSL — traffic never leaves the machine.

**Production — Supabase** (Streamlit Secrets):
```
DB_HOST=aws-0-us-west-2.pooler.supabase.com
DB_SSLMODE=verify-full
DB_SSLROOTCERT=certs/supabase-ca.crt
```
Uses `verify-full` with the Supabase CA certificate chain bundled in the repo. This verifies encryption, certificate authenticity, and hostname — full MITM protection.

## What's at risk without SSL

| Data in transit | Sensitivity |
|----------------|-------------|
| Speech transcripts (cleaned text) | Low — public speeches |
| Named entities (PER, ORG, LOC) | Low — extracted from public speeches |
| Embedding vectors (768-dim floats) | Low — derived from public text |
| User opinions (`user_opinions` table) | **Medium** — user-submitted content |
| DB credentials (in connection handshake) | **High** — reusable for direct DB access |

The main risk is **credential theft** during the connection handshake, not the speech data itself. An attacker who captures DB credentials can connect directly and read, modify, or delete all data.

## Provider-specific notes

### Supabase (current provider)
- SSL is enabled by default on all Supabase PostgreSQL instances
- Free tier uses IPv6-only for direct connections — use **Session Pooler** (`pooler.supabase.com`) for IPv4
- CA certificate chain extracted via `openssl s_client` and bundled at `certs/supabase-ca.crt`
- Chain: leaf (`*.pooler.supabase.com`) → intermediate (`Supabase Intermediate 2021 CA`) → root (`Supabase Root 2021 CA`)
- CA certs are public (they verify identity, not grant access) — safe to commit to the repo

## Related files

- `src/mcp/db.py` — MCP server DB connection (used by Streamlit app)
- `src/corpus/db_loader.py` — pipeline DB connection (used during ingestion)
- `docs/DEPLOYMENT_ARCHITECTURE.md` — deployment strategy
- `docs/decisions/008-db-ssl-for-remote-connections.md` — ADR for this decision
