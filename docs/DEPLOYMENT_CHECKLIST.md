# Deployment Checklist

Everything needed to take the Streamlit chatbot from localhost to the internet.

**Target:** Streamlit Community Cloud (free tier)
**Goal:** Working public URL for a single-user demo, with a plan for scaling later.

---

## Current State (2026-03-02)

The app runs perfectly on localhost. The architecture (in-process MCP calls, env-var-driven config, embedding provider switching) is deployment-friendly by design. But several concrete steps remain.

---

## Blockers

These must be resolved before the app can go live.

- [x] **1. Add `sslmode` to DB connections** (done 2026-03-02)
  - Added `sslmode` + conditional `sslrootcert` to `psycopg2.connect()` in both `src/mcp/db.py` and `src/corpus/db_loader.py`.
  - Defaults to `prefer` for local dev, configurable via `DB_SSLMODE` and `DB_SSLROOTCERT` env vars.
  - See "SSL Mode Decision" section below.

- [x] **2. Create `requirements-deploy.txt`** (done 2026-03-02)
  - 13 packages (streamlit, anthropic, fastmcp, pydantic, psycopg2-binary, pgvector, huggingface-hub, numpy, plotly, matplotlib, pandas, python-dotenv).
  - No PyTorch, Whisper, spaCy, or other pipeline deps. Fits in 1 GiB.

- [ ] **3. Set `EMBEDDING_PROVIDER=hf_api` in production**
  - If set to `local` (default), the app loads the 868 MB SentenceTransformer model into RAM on startup. That alone nearly fills the 1 GiB limit.
  - `hf_api` offloads query embedding to the HuggingFace Inference API (free tier, same model, ~200ms latency per query). Zero local model loading.
  - This is a config-only change (Streamlit Cloud secrets).
  - **Effort:** 1 minute.

- [x] **4. Provision Supabase PostgreSQL** (done 2026-03-02)
  - Project: `cepedaNLP` on Supabase free tier (us-west-2).
  - **IPv4 note:** Direct connection (`db.xxx.supabase.co`) is IPv6-only on free tier. Use **Session Pooler** instead.
  - Connection: `aws-0-us-west-2.pooler.supabase.com:5432`, user `postgres.airqmqvntfdvhivoenlj`, dbname `postgres`.
  - pgvector enabled, schema created, all data migrated:
    - 14 speeches, 825 entities, 1594 annotations, 174 chunks (768d embeddings), 2533 speaker segments, 4 opinions.
  - Vector similarity search verified working.
  - **Still TODO:** Download Supabase CA certificate for `verify-full` mode (currently using `require`).

- [ ] **5. Configure Streamlit Cloud secrets**
  - In the Streamlit Cloud dashboard (Settings > Secrets), add:
    ```toml
    ANTHROPIC_API_KEY = "sk-ant-..."
    DB_HOST = "aws-0-us-west-2.pooler.supabase.com"
    DB_PORT = "5432"
    DB_NAME = "postgres"
    DB_USER = "postgres.airqmqvntfdvhivoenlj"
    DB_PASSWORD = "..."
    DB_SSLMODE = "require"
    EMBEDDING_PROVIDER = "hf_api"
    HF_TOKEN = "hf_..."
    ```
  - Streamlit injects these as env vars automatically — no code changes needed.
  - **Effort:** 10 minutes.

---

## Should Do

Not strictly required to go live, but important for a polished deployment.

- [x] **6. Create `.env.example`** (done 2026-03-02)
  - Template with all 8 env vars at project root.

- [x] **7. Create `.streamlit/config.toml`** (done 2026-03-02)
  - Dark theme, headless mode, usage stats disabled.

- [x] **8. Add `runtime.txt`** (done 2026-03-02)
  - Pins `python-3.13` for Streamlit Cloud.

- [x] **9. Add ethical disclaimer to UI** (done 2026-03-02)
  - Added to sidebar in `src/frontend/app.py`. States it's an academic NLP project, not an official campaign tool, based only on public speech transcripts.

- [ ] **10. Write `README.md`**
  - Phase 7 task, but a public repo with no README looks unfinished.
  - **Effort:** 30 minutes.

---

## Nice-to-Haves

Can be deferred without affecting the deployment.

- [ ] **11. Remove dead dependencies from `requirements.txt`**
  - `fastapi` and `uvicorn` are still listed despite ADR 007 (MCP-only, no REST API).
  - Doesn't break anything, but it's misleading.
  - **Effort:** 2 minutes.

- [ ] **12. GitHub Actions CI**
  - Run `pytest` on push. Catches regressions before they reach production.
  - **Effort:** 30 minutes.

- [ ] **13. `packages.txt` for system dependencies**
  - Streamlit Cloud uses this for `apt-get` packages. Probably not needed since `psycopg2-binary` bundles its own `libpq`.
  - Only needed if install fails with missing system libraries.
  - **Effort:** 5 minutes (if needed).

- [ ] **14. Test with Claude Desktop as MCP client**
  - Checklist item, good for the interview demo ("connect Claude Desktop to my MCP server").
  - **Effort:** 15 minutes.

---

## SSL Mode Decision

**Question:** Should we use `require` or `verify-full`?

**Answer: Use `verify-full`.** It is the strongest mode and it costs nothing.

| Mode | Encryption | Cert verification | Hostname check | Protects against |
|------|-----------|-------------------|----------------|-----------------|
| `require` | Yes | No | No | Passive eavesdropping |
| `verify-ca` | Yes | Yes | No | Eavesdropping + forged certificates |
| `verify-full` | Yes | Yes | Yes | Eavesdropping + forged certs + DNS hijacking |

### Why `verify-full` is free

- **No monetary cost.** SSL/TLS is a protocol feature built into PostgreSQL and psycopg2. Both Supabase and AWS RDS include SSL at no extra charge. There is no "SSL tier" to pay for.
- **No performance cost.** The TLS handshake adds ~1-5ms to the initial connection. After that, encryption overhead is negligible (AES hardware acceleration handles it). Connections are reused within a session, so the handshake happens once, not per query.
- **One-time setup cost.** The only extra work compared to `require` is downloading the provider's CA certificate:
  - **Supabase:** download from the project dashboard (Settings > Database > SSL). Save as `certs/supabase-ca.crt` in the repo.
  - CA certificates are public (they verify identity, not grant access), so committing them to the repo is safe and standard practice.

### Why not just `require`

`require` encrypts the connection but doesn't verify *who* you're talking to. An attacker who can intercept network traffic (e.g., rogue Wi-Fi, compromised router) could present their own certificate and you'd connect happily — encrypted, but to the wrong server. `verify-full` prevents this by checking the certificate chain and hostname.

For a demo app with public speech data, `require` is honestly fine. But `verify-full` costs nothing and is the objectively correct choice. Use it.

### Implementation

```python
conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432"),
    dbname=os.getenv("DB_NAME", "cepeda_nlp"),
    user=os.getenv("DB_USER", "oscarm"),
    password=os.getenv("DB_PASSWORD", ""),
    sslmode=os.getenv("DB_SSLMODE", "prefer"),
    sslrootcert=os.getenv("DB_SSLROOTCERT", ""),
)
```

Production env vars (Supabase):
```
DB_SSLMODE=verify-full
DB_SSLROOTCERT=certs/supabase-ca.crt
```

Local dev stays on `prefer` (default) — localhost connections don't cross a network.

---

## RAM and Concurrent Users

**Question:** Will the app crash if multiple users connect simultaneously?

### How Streamlit handles concurrency

Streamlit runs a single Python process. Each browser tab (user session) runs in a separate thread within that process. They share:
- The Python interpreter and all imported modules (~200-300 MB)
- `@st.cache_resource` objects (shared singletons)

Each session owns:
- `st.session_state` — chat history, tool results, plotly figures
- Transient objects created during request handling

### Memory budget (Streamlit Community Cloud, 1 GiB)

| Component | RAM | When |
|-----------|-----|------|
| Python + Streamlit + imported modules | ~250-350 MB | Always (once, shared) |
| Per session: chat history + tool results | ~5-15 MB | Per active user |
| Per session: plotly charts (transient, sent to browser) | ~2-5 MB | During rendering |
| HuggingFace Inference API call overhead | ~1 MB | Per query (transient) |
| **Available for sessions** | **~650-750 MB** | |

### Estimated user capacity (1 GiB limit, `hf_api` mode)

| Users | Approx. total RAM | Status |
|-------|-------------------|--------|
| 1 | ~350 MB | Comfortable |
| 5 | ~400 MB | Fine |
| 10 | ~450 MB | Fine |
| 20 | ~550 MB | OK, but getting into territory where a long chat history could push it |
| 30+ | ~650+ MB | Risk of OOM kills |

These are rough estimates. Real-world RAM depends on conversation length (chat history accumulates), visualization complexity, and garbage collection timing.

### What kills RAM

1. **Long chat histories.** Each message with tool results can be 5-50 KB. A 50-message conversation with lots of entity searches and charts could reach 1-2 MB of session state. With 20 users all having long conversations, that adds up.
2. **Plotly figures in memory.** Charts are generated server-side, serialized to JSON, and sent to the browser. The Python objects are transient but exist during rendering.
3. **Concurrent Claude API calls.** The `anthropic` SDK holds request/response objects in memory during streaming. Multiple simultaneous streams add pressure, though each is small (~100 KB).

### What does NOT kill RAM (because of `hf_api`)

The SentenceTransformer model (868 MB) is the elephant in the room, but with `EMBEDDING_PROVIDER=hf_api`, it never loads. This is the single most important deployment decision — it makes the difference between "crashes with 1 user" and "handles 20 comfortably."

### Guardrails for the 1 GiB limit

**Already in place:**
- `EMBEDDING_PROVIDER=hf_api` — offloads the biggest RAM consumer
- MCP tools return data and release it — no persistent model objects
- Charts render to HTML and are sent to the browser (not held in Python memory)

**Should implement before worrying about scale:**
- Cap chat history length in `session_state` (e.g., keep last 40 messages, drop oldest). This prevents unbounded memory growth per session.
- Use `@st.cache_resource` for the Anthropic client (single shared instance instead of one per session).

**Not needed now, but available if needed:**
- Trim tool results before appending to history (store summaries, not full SQL result sets).
- Add a session timeout that clears state for idle users.

### The honest answer for a demo

For the interview demo with a single user (you showing it on your laptop), **RAM is not a concern at all.** The app will sit comfortably at ~350 MB.

If you share the URL and 10 people try it, it will still be fine.

If it goes viral and 50 people hit it simultaneously, Streamlit Community Cloud will either OOM-kill the process (it restarts automatically) or throttle connections. But that's a good problem to have, and it's not the scenario you're deploying for.

---

## Scalability Plan (for later)

When a single Streamlit Community Cloud instance is no longer enough:

### Tier 1: Stay on Streamlit Cloud, optimize ($0)
- Cap chat history (40 messages max)
- Add session timeouts (clear state after 30 min idle)
- Compress tool results before storing in session state
- This buys you ~50 concurrent users comfortably

### Tier 2: Containerized deployment ($5-15/month)
- **Fly.io**, **Railway**, or **GCP Cloud Run** — all support Docker, all have generous free tiers
- Use a `Dockerfile` with the slim `requirements-deploy.txt`
- Set memory limit to 2 GiB (or 4 GiB on Cloud Run)
- Handles 100+ concurrent users easily
- Auto-scaling: spin up more containers under load, scale to zero when idle

### Tier 3: Full production ($20-50/month)
- Multiple container instances behind a load balancer
- Redis for session state (shared across instances)
- Connection pooling (PgBouncer) for the database
- CDN for static assets
- Monitoring and alerting (Datadog, Sentry)
- This is enterprise territory — way beyond what a demo needs

### Platform comparison

| Platform | Free tier | RAM | Auto-scale | Deploy from |
|----------|-----------|-----|------------|-------------|
| Streamlit Cloud | Yes (1 GiB) | 1 GiB | No | GitHub push |
| Fly.io | 3 shared VMs | 256 MB-8 GiB | Yes | Dockerfile |
| Railway | $5 credit/month | Configurable | Yes | GitHub push |
| GCP Cloud Run | 2M requests/month | Up to 32 GiB | Yes (to zero) | Dockerfile |
| AWS ECS Fargate | None | Configurable | Yes | Docker image |

**Recommendation:** Start on Streamlit Cloud. If you outgrow it, move to Railway or Fly.io — both deploy from a Dockerfile with minimal config and cost $5-10/month for a small app.

---

## Deployment Steps (in order)

Once the blockers are resolved, the actual deploy process is:

1. Push the repo to GitHub (already done — `git@github.com:OscarMarulanda/cepedaNLP.git`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the GitHub repo
4. Set the entry point to `src/frontend/app.py`
5. Add all secrets in the Streamlit Cloud dashboard
6. Click Deploy
7. Wait ~2-3 minutes for install + boot
8. Test the public URL

### Post-deploy verification

- [ ] Chat responds to a basic question
- [ ] Citations include speech titles and dates
- [ ] Entity search returns results with charts
- [ ] Source chunk expanders show raw text and YouTube links
- [ ] Abuse detection triggers the Matrix rain easter egg
- [ ] Opinion submission works and persists

---

## Related Documents

- `docs/DEPLOYMENT_ARCHITECTURE.md` — architecture diagram, sync strategy, AWS cost estimates
- `docs/DB_CONNECTION_SECURITY.md` — SSL mode analysis, hop-by-hop security review
- `docs/decisions/008-db-ssl-for-remote-connections.md` — ADR for SSL requirement
- `docs/API_COST_ANALYSIS.md` — per-query and monthly API cost projections
- `docs/MCP_CLIENT_SETUP.md` — MCP connection instructions for Claude Desktop, Claude Code
