# Connecting to the CepedaNLP MCP Server

The CepedaNLP MCP server exposes 8 tools for querying a corpus of political speeches by Iván Cepeda. Any MCP-compatible AI agent can connect to it.

The server handles all database connections and embeddings internally. Clients connect to the SSE endpoint with an API key and call tools.

---

## Public SSE endpoint

A public instance is available — no cloning required:

```
https://cepeda-nlp-mcp.onrender.com/sse
```

> **Authentication:** The public endpoint requires an API key. Include it as a Bearer token in the `Authorization` header. Contact the project maintainer for a key.
>
> **Cold starts:** Render free tier spins down after 15 minutes of inactivity. The first request after idle may take 30-60 seconds while the server starts up. Subsequent requests are fast.
>
> **Rate limiting:** The endpoint is rate-limited to 30 requests per minute per IP address.

Use this URL in any of the client configurations below.

---

## Quick start (connect as a client)

### Claude Desktop

Add to your config file (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "cepeda-nlp": {
      "url": "https://cepeda-nlp-mcp.onrender.com/sse",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
      }
    }
  }
}
```

Restart Claude Desktop after editing.

### Claude Code

Add a `.mcp.json` file to your project root:

```json
{
  "mcpServers": {
    "cepeda-nlp": {
      "url": "https://cepeda-nlp-mcp.onrender.com/sse",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
      }
    }
  }
}
```

### Kiro (AWS)

Add to `.kiro/settings.json` or workspace config:

```json
{
  "mcpServers": {
    "cepeda-nlp": {
      "url": "https://cepeda-nlp-mcp.onrender.com/sse",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
      }
    }
  }
}
```

### Cursor

Go to **Settings > MCP Servers > Add Server** and configure:

- **Name:** `cepeda-nlp`
- **Type:** `sse`
- **URL:** `https://cepeda-nlp-mcp.onrender.com/sse`

Or add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "cepeda-nlp": {
      "url": "https://cepeda-nlp-mcp.onrender.com/sse",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
      }
    }
  }
}
```

### Any MCP client (generic)

Point your MCP client at the SSE endpoint: `https://cepeda-nlp-mcp.onrender.com/sse`. Include an `Authorization: Bearer YOUR_API_KEY` header with every request.

### Self-hosted endpoint

If you run your own instance (see [Self-hosting](#self-hosting-the-server) below), replace the public URL with your own: `http://<host>:<port>/sse`.

---

## Available tools

| # | Tool | Description | Parameters |
|---|------|-------------|------------|
| 1 | `retrieve_chunks` | Semantic search over speech fragments | `query` (str), `top_k` (int, default 5) |
| 2 | `list_speeches` | List all speeches with metadata | — |
| 3 | `get_speech_detail` | Full details of one speech | `speech_id` (int) |
| 4 | `search_entities` | Search named entities across corpus | `entity_text` (str), `entity_label` (str), `limit` (int, default 10) |
| 5 | `get_speech_entities` | Entities in one speech, grouped by NER label | `speech_id` (int) |
| 6 | `get_corpus_stats` | Corpus-wide statistics | — |
| 7 | `submit_opinion` | Save a user opinion | `opinion_text` (str), `will_win` (bool) |
| 8 | `get_opinions` | Retrieve user opinions + summary stats | `will_win` (bool, optional), `limit` (int, default 20) |

All tool descriptions and parameter hints are in Spanish, matching the speech corpus language.

## Verifying the connection

Once connected, ask the agent to call `get_corpus_stats`. You should see something like:

```json
{
  "speeches": 14,
  "total_words": 30963,
  "entities": 825,
  "annotations": 1594,
  "chunks": 174,
  "opinions": 4
}
```

---

## Self-hosting the server

This section is for **server operators** who want to run their own instance of the MCP server. Clients connecting to your endpoint do not need any of this — they just use the SSE URL.

### Prerequisites

1. **Python 3.13+** with a virtual environment:

   ```bash
   cd /path/to/cepedaNLP
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt        # slim deploy deps
   # or: pip install -r requirements-full.txt  # full pipeline + dev deps
   ```

2. **PostgreSQL** with the `cepeda_nlp` database populated (speeches, entities, chunks with embeddings) and the `pgvector` extension enabled.

3. **Environment variables** — create a `.env` file at the project root:

   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=cepeda_nlp
   DB_USER=<your db user>
   DB_PASSWORD=<your db password>
   DB_SSLMODE=prefer
   EMBEDDING_PROVIDER=local               # or hf_api
   HF_TOKEN=hf_...                        # only needed if EMBEDDING_PROVIDER=hf_api
   ```

   The `retrieve_chunks` tool embeds the query before searching. By default it loads `paraphrase-multilingual-mpnet-base-v2` locally (~868 MB). Set `EMBEDDING_PROVIDER=hf_api` to offload embedding to the HuggingFace Inference API instead (requires an HF token with "Inference Providers" permission).

### Running the server

| Transport | Command | Use case |
|-----------|---------|----------|
| **SSE** (recommended) | `fastmcp run src/mcp/server.py --transport sse --port 8000` | Remote agents, HTTP clients |
| **STDIO** | `python -m src.mcp.server` | Local agents (Claude Desktop, Claude Code) |

**SSE** is the recommended transport for serving external clients. The server binds to the specified port and clients connect to `http://<host>:<port>/sse`.

**STDIO** is useful when the server operator is also the client (e.g., running Claude Desktop on the same machine). In this case, the agent launches the server process directly.

### STDIO configuration (local use only)

When running server and client on the same machine, you can use STDIO instead of SSE. This requires the server operator to configure the agent with the Python path and project directory.

**Claude Desktop (STDIO):**

```json
{
  "mcpServers": {
    "cepeda-nlp": {
      "command": "/path/to/cepedaNLP/venv/bin/python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/cepedaNLP"
    }
  }
}
```

**Claude Code (STDIO):**

```json
{
  "mcpServers": {
    "cepeda-nlp": {
      "command": "./venv/bin/python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "."
    }
  }
}
```

The `.env` file in the project root provides database credentials automatically.

### Connecting to a remote database (e.g., Supabase)

To run the server locally but connect to a remote PostgreSQL, set the connection variables in `.env`:

```env
DB_HOST=<remote host>
DB_PORT=5432
DB_NAME=<database name>
DB_USER=<database user>
DB_PASSWORD=<database password>
DB_SSLMODE=verify-full
DB_SSLROOTCERT=certs/<provider>-ca.crt
EMBEDDING_PROVIDER=hf_api
HF_TOKEN=hf_...
```

No local PostgreSQL needed. Ensure the CA certificate file exists at the path specified by `DB_SSLROOTCERT`.

---

## Troubleshooting

### Client issues

| Problem | Fix |
|---------|-----|
| Connection refused / timeout | Verify the SSE endpoint URL is correct and the server is running. |
| `401 Unauthorized` | Check that your `Authorization: Bearer <key>` header is present and the key is correct. |
| `429 Too Many Requests` | You've exceeded the rate limit (30 req/min). Wait for the `Retry-After` seconds and try again. |
| Tools not appearing | Restart your MCP client after updating the configuration. |

### Server operator issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'src'` | Make sure `cwd` points to the project root, not `src/`. |
| `psycopg2.OperationalError: connection refused` | Start PostgreSQL or verify `DB_HOST` and `DB_PASSWORD` in `.env`. |
| `retrieve_chunks` is slow on first call | The embedding model (~868 MB) loads on first query. Use `EMBEDDING_PROVIDER=hf_api` to skip local loading. |
| `pgvector` import error | Install: `pip install pgvector` and ensure the `vector` extension is enabled in PostgreSQL. |
| `403 Forbidden` from HuggingFace | HF token lacks "Inference Providers" permission. Create a fine-grained token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). |
| SSL certificate error | Ensure `DB_SSLROOTCERT` points to the correct CA cert file (relative to project root). |
