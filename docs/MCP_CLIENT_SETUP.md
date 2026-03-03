# Connecting to the CepedaNLP MCP Server

The CepedaNLP MCP server exposes 9 tools for querying the political speech corpus. Any MCP-compatible AI agent can connect to it.

## Two ways to connect

| Mode | Database | Embedding | Use case |
|------|----------|-----------|----------|
| **Local** | Local PostgreSQL (`cepeda_nlp`) | Local model or HF API | Development, full pipeline access |
| **Production** | Supabase (deployed) | HF API | Demo, remote agents, no local setup needed |

Both use the same MCP server code — only the environment variables differ.

---

## Local setup

### Prerequisites

1. **PostgreSQL running** with the `cepeda_nlp` database populated (speeches, entities, chunks with embeddings).

2. **Python environment** with dependencies installed:

   ```bash
   cd /path/to/cepedaNLP
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements-full.txt
   ```

3. **Environment variables** — create a `.env` file at the project root (see `.env.example`):

   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=cepeda_nlp
   DB_USER=oscarm
   DB_PASSWORD=
   DB_SSLMODE=prefer
   ANTHROPIC_API_KEY=sk-ant-...       # only needed if the client uses Claude
   EMBEDDING_PROVIDER=local           # or hf_api (see below)
   HF_TOKEN=hf_...                    # only needed if EMBEDDING_PROVIDER=hf_api
   ```

   The `retrieve_chunks` tool embeds the query before searching. By default it loads `paraphrase-multilingual-mpnet-base-v2` locally (~868 MB). Set `EMBEDDING_PROVIDER=hf_api` to offload embedding to the HuggingFace Inference API instead.

---

## Production setup (Supabase)

To connect to the deployed production database without running a local PostgreSQL:

### Prerequisites

1. **Python environment** (same as local, but only needs the slim dependencies):

   ```bash
   cd /path/to/cepedaNLP
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment variables** — set in `.env`:

   ```env
   DB_HOST=aws-0-us-west-2.pooler.supabase.com
   DB_PORT=5432
   DB_NAME=postgres
   DB_USER=postgres.airqmqvntfdvhivoenlj
   DB_PASSWORD=<supabase password>
   DB_SSLMODE=verify-full
   DB_SSLROOTCERT=certs/supabase-ca.crt
   EMBEDDING_PROVIDER=hf_api
   HF_TOKEN=hf_...
   ```

   No local PostgreSQL needed. The HF token must have "Inference Providers" permission (create a fine-grained token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

## Transport modes

The server supports two transports:

| Transport | Command | Use case |
|-----------|---------|----------|
| **STDIO** | `python -m src.mcp.server` | Claude Desktop, local agents |
| **SSE**   | `fastmcp run src/mcp/server.py --transport sse --port 8000` | Remote agents, HTTP clients |

## Claude Desktop

Add this to your Claude Desktop config file:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**Local database:**

```json
{
  "mcpServers": {
    "cepeda-nlp": {
      "command": "/path/to/cepedaNLP/venv/bin/python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/cepedaNLP",
      "env": {
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "cepeda_nlp",
        "DB_USER": "oscarm",
        "EMBEDDING_PROVIDER": "local"
      }
    }
  }
}
```

**Production (Supabase):**

```json
{
  "mcpServers": {
    "cepeda-nlp": {
      "command": "/path/to/cepedaNLP/venv/bin/python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/cepedaNLP",
      "env": {
        "DB_HOST": "aws-0-us-west-2.pooler.supabase.com",
        "DB_PORT": "5432",
        "DB_NAME": "postgres",
        "DB_USER": "postgres.airqmqvntfdvhivoenlj",
        "DB_PASSWORD": "<supabase password>",
        "DB_SSLMODE": "verify-full",
        "DB_SSLROOTCERT": "certs/supabase-ca.crt",
        "EMBEDDING_PROVIDER": "hf_api",
        "HF_TOKEN": "<your HF token>"
      }
    }
  }
}
```

Replace `/path/to/cepedaNLP` with the actual project path. Restart Claude Desktop after editing.

## Claude Code

Add a `.mcp.json` file at the project root:

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

Claude Code reads `.mcp.json` automatically on startup. The `.env` file in the project root provides database credentials.

## Kiro (AWS)

Kiro uses the same MCP configuration format as Claude Code. Add to your project's `.kiro/settings.json` or the workspace config:

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

## Cursor

In Cursor, go to **Settings > MCP Servers > Add Server** and configure:

- **Name:** `cepeda-nlp`
- **Type:** `command`
- **Command:** `/path/to/cepedaNLP/venv/bin/python -m src.mcp.server`

Or add to `.cursor/mcp.json` in the project root:

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

## SSE transport (remote/HTTP agents)

For agents that connect over HTTP instead of STDIO:

```bash
# Start the server
cd /path/to/cepedaNLP
source venv/bin/activate
fastmcp run src/mcp/server.py --transport sse --port 8000
```

Then configure the client to connect to `http://localhost:8000/sse` (or the remote host). Example for Claude Desktop with SSE:

```json
{
  "mcpServers": {
    "cepeda-nlp": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

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
| 9 | `matrix_rain_easter_egg` | Trigger easter egg animation | `reason` (str) |

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

If you get a connection error, verify the database is reachable and the credentials in `.env` are correct.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'src'` | Make sure `cwd` points to the project root, not `src/` |
| `psycopg2.OperationalError: connection refused` | **Local:** Start PostgreSQL: `brew services start postgresql@17`. **Production:** Check `DB_HOST` and `DB_PASSWORD` in `.env` |
| `retrieve_chunks` is slow on first call | The embedding model (~868 MB) loads on first query. Subsequent calls are fast. Use `EMBEDDING_PROVIDER=hf_api` to skip local loading. |
| `pgvector` import error | Install: `pip install pgvector` and ensure the `vector` extension is enabled in PostgreSQL |
| `403 Forbidden` from HuggingFace | HF token lacks "Inference Providers" permission. Create a fine-grained token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with that permission enabled. |
| SSL certificate error with Supabase | Ensure `DB_SSLROOTCERT=certs/supabase-ca.crt` points to the bundled cert file (relative to project root). |
