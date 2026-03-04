# Syncing New Speeches to Production

After processing new speeches locally, you need to push the data to Supabase so the live frontend can serve them.

## Current State

| | Local (cepeda_nlp) | Production (Supabase) |
|---|---|---|
| Speeches | 16 | 14 |
| Chunks | 198 | 174 |
| Entities | 931 | 825 |
| Annotations | 1956 | 1594 |

The 2 new speeches (IDs 17-18) and their associated data exist only in the local DB.

## Option A: Re-run Pipeline Against Supabase (Recommended)

The pipeline is idempotent — it skips speeches already in the DB. Point it at Supabase and run it again; only the 2 missing speeches will be processed.

### Steps

1. **Back up your `.env`** (you'll restore it after):
   ```bash
   cp .env .env.local.bak
   ```

2. **Switch `.env` to Supabase**:
   ```
   DB_HOST=aws-0-us-west-2.pooler.supabase.com
   DB_PORT=5432
   DB_NAME=postgres
   DB_USER=postgres.airqmqvntfdvhivoenlj
   DB_PASSWORD=<your-supabase-password>
   DB_SSLMODE=verify-full
   DB_SSLROOTCERT=certs/supabase-ca.crt
   ```

3. **Run the pipeline** (same `--new=2`):
   ```bash
   source venv/bin/activate
   nohup python -m src.corpus.pipeline_runner --new=2 > data/pipeline_sync.log 2>&1 &
   ```
   The pipeline will download, diarize, transcribe, clean, analyze, chunk, and embed again — writing directly to Supabase. It skips the 14 speeches already there.

4. **Restore local `.env`**:
   ```bash
   mv .env.local.bak .env
   ```

**Pros:** Simple, no SQL export/import. Uses the same tested pipeline.
**Cons:** Re-does all processing (~25 min for 2 speeches). Uses bandwidth to re-download audio.

## Option B: pg_dump / psql Export-Import

Export only the new rows from local and import them into Supabase. Faster than re-processing but requires careful handling of serial IDs and foreign keys.

### Steps

1. **Export the new speeches and related data** (IDs 17-18):
   ```bash
   # Export speeches
   pg_dump -h localhost -U oscarm -d cepeda_nlp \
     --data-only --inserts --column-inserts \
     -t speeches \
     --where="id IN (17, 18)" \
     > data/sync_speeches.sql

   # Export entities
   pg_dump -h localhost -U oscarm -d cepeda_nlp \
     --data-only --inserts --column-inserts \
     -t entities \
     --where="speech_id IN (17, 18)" \
     > data/sync_entities.sql

   # Export annotations
   pg_dump -h localhost -U oscarm -d cepeda_nlp \
     --data-only --inserts --column-inserts \
     -t annotations \
     --where="speech_id IN (17, 18)" \
     > data/sync_annotations.sql

   # Export speaker_segments
   pg_dump -h localhost -U oscarm -d cepeda_nlp \
     --data-only --inserts --column-inserts \
     -t speaker_segments \
     --where="speech_id IN (17, 18)" \
     > data/sync_segments.sql

   # Export speech_chunks (includes embeddings)
   pg_dump -h localhost -U oscarm -d cepeda_nlp \
     --data-only --inserts --column-inserts \
     -t speech_chunks \
     --where="speech_id IN (17, 18)" \
     > data/sync_chunks.sql
   ```

2. **Import into Supabase** (in order — speeches first, then dependents):
   ```bash
   PGPASSWORD=<password> psql \
     -h aws-0-us-west-2.pooler.supabase.com \
     -U postgres.airqmqvntfdvhivoenlj \
     -d postgres \
     -f data/sync_speeches.sql \
     -f data/sync_entities.sql \
     -f data/sync_annotations.sql \
     -f data/sync_segments.sql \
     -f data/sync_chunks.sql
   ```

3. **Fix sequences** (serial IDs may conflict if Supabase sequences are behind):
   ```sql
   SELECT setval('speeches_id_seq', (SELECT MAX(id) FROM speeches));
   SELECT setval('entities_id_seq', (SELECT MAX(id) FROM entities));
   SELECT setval('annotations_id_seq', (SELECT MAX(id) FROM annotations));
   SELECT setval('speaker_segments_id_seq', (SELECT MAX(id) FROM speaker_segments));
   SELECT setval('speech_chunks_id_seq', (SELECT MAX(id) FROM speech_chunks));
   ```

**Pros:** Fast (no re-processing, just SQL transfer). Takes seconds.
**Cons:** Manual ID management. Must export in FK order. Embeddings in SQL inserts are very large (768 floats per chunk).

## Option C: Direct Pipeline to Supabase (For Future Runs)

For future speeches, skip local entirely — run the pipeline pointed at Supabase from the start.

```bash
# Set .env to Supabase, then:
nohup python -m src.corpus.pipeline_runner --new=5 > data/pipeline_run.log 2>&1 &
```

New speeches go directly to production. No sync step needed. This is the intended long-term workflow documented in `docs/DEPLOYMENT_ARCHITECTURE.md`.

## Verification

After syncing, verify on the live frontend or via SQL:

```bash
PGPASSWORD=<password> psql \
  -h aws-0-us-west-2.pooler.supabase.com \
  -U postgres.airqmqvntfdvhivoenlj \
  -d postgres \
  -c "SELECT id, title, word_count FROM speeches ORDER BY id DESC LIMIT 5;"
```

The frontend reads from Supabase on every request — no redeployment needed. New speeches appear immediately.
