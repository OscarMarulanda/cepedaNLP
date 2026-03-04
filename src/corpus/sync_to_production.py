"""Sync new speeches from local DB to Supabase production DB.

Connects to both databases simultaneously, finds speeches missing in
production (by youtube_url), and copies all related data (entities,
annotations, speaker_segments, speech_chunks with embeddings).

No .env swapping, no re-processing. Runs in seconds.

Usage:
    # Set Supabase credentials, then run:
    SUPABASE_PASSWORD=<password> python -m src.corpus.sync_to_production

    # Dry run (show what would be synced, don't write):
    SUPABASE_PASSWORD=<password> python -m src.corpus.sync_to_production --dry-run
"""

import argparse
import logging
import os
import sys

import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

try:
    from pgvector.psycopg2 import register_vector
    _HAS_PGVECTOR = True
except ImportError:
    _HAS_PGVECTOR = False

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Supabase connection defaults (override with env vars)
SUPABASE_HOST = os.getenv(
    "SUPABASE_HOST", "aws-0-us-west-2.pooler.supabase.com"
)
SUPABASE_PORT = os.getenv("SUPABASE_PORT", "5432")
SUPABASE_DB = os.getenv("SUPABASE_DB", "postgres")
SUPABASE_USER = os.getenv(
    "SUPABASE_USER", "postgres.airqmqvntfdvhivoenlj"
)
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD", "")
SUPABASE_SSLMODE = os.getenv("SUPABASE_SSLMODE", "verify-full")
SUPABASE_SSLROOTCERT = os.getenv(
    "SUPABASE_SSLROOTCERT", "certs/supabase-ca.crt"
)

# Tables to sync (in FK order — parent first)
DEPENDENT_TABLES = ["entities", "annotations", "speaker_segments", "speech_chunks"]


def get_local_conn():
    """Connect to local PostgreSQL using .env defaults."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "cepeda_nlp"),
        user=os.getenv("DB_USER", "oscarm"),
        password=os.getenv("DB_PASSWORD", ""),
        sslmode=os.getenv("DB_SSLMODE", "prefer"),
    )
    if _HAS_PGVECTOR:
        register_vector(conn)
    return conn


def get_supabase_conn():
    """Connect to Supabase production DB."""
    if not SUPABASE_PASSWORD:
        logger.error("SUPABASE_PASSWORD env var is required")
        sys.exit(1)

    ssl_kwargs = {}
    if SUPABASE_SSLROOTCERT and os.path.exists(SUPABASE_SSLROOTCERT):
        ssl_kwargs["sslrootcert"] = SUPABASE_SSLROOTCERT

    conn = psycopg2.connect(
        host=SUPABASE_HOST,
        port=SUPABASE_PORT,
        dbname=SUPABASE_DB,
        user=SUPABASE_USER,
        password=SUPABASE_PASSWORD,
        sslmode=SUPABASE_SSLMODE,
        **ssl_kwargs,
    )
    if _HAS_PGVECTOR:
        register_vector(conn)
    return conn


def find_missing_speeches(local_cur, prod_cur) -> list[dict]:
    """Find speeches in local DB that don't exist in production."""
    local_cur.execute(
        "SELECT id, title, youtube_url FROM speeches ORDER BY id"
    )
    local_speeches = local_cur.fetchall()

    prod_cur.execute("SELECT youtube_url, title FROM speeches")
    prod_urls = {row[0] for row in prod_cur.fetchall() if row[0]}

    missing = []
    for row in local_speeches:
        local_id, title, youtube_url = row
        if youtube_url and youtube_url not in prod_urls:
            missing.append({
                "local_id": local_id,
                "title": title,
                "youtube_url": youtube_url,
            })

    return missing


def get_speech_columns(cur) -> list[str]:
    """Get column names for the speeches table (excluding 'id')."""
    cur.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'speeches' AND column_name != 'id'
        ORDER BY ordinal_position
        """
    )
    return [row[0] for row in cur.fetchall()]


def get_table_columns(cur, table: str) -> list[str]:
    """Get column names for a table (excluding 'id')."""
    cur.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_name = %s AND column_name != 'id'
        ORDER BY ordinal_position
        """,
        (table,),
    )
    return [row[0] for row in cur.fetchall()]


def sync_speech(local_cur, prod_cur, local_id: int, title: str) -> int | None:
    """Copy a speech and all dependent rows from local to production.

    Returns the new speech ID in production, or None on failure.
    """
    # 1. Copy the speech row
    columns = get_speech_columns(local_cur)
    col_list = ", ".join(columns)
    local_cur.execute(
        f"SELECT {col_list} FROM speeches WHERE id = %s", (local_id,)
    )
    row = local_cur.fetchone()
    if not row:
        logger.warning("Speech %d not found in local DB", local_id)
        return None

    placeholders = ", ".join(["%s"] * len(columns))
    adapted_row = tuple(
        Json(v) if isinstance(v, (dict, list)) else v for v in row
    )
    prod_cur.execute(
        f"INSERT INTO speeches ({col_list}) VALUES ({placeholders}) RETURNING id",
        adapted_row,
    )
    new_id = prod_cur.fetchone()[0]
    logger.info("  Inserted speech '%s' → prod id=%d", title, new_id)

    # 2. Copy dependent tables
    for table in DEPENDENT_TABLES:
        columns = get_table_columns(local_cur, table)
        if "speech_id" not in columns:
            continue

        col_list = ", ".join(columns)
        local_cur.execute(
            f"SELECT {col_list} FROM {table} WHERE speech_id = %s",
            (local_id,),
        )
        rows = local_cur.fetchall()

        if not rows:
            continue

        # Replace local speech_id with new production ID
        speech_id_idx = columns.index("speech_id")
        remapped_rows = []
        for r in rows:
            r_list = list(r)
            r_list[speech_id_idx] = new_id
            remapped_rows.append(tuple(r_list))

        placeholders = ", ".join(["%s"] * len(columns))
        for r in remapped_rows:
            adapted = tuple(
                Json(v) if isinstance(v, (dict, list)) else v for v in r
            )
            prod_cur.execute(
                f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",
                adapted,
            )

        logger.info("  %s: %d rows", table, len(remapped_rows))

    return new_id


def main():
    parser = argparse.ArgumentParser(
        description="Sync new speeches from local DB to Supabase"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without writing",
    )
    args = parser.parse_args()

    local_conn = get_local_conn()
    prod_conn = get_supabase_conn()

    try:
        local_cur = local_conn.cursor()
        prod_cur = prod_conn.cursor()

        missing = find_missing_speeches(local_cur, prod_cur)

        if not missing:
            logger.info("Production is up to date — nothing to sync")
            return

        logger.info("Found %d speeches to sync:", len(missing))
        for s in missing:
            logger.info(
                "  [local id=%d] %s", s["local_id"], s["title"]
            )

        if args.dry_run:
            logger.info("Dry run — no changes made")
            return

        synced = 0
        for s in missing:
            new_id = sync_speech(
                local_cur, prod_cur, s["local_id"], s["title"]
            )
            if new_id is not None:
                synced += 1

        prod_conn.commit()
        logger.info(
            "Sync complete: %d/%d speeches pushed to production",
            synced, len(missing),
        )

    finally:
        local_conn.close()
        prod_conn.close()


if __name__ == "__main__":
    main()
