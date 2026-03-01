"""Lightweight DB connection for the MCP server.

Standalone connection factory that avoids importing heavy pipeline
modules (spaCy, pyannote, etc.). Uses the same env vars as
src/corpus/db_loader.get_connection().
"""

import logging
import os
from contextlib import contextmanager

import psycopg2
from dotenv import load_dotenv

try:
    from pgvector.psycopg2 import register_vector

    _HAS_PGVECTOR = True
except ImportError:
    _HAS_PGVECTOR = False

logger = logging.getLogger(__name__)

load_dotenv()


def get_connection():
    """Create a PostgreSQL connection from .env config."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "cepeda_nlp"),
        user=os.getenv("DB_USER", "oscarm"),
        password=os.getenv("DB_PASSWORD", ""),
    )
    if _HAS_PGVECTOR:
        register_vector(conn)
    return conn


@contextmanager
def db_connection():
    """Context manager that yields a DB connection and auto-closes."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()
