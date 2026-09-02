from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from app.config import settings

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def connect(url: str | None = None) -> psycopg.Connection:
    return psycopg.connect(url or settings.database_url, row_factory=dict_row)


@contextmanager
def get_conn(url: str | None = None):
    conn = connect(url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def apply_migrations(url: str | None = None) -> None:
    with get_conn(url) as conn:
        with conn.cursor() as cur:
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                cur.execute(path.read_text(encoding="utf-8"))
        ensure_embedding_column(conn, settings.embedding_dim)


def ensure_embedding_column(conn: psycopg.Connection, dim: int) -> None:
    """按配置维度设置 vector 列；空表可改 typmod。"""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'chunk_embeddings'::regclass
              AND attname = 'embedding'
            """
        )
        row = cur.fetchone()
        if row is None:
            return
        typmod = row["atttypmod"]
        current_dim = None if typmod is None or typmod < 0 else int(typmod)
        if current_dim == dim:
            return
        cur.execute("SELECT COUNT(*) AS n FROM chunk_embeddings")
        n = cur.fetchone()["n"]
        if n > 0 and current_dim is not None and current_dim != dim:
            raise RuntimeError(
                f"chunk_embeddings.embedding dim={current_dim} 与 EMBEDDING_DIM={dim} 不一致"
            )
        cur.execute(
            f"ALTER TABLE chunk_embeddings ALTER COLUMN embedding TYPE vector({int(dim)})"
        )
        cur.execute("DROP INDEX IF EXISTS chunk_embeddings_hnsw_idx")
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS chunk_embeddings_hnsw_idx
            ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)
            """
        )
