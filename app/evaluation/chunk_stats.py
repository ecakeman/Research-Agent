from __future__ import annotations

from statistics import median

from app.db import get_conn


def chunk_statistics(conn=None) -> dict:
    def _run(c):
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS n,
                       MIN(token_count) AS min_t,
                       MAX(token_count) AS max_t,
                       AVG(token_count)::float AS mean_t
                FROM chunks
                """
            )
            row = cur.fetchone()
            cur.execute("SELECT token_count FROM chunks ORDER BY token_count")
            tokens = [r["token_count"] for r in cur.fetchall()]
            cur.execute("SELECT COUNT(*) AS n FROM documents")
            docs = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM chunk_embeddings")
            embs = cur.fetchone()["n"]
            cur.execute(
                """
                SELECT chunk_type, COUNT(*) AS n
                FROM chunks GROUP BY chunk_type ORDER BY n DESC
                """
            )
            types = {r["chunk_type"]: r["n"] for r in cur.fetchall()}
            cur.execute(
                """
                SELECT heading_path, chunk_type, token_count,
                       left(content, 80) AS preview,
                       left(content_with_context, 120) AS ctx
                FROM chunks
                ORDER BY token_count
                LIMIT 3
                """
            )
            samples_short = [dict(r) for r in cur.fetchall()]
            cur.execute(
                """
                SELECT heading_path, chunk_type, token_count,
                       left(content_with_context, 160) AS ctx
                FROM chunks
                WHERE content_with_context ILIKE 'Document:%'
                ORDER BY token_count DESC
                LIMIT 3
                """
            )
            samples_ctx = [dict(r) for r in cur.fetchall()]
        med = int(median(tokens)) if tokens else None
        return {
            "documents": docs,
            "chunks": row["n"],
            "embeddings": embs,
            "min_token": row["min_t"],
            "max_token": row["max_t"],
            "mean_token": round(row["mean_t"], 2) if row["mean_t"] is not None else None,
            "median_token": med,
            "chunk_types": types,
            "samples_short": samples_short,
            "samples_context": samples_ctx,
        }

    if conn is not None:
        return _run(conn)
    with get_conn() as c:
        return _run(c)


def print_stats(stats: dict) -> None:
    print("Chunk quality (not an experiment metric)")
    print(f"documents:   {stats['documents']}")
    print(f"chunks:      {stats['chunks']}")
    print(f"embeddings:  {stats['embeddings']}")
    print(f"min token:   {stats['min_token']}")
    print(f"max token:   {stats['max_token']}")
    print(f"mean token:  {stats['mean_token']}")
    print(f"median token:{stats['median_token']}")
    print(f"types:       {stats['chunk_types']}")


def main() -> None:
    print_stats(chunk_statistics())


if __name__ == "__main__":
    main()
