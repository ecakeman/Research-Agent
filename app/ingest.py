from pathlib import Path

from app.ingestion.pipeline import ingest_path

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"


def main() -> None:
    stats = ingest_path(DOCS_DIR)
    print(
        f"ingest done documents_inserted={stats['documents_inserted']} "
        f"skipped={stats['documents_skipped']} chunks={stats['chunks']}"
    )


if __name__ == "__main__":
    main()
