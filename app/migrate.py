from app.db import apply_migrations


def main() -> None:
    apply_migrations()
    print("migrations applied")


if __name__ == "__main__":
    main()
