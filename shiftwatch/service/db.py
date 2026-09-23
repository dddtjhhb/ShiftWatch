"""PostgreSQL connection helpers and a minimal ordered-SQL migration runner."""
from importlib import resources

import psycopg
from psycopg.rows import dict_row

MIGRATION_LOCK_KEY = 72_110_415  # arbitrary constant for pg_advisory_lock


def connect(database_url: str, autocommit: bool = False) -> psycopg.Connection:
    return psycopg.connect(database_url, row_factory=dict_row, autocommit=autocommit)


def _migration_files() -> list[tuple[str, str]]:
    folder = resources.files("shiftwatch.service").joinpath("migrations")
    files = sorted(
        (entry.name, entry.read_text(encoding="utf-8"))
        for entry in folder.iterdir()
        if entry.name.endswith(".sql")
    )
    return files


def migrate(database_url: str) -> list[str]:
    """Apply pending migrations in filename order; safe to run from several processes."""
    applied = []
    with connect(database_url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,))
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            done = {
                row["name"]
                for row in conn.execute("SELECT name FROM schema_migrations").fetchall()
            }
            for name, sql in _migration_files():
                if name in done:
                    continue
                with conn.transaction():
                    conn.execute(sql)
                    conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (name,))
                applied.append(name)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_KEY,))
    return applied
