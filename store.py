import sqlite3
from pathlib import Path
from contextlib import contextmanager
import datetime


SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id TEXT PRIMARY KEY,
    name TEXT,
    path TEXT,
    ingested_at TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT,
    doc_name TEXT,
    page INTEGER,
    text TEXT,
    char_start INTEGER,
    char_end INTEGER
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    question TEXT,
    claim_text TEXT,
    stance TEXT,
    confidence REAL,
    chunk_id TEXT,
    doc_id TEXT,
    doc_name TEXT,
    page INTEGER,
    snippet TEXT,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc
ON chunks(doc_id);

CREATE INDEX IF NOT EXISTS idx_claims_q
ON claims(question);
"""


class Store:

    def __init__(self, path: Path):
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with self.conn() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row

        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def upsert_doc(self, doc_id, name, path):

        with self.conn() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO docs
                VALUES (?, ?, ?, ?)
                """,
                (
                    doc_id,
                    name,
                    str(path),
                    datetime.datetime.now().isoformat()
                )
            )

    def add_chunks(self, chunks):

        with self.conn() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO chunks
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.doc_name,
                        chunk.page,
                        chunk.text,
                        chunk.char_start,
                        chunk.char_end
                    )
                    for chunk in chunks
                ]
            )

    def all_chunks(self):

        with self.conn() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks"
            ).fetchall()

            return [dict(row) for row in rows]

    def get_chunk(self, chunk_id):

        with self.conn() as connection:
            row = connection.execute(
                """
                SELECT * FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,)
            ).fetchone()

            return dict(row) if row else None

    def clear(self):

        with self.conn() as connection:
            connection.executescript(
                """
                DELETE FROM claims;
                DELETE FROM chunks;
                DELETE FROM docs;
                """
            )