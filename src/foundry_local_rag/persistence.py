"""SQLite persistence for chunk records and precomputed embeddings."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .errors import PersistenceError

_CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    chunk_text TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
    UNIQUE (source_id, chunk_index)
)
"""

_INSERT_CHUNK = """
INSERT INTO chunks (
    source_id,
    chunk_index,
    chunk_text,
    embedding_json,
    embedding_dimension
) VALUES (?, ?, ?, ?, ?)
"""

_SELECT_CHUNKS = """
SELECT source_id, chunk_index, chunk_text, embedding_json, embedding_dimension
FROM chunks
ORDER BY source_id, chunk_index, id
"""


@dataclass(frozen=True)
class PersistedChunk:
    """A chunk and its precomputed embedding at the persistence boundary."""

    source_id: str
    chunk_index: int
    text: str
    embedding: tuple[float, ...]


def initialize_database(database_path: Path) -> None:
    """Explicitly create the database parent directory and required schema."""

    path = Path(database_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as connection:
            with connection:
                connection.execute(_CREATE_CHUNKS_TABLE)
    except (OSError, sqlite3.Error) as error:
        raise PersistenceError(f"Unable to initialize database: {path}") from error


def store_chunks(database_path: Path, chunks: Iterable[PersistedChunk]) -> None:
    """Atomically store validated chunk records in an initialized database."""

    rows = tuple(_record_to_row(chunk) for chunk in chunks)
    connection = _connect_existing_database(database_path)
    try:
        with closing(connection):
            with connection:
                connection.executemany(_INSERT_CHUNK, rows)
    except sqlite3.Error as error:
        raise PersistenceError("Unable to store chunk records") from error


def load_chunks(database_path: Path) -> tuple[PersistedChunk, ...]:
    """Load validated persisted chunk records in deterministic order."""

    connection = _connect_existing_database(database_path)
    try:
        with closing(connection):
            with closing(connection.execute(_SELECT_CHUNKS)) as cursor:
                rows = cursor.fetchall()
    except sqlite3.Error as error:
        raise PersistenceError("Unable to load chunk records") from error

    return tuple(_row_to_record(row) for row in rows)


def _connect_existing_database(database_path: Path) -> sqlite3.Connection:
    path = Path(database_path)
    if not path.is_file():
        raise PersistenceError(f"Database is not initialized: {path}")

    try:
        database_uri = f"{path.resolve().as_uri()}?mode=rw"
        return sqlite3.connect(database_uri, uri=True)
    except (OSError, sqlite3.Error) as error:
        raise PersistenceError(f"Unable to open database: {path}") from error


def _record_to_row(chunk: PersistedChunk) -> tuple[str, int, str, str, int]:
    if not isinstance(chunk.source_id, str) or not chunk.source_id.strip():
        raise PersistenceError("Chunk source_id must be a non-empty string")
    if isinstance(chunk.chunk_index, bool) or not isinstance(chunk.chunk_index, int):
        raise PersistenceError("Chunk index must be an integer")
    if chunk.chunk_index < 0:
        raise PersistenceError("Chunk index must not be negative")
    if not isinstance(chunk.text, str) or not chunk.text.strip():
        raise PersistenceError("Chunk text must be a non-empty string")

    embedding = _validated_embedding(chunk.embedding)
    embedding_json = json.dumps(list(embedding), separators=(",", ":"), allow_nan=False)
    return (
        chunk.source_id,
        chunk.chunk_index,
        chunk.text,
        embedding_json,
        len(embedding),
    )


def _row_to_record(row: tuple[object, ...]) -> PersistedChunk:
    source_id, chunk_index, text, embedding_json, embedding_dimension = row
    if not isinstance(source_id, str) or not source_id.strip():
        raise PersistenceError("Stored chunk source_id is invalid")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise PersistenceError("Stored chunk index is invalid")
    if not isinstance(text, str) or not text.strip():
        raise PersistenceError("Stored chunk text is invalid")
    if (
        isinstance(embedding_dimension, bool)
        or not isinstance(embedding_dimension, int)
        or embedding_dimension <= 0
    ):
        raise PersistenceError("Stored embedding dimension is invalid")
    if not isinstance(embedding_json, str):
        raise PersistenceError("Stored embedding data is invalid")

    try:
        decoded_embedding = json.loads(embedding_json)
    except json.JSONDecodeError as error:
        raise PersistenceError("Stored embedding data is malformed") from error
    if not isinstance(decoded_embedding, list):
        raise PersistenceError("Stored embedding data must be a JSON array")
    if len(decoded_embedding) != embedding_dimension:
        raise PersistenceError("Stored embedding dimension does not match vector data")

    return PersistedChunk(
        source_id=source_id,
        chunk_index=chunk_index,
        text=text,
        embedding=_validated_embedding(decoded_embedding),
    )


def _validated_embedding(values: Iterable[object]) -> tuple[float, ...]:
    try:
        embedding = tuple(values)
    except TypeError as error:
        raise PersistenceError("Embedding must be an iterable of numeric values") from error
    if not embedding:
        raise PersistenceError("Embedding must not be empty")

    normalized_values: list[float] = []
    for value in embedding:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PersistenceError("Embedding values must be numeric")
        normalized_value = float(value)
        if not math.isfinite(normalized_value):
            raise PersistenceError("Embedding values must be finite")
        normalized_values.append(normalized_value)
    return tuple(normalized_values)
