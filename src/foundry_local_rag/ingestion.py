"""Document ingestion orchestration using the existing local RAG boundaries."""

from __future__ import annotations

from pathlib import Path

from .embeddings import FoundryLocalEmbeddingAdapter
from .errors import EmbeddingError
from .persistence import (
    PersistedChunk,
    initialize_database,
    replace_source_chunks,
)
from .text_processing import chunk_paragraphs, read_text_file


def ingest_text_document(
    source_path: Path,
    database_path: Path,
    embedding_adapter: FoundryLocalEmbeddingAdapter,
) -> None:
    """Read, chunk, embed, and atomically persist one UTF-8 text document.

    The canonical absolute path identifies the source. Re-ingesting that path
    replaces its complete prior chunk set; the supplied adapter remains
    caller-owned.
    """

    path = Path(source_path)
    text = read_text_file(path)
    chunk_texts = chunk_paragraphs(text)
    embeddings = embedding_adapter.embed_texts(chunk_texts)
    if len(embeddings) != len(chunk_texts):
        raise EmbeddingError("Embedding count does not match document chunks")

    source_id = path.resolve().as_posix()
    records = tuple(
        PersistedChunk(source_id, index, chunk_text, embedding)
        for index, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings))
    )

    initialize_database(database_path)
    replace_source_chunks(database_path, source_id, records)
