import tempfile
import unittest
from pathlib import Path

from foundry_local_rag.errors import DocumentError, EmbeddingError
from foundry_local_rag.ingestion import ingest_text_document
from foundry_local_rag.persistence import PersistedChunk, load_chunks


class FakeEmbeddingAdapter:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []
        self.failure: Exception | None = None
        self.close_count = 0

    def embed_texts(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        self.inputs.append(list(texts))
        if self.failure is not None:
            raise self.failure
        return tuple((float(index), float(len(text))) for index, text in enumerate(texts))

    def close(self) -> None:
        self.close_count += 1


class IngestionTests(unittest.TestCase):
    def test_ingests_multiple_chunks_in_order_with_matching_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source_path = directory / "notes.txt"
            source_path.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
            database_path = directory / "chunks.sqlite3"
            adapter = FakeEmbeddingAdapter()

            ingest_text_document(source_path, database_path, adapter)  # type: ignore[arg-type]

            source_id = source_path.resolve().as_posix()
            self.assertEqual(adapter.inputs, [["First paragraph.", "Second paragraph."]])
            self.assertEqual(
                load_chunks(database_path),
                (
                    PersistedChunk(source_id, 0, "First paragraph.", (0.0, 16.0)),
                    PersistedChunk(source_id, 1, "Second paragraph.", (1.0, 17.0)),
                ),
            )
            self.assertEqual(adapter.close_count, 0)

    def test_reingestion_replaces_unchanged_and_shorter_source_without_stale_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source_path = directory / "notes.txt"
            database_path = directory / "chunks.sqlite3"
            adapter = FakeEmbeddingAdapter()
            source_path.write_text("One\n\nTwo", encoding="utf-8")

            ingest_text_document(source_path, database_path, adapter)  # type: ignore[arg-type]
            ingest_text_document(source_path, database_path, adapter)  # type: ignore[arg-type]
            self.assertEqual(len(load_chunks(database_path)), 2)

            source_path.write_text("Replacement", encoding="utf-8")
            ingest_text_document(source_path, database_path, adapter)  # type: ignore[arg-type]

            self.assertEqual(
                tuple((chunk.chunk_index, chunk.text) for chunk in load_chunks(database_path)),
                ((0, "Replacement"),),
            )

    def test_sources_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first_path = directory / "first.txt"
            second_path = directory / "second.txt"
            database_path = directory / "chunks.sqlite3"
            adapter = FakeEmbeddingAdapter()
            first_path.write_text("First", encoding="utf-8")
            second_path.write_text("Second", encoding="utf-8")

            ingest_text_document(first_path, database_path, adapter)  # type: ignore[arg-type]
            ingest_text_document(second_path, database_path, adapter)  # type: ignore[arg-type]
            first_path.write_text("First replacement", encoding="utf-8")
            ingest_text_document(first_path, database_path, adapter)  # type: ignore[arg-type]

            self.assertEqual(
                tuple((chunk.source_id, chunk.text) for chunk in load_chunks(database_path)),
                (
                    (first_path.resolve().as_posix(), "First replacement"),
                    (second_path.resolve().as_posix(), "Second"),
                ),
            )

    def test_read_failure_does_not_embed_or_change_persisted_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source_path = directory / "notes.txt"
            database_path = directory / "chunks.sqlite3"
            adapter = FakeEmbeddingAdapter()
            source_path.write_text("Original", encoding="utf-8")
            ingest_text_document(source_path, database_path, adapter)  # type: ignore[arg-type]
            before = load_chunks(database_path)
            source_path.unlink()

            with self.assertRaises(DocumentError):
                ingest_text_document(source_path, database_path, adapter)  # type: ignore[arg-type]

            self.assertEqual(adapter.inputs, [["Original"]])
            self.assertEqual(load_chunks(database_path), before)

    def test_embedding_failure_does_not_change_persisted_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source_path = directory / "notes.txt"
            database_path = directory / "chunks.sqlite3"
            adapter = FakeEmbeddingAdapter()
            source_path.write_text("Original", encoding="utf-8")
            ingest_text_document(source_path, database_path, adapter)  # type: ignore[arg-type]
            before = load_chunks(database_path)
            source_path.write_text("Changed", encoding="utf-8")
            adapter.failure = EmbeddingError("embedding failed")

            with self.assertRaisesRegex(EmbeddingError, "embedding failed"):
                ingest_text_document(source_path, database_path, adapter)  # type: ignore[arg-type]

            self.assertEqual(load_chunks(database_path), before)


if __name__ == "__main__":
    unittest.main()
