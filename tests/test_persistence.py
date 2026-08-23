import json
import math
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from foundry_local_rag.errors import ApplicationError, PersistenceError
from foundry_local_rag.persistence import (
    PersistedChunk,
    initialize_database,
    load_chunks,
    replace_source_chunks,
    store_chunks,
)


class PersistenceTests(unittest.TestCase):
    def test_initialize_creates_schema_and_is_safe_to_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "nested" / "chunks.sqlite3"

            initialize_database(database_path)
            record = PersistedChunk("source", 0, "text", (1.0,))
            store_chunks(database_path, (record,))
            initialize_database(database_path)

            self.assertTrue(database_path.is_file())
            self.assertEqual(load_chunks(database_path), (record,))

    def test_round_trip_preserves_unicode_quotes_and_embedding_order(self) -> None:
        record = PersistedChunk(
            source_id="C:/Kaynaklar/O'Reilly.txt",
            chunk_index=2,
            text="Merhaba, d\u00fcnya. O'Reilly'nin notu.",
            embedding=(1, -1.5, 3.0),
        )
        expected_record = PersistedChunk(
            source_id=record.source_id,
            chunk_index=record.chunk_index,
            text=record.text,
            embedding=(1.0, -1.5, 3.0),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "database folder" / "chunks.sqlite3"
            initialize_database(database_path)
            store_chunks(database_path, (record,))

            self.assertEqual(load_chunks(database_path), (expected_record,))

    def test_load_order_is_source_then_chunk_index(self) -> None:
        records = (
            PersistedChunk("source-b", 1, "B second", (2.0,)),
            PersistedChunk("source-a", 1, "A second", (3.0,)),
            PersistedChunk("source-a", 0, "A first", (1.0,)),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "chunks.sqlite3"
            initialize_database(database_path)
            store_chunks(database_path, records)

            self.assertEqual(
                load_chunks(database_path),
                (records[2], records[1], records[0]),
            )

    def test_store_and_load_missing_database_do_not_create_file(self) -> None:
        record = PersistedChunk("source", 0, "text", (1.0,))

        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "missing.sqlite3"

            with self.assertRaisesRegex(PersistenceError, "not initialized"):
                store_chunks(database_path, (record,))
            self.assertFalse(database_path.exists())

            with self.assertRaisesRegex(PersistenceError, "not initialized"):
                load_chunks(database_path)
            self.assertFalse(database_path.exists())

    def test_invalid_input_and_duplicate_batch_roll_back(self) -> None:
        record = PersistedChunk("source", 0, "text", (1.0,))

        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "chunks.sqlite3"
            initialize_database(database_path)

            with self.assertRaisesRegex(PersistenceError, "Embedding must not be empty"):
                store_chunks(
                    database_path,
                    (PersistedChunk("source", 1, "invalid", ()),),
                )
            self.assertEqual(load_chunks(database_path), ())

            with self.assertRaises(PersistenceError):
                store_chunks(database_path, (record, record))
            self.assertEqual(load_chunks(database_path), ())

    def test_replace_source_chunks_rejects_records_for_another_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "chunks.sqlite3"
            initialize_database(database_path)

            with self.assertRaisesRegex(PersistenceError, "belong to the replacement source"):
                replace_source_chunks(
                    database_path,
                    "source-a",
                    (PersistedChunk("source-b", 0, "text", (1.0,)),),
                )

            self.assertEqual(load_chunks(database_path), ())

    def test_replace_source_chunks_is_atomic_and_preserves_other_sources(self) -> None:
        old_source = PersistedChunk("source-a", 0, "old", (1.0,))
        other_source = PersistedChunk("source-b", 0, "other", (2.0,))

        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "chunks.sqlite3"
            initialize_database(database_path)
            store_chunks(database_path, (old_source, other_source))

            with closing(sqlite3.connect(database_path)) as connection:
                with connection:
                    connection.execute(
                        """
                        CREATE TRIGGER fail_source_a_insert
                        BEFORE INSERT ON chunks
                        WHEN NEW.source_id = 'source-a'
                        BEGIN
                            SELECT RAISE(ABORT, 'forced failure');
                        END
                        """
                    )

            with self.assertRaises(PersistenceError):
                replace_source_chunks(
                    database_path,
                    "source-a",
                    (PersistedChunk("source-a", 0, "new", (3.0,)),),
                )

            self.assertEqual(load_chunks(database_path), (old_source, other_source))

    def test_rejects_invalid_embedding_values(self) -> None:
        invalid_embeddings = (
            (True,),
            ("1.0",),
            (math.nan,),
            (math.inf,),
            (-math.inf,),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "chunks.sqlite3"
            initialize_database(database_path)

            for embedding in invalid_embeddings:
                with self.subTest(embedding=embedding):
                    with self.assertRaises(PersistenceError):
                        store_chunks(
                            database_path,
                            (PersistedChunk("source", 0, "text", embedding),),
                        )

    def test_rejects_malformed_and_dimension_mismatched_stored_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "chunks.sqlite3"
            initialize_database(database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO chunks (
                            source_id, chunk_index, chunk_text, embedding_json, embedding_dimension
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        ("source", 0, "text", "not-json", 1),
                    )

            with self.assertRaisesRegex(PersistenceError, "malformed"):
                load_chunks(database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                with connection:
                    connection.execute("DELETE FROM chunks")
                    connection.execute(
                        """
                        INSERT INTO chunks (
                            source_id, chunk_index, chunk_text, embedding_json, embedding_dimension
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        ("source", 0, "text", json.dumps([1.0]), 2),
                    )

            with self.assertRaisesRegex(PersistenceError, "does not match"):
                load_chunks(database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                with connection:
                    connection.execute("DELETE FROM chunks")
                    connection.execute(
                        """
                        INSERT INTO chunks (
                            source_id, chunk_index, chunk_text, embedding_json, embedding_dimension
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        ("source", 0, "text", json.dumps([True]), 1),
                    )

            with self.assertRaisesRegex(PersistenceError, "must be numeric"):
                load_chunks(database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                with connection:
                    connection.execute("DELETE FROM chunks")
                    connection.execute(
                        """
                        INSERT INTO chunks (
                            source_id, chunk_index, chunk_text, embedding_json, embedding_dimension
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        ("source", 0, "text", json.dumps({"value": 1.0}), 1),
                    )

            with self.assertRaisesRegex(PersistenceError, "must be a JSON array"):
                load_chunks(database_path)

    def test_existing_database_without_schema_raises_persistence_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "empty.sqlite3"

            with closing(sqlite3.connect(database_path)):
                pass

            with self.assertRaises(PersistenceError):
                load_chunks(database_path)

    def test_connections_are_closed_after_operations(self) -> None:
        record = PersistedChunk("source", 0, "text", (1.0,))

        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "chunks.sqlite3"
            initialize_database(database_path)
            store_chunks(database_path, (record,))
            load_chunks(database_path)

            database_path.unlink()
            self.assertFalse(database_path.exists())

    def test_persistence_error_uses_application_error_contract(self) -> None:
        self.assertTrue(issubclass(PersistenceError, ApplicationError))


if __name__ == "__main__":
    unittest.main()
