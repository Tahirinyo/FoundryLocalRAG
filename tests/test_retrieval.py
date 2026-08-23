import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from foundry_local_rag.errors import ApplicationError, EmbeddingError, PersistenceError, RetrievalError
from foundry_local_rag.persistence import PersistedChunk, initialize_database, store_chunks
from foundry_local_rag.retrieval import RetrievedChunk, cosine_similarity, retrieve_chunks


class FakeEmbeddingAdapter:
    def __init__(self, embedding: tuple[float, ...] = (1.0, 0.0)) -> None:
        self.embedding = embedding
        self.inputs: list[str] = []
        self.failure: Exception | None = None

    def embed_text(self, text: str) -> tuple[float, ...]:
        self.inputs.append(text)
        if self.failure is not None:
            raise self.failure
        return self.embedding


class RetrievalTests(unittest.TestCase):
    def make_database(self, records: tuple[PersistedChunk, ...]) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        directory = tempfile.TemporaryDirectory()
        database_path = Path(directory.name) / "chunks.sqlite3"
        initialize_database(database_path)
        store_chunks(database_path, records)
        return directory, database_path

    def test_cosine_similarity_handles_known_vectors(self) -> None:
        self.assertAlmostEqual(cosine_similarity((1.0, 0.0), (1.0, 0.0)), 1.0)
        self.assertAlmostEqual(cosine_similarity((1.0, 0.0), (0.0, 1.0)), 0.0)
        self.assertAlmostEqual(cosine_similarity((1.0, 1.0), (-1.0, -1.0)), -1.0)

    def test_ranks_and_truncates_results_without_reembedding_documents(self) -> None:
        directory, database_path = self.make_database(
            (
                PersistedChunk("source-b", 0, "opposite", (-1.0, 0.0)),
                PersistedChunk("source-a", 1, "near", (2.0, 1.0)),
                PersistedChunk("source-a", 0, "best", (1.0, 0.0)),
            )
        )
        self.addCleanup(directory.cleanup)
        adapter = FakeEmbeddingAdapter()

        results = retrieve_chunks("question", database_path, adapter, 2)  # type: ignore[arg-type]

        self.assertEqual(adapter.inputs, ["question"])
        self.assertEqual(
            tuple((result.source_id, result.chunk_index, result.text) for result in results),
            (("source-a", 0, "best"), ("source-a", 1, "near")),
        )
        self.assertAlmostEqual(results[0].score, 1.0)
        self.assertAlmostEqual(results[1].score, 2.0 / (5.0 ** 0.5))

    def test_returns_all_available_results_when_k_is_larger(self) -> None:
        directory, database_path = self.make_database(
            (PersistedChunk("source", 0, "text", (1.0, 0.0)),)
        )
        self.addCleanup(directory.cleanup)

        results = retrieve_chunks("question", database_path, FakeEmbeddingAdapter(), 5)  # type: ignore[arg-type]

        self.assertEqual(len(results), 1)

    def test_empty_database_returns_empty_results_after_one_query_embedding(self) -> None:
        directory, database_path = self.make_database(())
        self.addCleanup(directory.cleanup)
        adapter = FakeEmbeddingAdapter()

        self.assertEqual(retrieve_chunks("question", database_path, adapter, 1), ())  # type: ignore[arg-type]
        self.assertEqual(adapter.inputs, ["question"])

    def test_equal_scores_use_source_then_chunk_index_order(self) -> None:
        directory, database_path = self.make_database(
            (
                PersistedChunk("source-b", 0, "B", (1.0, 0.0)),
                PersistedChunk("source-a", 1, "A second", (1.0, 0.0)),
                PersistedChunk("source-a", 0, "A first", (1.0, 0.0)),
            )
        )
        self.addCleanup(directory.cleanup)

        results = retrieve_chunks("question", database_path, FakeEmbeddingAdapter(), 3)  # type: ignore[arg-type]

        self.assertEqual(
            tuple((result.source_id, result.chunk_index) for result in results),
            (("source-a", 0), ("source-a", 1), ("source-b", 0)),
        )

    def test_result_metadata_stays_with_the_matching_score(self) -> None:
        directory, database_path = self.make_database(
            (
                PersistedChunk("first", 4, "first text", (0.0, 1.0)),
                PersistedChunk("second", 2, "second text", (1.0, 0.0)),
            )
        )
        self.addCleanup(directory.cleanup)

        results = retrieve_chunks("question", database_path, FakeEmbeddingAdapter(), 2)  # type: ignore[arg-type]

        self.assertEqual(
            results,
            (
                RetrievedChunk("second", 2, "second text", 1.0),
                RetrievedChunk("first", 4, "first text", 0.0),
            ),
        )

    def test_invalid_k_is_rejected_before_embedding(self) -> None:
        adapter = FakeEmbeddingAdapter()
        for top_k in (True, False, 0, -1, 1.5, "1", None):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(RetrievalError, "top_k"):
                    retrieve_chunks("question", Path("missing.sqlite3"), adapter, top_k)  # type: ignore[arg-type]
        self.assertEqual(adapter.inputs, [])

    def test_dimension_mismatch_is_rejected(self) -> None:
        directory, database_path = self.make_database(
            (PersistedChunk("source", 0, "text", (1.0, 0.0, 0.0)),)
        )
        self.addCleanup(directory.cleanup)

        with self.assertRaisesRegex(RetrievalError, "dimensions"):
            retrieve_chunks("question", database_path, FakeEmbeddingAdapter(), 1)  # type: ignore[arg-type]

    def test_zero_query_and_persisted_vectors_are_rejected(self) -> None:
        directory, database_path = self.make_database(
            (PersistedChunk("source", 0, "text", (1.0, 0.0)),)
        )
        self.addCleanup(directory.cleanup)
        with self.assertRaisesRegex(RetrievalError, "zero-magnitude"):
            retrieve_chunks("question", database_path, FakeEmbeddingAdapter((0.0, 0.0)), 1)  # type: ignore[arg-type]

        directory, database_path = self.make_database(
            (PersistedChunk("source", 0, "text", (0.0, 0.0)),)
        )
        self.addCleanup(directory.cleanup)
        with self.assertRaisesRegex(RetrievalError, "zero-magnitude"):
            retrieve_chunks("question", database_path, FakeEmbeddingAdapter(), 1)  # type: ignore[arg-type]

    def test_malformed_and_non_finite_stored_vectors_propagate_persistence_errors(self) -> None:
        for embedding_json in ("not-json", json.dumps([float("nan")])):
            with self.subTest(embedding_json=embedding_json):
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
                                ("source", 0, "text", embedding_json, 1),
                            )

                    with self.assertRaises(PersistenceError):
                        retrieve_chunks("question", database_path, FakeEmbeddingAdapter(), 1)  # type: ignore[arg-type]

    def test_embedding_failures_propagate_without_loading_or_ranking(self) -> None:
        adapter = FakeEmbeddingAdapter()
        adapter.failure = EmbeddingError("embedding failed")

        with self.assertRaisesRegex(EmbeddingError, "embedding failed"):
            retrieve_chunks("question", Path("missing.sqlite3"), adapter, 1)  # type: ignore[arg-type]
        self.assertEqual(adapter.inputs, ["question"])

    def test_retrieval_error_uses_application_error_contract(self) -> None:
        self.assertTrue(issubclass(RetrievalError, ApplicationError))


if __name__ == "__main__":
    unittest.main()
