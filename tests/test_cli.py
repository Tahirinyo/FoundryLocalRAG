import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from foundry_local_rag.answering import AnswerResult
from foundry_local_rag.cli import main
from foundry_local_rag.config import AppConfig
from foundry_local_rag.errors import ApplicationError, DocumentError
from foundry_local_rag.retrieval import RetrievedChunk


class FakeAdapter:
    def __init__(self, close_error: Exception | None = None) -> None:
        self.close_count = 0
        self.close_error = close_error

    def close(self) -> None:
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


class CliTests(unittest.TestCase):
    def test_help_starts_successfully(self) -> None:
        stdout = io.StringIO()

        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("foundry-local-rag", stdout.getvalue())
        self.assertIn("ingest", stdout.getvalue())
        self.assertIn("ask", stdout.getvalue())

    def test_ingest_invokes_application_boundary_and_closes_embedding_adapter(self) -> None:
        stdout = io.StringIO()
        embedding = FakeAdapter()
        config = AppConfig("embedding", "chat", Path("database.sqlite3"))

        with (
            patch("foundry_local_rag.cli.load_config", return_value=config),
            patch("foundry_local_rag.cli.FoundryLocalEmbeddingAdapter", return_value=embedding),
            patch("foundry_local_rag.cli.FoundryLocalChatAdapter") as chat_factory,
            patch("foundry_local_rag.cli.ingest_text_document") as ingest,
            redirect_stdout(stdout),
        ):
            status = main(["ingest", "notes.txt"])

        self.assertEqual(status, 0)
        ingest.assert_called_once_with(Path("notes.txt"), config.database_path, embedding)
        chat_factory.assert_not_called()
        self.assertEqual(embedding.close_count, 1)
        self.assertEqual(stdout.getvalue(), "Ingested: notes.txt\n")

    def test_ingest_application_error_is_presented_and_adapter_is_closed(self) -> None:
        stderr = io.StringIO()
        embedding = FakeAdapter()
        error = DocumentError("document failed")

        with (
            patch("foundry_local_rag.cli.FoundryLocalEmbeddingAdapter", return_value=embedding),
            patch("foundry_local_rag.cli.ingest_text_document", side_effect=error),
            redirect_stderr(stderr),
        ):
            status = main(["ingest", "notes.txt"])

        self.assertEqual(status, 2)
        self.assertEqual(stderr.getvalue(), "error: document failed\n")
        self.assertEqual(embedding.close_count, 1)

    def test_ingest_failure_remains_primary_when_cleanup_fails(self) -> None:
        stderr = io.StringIO()
        embedding = FakeAdapter(ApplicationError("embedding cleanup failed"))

        with (
            patch("foundry_local_rag.cli.FoundryLocalEmbeddingAdapter", return_value=embedding),
            patch(
                "foundry_local_rag.cli.ingest_text_document",
                side_effect=ApplicationError("ingestion failed"),
            ),
            redirect_stderr(stderr),
        ):
            status = main(["ingest", "notes.txt"])

        self.assertEqual(status, 2)
        self.assertEqual(stderr.getvalue(), "error: ingestion failed\n")
        self.assertEqual(embedding.close_count, 1)

    def test_ask_prints_answer_and_sources_in_result_order(self) -> None:
        stdout = io.StringIO()
        embedding = FakeAdapter()
        chat = FakeAdapter()
        result = AnswerResult(
            "grounded answer",
            (
                RetrievedChunk("source-b", 2, "second", 0.8),
                RetrievedChunk("source-a", 0, "first", 1.0),
            ),
        )
        questions: list[str] = []

        def answer(question: str) -> AnswerResult:
            questions.append(question)
            return result

        orchestrator = SimpleNamespace(answer=answer)
        config = AppConfig("embedding", "chat", Path("database.sqlite3"))

        with (
            patch("foundry_local_rag.cli.load_config", return_value=config),
            patch("foundry_local_rag.cli.FoundryLocalEmbeddingAdapter", return_value=embedding),
            patch("foundry_local_rag.cli.FoundryLocalChatAdapter", return_value=chat),
            patch("foundry_local_rag.cli.AnswerOrchestrator", return_value=orchestrator),
            redirect_stdout(stdout),
        ):
            status = main(["ask", "question"])

        self.assertEqual(status, 0)
        self.assertEqual(
            stdout.getvalue(),
            "Answer:\ngrounded answer\n\nSources:\n"
            "- source-b (chunk 2)\n- source-a (chunk 0)\n",
        )
        self.assertEqual(embedding.close_count, 1)
        self.assertEqual(chat.close_count, 1)
        self.assertEqual(questions, ["question"])

    def test_ask_without_sources_prints_none(self) -> None:
        stdout = io.StringIO()
        embedding = FakeAdapter()
        chat = FakeAdapter()
        result = AnswerResult("insufficient context", ())
        orchestrator = SimpleNamespace(answer=lambda question: result)

        with (
            patch("foundry_local_rag.cli.FoundryLocalEmbeddingAdapter", return_value=embedding),
            patch("foundry_local_rag.cli.FoundryLocalChatAdapter", return_value=chat),
            patch("foundry_local_rag.cli.AnswerOrchestrator", return_value=orchestrator),
            redirect_stdout(stdout),
        ):
            status = main(["ask", "question"])

        self.assertEqual(status, 0)
        self.assertEqual(
            stdout.getvalue(),
            "Answer:\ninsufficient context\n\nSources:\n- none\n",
        )
        self.assertEqual(embedding.close_count, 1)
        self.assertEqual(chat.close_count, 1)

    def test_ask_failure_closes_both_adapters(self) -> None:
        embedding = FakeAdapter()
        chat = FakeAdapter()

        with (
            patch("foundry_local_rag.cli.FoundryLocalEmbeddingAdapter", return_value=embedding),
            patch("foundry_local_rag.cli.FoundryLocalChatAdapter", return_value=chat),
            patch(
                "foundry_local_rag.cli.AnswerOrchestrator",
                side_effect=ApplicationError("answer failed"),
            ),
            redirect_stderr(io.StringIO()),
        ):
            status = main(["ask", "question"])

        self.assertEqual(status, 2)
        self.assertEqual(embedding.close_count, 1)
        self.assertEqual(chat.close_count, 1)

    def test_ask_failure_remains_primary_when_cleanup_fails(self) -> None:
        embedding = FakeAdapter(ApplicationError("embedding cleanup failed"))
        chat = FakeAdapter()
        stderr = io.StringIO()

        with (
            patch("foundry_local_rag.cli.FoundryLocalEmbeddingAdapter", return_value=embedding),
            patch("foundry_local_rag.cli.FoundryLocalChatAdapter", return_value=chat),
            patch(
                "foundry_local_rag.cli.AnswerOrchestrator",
                side_effect=ApplicationError("answer failed"),
            ),
            redirect_stderr(stderr),
        ):
            status = main(["ask", "question"])

        self.assertEqual(status, 2)
        self.assertEqual(stderr.getvalue(), "error: answer failed\n")
        self.assertEqual(embedding.close_count, 1)
        self.assertEqual(chat.close_count, 1)

    def test_chat_construction_failure_closes_embedding_only(self) -> None:
        embedding = FakeAdapter()
        stderr = io.StringIO()

        with (
            patch("foundry_local_rag.cli.FoundryLocalEmbeddingAdapter", return_value=embedding),
            patch(
                "foundry_local_rag.cli.FoundryLocalChatAdapter",
                side_effect=ApplicationError("chat initialization failed"),
            ) as chat_factory,
            redirect_stderr(stderr),
        ):
            status = main(["ask", "question"])

        self.assertEqual(status, 2)
        self.assertEqual(stderr.getvalue(), "error: chat initialization failed\n")
        chat_factory.assert_called_once()
        self.assertEqual(embedding.close_count, 1)

    def test_chat_construction_failure_remains_primary_when_cleanup_fails(self) -> None:
        embedding = FakeAdapter(ApplicationError("embedding cleanup failed"))
        stderr = io.StringIO()

        with (
            patch("foundry_local_rag.cli.FoundryLocalEmbeddingAdapter", return_value=embedding),
            patch(
                "foundry_local_rag.cli.FoundryLocalChatAdapter",
                side_effect=ApplicationError("chat initialization failed"),
            ),
            redirect_stderr(stderr),
        ):
            status = main(["ask", "question"])

        self.assertEqual(status, 2)
        self.assertEqual(stderr.getvalue(), "error: chat initialization failed\n")
        self.assertEqual(embedding.close_count, 1)

    def test_successful_command_reports_standalone_cleanup_failure(self) -> None:
        stderr = io.StringIO()
        stdout = io.StringIO()
        embedding = FakeAdapter(ApplicationError("embedding cleanup failed"))

        with (
            patch("foundry_local_rag.cli.FoundryLocalEmbeddingAdapter", return_value=embedding),
            patch("foundry_local_rag.cli.ingest_text_document"),
            redirect_stderr(stderr),
            redirect_stdout(stdout),
        ):
            status = main(["ingest", "notes.txt"])

        self.assertEqual(status, 2)
        self.assertEqual(stderr.getvalue(), "error: embedding cleanup failed\n")
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(embedding.close_count, 1)

    def test_expected_application_error_is_presented_to_user(self) -> None:
        stderr = io.StringIO()

        with patch(
            "foundry_local_rag.cli._run",
            side_effect=ApplicationError("expected failure"),
        ), redirect_stderr(stderr):
            status = main(["ask", "question"])

        self.assertEqual(status, 2)
        self.assertEqual(stderr.getvalue(), "error: expected failure\n")

    def test_unexpected_error_is_not_masked(self) -> None:
        with patch(
            "foundry_local_rag.cli._run",
            side_effect=RuntimeError("unexpected failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected failure"):
                main(["ask", "question"])

    def test_missing_argument_and_unknown_command_are_usage_errors(self) -> None:
        for arguments in (("ingest",), ("ask",), ("unknown",)):
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit) as raised:
                    main(list(arguments))
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
