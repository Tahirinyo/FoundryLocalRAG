import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from foundry_local_rag.cli import main
from foundry_local_rag.errors import ApplicationError


class CliTests(unittest.TestCase):
    def test_help_starts_successfully(self) -> None:
        stdout = io.StringIO()

        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("foundry-local-rag", stdout.getvalue())

    def test_startup_prints_configuration_without_loading_models(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            status = main([])

        self.assertEqual(status, 0)
        self.assertIn("application foundation", stdout.getvalue())

    def test_expected_application_error_is_presented_to_user(self) -> None:
        stderr = io.StringIO()

        with patch(
            "foundry_local_rag.cli._run",
            side_effect=ApplicationError("expected failure"),
        ), redirect_stderr(stderr):
            status = main([])

        self.assertEqual(status, 2)
        self.assertEqual(stderr.getvalue(), "error: expected failure\n")

    def test_unexpected_error_is_not_masked(self) -> None:
        with patch(
            "foundry_local_rag.cli._run",
            side_effect=RuntimeError("unexpected failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected failure"):
                main([])


if __name__ == "__main__":
    unittest.main()
