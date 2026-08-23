import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from foundry_local_rag.errors import DocumentError
from foundry_local_rag.text_processing import chunk_paragraphs, read_text_file


class ReadTextFileTests(unittest.TestCase):
    def test_reads_utf8_text_without_changing_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notes.txt"
            expected_text = "Merhaba, d\u00fcnya.\n"
            path.write_text(expected_text, encoding="utf-8")

            self.assertEqual(read_text_file(path), expected_text)

    def test_accepts_case_insensitive_txt_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notes.TXT"
            path.write_text("content", encoding="utf-8")

            self.assertEqual(read_text_file(path), "content")

    def test_rejects_unsupported_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notes.md"
            path.write_text("content", encoding="utf-8")

            with self.assertRaisesRegex(DocumentError, "Unsupported document type"):
                read_text_file(path)

    def test_rejects_missing_file_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            text_directory = directory / "folder.txt"
            text_directory.mkdir()

            with self.assertRaisesRegex(DocumentError, "does not exist"):
                read_text_file(directory / "missing.txt")
            with self.assertRaisesRegex(DocumentError, "not a file"):
                read_text_file(text_directory)

    def test_wraps_unreadable_file_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notes.txt"
            path.write_text("content", encoding="utf-8")

            with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
                with self.assertRaisesRegex(DocumentError, "Unable to read"):
                    read_text_file(path)

    def test_rejects_invalid_utf8_and_empty_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            invalid_path = directory / "invalid.txt"
            invalid_path.write_bytes(b"\xff")

            with self.assertRaisesRegex(DocumentError, "not valid UTF-8"):
                read_text_file(invalid_path)

            for name, contents in (("empty.txt", ""), ("blank.txt", " \n\t ")):
                with self.subTest(name=name):
                    path = directory / name
                    path.write_text(contents, encoding="utf-8")

                    with self.assertRaisesRegex(DocumentError, "is empty"):
                        read_text_file(path)


class ChunkParagraphsTests(unittest.TestCase):
    def test_returns_single_paragraph_as_an_immutable_tuple(self) -> None:
        chunks = chunk_paragraphs("  One paragraph\nwith a line break.  ")

        self.assertEqual(chunks, ("One paragraph\nwith a line break.",))
        self.assertIsInstance(chunks, tuple)

    def test_normalizes_newlines_and_preserves_paragraph_order(self) -> None:
        text = "  First\r\nline  \r\n \r\n\r\nSecond\rthird  "

        self.assertEqual(
            chunk_paragraphs(text),
            ("First\nline", "Second\nthird"),
        )

    def test_is_deterministic_for_multiple_paragraphs(self) -> None:
        text = "First\n\n\nSecond\n \nThird"

        self.assertEqual(
            chunk_paragraphs(text),
            ("First", "Second", "Third"),
        )
        self.assertEqual(chunk_paragraphs(text), chunk_paragraphs(text))

    def test_rejects_empty_or_whitespace_only_text(self) -> None:
        for text in ("", " \r\n\t "):
            with self.subTest(text=text):
                with self.assertRaisesRegex(DocumentError, "Text content is empty"):
                    chunk_paragraphs(text)


if __name__ == "__main__":
    unittest.main()
