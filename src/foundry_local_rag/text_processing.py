"""UTF-8 text-file reading and deterministic paragraph chunking."""

from __future__ import annotations

import re
from pathlib import Path

from .errors import DocumentError

_PARAGRAPH_SEPARATOR = re.compile(r"\n[^\S\r\n]*\n+")


def read_text_file(path: Path) -> str:
    """Read a non-empty UTF-8 text file without modifying its contents."""

    source_path = Path(path)
    if source_path.suffix.lower() != ".txt":
        raise DocumentError(f"Unsupported document type: {source_path}")
    if not source_path.exists():
        raise DocumentError(f"Text file does not exist: {source_path}")
    if not source_path.is_file():
        raise DocumentError(f"Text path is not a file: {source_path}")

    try:
        text = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise DocumentError(f"Text file is not valid UTF-8: {source_path}") from error
    except OSError as error:
        raise DocumentError(f"Unable to read text file: {source_path}") from error

    if not text.strip():
        raise DocumentError(f"Text file is empty: {source_path}")
    return text


def chunk_paragraphs(text: str) -> tuple[str, ...]:
    """Return trimmed paragraphs in source order using normalized line endings."""

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized_text.strip():
        raise DocumentError("Text content is empty")

    return tuple(
        paragraph.strip()
        for paragraph in _PARAGRAPH_SEPARATOR.split(normalized_text)
        if paragraph.strip()
    )
