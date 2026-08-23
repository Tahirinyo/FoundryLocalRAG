"""Command-line entry point for the application foundation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .config import load_config
from .errors import ApplicationError


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the currently supported foundation commands."""

    return argparse.ArgumentParser(
        prog="foundry-local-rag",
        description="Local Foundry RAG application foundation (no RAG operations yet).",
    )


def _run() -> int:
    config = load_config()
    print("Foundry Local RAG application foundation")
    print(f"Embedding model: {config.embedding_model_id}")
    print(f"Chat model: {config.chat_model_id}")
    print(f"Database path: {config.database_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process status code."""

    parser = build_parser()
    parser.parse_args(argv)
    try:
        return _run()
    except ApplicationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
