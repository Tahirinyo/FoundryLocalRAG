"""Minimal command-line interface for the local RAG application."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .answering import AnswerOrchestrator
from .chat import FoundryLocalChatAdapter
from .config import AppConfig, load_config
from .embeddings import FoundryLocalEmbeddingAdapter
from .errors import ApplicationError
from .ingestion import ingest_text_document


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the supported application commands."""

    parser = argparse.ArgumentParser(
        prog="foundry-local-rag",
        description="A local document question-and-answer assistant.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="ingest a .txt document")
    ingest_parser.add_argument("path", help="path to the .txt document")

    ask_parser = commands.add_parser("ask", help="ask a question about ingested documents")
    ask_parser.add_argument("question", help="question to answer")
    return parser


def _run(arguments: argparse.Namespace) -> int:
    """Run one parsed command and return its process status code."""

    config = load_config()
    if arguments.command == "ingest":
        return _run_ingest(arguments.path, config)
    if arguments.command == "ask":
        return _run_ask(arguments.question, config)
    raise ApplicationError(f"Unknown command: {arguments.command}")


def _run_ingest(path: str, config: AppConfig) -> int:
    """Run ingestion with an embedding adapter owned by this command."""

    embedding_adapter = FoundryLocalEmbeddingAdapter(config)
    try:
        ingest_text_document(
            Path(path),
            config.database_path,
            embedding_adapter,
        )
    except BaseException as error:
        _close_adapters(embedding_adapter, primary_error=error)
        raise
    else:
        _close_adapters(embedding_adapter)

    print(f"Ingested: {path}")
    return 0


def _run_ask(question: str, config: AppConfig) -> int:
    """Run answer orchestration with command-owned local adapters."""

    embedding_adapter = FoundryLocalEmbeddingAdapter(config)
    chat_adapter: FoundryLocalChatAdapter | None = None
    try:
        chat_adapter = FoundryLocalChatAdapter(config)
        orchestrator = AnswerOrchestrator(
            config.database_path,
            embedding_adapter,
            chat_adapter,
        )
        result = orchestrator.answer(question)
    except BaseException as error:
        _close_adapters(embedding_adapter, chat_adapter, primary_error=error)
        raise
    else:
        _close_adapters(embedding_adapter, chat_adapter)

    print("Answer:")
    print(result.answer)
    print()
    print("Sources:")
    if result.sources:
        for source in result.sources:
            print(f"- {source.source_id} (chunk {source.chunk_index})")
    else:
        print("- none")
    return 0


def _close_adapters(
    *adapters: object | None,
    primary_error: BaseException | None = None,
) -> None:
    """Close adapters without replacing an already-active command failure."""

    first_error: Exception | None = None
    for adapter in adapters:
        if adapter is None:
            continue
        try:
            adapter.close()  # type: ignore[attr-defined]
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None and primary_error is None:
        raise first_error


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process status code."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return _run(arguments)
    except ApplicationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
