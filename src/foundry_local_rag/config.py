"""Application configuration and local path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigurationError

EMBEDDING_MODEL_ID = "qwen3-embedding-0.6b-generic-cpu:1"
CHAT_MODEL_ID = "qwen2.5-0.5b-instruct-generic-cpu:4"

_EMBEDDING_MODEL_ENV = "FOUNDRY_LOCAL_RAG_EMBEDDING_MODEL"
_CHAT_MODEL_ENV = "FOUNDRY_LOCAL_RAG_CHAT_MODEL"
_DATABASE_PATH_ENV = "FOUNDRY_LOCAL_RAG_DATABASE_PATH"


@dataclass(frozen=True)
class AppConfig:
    """Validated settings shared by the application."""

    embedding_model_id: str
    chat_model_id: str
    database_path: Path


def _required_text(value: str, setting_name: str) -> str:
    value = value.strip()
    if not value:
        raise ConfigurationError(f"{setting_name} must not be empty")
    return value


def load_config(environ: dict[str, str] | None = None) -> AppConfig:
    """Load validated settings from defaults and optional environment values."""

    values = os.environ if environ is None else environ
    embedding_model_id = _required_text(
        values.get(_EMBEDDING_MODEL_ENV, EMBEDDING_MODEL_ID),
        _EMBEDDING_MODEL_ENV,
    )
    chat_model_id = _required_text(
        values.get(_CHAT_MODEL_ENV, CHAT_MODEL_ID),
        _CHAT_MODEL_ENV,
    )

    database_value = values.get(
        _DATABASE_PATH_ENV,
        str(Path.cwd() / "data" / "rag.sqlite3"),
    ).strip()
    if not database_value:
        raise ConfigurationError(f"{_DATABASE_PATH_ENV} must not be empty")

    database_path = Path(database_value).expanduser()
    if not database_path.is_absolute():
        database_path = Path.cwd() / database_path

    return AppConfig(
        embedding_model_id=embedding_model_id,
        chat_model_id=chat_model_id,
        database_path=database_path,
    )
