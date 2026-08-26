"""Foundry Local embedding integration boundary."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol

from .config import EMBEDDING_MODEL_ID, AppConfig
from .errors import EmbeddingError

_APP_NAME = "foundry_local_rag"


class _EmbeddingRuntime(Protocol):
    def generate_embedding(self, text: str) -> object: ...

    def generate_embeddings(self, texts: list[str]) -> object: ...

    def close(self) -> None: ...


_RuntimeFactory = Callable[[str, Path], _EmbeddingRuntime]


class FoundryLocalEmbeddingAdapter:
    """Generate validated embeddings through one reusable local model client."""

    def __init__(
        self,
        config: AppConfig,
        model_cache_dir: Path | None = None,
        *,
        _runtime_factory: _RuntimeFactory | None = None,
    ) -> None:
        self._model_id = config.embedding_model_id
        cache_path = (
            Path.cwd() / "model-cache"
            if model_cache_dir is None
            else Path(model_cache_dir).expanduser()
        )
        if not cache_path.is_absolute():
            cache_path = Path.cwd() / cache_path
        self._model_cache_dir = cache_path
        self._runtime_factory = _runtime_factory or _create_foundry_runtime
        self._runtime: _EmbeddingRuntime | None = None
        self._closed = False

    def embed_text(self, text: str) -> tuple[float, ...]:
        """Generate one validated embedding for one non-empty text input."""

        _validate_text(text)
        runtime = self._get_runtime()
        try:
            response = runtime.generate_embedding(text)
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError("Unable to generate embedding") from error

        return _extract_embeddings(response, expected_count=1)[0]

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        """Generate one ordered, validated embedding per input in one batch."""

        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise EmbeddingError("Embedding inputs must be a sequence of strings")
        batch = list(texts)
        if not batch:
            raise EmbeddingError("Embedding inputs must not be empty")
        for text in batch:
            _validate_text(text)

        runtime = self._get_runtime()
        try:
            response = runtime.generate_embeddings(batch)
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError("Unable to generate embeddings") from error

        embeddings = _extract_embeddings(response, expected_count=len(batch))
        dimensions = {len(embedding) for embedding in embeddings}
        if len(dimensions) != 1:
            raise EmbeddingError("Batch embeddings must have consistent dimensions")
        return embeddings

    def close(self) -> None:
        """Release a model loaded by this adapter; safe to call repeatedly."""

        if self._closed:
            return
        if self._runtime is None:
            self._closed = True
            return
        try:
            self._runtime.close()
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError("Unable to unload embedding model") from error
        self._runtime = None
        self._closed = True

    def _get_runtime(self) -> _EmbeddingRuntime:
        if self._closed:
            raise EmbeddingError("Embedding adapter is closed")
        if self._model_id != EMBEDDING_MODEL_ID:
            raise EmbeddingError(
                f"Embedding model must be the approved variant: {EMBEDDING_MODEL_ID}"
            )
        if self._runtime is None:
            try:
                self._runtime = self._runtime_factory(
                    self._model_id,
                    self._model_cache_dir,
                )
            except EmbeddingError:
                raise
            except Exception as error:
                raise EmbeddingError("Unable to initialize embedding runtime") from error
        return self._runtime


class _FoundryLocalRuntime:
    def __init__(self, model: object, client: object, owns_model_load: bool) -> None:
        self._model = model
        self._client = client
        self._owns_model_load = owns_model_load
        self._closed = False

    def generate_embedding(self, text: str) -> object:
        return self._client.generate_embedding(text)  # type: ignore[attr-defined]

    def generate_embeddings(self, texts: list[str]) -> object:
        return self._client.generate_embeddings(texts)  # type: ignore[attr-defined]

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_model_load:
            try:
                self._model.unload()  # type: ignore[attr-defined]
            except Exception as error:
                raise EmbeddingError("Unable to unload embedding model") from error
        self._closed = True


def _create_foundry_runtime(model_id: str, model_cache_dir: Path) -> _EmbeddingRuntime:
    try:
        from foundry_local_sdk import Configuration, FoundryLocalManager
    except Exception as error:
        raise EmbeddingError("Foundry Local SDK is unavailable") from error

    model = None
    loaded_by_adapter = False
    try:
        manager = FoundryLocalManager.instance
        if manager is None:
            sdk_config = Configuration(
                app_name=_APP_NAME,
                model_cache_dir=str(model_cache_dir),
            )
            FoundryLocalManager.initialize(sdk_config)
            manager = FoundryLocalManager.instance
        if manager is None:
            raise EmbeddingError("Foundry Local SDK did not initialize")

        model = manager.catalog.get_model_variant(model_id)
        if model is None:
            raise EmbeddingError(f"Embedding model variant is unavailable: {model_id}")
        if model.id != EMBEDDING_MODEL_ID:
            raise EmbeddingError(
                f"Foundry Local resolved an unexpected embedding model: {model.id}"
            )
        if not model.is_cached:
            raise EmbeddingError(f"Embedding model is not cached locally: {model_id}")

        if not model.is_loaded:
            model.load()
            loaded_by_adapter = True
        client = model.get_embedding_client()
        return _FoundryLocalRuntime(model, client, loaded_by_adapter)
    except EmbeddingError as error:
        if loaded_by_adapter and model is not None:
            _unload_after_initialization_failure(model, error)
        raise
    except Exception as error:
        if loaded_by_adapter and model is not None:
            _unload_after_initialization_failure(model, error)
        raise EmbeddingError("Unable to initialize Foundry Local embedding model") from error


def _unload_after_initialization_failure(
    model: object,
    initialization_error: Exception,
) -> None:
    try:
        model.unload()  # type: ignore[attr-defined]
    except Exception as cleanup_error:
        raise EmbeddingError(
            "Unable to unload embedding model after initialization failure: "
            f"{initialization_error}"
        ) from cleanup_error


def _validate_text(text: object) -> None:
    if not isinstance(text, str) or not text.strip():
        raise EmbeddingError("Embedding text must be a non-empty string")


def _extract_embeddings(
    response: object,
    expected_count: int,
) -> tuple[tuple[float, ...], ...]:
    try:
        data = response.data  # type: ignore[attr-defined]
        if isinstance(data, (str, bytes)):
            raise TypeError
        items = tuple(data)
    except (AttributeError, TypeError) as error:
        raise EmbeddingError("Embedding response data is malformed") from error
    if len(items) != expected_count:
        raise EmbeddingError("Embedding response count does not match input count")

    embeddings: list[tuple[float, ...]] = []
    for item in items:
        try:
            values = item.embedding  # type: ignore[attr-defined]
        except AttributeError as error:
            raise EmbeddingError("Embedding response item is malformed") from error
        embeddings.append(_validated_embedding(values))
    return tuple(embeddings)


def _validated_embedding(values: Iterable[object]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise EmbeddingError("Embedding vector must contain numeric values")
    try:
        embedding = tuple(values)
    except TypeError as error:
        raise EmbeddingError("Embedding vector must be an iterable") from error
    if not embedding:
        raise EmbeddingError("Embedding vector must not be empty")

    normalized: list[float] = []
    for value in embedding:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingError("Embedding vector values must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingError("Embedding vector values must be finite")
        normalized.append(number)
    return tuple(normalized)
