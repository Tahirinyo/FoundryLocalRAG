"""Foundry Local chat-model integration boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from .config import CHAT_MODEL_ID, AppConfig
from .errors import ChatError
from .prompting import PromptMessage

_APP_NAME = "foundry_local_rag"


class _ChatRuntime(Protocol):
    def complete_chat(self, messages: list[dict[str, str]]) -> object: ...

    def close(self) -> None: ...


_RuntimeFactory = Callable[[str, Path], _ChatRuntime]


class FoundryLocalChatAdapter:
    """Generate validated chat answers through one reusable local model client."""

    def __init__(
        self,
        config: AppConfig,
        model_cache_dir: Path | None = None,
        *,
        _runtime_factory: _RuntimeFactory | None = None,
    ) -> None:
        self._model_id = config.chat_model_id
        cache_path = (
            Path.cwd() / "model-cache"
            if model_cache_dir is None
            else Path(model_cache_dir).expanduser()
        )
        if not cache_path.is_absolute():
            cache_path = Path.cwd() / cache_path
        self._model_cache_dir = cache_path
        self._runtime_factory = _runtime_factory or _create_foundry_runtime
        self._runtime: _ChatRuntime | None = None
        self._closed = False

    def complete(self, messages: Sequence[PromptMessage]) -> str:
        """Return one validated answer for ordered, structured prompt messages."""

        native_messages = _validated_messages(messages)
        runtime = self._get_runtime()
        try:
            response = runtime.complete_chat(native_messages)
        except ChatError:
            raise
        except Exception as error:
            raise ChatError("Unable to generate chat completion") from error
        return _extract_answer(response)

    def close(self) -> None:
        """Release a model loaded by this adapter; safe to call repeatedly."""

        if self._closed:
            return
        if self._runtime is None:
            self._closed = True
            return
        try:
            self._runtime.close()
        except ChatError:
            raise
        except Exception as error:
            raise ChatError("Unable to unload chat model") from error
        self._runtime = None
        self._closed = True

    def _get_runtime(self) -> _ChatRuntime:
        if self._closed:
            raise ChatError("Chat adapter is closed")
        if self._model_id != CHAT_MODEL_ID:
            raise ChatError(
                f"Chat model must be the approved variant: {CHAT_MODEL_ID}"
            )
        if self._runtime is None:
            try:
                self._runtime = self._runtime_factory(
                    self._model_id,
                    self._model_cache_dir,
                )
            except ChatError:
                raise
            except Exception as error:
                raise ChatError("Unable to initialize chat runtime") from error
        return self._runtime


class _FoundryLocalRuntime:
    def __init__(self, model: object, client: object, owns_model_load: bool) -> None:
        self._model = model
        self._client = client
        self._owns_model_load = owns_model_load
        self._closed = False

    def complete_chat(self, messages: list[dict[str, str]]) -> object:
        return self._client.complete_chat(messages)  # type: ignore[attr-defined]

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_model_load:
            try:
                self._model.unload()  # type: ignore[attr-defined]
            except Exception as error:
                raise ChatError("Unable to unload chat model") from error
        self._closed = True


def _create_foundry_runtime(model_id: str, model_cache_dir: Path) -> _ChatRuntime:
    try:
        from foundry_local_sdk import Configuration, FoundryLocalManager
    except Exception as error:
        raise ChatError("Foundry Local SDK is unavailable") from error

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
            raise ChatError("Foundry Local SDK did not initialize")

        model = manager.catalog.get_model_variant(model_id)
        if model is None:
            raise ChatError(f"Chat model variant is unavailable: {model_id}")
        if model.id != CHAT_MODEL_ID:
            raise ChatError(
                f"Foundry Local resolved an unexpected chat model: {model.id}"
            )
        if not model.is_cached:
            raise ChatError(f"Chat model is not cached locally: {model_id}")

        if not model.is_loaded:
            model.load()
            loaded_by_adapter = True
        client = model.get_chat_client()
        return _FoundryLocalRuntime(model, client, loaded_by_adapter)
    except ChatError as error:
        if loaded_by_adapter and model is not None:
            _unload_after_initialization_failure(model, error)
        raise
    except Exception as error:
        if loaded_by_adapter and model is not None:
            _unload_after_initialization_failure(model, error)
        raise ChatError("Unable to initialize Foundry Local chat model") from error


def _unload_after_initialization_failure(
    model: object,
    initialization_error: Exception,
) -> None:
    try:
        model.unload()  # type: ignore[attr-defined]
    except Exception as cleanup_error:
        raise ChatError(
            "Unable to unload chat model after initialization failure: "
            f"{initialization_error}"
        ) from cleanup_error


def _validated_messages(messages: object) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ChatError("Chat messages must be a sequence")
    if not messages:
        raise ChatError("Chat messages must not be empty")

    native_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, PromptMessage):
            raise ChatError("Chat messages must contain PromptMessage values")
        if message.role not in ("system", "user"):
            raise ChatError("Chat message role must be system or user")
        if not isinstance(message.content, str) or not message.content.strip():
            raise ChatError("Chat message content must be a non-empty string")
        native_messages.append({"role": message.role, "content": message.content})
    return native_messages


def _extract_answer(response: object) -> str:
    try:
        choices = response.choices  # type: ignore[attr-defined]
        if isinstance(choices, (str, bytes)):
            raise TypeError
        items = tuple(choices)
        if not items:
            raise ValueError
        content = items[0].message.content
    except (AttributeError, TypeError, ValueError) as error:
        raise ChatError("Chat completion response is malformed") from error
    if not isinstance(content, str) or not content.strip():
        raise ChatError("Chat completion answer is missing or empty")
    return content
