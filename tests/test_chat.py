import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from foundry_local_rag.chat import FoundryLocalChatAdapter, _create_foundry_runtime
from foundry_local_rag.config import AppConfig, CHAT_MODEL_ID, EMBEDDING_MODEL_ID
from foundry_local_rag.errors import ApplicationError, ChatError
from foundry_local_rag.prompting import PromptMessage


def _config(model_id: str = CHAT_MODEL_ID) -> AppConfig:
    return AppConfig(EMBEDDING_MODEL_ID, model_id, Path("unused.sqlite3"))


def _response(content: object = "answer") -> object:
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


class FakeRuntime:
    def __init__(self) -> None:
        self.response: object = _response()
        self.inputs: list[list[dict[str, str]]] = []
        self.close_count = 0

    def complete_chat(self, messages: list[dict[str, str]]) -> object:
        self.inputs.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def close(self) -> None:
        self.close_count += 1


class AdapterTests(unittest.TestCase):
    def make_adapter(
        self,
        runtime: FakeRuntime | None = None,
        model_cache_dir: Path | None = None,
    ) -> tuple[FoundryLocalChatAdapter, FakeRuntime, Mock]:
        selected_runtime = runtime or FakeRuntime()
        factory = Mock(return_value=selected_runtime)
        adapter = FoundryLocalChatAdapter(
            _config(),
            model_cache_dir=model_cache_dir,
            _runtime_factory=factory,
        )
        return adapter, selected_runtime, factory

    def test_completion_is_lazy_and_preserves_message_order_roles_and_content(self) -> None:
        cache_path = Path.cwd() / "custom-cache"
        adapter, runtime, factory = self.make_adapter(model_cache_dir=cache_path)
        runtime.response = _response("  Merhaba, d\u00fcnya!  ")
        messages = (
            PromptMessage("system", "  system text  "),
            PromptMessage("user", "context"),
            PromptMessage("user", "question"),
        )

        self.assertEqual(factory.call_count, 0)
        answer = adapter.complete(messages)

        self.assertEqual(answer, "  Merhaba, d\u00fcnya!  ")
        self.assertEqual(
            runtime.inputs,
            [[
                {"role": "system", "content": "  system text  "},
                {"role": "user", "content": "context"},
                {"role": "user", "content": "question"},
            ]],
        )
        factory.assert_called_once_with(CHAT_MODEL_ID, cache_path)

    def test_invalid_messages_are_rejected_before_initialization(self) -> None:
        invalid_messages = (
            None,
            "message",
            b"message",
            (),
            ("message",),
            (PromptMessage("assistant", "answer"),),
            (PromptMessage("user", ""),),
            (PromptMessage("user", " \t\n"),),
            (PromptMessage("user", 1),),
        )
        for messages in invalid_messages:
            with self.subTest(messages=messages):
                adapter, _, factory = self.make_adapter()
                with self.assertRaises(ChatError):
                    adapter.complete(messages)  # type: ignore[arg-type]
                factory.assert_not_called()

    def test_non_sequence_messages_are_rejected(self) -> None:
        adapter, _, factory = self.make_adapter()

        with self.assertRaises(ChatError):
            adapter.complete(iter((PromptMessage("user", "question"),)))  # type: ignore[arg-type]

        factory.assert_not_called()

    def test_runtime_is_created_once_and_reused(self) -> None:
        adapter, runtime, factory = self.make_adapter()
        messages = (PromptMessage("user", "question"),)

        adapter.complete(messages)
        adapter.complete(messages)

        factory.assert_called_once()
        self.assertEqual(len(runtime.inputs), 2)

    def test_inference_failure_is_not_reported_as_success(self) -> None:
        adapter, runtime, _ = self.make_adapter()
        runtime.response = RuntimeError("inference failed")

        with self.assertRaises(ChatError) as caught:
            adapter.complete((PromptMessage("user", "question"),))

        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_malformed_responses_are_rejected(self) -> None:
        malformed = (
            object(),
            types.SimpleNamespace(choices="invalid"),
            types.SimpleNamespace(choices=[]),
            types.SimpleNamespace(choices=[object()]),
            types.SimpleNamespace(choices=[types.SimpleNamespace(message=object())]),
        )
        for response in malformed:
            with self.subTest(response=response):
                adapter, runtime, _ = self.make_adapter()
                runtime.response = response
                with self.assertRaisesRegex(ChatError, "malformed"):
                    adapter.complete((PromptMessage("user", "question"),))

    def test_missing_or_empty_answer_is_rejected(self) -> None:
        for content in (None, 1, "", " \t\n"):
            with self.subTest(content=content):
                adapter, runtime, _ = self.make_adapter()
                runtime.response = _response(content)
                with self.assertRaisesRegex(ChatError, "missing or empty"):
                    adapter.complete((PromptMessage("user", "question"),))

    def test_runtime_factory_failure_is_translated(self) -> None:
        factory = Mock(side_effect=RuntimeError("initialization failed"))
        adapter = FoundryLocalChatAdapter(_config(), _runtime_factory=factory)

        with self.assertRaises(ChatError) as caught:
            adapter.complete((PromptMessage("user", "question"),))

        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_unapproved_configured_model_is_rejected(self) -> None:
        factory = Mock()
        adapter = FoundryLocalChatAdapter(
            _config("another-model"),
            _runtime_factory=factory,
        )

        with self.assertRaisesRegex(ChatError, "approved variant"):
            adapter.complete((PromptMessage("user", "question"),))
        factory.assert_not_called()

    def test_close_is_idempotent_and_prevents_reopening(self) -> None:
        adapter, runtime, _ = self.make_adapter()
        adapter.complete((PromptMessage("user", "question"),))

        adapter.close()
        adapter.close()

        self.assertEqual(runtime.close_count, 1)
        with self.assertRaisesRegex(ChatError, "closed"):
            adapter.complete((PromptMessage("user", "question"),))

    def test_close_before_initialization_has_no_runtime_side_effect(self) -> None:
        adapter, _, factory = self.make_adapter()

        adapter.close()
        adapter.close()

        factory.assert_not_called()

    def test_close_failure_is_translated_and_retryable(self) -> None:
        runtime = FakeRuntime()
        runtime.close = Mock(side_effect=(RuntimeError("unload failed"), None))
        adapter, _, _ = self.make_adapter(runtime)
        adapter.complete((PromptMessage("user", "question"),))

        with self.assertRaises(ChatError) as caught:
            adapter.close()
        adapter.close()
        adapter.close()

        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(runtime.close.call_count, 2)

    def test_chat_error_uses_application_error_contract(self) -> None:
        self.assertTrue(issubclass(ChatError, ApplicationError))


class FoundryRuntimeTests(unittest.TestCase):
    def make_sdk(
        self,
        *,
        model: Mock | None,
        existing_manager: object | None = None,
    ) -> tuple[types.ModuleType, type, Mock]:
        catalog = Mock()
        catalog.get_model_variant.return_value = model
        manager = existing_manager or types.SimpleNamespace(catalog=catalog)

        class Configuration:
            def __init__(self, **values: object) -> None:
                self.values = values

        class FoundryLocalManager:
            instance = existing_manager
            initialize = Mock(
                side_effect=lambda _config: setattr(
                    FoundryLocalManager,
                    "instance",
                    manager,
                )
            )

        module = types.ModuleType("foundry_local_sdk")
        module.Configuration = Configuration
        module.FoundryLocalManager = FoundryLocalManager
        return module, FoundryLocalManager, catalog

    def test_exact_variant_is_initialized_loaded_completed_and_unloaded(self) -> None:
        client = Mock()
        client.complete_chat.return_value = _response("answer")
        model = Mock()
        model.id = CHAT_MODEL_ID
        model.is_cached = True
        model.is_loaded = False
        model.get_chat_client.return_value = client
        sdk, manager_type, catalog = self.make_sdk(model=model)
        messages = [{"role": "user", "content": "question"}]

        with patch.dict("sys.modules", {"foundry_local_sdk": sdk}):
            runtime = _create_foundry_runtime(
                CHAT_MODEL_ID,
                Path.cwd() / "model-cache",
            )
            runtime.complete_chat(messages)
            runtime.close()
            runtime.close()

        manager_type.initialize.assert_called_once()
        sdk_config = manager_type.initialize.call_args.args[0]
        self.assertEqual(sdk_config.values["app_name"], "foundry_local_rag")
        self.assertEqual(
            sdk_config.values["model_cache_dir"],
            str(Path.cwd() / "model-cache"),
        )
        catalog.get_model_variant.assert_called_once_with(CHAT_MODEL_ID)
        model.load.assert_called_once_with()
        model.get_chat_client.assert_called_once_with()
        client.complete_chat.assert_called_once_with(messages)
        model.unload.assert_called_once_with()
        client.stream_chat.assert_not_called()

    def test_existing_manager_and_loaded_model_are_not_reinitialized_or_unloaded(self) -> None:
        model = Mock()
        model.id = CHAT_MODEL_ID
        model.is_cached = True
        model.is_loaded = True
        model.get_chat_client.return_value = Mock()
        catalog = Mock()
        catalog.get_model_variant.return_value = model
        existing_manager = types.SimpleNamespace(catalog=catalog)
        sdk, manager_type, _ = self.make_sdk(
            model=model,
            existing_manager=existing_manager,
        )

        with patch.dict("sys.modules", {"foundry_local_sdk": sdk}):
            runtime = _create_foundry_runtime(
                CHAT_MODEL_ID,
                Path.cwd() / "model-cache",
            )
            runtime.close()

        manager_type.initialize.assert_not_called()
        model.load.assert_not_called()
        model.unload.assert_not_called()

    def test_missing_wrong_or_uncached_model_is_rejected_without_download(self) -> None:
        wrong = Mock()
        wrong.id = "wrong-model:1"
        wrong.is_cached = True
        uncached = Mock()
        uncached.id = CHAT_MODEL_ID
        uncached.is_cached = False
        uncached.is_loaded = False

        for model in (None, wrong, uncached):
            with self.subTest(model=model):
                sdk, _, _ = self.make_sdk(model=model)
                with patch.dict("sys.modules", {"foundry_local_sdk": sdk}):
                    with self.assertRaises(ChatError):
                        _create_foundry_runtime(
                            CHAT_MODEL_ID,
                            Path.cwd() / "model-cache",
                        )
                if model is not None:
                    model.download.assert_not_called()
                    model.load.assert_not_called()

    def test_model_load_failure_is_translated(self) -> None:
        model = Mock()
        model.id = CHAT_MODEL_ID
        model.is_cached = True
        model.is_loaded = False
        model.load.side_effect = RuntimeError("load failed")
        sdk, _, _ = self.make_sdk(model=model)

        with patch.dict("sys.modules", {"foundry_local_sdk": sdk}):
            with self.assertRaises(ChatError) as caught:
                _create_foundry_runtime(CHAT_MODEL_ID, Path.cwd() / "model-cache")

        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        model.get_chat_client.assert_not_called()
        model.unload.assert_not_called()

    def test_client_creation_failure_unloads_adapter_owned_model(self) -> None:
        model = Mock()
        model.id = CHAT_MODEL_ID
        model.is_cached = True
        model.is_loaded = False
        model.get_chat_client.side_effect = RuntimeError("client failed")
        sdk, _, _ = self.make_sdk(model=model)

        with patch.dict("sys.modules", {"foundry_local_sdk": sdk}):
            with self.assertRaises(ChatError) as caught:
                _create_foundry_runtime(CHAT_MODEL_ID, Path.cwd() / "model-cache")

        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        model.load.assert_called_once_with()
        model.unload.assert_called_once_with()

    def test_client_creation_and_compensating_unload_failures_are_both_observable(self) -> None:
        model = Mock()
        model.id = CHAT_MODEL_ID
        model.is_cached = True
        model.is_loaded = False
        model.get_chat_client.side_effect = RuntimeError("client failed")
        model.unload.side_effect = RuntimeError("unload failed")
        sdk, _, _ = self.make_sdk(model=model)
        adapter = FoundryLocalChatAdapter(
            _config(),
            _runtime_factory=_create_foundry_runtime,
        )

        with patch.dict("sys.modules", {"foundry_local_sdk": sdk}):
            with self.assertRaises(ChatError) as caught:
                adapter.complete((PromptMessage("user", "question"),))

        self.assertIn("client failed", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(str(caught.exception.__cause__), "unload failed")
        model.load.assert_called_once_with()
        model.get_chat_client.assert_called_once_with()
        model.unload.assert_called_once_with()

    def test_owned_model_unload_failure_can_be_retried(self) -> None:
        model = Mock()
        model.id = CHAT_MODEL_ID
        model.is_cached = True
        model.is_loaded = False
        model.unload.side_effect = (RuntimeError("unload failed"), None)
        model.get_chat_client.return_value = Mock()
        sdk, _, _ = self.make_sdk(model=model)

        with patch.dict("sys.modules", {"foundry_local_sdk": sdk}):
            runtime = _create_foundry_runtime(
                CHAT_MODEL_ID,
                Path.cwd() / "model-cache",
            )
            with self.assertRaises(ChatError):
                runtime.close()
            runtime.close()
            runtime.close()

        self.assertEqual(model.unload.call_count, 2)


if __name__ == "__main__":
    unittest.main()
