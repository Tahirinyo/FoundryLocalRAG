import math
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from foundry_local_rag.config import AppConfig, CHAT_MODEL_ID, EMBEDDING_MODEL_ID
from foundry_local_rag.embeddings import (
    FoundryLocalEmbeddingAdapter,
    _create_foundry_runtime,
)
from foundry_local_rag.errors import ApplicationError, EmbeddingError


def _config(model_id: str = EMBEDDING_MODEL_ID) -> AppConfig:
    return AppConfig(model_id, CHAT_MODEL_ID, Path("unused.sqlite3"))


def _response(*embeddings: object) -> object:
    return types.SimpleNamespace(
        data=[types.SimpleNamespace(embedding=embedding) for embedding in embeddings]
    )


class FakeRuntime:
    def __init__(self) -> None:
        self.single_response = _response((1.0, 2.0))
        self.batch_response = _response((1.0, 2.0), (3.0, 4.0))
        self.single_inputs: list[str] = []
        self.batch_inputs: list[list[str]] = []
        self.close_count = 0

    def generate_embedding(self, text: str) -> object:
        self.single_inputs.append(text)
        if isinstance(self.single_response, Exception):
            raise self.single_response
        return self.single_response

    def generate_embeddings(self, texts: list[str]) -> object:
        self.batch_inputs.append(texts)
        if isinstance(self.batch_response, Exception):
            raise self.batch_response
        return self.batch_response

    def close(self) -> None:
        self.close_count += 1


class AdapterTests(unittest.TestCase):
    def make_adapter(
        self,
        runtime: FakeRuntime | None = None,
        model_cache_dir: Path | None = None,
    ) -> tuple[FoundryLocalEmbeddingAdapter, FakeRuntime, Mock]:
        selected_runtime = runtime or FakeRuntime()
        factory = Mock(return_value=selected_runtime)
        adapter = FoundryLocalEmbeddingAdapter(
            _config(),
            model_cache_dir=model_cache_dir,
            _runtime_factory=factory,
        )
        return adapter, selected_runtime, factory

    def test_single_embedding_is_lazy_preserves_unicode_and_normalizes_floats(self) -> None:
        cache_path = Path.cwd() / "custom-cache"
        adapter, runtime, factory = self.make_adapter(model_cache_dir=cache_path)
        runtime.single_response = _response((1, -2.5, 3.0))

        self.assertEqual(factory.call_count, 0)
        result = adapter.embed_text("  Merhaba, dünya!  ")

        self.assertEqual(result, (1.0, -2.5, 3.0))
        self.assertIsInstance(result, tuple)
        self.assertEqual(runtime.single_inputs, ["  Merhaba, dünya!  "])
        factory.assert_called_once_with(EMBEDDING_MODEL_ID, cache_path)

    def test_batch_uses_one_native_call_and_preserves_order(self) -> None:
        adapter, runtime, _ = self.make_adapter()
        runtime.batch_response = _response((1, 2), (3, 4))

        result = adapter.embed_texts(("first", "second"))

        self.assertEqual(result, ((1.0, 2.0), (3.0, 4.0)))
        self.assertEqual(runtime.batch_inputs, [["first", "second"]])
        self.assertEqual(runtime.single_inputs, [])

    def test_one_item_batch_is_supported(self) -> None:
        adapter, runtime, _ = self.make_adapter()
        runtime.batch_response = _response((1.0,))

        self.assertEqual(adapter.embed_texts(["one"]), ((1.0,),))

    def test_invalid_single_inputs_are_rejected_before_initialization(self) -> None:
        for value in (None, 1, "", " \t\r\n"):
            with self.subTest(value=value):
                adapter, _, factory = self.make_adapter()
                with self.assertRaises(EmbeddingError):
                    adapter.embed_text(value)  # type: ignore[arg-type]
                factory.assert_not_called()

    def test_invalid_batch_inputs_are_rejected_before_initialization(self) -> None:
        invalid_inputs = ("text", b"text", (), ["valid", "  "], ["valid", 1])
        for value in invalid_inputs:
            with self.subTest(value=value):
                adapter, _, factory = self.make_adapter()
                with self.assertRaises(EmbeddingError):
                    adapter.embed_texts(value)  # type: ignore[arg-type]
                factory.assert_not_called()

    def test_non_sequence_batch_is_rejected(self) -> None:
        adapter, _, factory = self.make_adapter()

        with self.assertRaises(EmbeddingError):
            adapter.embed_texts(iter(("one", "two")))  # type: ignore[arg-type]

        factory.assert_not_called()

    def test_response_count_must_match_input_count(self) -> None:
        adapter, runtime, _ = self.make_adapter()
        runtime.batch_response = _response((1.0,))

        with self.assertRaisesRegex(EmbeddingError, "count"):
            adapter.embed_texts(["one", "two"])

    def test_malformed_response_structures_are_rejected(self) -> None:
        malformed = (
            object(),
            types.SimpleNamespace(data="invalid"),
            types.SimpleNamespace(data=[object()]),
        )
        for response in malformed:
            with self.subTest(response=response):
                adapter, runtime, _ = self.make_adapter()
                runtime.single_response = response
                with self.assertRaises(EmbeddingError):
                    adapter.embed_text("text")

    def test_invalid_vectors_are_rejected(self) -> None:
        invalid_vectors = (
            (),
            "123",
            object(),
            (True,),
            ("1",),
            (math.nan,),
            (math.inf,),
            (-math.inf,),
        )
        for vector in invalid_vectors:
            with self.subTest(vector=vector):
                adapter, runtime, _ = self.make_adapter()
                runtime.single_response = _response(vector)
                with self.assertRaises(EmbeddingError):
                    adapter.embed_text("text")

    def test_batch_dimensions_must_be_consistent(self) -> None:
        adapter, runtime, _ = self.make_adapter()
        runtime.batch_response = _response((1.0,), (2.0, 3.0))

        with self.assertRaisesRegex(EmbeddingError, "consistent dimensions"):
            adapter.embed_texts(["one", "two"])

    def test_inference_failures_are_not_reported_as_success(self) -> None:
        adapter, runtime, _ = self.make_adapter()
        runtime.single_response = RuntimeError("single failure")

        with self.assertRaises(EmbeddingError) as caught:
            adapter.embed_text("text")
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

        adapter, runtime, _ = self.make_adapter()
        runtime.batch_response = RuntimeError("batch failure")
        with self.assertRaises(EmbeddingError) as caught:
            adapter.embed_texts(["one", "two"])
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_runtime_is_created_once_and_reused(self) -> None:
        adapter, runtime, factory = self.make_adapter()
        runtime.batch_response = _response((3.0, 4.0))

        adapter.embed_text("one")
        adapter.embed_text("two")
        adapter.embed_texts(["three"])

        factory.assert_called_once()

    def test_runtime_factory_failure_is_translated(self) -> None:
        factory = Mock(side_effect=RuntimeError("initialization failed"))
        adapter = FoundryLocalEmbeddingAdapter(_config(), _runtime_factory=factory)

        with self.assertRaises(EmbeddingError) as caught:
            adapter.embed_text("text")

        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_unapproved_configured_model_is_rejected(self) -> None:
        factory = Mock()
        adapter = FoundryLocalEmbeddingAdapter(
            _config("another-model"),
            _runtime_factory=factory,
        )

        with self.assertRaisesRegex(EmbeddingError, "approved variant"):
            adapter.embed_text("text")
        factory.assert_not_called()

    def test_close_is_idempotent_and_prevents_reopening(self) -> None:
        adapter, runtime, _ = self.make_adapter()
        adapter.embed_text("text")

        adapter.close()
        adapter.close()

        self.assertEqual(runtime.close_count, 1)
        with self.assertRaisesRegex(EmbeddingError, "closed"):
            adapter.embed_text("text")

    def test_close_before_initialization_has_no_runtime_side_effect(self) -> None:
        adapter, _, factory = self.make_adapter()

        adapter.close()
        adapter.close()

        factory.assert_not_called()

    def test_close_failure_is_translated(self) -> None:
        runtime = FakeRuntime()
        runtime.close = Mock(
            side_effect=(RuntimeError("unload failed"), None),
        )
        adapter, _, _ = self.make_adapter(runtime)
        adapter.embed_text("text")

        with self.assertRaises(EmbeddingError) as caught:
            adapter.close()
        adapter.close()
        adapter.close()

        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(runtime.close.call_count, 2)

    def test_embedding_error_uses_application_error_contract(self) -> None:
        self.assertTrue(issubclass(EmbeddingError, ApplicationError))


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
            initialize = Mock(side_effect=lambda _config: setattr(
                FoundryLocalManager,
                "instance",
                manager,
            ))

        module = types.ModuleType("foundry_local_sdk")
        module.Configuration = Configuration
        module.FoundryLocalManager = FoundryLocalManager
        return module, FoundryLocalManager, catalog

    def test_exact_variant_is_initialized_loaded_reused_and_unloaded(self) -> None:
        client = Mock()
        client.generate_embedding.return_value = _response((1.0, 2.0))
        model = Mock()
        model.id = EMBEDDING_MODEL_ID
        model.is_cached = True
        model.is_loaded = False
        model.get_embedding_client.return_value = client
        sdk, manager_type, catalog = self.make_sdk(model=model)

        with patch.dict("sys.modules", {"foundry_local_sdk": sdk}):
            runtime = _create_foundry_runtime(
                EMBEDDING_MODEL_ID,
                Path.cwd() / "model-cache",
            )
            runtime.generate_embedding("text")
            runtime.close()
            runtime.close()

        manager_type.initialize.assert_called_once()
        sdk_config = manager_type.initialize.call_args.args[0]
        self.assertEqual(sdk_config.values["app_name"], "foundry_local_rag")
        self.assertEqual(
            sdk_config.values["model_cache_dir"],
            str(Path.cwd() / "model-cache"),
        )
        catalog.get_model_variant.assert_called_once_with(EMBEDDING_MODEL_ID)
        model.load.assert_called_once_with()
        model.get_embedding_client.assert_called_once_with()
        client.generate_embedding.assert_called_once_with("text")
        model.unload.assert_called_once_with()

    def test_existing_manager_and_loaded_model_are_not_reinitialized_or_unloaded(self) -> None:
        client = Mock()
        model = Mock()
        model.id = EMBEDDING_MODEL_ID
        model.is_cached = True
        model.is_loaded = True
        model.get_embedding_client.return_value = client
        catalog = Mock()
        catalog.get_model_variant.return_value = model
        existing_manager = types.SimpleNamespace(catalog=catalog)
        sdk, manager_type, _ = self.make_sdk(
            model=model,
            existing_manager=existing_manager,
        )

        with patch.dict("sys.modules", {"foundry_local_sdk": sdk}):
            runtime = _create_foundry_runtime(
                EMBEDDING_MODEL_ID,
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
        uncached.id = EMBEDDING_MODEL_ID
        uncached.is_cached = False
        uncached.is_loaded = False

        for model in (None, wrong, uncached):
            with self.subTest(model=model):
                sdk, _, _ = self.make_sdk(model=model)
                with patch.dict("sys.modules", {"foundry_local_sdk": sdk}):
                    with self.assertRaises(EmbeddingError):
                        _create_foundry_runtime(
                            EMBEDDING_MODEL_ID,
                            Path.cwd() / "model-cache",
                        )
                if model is not None:
                    model.download.assert_not_called()
                    model.load.assert_not_called()

    def test_client_creation_failure_unloads_adapter_owned_model(self) -> None:
        model = Mock()
        model.id = EMBEDDING_MODEL_ID
        model.is_cached = True
        model.is_loaded = False
        model.get_embedding_client.side_effect = RuntimeError("client failed")
        sdk, _, _ = self.make_sdk(model=model)

        with patch.dict("sys.modules", {"foundry_local_sdk": sdk}):
            with self.assertRaises(EmbeddingError) as caught:
                _create_foundry_runtime(
                    EMBEDDING_MODEL_ID,
                    Path.cwd() / "model-cache",
                )

        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        model.load.assert_called_once_with()
        model.unload.assert_called_once_with()

    def test_client_creation_and_compensating_unload_failures_are_both_observable(self) -> None:
        model = Mock()
        model.id = EMBEDDING_MODEL_ID
        model.is_cached = True
        model.is_loaded = False
        model.get_embedding_client.side_effect = RuntimeError("client failed")
        model.unload.side_effect = RuntimeError("unload failed")
        sdk, _, _ = self.make_sdk(model=model)
        adapter = FoundryLocalEmbeddingAdapter(
            _config(),
            _runtime_factory=_create_foundry_runtime,
        )

        with patch.dict("sys.modules", {"foundry_local_sdk": sdk}):
            with self.assertRaises(EmbeddingError) as caught:
                adapter.embed_text("text")

        self.assertIn("client failed", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(str(caught.exception.__cause__), "unload failed")
        model.load.assert_called_once_with()
        model.get_embedding_client.assert_called_once_with()
        model.unload.assert_called_once_with()

    def test_owned_model_unload_failure_can_be_retried(self) -> None:
        model = Mock()
        model.id = EMBEDDING_MODEL_ID
        model.is_cached = True
        model.is_loaded = False
        model.unload.side_effect = (RuntimeError("unload failed"), None)
        model.get_embedding_client.return_value = Mock()
        sdk, _, _ = self.make_sdk(model=model)

        with patch.dict("sys.modules", {"foundry_local_sdk": sdk}):
            runtime = _create_foundry_runtime(
                EMBEDDING_MODEL_ID,
                Path.cwd() / "model-cache",
            )
            with self.assertRaises(EmbeddingError):
                runtime.close()
            runtime.close()
            runtime.close()

        self.assertEqual(model.unload.call_count, 2)


if __name__ == "__main__":
    unittest.main()
