import os
import unittest
from pathlib import Path
from unittest.mock import patch

import foundry_local_rag.config as config_module
from foundry_local_rag.config import (
    CHAT_MODEL_ID,
    EMBEDDING_MODEL_ID,
    load_config,
)
from foundry_local_rag.errors import ConfigurationError


class LoadConfigTests(unittest.TestCase):
    def test_defaults_use_t01_models_and_working_directory_database_path(self) -> None:
        config = load_config({})

        self.assertEqual(config.embedding_model_id, EMBEDDING_MODEL_ID)
        self.assertEqual(config.chat_model_id, CHAT_MODEL_ID)
        self.assertIsInstance(config.database_path, Path)
        self.assertEqual(
            config.database_path,
            Path.cwd() / "data" / "rag.sqlite3",
        )

    def test_default_path_does_not_depend_on_package_file_location(self) -> None:
        expected_path = Path.cwd() / "data" / "rag.sqlite3"

        with patch.object(
            config_module,
            "__file__",
            "C:/isolated/site-packages/foundry_local_rag/config.py",
        ):
            config = load_config({})

        self.assertEqual(config.database_path, expected_path)

    def test_environment_overrides_are_applied(self) -> None:
        config = load_config(
            {
                "FOUNDRY_LOCAL_RAG_EMBEDDING_MODEL": "embedding-test",
                "FOUNDRY_LOCAL_RAG_CHAT_MODEL": "chat-test",
                "FOUNDRY_LOCAL_RAG_DATABASE_PATH": "local/test.sqlite3",
            }
        )

        self.assertEqual(config.embedding_model_id, "embedding-test")
        self.assertEqual(config.chat_model_id, "chat-test")
        self.assertEqual(config.database_path, Path.cwd() / "local" / "test.sqlite3")

    def test_absolute_database_override_is_preserved(self) -> None:
        absolute_path = Path(os.environ.get("TEMP", ".")) / "rag.sqlite3"

        config = load_config(
            {"FOUNDRY_LOCAL_RAG_DATABASE_PATH": str(absolute_path)}
        )

        self.assertEqual(config.database_path, absolute_path)

    def test_empty_values_raise_configuration_error(self) -> None:
        for variable in (
            "FOUNDRY_LOCAL_RAG_EMBEDDING_MODEL",
            "FOUNDRY_LOCAL_RAG_CHAT_MODEL",
            "FOUNDRY_LOCAL_RAG_DATABASE_PATH",
        ):
            with self.subTest(variable=variable):
                with self.assertRaises(ConfigurationError):
                    load_config({variable: "  "})

    def test_default_loading_does_not_use_process_environment(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = load_config()

        self.assertEqual(config.embedding_model_id, EMBEDDING_MODEL_ID)
