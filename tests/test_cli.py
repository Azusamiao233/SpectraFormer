import unittest
from importlib.util import find_spec
from unittest.mock import patch

from mssa_transformer.cli import MODEL_REGISTRY, format_model_list, main


class CliTests(unittest.TestCase):
    def test_model_keys_are_unique_and_modules_are_present(self):
        self.assertGreaterEqual(len(MODEL_REGISTRY), 8)
        for key, entry in MODEL_REGISTRY.items():
            self.assertEqual(key, entry.key)
            self.assertTrue(entry.module)
            self.assertTrue(entry.description)
            self.assertIsNotNone(find_spec(entry.module), entry.module)

    def test_model_list_contains_main_versions(self):
        rendered = format_model_list()
        for key in ("structured", "transformer", "mssa-mlp", "spectral-modal"):
            self.assertIn(key, rendered)
        self.assertNotIn("gdn", MODEL_REGISTRY)
        self.assertNotIn("gat-vae", MODEL_REGISTRY)

    def test_list_models_does_not_import_training_dependencies(self):
        with patch("builtins.print") as mocked_print:
            result = main(["--list-models"])
        self.assertEqual(result, 0)
        self.assertTrue(mocked_print.called)

    def test_legacy_arguments_default_to_structured_experiment(self):
        with patch("mssa_transformer.cli.run_model", return_value=0) as mocked_run:
            result = main(["--plan"])
        self.assertEqual(result, 0)
        self.assertEqual(mocked_run.call_args.args[0].key, "structured")
        self.assertEqual(mocked_run.call_args.args[1], ["--plan"])

    def test_explicit_model_receives_forwarded_arguments(self):
        with patch("mssa_transformer.cli.run_model", return_value=0) as mocked_run:
            result = main(["--model", "structured", "--", "--plan"])
        self.assertEqual(result, 0)
        self.assertEqual(mocked_run.call_args.args[0].key, "structured")
        self.assertEqual(mocked_run.call_args.args[1], ["--", "--plan"])


if __name__ == "__main__":
    unittest.main()
