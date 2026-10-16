"""Offline contract tests; no model weights or ML runtime are loaded."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from korean_600m.chat import PROVENANCE_FILE, generate_reply, load_chat_model, parse_args as chat_args, validate_adapter
from korean_600m.config import config_fingerprint, sha256_file
from korean_600m.evaluate import EVAL_CASES, case_signals, main as evaluate_main, response_sanity, run_evaluation, seeded_generate_reply
from korean_600m.export import export_merged, validate_output_path


class InferenceContracts(unittest.TestCase):
    def setUp(self):
        self.config = {"model_name": "Qwen/Qwen3-0.6B", "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca"}

    def test_adapter_requires_matching_base_and_revision_and_safetensors(self):
        with tempfile.TemporaryDirectory(prefix=".test-inference-", dir=Path(__file__).resolve().parents[1]) as temporary:
            adapter = Path(temporary)
            metadata = {"base_model_name_or_path": self.config["model_name"],
                        "revision": self.config["model_revision"], "peft_type": "LORA"}
            (adapter / "adapter_config.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                validate_adapter(adapter, self.config)
            (adapter / "adapter_model.safetensors").write_bytes(b"test fixture only")
            self.assertEqual(validate_adapter(adapter, self.config), metadata)
            for key, replacement in (("revision", None), ("revision", "f" * 40),
                                     ("base_model_name_or_path", "another/model"), ("peft_type", "IA3")):
                with self.subTest(key=key, replacement=replacement):
                    invalid = dict(metadata, **{key: replacement})
                    (adapter / "adapter_config.json").write_text(json.dumps(invalid), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        validate_adapter(adapter, self.config)

    def test_source_options_are_exclusive(self):
        self.assertEqual(chat_args([]).adapter, Path("artifacts_korean_600m/adapter"))
        self.assertTrue(chat_args(["--base-model"]).base_model)
        with self.assertRaises(SystemExit):
            chat_args(["--base-model", "--merged", "somewhere"])
        with self.assertRaises(SystemExit):
            chat_args(["--base-model", "--adapter", "somewhere"])

    def test_invalid_generation_arguments_fail_before_runtime_import(self):
        for arguments in ({"max_new_tokens": 64}, {"max_new_tokens": 0},
                          {"max_new_tokens": 32, "temperature": 0},
                          {"max_new_tokens": 32, "temperature": float("nan")}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                generate_reply(None, None, [], max_seq_length=64, **arguments)

    def test_cuda_unavailable_never_downloads_or_falls_back(self):
        model_loader, tokenizer_loader = Mock(), Mock()
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        fake_transformers = SimpleNamespace(AutoModelForCausalLM=model_loader, AutoTokenizer=tokenizer_loader)
        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                load_chat_model(self.config, adapter=None, merged=None, device="cuda")
        model_loader.from_pretrained.assert_not_called()
        tokenizer_loader.from_pretrained.assert_not_called()

    def test_cuda_precision_requires_native_bfloat16(self):
        model_loader, tokenizer_loader = Mock(), Mock()
        tokenizer_loader.from_pretrained.return_value = SimpleNamespace(eos_token_id=1, pad_token_id=0)
        supported = Mock(return_value=False)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
                                    is_bf16_supported=supported),
                                    float32="float32", float16="float16", bfloat16="bfloat16")
        fake_transformers = SimpleNamespace(AutoModelForCausalLM=model_loader, AutoTokenizer=tokenizer_loader)
        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
            load_chat_model(self.config, adapter=None, merged=None, device="cuda")
        supported.assert_called_once_with(including_emulation=False)
        self.assertEqual(model_loader.from_pretrained.call_args.kwargs["torch_dtype"], "float16")

    def test_export_preserves_adapter_and_existing_outputs(self):
        with tempfile.TemporaryDirectory(prefix=".test-inference-", dir=Path(__file__).resolve().parents[1]) as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            output = root / "merged"
            self.assertEqual(validate_output_path(output, adapter), output.resolve())
            for invalid in (adapter, root, adapter / "nested"):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    validate_output_path(invalid, adapter)
            output.mkdir()
            (output / "keep.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_output_path(output, adapter)
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "preserve")


class ExportContracts(unittest.TestCase):
    def create_export(self, root):
        config = {"model_name": "Qwen/Qwen3-0.6B", "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                  "system_prompt": "한국어로 답변합니다.", "max_seq_length": 2048}
        adapter, output = root / "adapter", root / "merged"
        adapter.mkdir()
        metadata = {"base_model_name_or_path": config["model_name"], "revision": config["model_revision"], "peft_type": "LORA"}
        (adapter / "adapter_config.json").write_text(json.dumps(metadata), encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"dummy adapter weights; not a model")
        training = {"base_model": config["model_name"], "base_revision": config["model_revision"],
                    "data_manifest": {"sources": [{"dataset": "fixture", "license": "CC-BY-4.0"}]}}
        (adapter / PROVENANCE_FILE).write_text(json.dumps(training), encoding="utf-8")
        merged_model, model, tokenizer = Mock(), Mock(), Mock()
        model.merge_and_unload.return_value = merged_model
        merged_model.parameters.return_value = [SimpleNamespace(numel=lambda: 596049920)]

        def save_model(path, **kwargs):
            (path / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"dummy merged weights; not a model")

        def save_tokenizer(path):
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")

        merged_model.save_pretrained.side_effect = save_model
        tokenizer.save_pretrained.side_effect = save_tokenizer
        with patch("korean_600m.export.load_chat_model", return_value=(model, tokenizer)):
            provenance = export_merged(config, adapter=adapter, output=output, device="cuda")
        return config, adapter, output, provenance

    def test_valid_reuse_checks_hashes_without_loading_model(self):
        with tempfile.TemporaryDirectory(prefix=".test-inference-", dir=Path(__file__).resolve().parents[1]) as temporary:
            config, adapter, output, original = self.create_export(Path(temporary))
            with patch("korean_600m.export.load_chat_model", side_effect=AssertionError("No model load during reuse")):
                reused = export_merged(config, adapter=adapter, output=output, device="cuda", reuse_existing=True)
                with self.assertRaises(ValueError):
                    export_merged(config, adapter=adapter, output=output, device="cuda")
            self.assertEqual(reused, original)
            self.assertEqual(reused["source_adapter_sha256"], sha256_file(adapter / "adapter_model.safetensors"))
            self.assertEqual(reused["config_sha256"], config_fingerprint(config))
            self.assertEqual(reused["training_provenance"]["data_manifest"]["sources"][0]["license"], "CC-BY-4.0")
            self.assertIn("tokenizer.json", reused["artifact_sha256"])

    def test_reuse_rejects_changed_source_and_corrupt_files(self):
        changes = ("adapter", "config", "base", "weights", "tokenizer", "missing_weights", "missing_manifest")
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory(
                    prefix=".test-inference-", dir=Path(__file__).resolve().parents[1]) as temporary:
                config, adapter, output, provenance = self.create_export(Path(temporary))
                if change == "adapter":
                    (adapter / "adapter_model.safetensors").write_bytes(b"changed adapter")
                elif change == "config":
                    config["max_seq_length"] = 1024
                elif change == "base":
                    provenance["base_revision"] = "f" * 40
                    (output / PROVENANCE_FILE).write_text(json.dumps(provenance), encoding="utf-8")
                elif change == "weights":
                    (output / "model.safetensors").write_bytes(b"corrupted weights")
                elif change == "tokenizer":
                    (output / "tokenizer.json").write_text('{"corrupted":true}', encoding="utf-8")
                elif change == "missing_weights":
                    (output / "model.safetensors").unlink()
                else:
                    provenance.pop("artifact_sha256")
                    (output / PROVENANCE_FILE).write_text(json.dumps(provenance), encoding="utf-8")
                with patch("korean_600m.export.load_chat_model", side_effect=AssertionError("No model load for invalid reuse")):
                    with self.assertRaises(ValueError):
                        export_merged(config, adapter=adapter, output=output, device="cuda", reuse_existing=True)


class EvaluationContracts(unittest.TestCase):
    def test_seeded_sampling_restores_rng_scope(self):
        events = []

        @contextmanager
        def fork_rng():
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        fake_torch = SimpleNamespace(random=SimpleNamespace(fork_rng=fork_rng),
                                     manual_seed=lambda seed: events.append(seed))
        with patch.dict(sys.modules, {"torch": fake_torch}), \
                patch("korean_600m.evaluate.generate_reply", return_value="한국어 답변입니다.") as generate:
            reply = seeded_generate_reply(None, None, [], max_seq_length=2048, max_new_tokens=192, seed=123)
        self.assertEqual(reply, "한국어 답변입니다.")
        self.assertEqual(events, ["enter", 123, "exit"])
        self.assertTrue(generate.call_args.kwargs["do_sample"])
        self.assertEqual(generate.call_args.kwargs["temperature"], 0.7)

    def test_sanity_catches_wrong_language_and_loops(self):
        self.assertTrue(response_sanity("안녕하세요. 무엇을 도와드릴까요?")["passed"])
        self.assertFalse(response_sanity("Hello, what can I help you with?")["passed"])
        self.assertFalse(response_sanity("这是中文回答。")["passed"])
        self.assertFalse(response_sanity("안녕하세요 도와드릴게요 오늘도 " * 7)["passed"])
        self.assertFalse(response_sanity("가" * 30)["passed"])
        self.assertFalse(response_sanity("<think>안녕하세요</think>")["passed"])
        self.assertFalse(response_sanity("")["passed"])
        self.assertTrue(response_sanity("19", min_hangul_ratio=0)["passed"])

    def test_case_specific_signals(self):
        cases = {case.name: case for case in EVAL_CASES}
        self.assertTrue(all(case_signals(cases["memory_correction"], "지연").values()))
        self.assertFalse(all(case_signals(cases["memory_correction"], "민수 또는 지연").values()))
        self.assertTrue(all(case_signals(cases["three_items"], "- 물\n- 신발\n- 모자").values()))
        self.assertFalse(all(case_signals(cases["three_items"], "- 물\n- 모자").values()))
        self.assertFalse(all(case_signals(cases["arithmetic"], "19개입니다.").values()))

    def test_multiturn_evaluation_uses_actual_previous_reply(self):
        memory_case = next(case for case in EVAL_CASES if case.name == "memory_correction")
        calls = []
        answers = iter(("민수님, 반갑습니다.", "지연님으로 기억하겠습니다.", "지연"))

        def fake_generate(model, tokenizer, messages, **kwargs):
            calls.append([dict(message) for message in messages])
            self.assertEqual(kwargs["seed"], 42 + len(calls) - 1)
            return next(answers)

        with patch("korean_600m.evaluate.EVAL_CASES", (memory_case,)), \
                patch("korean_600m.evaluate.seeded_generate_reply", side_effect=fake_generate):
            results = run_evaluation(None, None, {"system_prompt": "한국어로 대화합니다.", "max_seq_length": 512, "seed": 42}, 64)
        self.assertTrue(results[0]["passed"])
        self.assertEqual(calls[2][2], {"role": "assistant", "content": "민수님, 반갑습니다."})
        self.assertEqual(calls[2][4], {"role": "assistant", "content": "지연님으로 기억하겠습니다."})

    def test_report_keeps_failed_gate_and_requires_explicit_allow_fail(self):
        failed = [{"name": "failed", "passed": False, "checks": {}, "turns": [
            {"prompt": "안녕", "response": "", "sanity": {"passed": False, "reasons": ["empty_response"]}},
        ]}]
        config = {"model_name": "Qwen/Qwen3-0.6B", "model_revision": "a" * 40, "seed": 42}
        with tempfile.TemporaryDirectory(prefix=".test-inference-", dir=Path(__file__).resolve().parents[1]) as temporary:
            report_path = Path(temporary) / "report.json"
            args = ["--base-model", "--report", str(report_path)]
            with patch("korean_600m.evaluate.load_config", return_value=config), \
                    patch("korean_600m.evaluate.load_chat_model", return_value=(None, None)), \
                    patch("korean_600m.evaluate.run_evaluation", return_value=failed):
                with self.assertRaises(SystemExit) as raised:
                    evaluate_main(args)
                self.assertEqual(raised.exception.code, 1)
                evaluate_main([*args, "--allow-fail"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["heuristic_gate_passed"])
            self.assertFalse(report["conversation_quality_verified"])
            self.assertTrue(report["human_rubric"]["criteria"])
            self.assertTrue(report["generation"]["do_sample"])
            self.assertEqual(report["generation"]["seed"], 42)


if __name__ == "__main__":
    unittest.main()
