"""Offline integration checks; no tokenizer training, backward pass, or optimizer step."""

from __future__ import annotations

from array import array
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from native_600m.checkpoint import load_model, read_checkpoint, save_checkpoint
from native_600m.config import load_config, sha256_file
from native_600m.model import ModelConfig, NativeLM
from native_600m.runtime import require_cuda
from native_600m.train import learning_rate, stage_options
from native_600m.training_data import TokenWindows, sample_index
from run_native_600m import build_commands, parse_args


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_addressable_sampler_has_no_duplicates_and_resumes_exactly(self):
        for length in (1, 2, 7, 20, 100):
            for epoch in range(3):
                actual = [sample_index(epoch * length + position, length, 42) for position in range(length)]
                self.assertEqual(sorted(actual), list(range(length)))
            full = [sample_index(index, length, 42) for index in range(200)]
            self.assertEqual(full[73:], [sample_index(index, length, 42) for index in range(73, 200)])

    def test_target_mask_is_shifted_with_next_token_and_empty_windows_are_excluded(self):
        values = array("H", [1, 3, 8, 6, 4, 9, 6, 5, 10, 11, 6, 2, 1])
        token_file = self.root / "tokens.bin"
        token_file.write_bytes(values.tobytes())
        mask_file = self.root / "mask.bin"
        mask_file.write_bytes(bytes([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]))
        entry = {"tokens_file": "tokens.bin", "mask_file": "mask.bin"}
        dataset = TokenWindows(self.root, entry, seq_len=4)
        self.assertEqual(len(dataset), 2)
        x0, y0 = dataset[0]
        self.assertEqual(x0.tolist(), [4, 9, 6, 5])
        self.assertEqual(y0.tolist(), [-100, -100, -100, 10])
        _, y1 = dataset[1]
        self.assertEqual(y1.tolist(), [11, 6, -100, -100])

    def test_checkpoint_and_export_restore_identical_logits_and_tied_weights(self):
        try:
            from safetensors.torch import load_file
        except ImportError:
            self.skipTest("Install safetensors to verify exported weights.")
        from native_600m.export import export_model

        config = ModelConfig(vocab_size=263, max_seq_len=32, d_model=32, n_layers=2,
                             n_heads=4, n_kv_heads=2, d_ff=64)
        model = NativeLM(config).eval()
        tokenizer_path = self.root / "tokenizer.json"
        tokenizer_path.write_text("manual tokenizer fixture", encoding="utf-8")
        checkpoint = self.root / "best.pt"
        save_checkpoint(checkpoint, model, tokenizer_sha256=sha256_file(tokenizer_path), stage="pretrain",
                        torch_rng=torch.get_rng_state(), python_rng=__import__("random").getstate())
        restored, metadata = load_model(checkpoint, device="cpu")
        self.assertEqual(metadata["config"], asdict(config))
        self.assertIs(restored.lm_head.weight, restored.token_embedding.weight)
        input_ids = torch.tensor([[1, 8, 10, 12]])
        with torch.inference_mode():
            expected = model(input_ids)[0]
            torch.testing.assert_close(expected, restored(input_ids)[0], rtol=0, atol=0)
        output = self.root / "model"
        export_model(checkpoint, tokenizer_path, output)
        exported, _ = load_model(output, device="cpu")
        self.assertIs(exported.lm_head.weight, exported.token_embedding.weight)
        self.assertNotIn("lm_head.weight", load_file(str(output / "model.safetensors")))
        with torch.inference_mode():
            torch.testing.assert_close(expected, exported(input_ids)[0], rtol=0, atol=0)
        export_model(checkpoint, tokenizer_path, output)
        (output / "model_config.json").write_text("corrupted", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_model(output, device="cpu")

    def test_h200_profile_preserves_effective_batch_and_respects_overrides(self):
        from native_600m.train import parse_args as train_args

        config = load_config()
        args = train_args(["--stage", "pretrain", "--output-dir", "unused"])
        balanced = stage_options(config, args, gpu_name="Other GPU", gpu_memory_gb=24)
        h200 = stage_options(config, args, gpu_name="NVIDIA H200", gpu_memory_gb=141)
        self.assertEqual(h200["batch_size"], 8)
        self.assertEqual(h200["batch_size"] * h200["grad_accumulation"],
                         balanced["batch_size"] * balanced["grad_accumulation"])
        self.assertEqual(stage_options(config, args, gpu_name="NVIDIA H200 MIG", gpu_memory_gb=18), balanced)
        args.batch_size = 2
        self.assertEqual(stage_options(config, args, gpu_name="NVIDIA H200", gpu_memory_gb=141)["batch_size"], 2)

    def test_generated_commands_match_all_parsers_without_importing_model_loaders(self):
        from native_600m import evaluate, export, prepare, tokenize, tokenizer, train

        parsers = {"native_600m.prepare": prepare.parse_args, "native_600m.tokenizer": tokenizer.parse_args,
                   "native_600m.tokenize": tokenize.parse_args, "native_600m.train": train.parse_args,
                   "native_600m.export": export.parse_args, "native_600m.evaluate": evaluate.parse_args}
        commands = build_commands(parse_args(["--plan", "--reuse-prepared"]))
        self.assertEqual(len(commands), 7)
        self.assertEqual(len(build_commands(parse_args(["--plan", "--prepare-only"]))), 3)
        for command in commands:
            parsers[command[2]](command[3:])
            self.assertNotIn("Qwen", " ".join(command))

    def test_preparation_worker_setting_is_forwarded_and_validated(self):
        from native_600m.prepare import parse_args as prepare_args
        from native_600m.tokenize import parse_args as tokenize_args
        from native_600m.tokenizer import parse_args as tokenizer_args
        from native_600m.train import parse_args as train_args

        commands = build_commands(parse_args(["--plan", "--cpu-workers", "24"]))
        self.assertEqual(parse_args([]).prepare_workers, 24)
        self.assertEqual(prepare_args(commands[0][3:]).workers, 24)
        self.assertEqual(tokenizer_args(commands[1][3:]).workers, 24)
        self.assertEqual(tokenize_args(commands[2][3:]).workers, 24)
        self.assertEqual(train_args(commands[3][3:]).cpu_workers, 24)
        self.assertEqual(train_args(commands[4][3:]).cpu_workers, 24)
        self.assertEqual(parse_args(["--prepare-workers", "3"]).prepare_workers, 3)
        for workers in ("0", "-1"):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "prepare-workers"):
                build_commands(parse_args(["--plan", "--prepare-workers", workers]))

    def test_cuda_guard_precedes_model_training(self):
        from native_600m.train import parse_args as train_args, train

        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "available CUDA GPU"):
                train(train_args(["--stage", "pretrain", "--output-dir", "unused"]))

    def test_cuda_guard_requires_one_visible_gpu_on_any_platform(self):
        with patch("torch.cuda.is_available", return_value=True):
            for count in (0, 2):
                with self.subTest(count=count), patch("torch.cuda.device_count", return_value=count):
                    with self.assertRaisesRegex(RuntimeError, "one visible GPU"):
                        require_cuda()
            with patch("torch.cuda.device_count", return_value=1):
                self.assertIs(require_cuda(), torch)

    def test_prepare_only_runs_without_cuda(self):
        from run_native_600m import main

        with patch("sys.argv", ["run_native_600m.py", "--prepare-only"]), \
             patch("run_native_600m.require_cuda", side_effect=AssertionError("CUDA is not needed")), \
             patch("run_native_600m.subprocess.run") as run:
            main()
        self.assertEqual([call.args[0][2] for call in run.call_args_list],
                         ["native_600m.prepare", "native_600m.tokenizer", "native_600m.tokenize"])

    def test_learning_rate_reaches_peak_and_final_minimum(self):
        rates = [learning_rate(step, 10, 0.001, 0.0001, 2) for step in range(10)]
        self.assertEqual(rates[0], 0.0005)
        self.assertEqual(rates[1], 0.001)
        self.assertEqual(rates[-1], 0.0001)


if __name__ == "__main__":
    unittest.main()
