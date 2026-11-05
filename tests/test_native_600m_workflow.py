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
from native_600m.train import checkpoint_activations, learning_rate, resolve_run_identity, stage_options
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

    def test_rtx_pro_6000_profile_preserves_both_stage_budgets(self):
        from native_600m.train import parse_args as train_args

        config = load_config()
        for stage in ("pretrain", "sft"):
            args = train_args(["--stage", stage, "--output-dir", "unused"])
            with self.subTest(stage=stage):
                actual = stage_options(config, args, gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                       gpu_memory_gb=96)
                self.assertEqual(actual["batch_size"], 8)
                self.assertEqual(actual["batch_size"] * actual["grad_accumulation"],
                                 config[stage]["batch_size"] * config[stage]["grad_accumulation"])
                self.assertEqual(stage_options(config, args, gpu_name="NVIDIA RTX PRO 6000 MIG", gpu_memory_gb=24),
                                 config[stage])
                args.gpu_profile = "balanced"
                self.assertEqual(stage_options(config, args, gpu_name="NVIDIA RTX PRO 6000", gpu_memory_gb=96),
                                 config[stage])
                args.gpu_profile = "rtx-pro-6000"
                self.assertEqual(stage_options(config, args), actual)
                args.batch_size = 4
                args.grad_accumulation = 16 if stage == "pretrain" else 8
                overridden = stage_options(config, args)
                self.assertEqual(overridden["batch_size"], 4)
                self.assertEqual(overridden["grad_accumulation"], args.grad_accumulation)

    def test_gpu_upgrade_keeps_requested_batch_and_accepts_existing_run_identity(self):
        from native_600m.train import parse_args as train_args

        config = load_config()
        for stage in ("pretrain", "sft"):
            for profile in ("auto", "rtx-pro-6000"):
                args = train_args(["--stage", stage, "--output-dir", "unused", "--gpu-profile", profile])
                requested = stage_options(config, args, gpu_name="NVIDIA RTX PRO 6000", gpu_memory_gb=96)
                saved = dict(config[stage])
                with self.subTest(stage=stage, profile=profile):
                    previous_identity = {"options": saved, "stage": stage, "max_steps": 100}
                    actual = resolve_run_identity({**previous_identity, "options": requested}, previous_identity)
                    self.assertEqual(actual, previous_identity)
                    self.assertEqual(requested["batch_size"], 8)
                    self.assertEqual(saved["batch_size"], 1)

    def test_gpu_upgrade_rejects_learning_data_and_effective_batch_changes(self):
        from copy import deepcopy
        from native_600m.train import parse_args as train_args

        config = load_config()
        args = train_args(["--stage", "pretrain", "--output-dir", "unused", "--gpu-profile", "rtx-pro-6000"])
        requested = stage_options(config, args)
        saved = dict(config["pretrain"])
        identity = {"options": requested, "config_sha256": "config", "stage": "pretrain",
                    "precision": "torch.bfloat16", "token_manifest_sha256": "tokens",
                    "tokenizer_sha256": "tokenizer", "max_steps": 91553, "init_checkpoint_sha256": None}
        for change in ({"lr": 0.01}, {"seq_len": 128}, {"target_tokens": 10}, {"max_eval_batches": 1},
                       {"grad_accumulation": 32}, {"batch_size": True}, {"batch_size": 0}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "configuration or data changed"):
                resolve_run_identity(identity, {**identity, "options": {**saved, **change}})
        for previous in (None, {}, {"options": None}):
            with self.assertRaises(ValueError):
                resolve_run_identity(identity, previous)
        for name in ("config_sha256", "stage", "precision", "token_manifest_sha256", "tokenizer_sha256",
                     "max_steps", "init_checkpoint_sha256"):
            changed = deepcopy(identity)
            changed[name] = "different"
            with self.subTest(name=name), self.assertRaises(ValueError):
                resolve_run_identity(identity, changed)
        args.batch_size, args.grad_accumulation = 4, 16
        explicit = stage_options(config, args)
        self.assertEqual(resolve_run_identity({**identity, "options": explicit}, {**identity, "options": saved}),
                         {**identity, "options": saved})

    def test_rtx_pro_profile_is_forwarded_to_both_training_stages(self):
        from native_600m.train import parse_args as train_args

        commands = build_commands(parse_args(["--plan", "--gpu-profile", "rtx-pro-6000"]))
        for command in commands[3:5]:
            self.assertEqual(train_args(command[3:]).gpu_profile, "rtx-pro-6000")

    def test_full_rtx_uses_vram_for_activations_and_balanced_retains_recomputation(self):
        from native_600m.train import parse_args as train_args

        args = train_args(["--stage", "pretrain", "--output-dir", "unused"])
        for profile in ("auto", "rtx-pro-6000"):
            args.gpu_profile = profile
            with self.subTest(profile=profile):
                self.assertFalse(checkpoint_activations(args, gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                                        gpu_memory_gb=96))
                self.assertTrue(checkpoint_activations(args, gpu_name="NVIDIA RTX PRO 6000 MIG", gpu_memory_gb=24))
                self.assertTrue(checkpoint_activations(args, gpu_name="NVIDIA A100", gpu_memory_gb=80))
        args.gpu_profile = "balanced"
        self.assertTrue(checkpoint_activations(args, gpu_name="NVIDIA RTX PRO 6000", gpu_memory_gb=96))

    def test_full_h100_retains_activations_but_smaller_partitions_recompute(self):
        from native_600m.train import parse_args as train_args

        args = train_args(["--stage", "pretrain", "--output-dir", "unused"])
        for profile in ("auto", "h100"):
            args.gpu_profile = profile
            with self.subTest(profile=profile):
                self.assertFalse(checkpoint_activations(args, gpu_name="NVIDIA H100 80GB HBM3",
                                                        gpu_memory_gb=79))
                self.assertTrue(checkpoint_activations(args, gpu_name="NVIDIA H100 MIG 3g.40gb",
                                                       gpu_memory_gb=39))
                self.assertTrue(checkpoint_activations(args, gpu_name="NVIDIA A100", gpu_memory_gb=80))
                self.assertTrue(checkpoint_activations(args))
        args.gpu_profile = "balanced"
        self.assertTrue(checkpoint_activations(args, gpu_name="NVIDIA H100 80GB HBM3", gpu_memory_gb=79))

    def test_300m_workflow_is_isolated_from_600m_and_h100_auto_keeps_effective_batch(self):
        from native_600m.config import ROOT
        from native_600m.train import parse_args as train_args

        args = parse_args([])
        config = load_config(args.config)
        self.assertEqual(config["pretrain"]["target_tokens"], 6_000_000_000)
        self.assertEqual(config["pretrain"]["warmup_steps"], 1000)
        self.assertEqual(args.artifacts_dir, ROOT / "artifacts_native_300m")
        commands = build_commands(args)
        for command in commands[3:5]:
            options = train_args(command[3:])
            self.assertEqual(options.config, args.config)
            self.assertEqual(options.output_dir.parent, args.artifacts_dir)
            actual = stage_options(config, options, gpu_name="NVIDIA H100 80GB HBM3", gpu_memory_gb=79)
            self.assertEqual(actual["batch_size"], 8)
            self.assertEqual(actual["batch_size"] * actual["grad_accumulation"],
                             64 if options.stage == "pretrain" else 32)
            self.assertEqual(stage_options(config, options, gpu_name="NVIDIA H100 MIG", gpu_memory_gb=39),
                             config[options.stage])
        legacy = parse_args(["--config", str(ROOT / "configs" / "native_600m.json"),
                             "--artifacts-dir", str(ROOT / "artifacts_native_600m")])
        legacy_command = build_commands(legacy)[3]
        self.assertEqual(train_args(legacy_command[3:]).output_dir, legacy.artifacts_dir / "pretrain")
        self.assertNotEqual(args.artifacts_dir, legacy.artifacts_dir)

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
        compiled = build_commands(parse_args(["--plan", "--compile-model"]))
        self.assertTrue(all(train.parse_args(command[3:]).compile_model for command in compiled[3:5]))
        self.assertTrue(all("--compile-model" not in command for command in compiled[:3] + compiled[5:]))

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
