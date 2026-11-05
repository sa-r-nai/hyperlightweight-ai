"""CPU-only tiny-model checks for checkpoint reuse and equivalent batch regrouping."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from native_600m.checkpoint import read_checkpoint, save_checkpoint
from native_600m.config import config_fingerprint, load_config, sha256_file
from native_600m.model import ModelConfig, NativeLM
from native_600m.train import checkpoint_due, evaluate, parse_args, resolve_run_identity, stage_options, train
from native_600m.training_data import TokenWindows


class TrainingResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model_config = ModelConfig(vocab_size=263, max_seq_len=8, d_model=16, n_layers=1,
                                        n_heads=2, n_kv_heads=1, d_ff=24)
        self.values = ((np.arange(8 * 300 + 1) * 17) % 255 + 8).astype("<u2")
        self.values.tofile(self.root / "tokens.bin")
        (np.arange(len(self.values)) % 3 != 0).astype("u1").tofile(self.root / "mask.bin")

    def dataset(self, stage):
        entry = {"tokens_file": "tokens.bin"}
        if stage == "sft":
            entry["mask_file"] = "mask.bin"
        dataset = TokenWindows(self.root, entry, seq_len=8)
        # Windows retains mapped-file handles until they are explicitly closed.
        self.addCleanup(dataset.tokens._mmap.close)
        if dataset.mask is not None:
            self.addCleanup(dataset.mask._mmap.close)
        return dataset

    @staticmethod
    def optimizer(model):
        return torch.optim.AdamW(model.parameters(), lr=0.0003, betas=(0.9, 0.95), weight_decay=0.1)

    @staticmethod
    def update(model, optimizer, dataset, *, step, batch_size, accumulation):
        batches = [dataset.batch((step * accumulation + micro) * batch_size, batch_size, seed=42)
                   for micro in range(accumulation)]
        count = sum(int(y.ne(-100).sum()) for _, y in batches)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        for x, y in batches:
            _, loss = model(x, y, loss_reduction="sum")
            (loss / count).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return torch.cat([x for x, _ in batches]), torch.cat([y for _, y in batches]), count

    def test_actual_optimizer_and_sample_cursor_survive_batch_one_to_sixteen_resume(self):
        for stage, effective_batch in (("pretrain", 64), ("sft", 32)):
            with self.subTest(stage=stage):
                torch.manual_seed(7)
                dataset = self.dataset(stage)
                reference = NativeLM(self.model_config)
                # Resume a recomputing run with activations retained in memory.
                reference.enable_gradient_checkpointing()
                old_optimizer = self.optimizer(reference)
                old_identity = {"stage": stage, "max_steps": 100,
                                "options": {"batch_size": 1, "grad_accumulation": effective_batch,
                                            "seq_len": 8, "max_eval_batches": 5}}
                self.update(reference, old_optimizer, dataset, step=0, batch_size=1, accumulation=effective_batch)
                checkpoint_path = self.root / f"{stage}.pt"
                save_checkpoint(checkpoint_path, reference, stage=stage, step=1, best_loss=4.0,
                                run_identity=old_identity, optimizer=old_optimizer.state_dict(), scaler={},
                                torch_rng=torch.get_rng_state(), cuda_rng=[], python_rng=random.getstate())
                saved = read_checkpoint(checkpoint_path)
                execution = {**old_identity["options"], "batch_size": 16, "grad_accumulation": effective_batch // 16}
                identity = resolve_run_identity({**old_identity, "options": execution}, saved["run_identity"])
                self.assertEqual(identity, old_identity)
                self.assertEqual(saved["step"], 1)
                restored = NativeLM(self.model_config)
                self.assertFalse(restored.gradient_checkpointing)
                new_optimizer = self.optimizer(restored)
                restored.load_state_dict(saved["model"])
                new_optimizer.load_state_dict(saved["optimizer"])
                torch.set_rng_state(saved["torch_rng"])
                random.setstate(saved["python_rng"])
                for before, after in zip(reference.parameters(), restored.parameters()):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
                for old, new in zip(old_optimizer.state.values(), new_optimizer.state.values()):
                    for name in ("step", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)
                old_inputs, old_targets, old_count = self.update(
                    reference, old_optimizer, dataset, step=saved["step"], batch_size=1, accumulation=effective_batch)
                new_inputs, new_targets, new_count = self.update(
                    restored, new_optimizer, dataset, step=saved["step"], batch_size=16, accumulation=effective_batch // 16)
                self.assertTrue(torch.equal(old_inputs, new_inputs))
                self.assertTrue(torch.equal(old_targets, new_targets))
                self.assertEqual(old_count, new_count)
                for before, after in zip(reference.parameters(), restored.parameters()):
                    torch.testing.assert_close(before, after, rtol=2e-4, atol=2e-6)
                # Reusing the original identity also makes a second resume safe
                # if interruption happens between manifest and checkpoint writes.
                save_checkpoint(checkpoint_path, restored, step=2, run_identity=identity,
                                execution_options=execution, optimizer=new_optimizer.state_dict())
                second = read_checkpoint(checkpoint_path)
                self.assertEqual(second["step"], 2)
                self.assertEqual(second["execution_options"]["batch_size"], 16)
                self.assertEqual(resolve_run_identity({**old_identity, "options": execution}, second["run_identity"]),
                                 old_identity)

    def test_validation_keeps_the_original_sample_budget_after_batch_regrouping(self):
        dataset = self.dataset("sft")
        model = NativeLM(self.model_config)
        accessed = []

        class TrackedWindows:
            def __len__(self):
                return len(dataset)

            def __getitem__(self, index):
                accessed.append(index)
                return dataset[index]

        # Evaluation explicitly uses CUDA autocast in production; the tiny
        # fixture uses CPU float32 without requesting any GPU resources.
        with patch("torch.autocast", side_effect=lambda *args, **kwargs: nullcontext()):
            original = evaluate(model, TrackedWindows(), batch_size=1, max_batches=5,
                                device="cpu", dtype=torch.float32)
            self.assertEqual(accessed, list(range(5)))
            accessed.clear()
            regrouped = evaluate(model, TrackedWindows(), batch_size=8, max_batches=1, max_windows=5,
                                 device="cpu", dtype=torch.float32)
        self.assertEqual(accessed, list(range(5)))
        self.assertAlmostEqual(original, regrouped, places=5)
        self.assertTrue(model.training)

    def test_compilation_preserves_parameters_gradients_and_checkpoint_keys(self):
        reference = NativeLM(self.model_config)
        compiled = NativeLM(self.model_config)
        compiled.load_state_dict(reference.state_dict())
        parameters = tuple(compiled.parameters())
        keys = set(compiled.state_dict())
        # Exercise real graph capture/autograd on CPU without requiring CUDA,
        # Triton, or a Windows C++ compiler. CUDA performance is measured remotely.
        compiled.compile(backend="aot_eager", dynamic=False)
        self.assertTrue(all(before is after for before, after in zip(parameters, compiled.parameters())))
        x, y = self.dataset("sft").batch(0, 2, seed=42)
        expected, reference_loss = reference(x, y, loss_reduction="sum")
        actual, compiled_loss = compiled(x, y, loss_reduction="sum")
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(compiled_loss, reference_loss)
        reference_loss.backward()
        compiled_loss.backward()
        for before, after in zip(reference.parameters(), compiled.parameters()):
            torch.testing.assert_close(before.grad, after.grad)
        path = self.root / "compiled.pt"
        save_checkpoint(path, compiled, stage="pretrain", step=1)
        checkpoint = read_checkpoint(path)
        self.assertEqual(set(checkpoint["model"]), keys)
        restored = NativeLM(self.model_config)
        restored.load_state_dict(checkpoint["model"], strict=True)
        with torch.inference_mode():
            torch.testing.assert_close(restored(x)[0], expected)

    def test_training_entrypoint_resumes_a_checkpoint_at_batch_eight_and_saves_the_first_new_step(self):
        config = load_config()
        config["model"] = asdict(self.model_config)
        config["tokenizer"]["vocab_size"] = 263
        config["pretrain"]["seq_len"] = config["sft"]["seq_len"] = 8
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        manifest = {"outputs": {name: {"tokens_file": "tokens.bin"}
                                for name in ("pretrain_train", "pretrain_validation")}}
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        output = self.root / "run"
        output.mkdir()
        identity = {"config_sha256": config_fingerprint(config), "stage": "pretrain",
                    "token_manifest_sha256": sha256_file(self.root / "manifest.json"),
                    "tokenizer_sha256": "manual-fixture", "options": dict(config["pretrain"]),
                    "max_steps": 3, "precision": str(torch.bfloat16), "init_checkpoint_sha256": None}
        (output / "run_manifest.json").write_text(json.dumps({"identity": identity}), encoding="utf-8")
        original = NativeLM(self.model_config)
        optimizer = self.optimizer(original)
        self.update(original, optimizer, self.dataset("pretrain"), step=0, batch_size=1, accumulation=64)
        save_checkpoint(output / "last.pt", original, step=1, best_loss=None, run_identity=identity,
                        optimizer=optimizer.state_dict(), scaler={}, torch_rng=torch.get_rng_state(),
                        cuda_rng=[], python_rng=random.getstate())
        args = parse_args(["--config", str(config_path), "--tokens-dir", str(self.root),
                           "--tokenizer", str(self.root / "manual-tokenizer.json"), "--output-dir", str(output),
                           "--stage", "pretrain", "--gpu-profile", "rtx-pro-6000", "--max-steps", "3",
                           "--compile-model"])
        tensor_to = torch.Tensor.to
        adamw = torch.optim.AdamW

        def cpu_tensor_to(tensor, *values, **keywords):
            def cpu_device(value):
                if isinstance(value, (str, torch.device)) and str(value).startswith("cuda"):
                    return "cpu"
                return value
            return tensor_to(tensor, *(cpu_device(value) for value in values),
                             **{key: cpu_device(value) if key == "device" else value
                                for key, value in keywords.items()})

        def cpu_optimizer(*values, **keywords):
            return adamw(*values, **{**keywords, "fused": False})

        # Exercise the production control flow with real tiny-model gradients
        # and files, replacing CUDA operations with CPU-only equivalents.
        with ExitStack() as stack:
            replacements = [
                patch("native_600m.train.require_cuda", return_value=torch),
                patch("native_600m.tokenizer.NativeTokenizer.load",
                      return_value=SimpleNamespace(vocab_size=263, fingerprint="manual-fixture")),
                patch("native_600m.tokenize.validate_tokenized", return_value=manifest),
                patch("torch.cuda.get_device_name", return_value="NVIDIA RTX PRO 6000"),
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=96 * 1024**3)),
                patch("torch.cuda.get_device_capability", return_value=(12, 0)),
                patch("torch.cuda.is_bf16_supported", return_value=True),
                patch("torch.cuda.synchronize"), patch("torch.cuda.max_memory_allocated", return_value=0),
                patch("torch.cuda.max_memory_reserved", return_value=0),
                patch("torch.cuda.manual_seed_all"), patch("torch.cuda.get_rng_state_all", return_value=[]),
                patch("torch.cuda.set_rng_state_all"), patch("torch.Tensor.to", new=cpu_tensor_to),
                patch("torch.optim.AdamW", new=cpu_optimizer),
                patch("torch.autocast", side_effect=lambda *args, **kwargs: nullcontext()),
            ]
            for replacement in replacements:
                stack.enter_context(replacement)
            compiler = stack.enter_context(patch.object(NativeLM, "compile", autospec=True))
            saved = stack.enter_context(patch("native_600m.train.save_checkpoint", wraps=save_checkpoint))
            train(args)
        self.assertEqual(compiler.call_count, 1)
        self.assertEqual(compiler.call_args.kwargs, {"backend": "inductor", "mode": "default", "dynamic": False})
        last_saves = [call.kwargs["step"] for call in saved.call_args_list if call.args[0].name == "last.pt"]
        self.assertEqual(last_saves, [2, 3])
        last = read_checkpoint(output / "last.pt")
        self.assertEqual(last["step"], 3)
        self.assertEqual(last["run_identity"], identity)
        self.assertTrue(last["compile_model"])
        self.assertEqual(last["execution_options"]["batch_size"], 8)
        self.assertEqual(last["execution_options"]["grad_accumulation"], 8)
        self.assertEqual(last["validation_window_budget"], 32)
        self.assertTrue(all(int(state["step"]) == 3 for state in last["optimizer"]["state"].values()))
        completed = json.loads((output / "completed.json").read_text(encoding="utf-8"))
        self.assertEqual(completed["identity"], identity)
        for name in ("best.pt", "last.pt"):
            self.assertEqual(completed["hashes"][name], sha256_file(output / name))

    def test_first_resumed_step_and_five_minute_boundary_are_saved_before_step_one_thousand(self):
        settings = {"start_step": 130, "max_steps": 91553, "interval_steps": 1000,
                    "elapsed_seconds": 0, "interval_seconds": 300}
        self.assertTrue(checkpoint_due(131, **settings))
        self.assertFalse(checkpoint_due(132, **settings))
        self.assertFalse(checkpoint_due(176, **{**settings, "elapsed_seconds": 299.9}))
        self.assertTrue(checkpoint_due(176, **{**settings, "elapsed_seconds": 300}))
        self.assertTrue(checkpoint_due(1000, **settings))
        self.assertTrue(checkpoint_due(132, **settings, evaluated=True))
        self.assertTrue(checkpoint_due(91553, **settings))

    def test_checkpoint_timer_is_runtime_only_and_rejects_invalid_intervals(self):
        config = load_config()
        original = stage_options(config, parse_args(["--stage", "pretrain", "--output-dir", "unused"]))
        faster = stage_options(config, parse_args(["--stage", "pretrain", "--output-dir", "unused",
                                                   "--checkpoint-seconds", "60"]))
        self.assertEqual(original, faster)
        for interval in ("0", "-1", "nan", "inf"):
            with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, "checkpoint-seconds"):
                stage_options(config, parse_args(["--stage", "pretrain", "--output-dir", "unused",
                                                  f"--checkpoint-seconds={interval}"]))


if __name__ == "__main__":
    unittest.main()
