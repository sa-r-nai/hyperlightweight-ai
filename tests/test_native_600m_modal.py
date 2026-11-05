"""Offline Modal command checks; no cloud calls, downloads, or model training."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("modal") is not None:
    import modaltest
else:
    modaltest = None


@unittest.skipIf(modaltest is None, "Install requirements-modal.txt to check the Modal launcher.")
class ModalWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.artifacts = Path(temporary.name) / "artifacts"

    def test_preparation_reuses_only_existing_stages(self):
        from native_600m.prepare import parse_args

        commands = modaltest._preparation_commands(self.artifacts)
        self.assertEqual(parse_args(commands[0][3:]).workers, 24)
        self.assertTrue(all("--reuse-prepared" not in command for command in commands))
        for relative in ("data/manifest.json", "tokenizer/tokenizer.json"):
            marker = self.artifacts / relative
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}", encoding="utf-8")
        commands = modaltest._preparation_commands(self.artifacts)
        self.assertEqual(["--reuse-prepared" in command for command in commands], [True, True, False])

    def test_gpu_pipeline_contains_training_export_and_evaluation(self):
        from native_600m import evaluate, export, train
        from native_600m.config import load_config

        self.assertEqual(modaltest.ARTIFACTS_DIR, Path("/training/artifacts_native_300m"))
        commands = modaltest._training_commands(self.artifacts)
        self.assertEqual([command[2] for command in commands],
                         ["native_600m.train", "native_600m.train", "native_600m.export", "native_600m.evaluate"])
        pretrain, sft = [train.parse_args(command[3:]) for command in commands[:2]]
        self.assertEqual((pretrain.stage, sft.stage), ("pretrain", "sft"))
        self.assertEqual((pretrain.gpu_profile, sft.gpu_profile), ("h100", "h100"))
        self.assertEqual(load_config(pretrain.config)["name"], "NativeKoreanLM-300M")
        self.assertEqual((pretrain.batch_size, pretrain.grad_accumulation), (16, 4))
        self.assertEqual((sft.batch_size, sft.grad_accumulation), (16, 2))
        self.assertEqual(sft.init_from, pretrain.output_dir / "best.pt")
        self.assertEqual(export.parse_args(commands[2][3:]).checkpoint, sft.output_dir / "best.pt")
        self.assertEqual(evaluate.parse_args(commands[3][3:]).checkpoint, sft.output_dir / "best.pt")

    def test_smoke_uses_separate_checkpoints_and_only_two_steps(self):
        from native_600m.train import parse_args

        commands = modaltest._training_commands(self.artifacts, smoke=True)
        self.assertEqual(len(commands), 1)
        args = parse_args(commands[0][3:])
        self.assertEqual(args.output_dir, self.artifacts.with_name("artifacts_smoke") / "pretrain")
        self.assertEqual(args.max_steps, 2)
        self.assertEqual((args.seq_len, args.batch_size, args.grad_accumulation), (128, 1, 1))
        self.assertEqual(args.tokens_dir, self.artifacts / "tokens")
        self.assertNotIn("--max-steps", modaltest._training_commands(self.artifacts)[0])

    def test_batch_override_preserves_effective_batch_and_rejects_invalid_values(self):
        from native_600m.train import parse_args

        for batch in (8, 16, 32):
            with self.subTest(batch=batch):
                commands = modaltest._training_commands(self.artifacts, micro_batch_size=batch)
                pretrain, sft = [parse_args(command[3:]) for command in commands[:2]]
                self.assertEqual(pretrain.batch_size * pretrain.grad_accumulation, 64)
                self.assertEqual(sft.batch_size * sft.grad_accumulation, 32)
                self.assertEqual(pretrain.batch_size, batch)
                self.assertEqual(sft.batch_size, batch)
        for batch in (0, -1, 3, 64, True, 8.0):
            with self.subTest(batch=batch), self.assertRaisesRegex(ValueError, "micro-batch-size"):
                modaltest._training_commands(self.artifacts, micro_batch_size=batch)

    def test_compilation_is_opt_in_and_only_forwarded_to_training(self):
        from native_600m.train import parse_args

        self.assertTrue(all("--compile-model" not in command
                            for command in modaltest._training_commands(self.artifacts)))
        commands = modaltest._training_commands(self.artifacts, compile_model=True)
        self.assertTrue(all(parse_args(command[3:]).compile_model for command in commands[:2]))
        self.assertTrue(all("--compile-model" not in command for command in commands[2:]))
        smoke = modaltest._training_commands(self.artifacts, smoke=True, compile_model=True)
        self.assertTrue(parse_args(smoke[0][3:]).compile_model)

    def test_failed_command_commits_saved_results_and_propagates_failure(self):
        command = ["python", "-m", "native_600m.evaluate"]
        with patch("run_native_600m.subprocess.run", side_effect=subprocess.CalledProcessError(1, command)) as run, \
             patch("modaltest.volume") as volume:
            with self.assertRaises(subprocess.CalledProcessError):
                modaltest._run_command(command)
        run.assert_called_once_with(command, cwd=modaltest.REMOTE_ROOT, check=True)
        volume.commit.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
