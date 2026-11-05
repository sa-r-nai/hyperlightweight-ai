"""Offline recovery checks; no Modal calls, downloads, or model training."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from native_600m.config import ROOT, load_config
from native_600m.prepare import prepare_records, public_source
from run_native_600m import run_command


class PreparationExitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.data_dir = self.root / "data"
        self.config_path = self.root / "config.json"
        self.command = [sys.executable, "-m", "native_600m.prepare", "--config", str(self.config_path),
                        "--output-dir", str(self.data_dir)]

    def prepare_fixture(self):
        config = load_config()
        config["data"].update(pretrain_train_documents=5, pretrain_validation_documents=2,
                              chat_train_records=4, chat_validation_records=2, min_text_characters=20)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        documents = [{"text": f"한국어 문서 {index}번입니다. 오늘의 이야기를 함께 읽어 봅니다."} for index in range(100)]
        chats = [{"messages": [{"role": "user", "content": f"오늘의 계획 {index}번을 알려 주세요."},
                               {"role": "assistant", "content": "중요한 일을 먼저 정리하고 쉬는 시간도 정해 보세요."}]}
                 for index in range(100)]
        prepare_records(documents, chats, self.data_dir, config,
                        sources={stage: public_source(stage) for stage in ("pretrain", "chat")},
                        shuffle_buffer_size=5)

    def abort_then_verify(self, command, **options):
        if "--reuse-prepared" not in command:
            raise subprocess.CalledProcessError(-signal.SIGABRT, command)
        # The second process performs real offline artifact validation. The
        # saved original runner avoids recursively calling the patched mock.
        return self.original_run(command, **options, capture_output=True, text=True)

    def test_completed_data_survives_preparation_shutdown_abort(self):
        self.prepare_fixture()
        self.command.extend(["--workers", "2"])
        before = {path.name: path.read_bytes() for path in self.data_dir.iterdir()}
        self.original_run = subprocess.run
        with patch("run_native_600m.subprocess.run", side_effect=self.abort_then_verify) as run:
            run_command(self.command)
        self.assertEqual(run.call_count, 2)
        self.assertIn("--reuse-prepared", run.call_args.args[0])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.data_dir.iterdir()})

    def test_corrupt_completed_data_cannot_pass_recovery(self):
        self.prepare_fixture()
        target = self.data_dir / "pretrain_train.jsonl"
        target.write_text("corrupted\n", encoding="utf-8")
        self.original_run = subprocess.run
        with patch("run_native_600m.subprocess.run", side_effect=self.abort_then_verify) as run:
            with self.assertRaises(subprocess.CalledProcessError) as error:
                run_command(self.command)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(error.exception.returncode, 1)
        self.assertIn("hash mismatch", error.exception.stderr)
        self.assertEqual(target.read_text(encoding="utf-8"), "corrupted\n")

    def test_modal_command_helper_uses_verified_recovery_and_commits(self):
        self.prepare_fixture()
        # Execute only the plain command helper from its AST. Do not import
        # the Modal SDK, instantiate the App, or invoke any Modal function.
        source = ast.parse((ROOT / "modaltest.py").read_text(encoding="utf-8"))
        helper = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_run_command")
        namespace = {"shlex": shlex, "REMOTE_ROOT": ROOT, "volume": Mock()}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "modaltest.py", "exec"), namespace)
        self.original_run = subprocess.run
        with patch("run_native_600m.subprocess.run", side_effect=self.abort_then_verify):
            namespace["_run_command"](self.command)
        namespace["volume"].commit.assert_called_once_with()

    def test_modal_cpu_and_pipeline_budgets_are_twenty_four_without_loading_modal(self):
        from native_600m import prepare, tokenize, tokenizer, train
        from native_600m.config import load_config

        source = ast.parse((ROOT / "modaltest.py").read_text(encoding="utf-8"))
        cpu_budgets = []
        for node in source.body:
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                            and decorator.func.attr == "function"):
                        cpu_budgets.extend(keyword.value.value for keyword in decorator.keywords if keyword.arg == "cpu")
        self.assertEqual(cpu_budgets, [24, 24])
        helper = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_workflow_commands")
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "modaltest.py", "exec"), namespace)
        commands = namespace["_workflow_commands"](self.root / "artifacts")
        for command, parser in zip(commands[:3], (prepare.parse_args, tokenizer.parse_args, tokenize.parse_args)):
            self.assertEqual(parser(command[3:]).workers, 24)
        for command in commands[3:5]:
            args = train.parse_args(command[3:])
            self.assertEqual(args.cpu_workers, 24)
            self.assertEqual(args.gpu_profile, "h100")
            options = train.stage_options(load_config(args.config), args)
            self.assertEqual(options["batch_size"], 8)
            self.assertEqual(options["grad_accumulation"], 8 if args.stage == "pretrain" else 4)
            self.assertFalse(train.checkpoint_activations(args, gpu_name="NVIDIA H100 80GB HBM3",
                                                          gpu_memory_gb=79))

    def test_modal_requests_one_h100_sxm5_and_a_cuda_pytorch_build(self):
        source = ast.parse((ROOT / "modaltest.py").read_text(encoding="utf-8"))
        resources = {}
        for node in source.body:
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                            and decorator.func.attr == "function"):
                        resources[node.name] = {keyword.arg: ast.literal_eval(keyword.value)
                                                for keyword in decorator.keywords
                                                if keyword.arg in ("gpu", "cpu", "memory", "max_containers")}
        self.assertNotIn("gpu", resources["prepare_data"])
        self.assertEqual(resources["train"], {"gpu": "H100!", "cpu": 24,
                                              "memory": 65536, "max_containers": 1})
        image = next(node.value for node in source.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "image" for target in node.targets))
        calls = []
        while isinstance(image, ast.Call) and isinstance(image.func, ast.Attribute):
            calls.append(image)
            image = image.func.value
        calls.reverse()
        torch_install = next(call for call in calls if call.func.attr == "pip_install")
        self.assertEqual([ast.literal_eval(arg) for arg in torch_install.args], ["torch==2.8.0+cu128"])
        self.assertEqual({kw.arg: ast.literal_eval(kw.value) for kw in torch_install.keywords},
                         {"index_url": "https://download.pytorch.org/whl/cu128"})
        requirements = next(call for call in calls if call.func.attr == "pip_install_from_requirements")
        self.assertLess(calls.index(torch_install), calls.index(requirements))

    def test_modal_reuses_prepared_data_and_keeps_smoke_checkpoints_separate_without_sdk(self):
        from native_600m.train import parse_args as train_args

        source = ast.parse((ROOT / "modaltest.py").read_text(encoding="utf-8"))
        helpers = [node for node in source.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("_workflow_commands", "_preparation_commands", "_training_commands")]
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "modaltest.py", "exec"), namespace)
        artifacts = self.root / "artifacts"
        for relative in ("data/manifest.json", "tokenizer/tokenizer.json", "tokens/manifest.json"):
            marker = artifacts / relative
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}", encoding="utf-8")
        self.assertTrue(all("--reuse-prepared" in command
                            for command in namespace["_preparation_commands"](artifacts)))
        commands = namespace["_training_commands"](artifacts)
        self.assertEqual([command[2] for command in commands],
                         ["native_600m.train", "native_600m.train", "native_600m.export", "native_600m.evaluate"])
        pretrain, sft = [train_args(command[3:]) for command in commands[:2]]
        self.assertEqual(sft.init_from, pretrain.output_dir / "best.pt")
        smoke = train_args(namespace["_training_commands"](artifacts, smoke=True)[0][3:])
        self.assertEqual(smoke.output_dir, artifacts.with_name("artifacts_smoke") / "pretrain")
        self.assertEqual((smoke.max_steps, smoke.seq_len, smoke.batch_size, smoke.grad_accumulation), (2, 128, 1, 1))

    def test_missing_manifest_does_not_trigger_verification(self):
        failure = subprocess.CalledProcessError(-signal.SIGABRT, self.command)
        with patch("run_native_600m.subprocess.run", side_effect=failure) as run:
            with self.assertRaises(subprocess.CalledProcessError) as error:
                run_command(self.command)
        self.assertIs(error.exception, failure)
        run.assert_called_once_with(self.command, cwd=ROOT, check=True)

    def test_other_errors_stages_and_verification_aborts_propagate(self):
        self.data_dir.mkdir()
        (self.data_dir / "manifest.json").write_text("{}", encoding="utf-8")
        cases = [
            (self.command, 1), (self.command, -signal.SIGTERM),
            ([sys.executable, "-m", "native_600m.train"], -signal.SIGABRT),
            ([sys.executable, "-m", "native_600m.evaluate"], -signal.SIGABRT),
            ([*self.command, "--reuse-prepared"], -signal.SIGABRT),
        ]
        for command, returncode in cases:
            with self.subTest(command=command, returncode=returncode):
                failure = subprocess.CalledProcessError(returncode, command)
                with patch("run_native_600m.subprocess.run", side_effect=failure) as run:
                    with self.assertRaises(subprocess.CalledProcessError) as error:
                        run_command(command)
                self.assertIs(error.exception, failure)
                self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
