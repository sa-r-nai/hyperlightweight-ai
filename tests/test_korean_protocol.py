"""Offline tests: no model loading, forward/backward pass, or optimizer steps."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from korean_600m.config import load_config, sha256_file, write_json
from korean_600m.protocol import AssistantCollator, build_prompt, encode_conversation
from korean_600m.runtime import require_studio
from korean_600m.train import resolve_resume, verify_completed_run
from run_qwen_600m_legacy import build_commands, parse_args


class CharacterTokenizer:
    eos_token_id = 3

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and enable_thinking is False
        text = "".join(f"<{m['role']}>{m['content']}\n" for m in messages)
        return self.encode(text + "<assistant><think>\n\n</think>\n\n")


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.messages = [{"role": "system", "content": "한국어 도우미"},
                         {"role": "user", "content": "내 이름은 민지야"},
                         {"role": "assistant", "content": "반가워요, 민지님."},
                         {"role": "user", "content": "내 이름을 기억해?"},
                         {"role": "assistant", "content": "민지님이에요."}]

    def test_every_assistant_turn_has_masked_prompt_and_supervised_eos(self):
        rows = encode_conversation(self.tokenizer, self.messages, 1024)
        self.assertEqual(len(rows), 2)
        for row, turn in zip(rows, (2, 4)):
            expected = self.tokenizer.encode(self.messages[turn]["content"]) + [3]
            self.assertEqual([token for token in row["labels"] if token != -100], expected)
            prompt = build_prompt(self.tokenizer, self.messages[:turn], 1024)
            self.assertEqual(row["input_ids"], prompt + expected)
            self.assertTrue(all(label == -100 for label in row["labels"][:len(prompt)]))

    def test_context_removes_complete_pairs_without_cutting_latest_turn(self):
        minimal = build_prompt(self.tokenizer, [self.messages[0], self.messages[3]], 1024)
        result = build_prompt(self.tokenizer, self.messages[:4], len(minimal))
        self.assertEqual(result, minimal)
        with self.assertRaises(ValueError):
            build_prompt(self.tokenizer, self.messages[:2], 4)

    def test_oversized_responses_are_skipped_instead_of_teaching_false_eos(self):
        messages = self.messages[:2] + [{"role": "assistant", "content": "가" * 500}]
        self.assertEqual(encode_conversation(self.tokenizer, messages, 128), [])

    def test_padding_masks_labels_and_attention(self):
        rows = encode_conversation(self.tokenizer, self.messages, 1024)
        batch = AssistantCollator(0)(rows)
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        for index, row in enumerate(rows):
            self.assertTrue((batch["labels"][index, len(row["input_ids"]):] == -100).all())
            self.assertTrue((batch["attention_mask"][index, len(row["input_ids"]):] == 0).all())

    def test_training_is_blocked_outside_studio(self):
        with patch("korean_600m.runtime.Path.is_dir", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Local training is disabled"):
                require_studio()

    def test_resume_selects_last_complete_checkpoint_and_does_not_use_old_pt(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root / "best.pt").touch()
            self.assertIsNone(resolve_resume(root, "auto"))
            for step in (2, 4):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json",
                             "adapter_model.safetensors", "rng_state.pth"):
                    (checkpoint / name).touch()
            (root / "checkpoint-4" / "optimizer.pt").unlink()
            self.assertEqual(resolve_resume(root, "auto"), root / "checkpoint-2")
            with self.assertRaises(ValueError):
                resolve_resume(root, "auto", require_scaler=True)
            (root / "checkpoint-2" / "scaler.pt").touch()
            self.assertEqual(resolve_resume(root, "auto", require_scaler=True), root / "checkpoint-2")
            with self.assertRaises(ValueError):
                resolve_resume(root, "none")

    def test_plan_always_uses_cuda_and_new_artifact_paths(self):
        args = parse_args(["--plan"])
        commands = build_commands(args)
        self.assertEqual(len(commands), 5)
        self.assertIn("--base-model", commands[1])
        self.assertIn("--allow-fail", commands[1])
        self.assertNotIn("--allow-fail", commands[-1])
        self.assertIn("korean_600m.train", commands[2])
        self.assertTrue(all("native_200m" not in " ".join(command) for command in commands))

    def test_generated_commands_match_every_entry_point_parser(self):
        from korean_600m import evaluate, export, prepare, train

        parsers = {"korean_600m.prepare": prepare.parse_args,
                   "korean_600m.evaluate": evaluate.parse_args,
                   "korean_600m.train": train.parse_args,
                   "korean_600m.export": export.parse_args}
        for command in build_commands(parse_args(["--plan", "--reuse-prepared"])):
            with self.subTest(module=command[2]):
                parsers[command[2]](command[3:])

    def test_completed_run_is_reused_only_when_adapter_files_match(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            marker = root / "completed.json"
            identity = {"configuration": "test"}
            self.assertFalse(verify_completed_run(marker, identity, adapter))
            names = ("adapter_model.safetensors", "adapter_config.json", "korean_600m_provenance.json",
                     "tokenizer.json", "tokenizer_config.json")
            for name in names:
                (adapter / name).write_text("fixture", encoding="utf-8")
            write_json(marker, {"identity": identity, "adapter_dir": str(adapter.resolve()),
                                "adapter_files": {name: sha256_file(adapter / name) for name in names}})
            self.assertTrue(verify_completed_run(marker, identity, adapter))
            with self.assertRaises(ValueError):
                verify_completed_run(marker, {"configuration": "changed"}, adapter)
            (adapter / "adapter_model.safetensors").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_completed_run(marker, identity, adapter)

    def test_config_rejects_unpinned_model_and_invalid_intervals(self):
        original = load_config()
        for updates in ({"model_revision": "main"}, {"max_seq_length": 0},
                        {"training": {**original["training"], "save_steps": 201}}):
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps({**original, **updates}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
