from __future__ import annotations

import json
import tempfile
import unittest
from array import array
from pathlib import Path

from native_tokenizer import NativeTokenizer
from prepare_public_data import (
    DEFAULT_SYSTEM_PROMPT,
    normalize_ascii_text,
    normalize_messages,
    write_chat_stream,
    write_pretraining_stream,
)
from tokenize_native_data import tokenize_to_binary


class PublicDataTests(unittest.TestCase):
    def test_ascii_normalization_handles_common_english_punctuation(self) -> None:
        value = normalize_ascii_text("Caf\u00e9 \u201cnotes\u201d \u2014 useful\u2026")
        self.assertEqual(value, 'Cafe "notes" - useful...')
        self.assertIsNone(normalize_ascii_text("\ud55c\uae00 \ubb38\uc7a5\ub9cc \uc788\uc2b5\ub2c8\ub2e4"))

    def test_chat_normalization_adds_system_and_rejects_bad_roles(self) -> None:
        messages = normalize_messages(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello!"},
            ],
            max_characters=1000,
        )
        self.assertIsNotNone(messages)
        assert messages is not None
        self.assertEqual(messages[0], {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
        self.assertIsNone(
            normalize_messages(
                [
                    {"role": "user", "content": "One"},
                    {"role": "user", "content": "Two"},
                ],
                max_characters=1000,
            )
        )

    def test_stream_writers_split_deduplicate_and_hash_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrain_stats = write_pretraining_stream(
                [
                    {"text": "A sufficiently long first English training document."},
                    {"text": "A sufficiently long first English training document."},
                    {"text": "A sufficiently long second English training document."},
                    {"text": "A sufficiently long third English training document."},
                ],
                train_path=root / "pretrain_train.jsonl",
                validation_path=root / "pretrain_validation.jsonl",
                train_count=2,
                validation_count=1,
                min_characters=20,
                max_characters=1000,
            )
            self.assertEqual(pretrain_stats["rejected_records"], 1)
            self.assertEqual(pretrain_stats["train_records"], 2)

            chat_stats = write_chat_stream(
                [
                    {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
                    {"messages": [{"role": "user", "content": "Plan this"}, {"role": "assistant", "content": "First, define the goal."}]},
                ],
                train_path=root / "chat_train.jsonl",
                validation_path=root / "chat_validation.jsonl",
                train_count=1,
                validation_count=1,
                max_characters=1000,
            )
            self.assertEqual(chat_stats["train_records"], 1)
            self.assertEqual(chat_stats["validation_records"], 1)
            for path in root.glob("*.jsonl"):
                for line in path.read_text(encoding="ascii").splitlines():
                    json.loads(line)

    def test_binary_chat_tokenization_writes_assistant_mask(self) -> None:
        tokenizer = NativeTokenizer()
        documents = [
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello!"},
            ]
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chat.json"
            manifest = tokenize_to_binary(
                documents,
                tokenizer,
                output_manifest=output,
                assistant_only_loss=True,
            )
            token_path = output.with_name("chat.tokens.bin")
            mask_path = output.with_name("chat.mask.bin")
            self.assertEqual(token_path.stat().st_size, int(manifest["token_count"]) * 2)
            self.assertEqual(mask_path.stat().st_size, int(manifest["token_count"]))
            masks = array("B")
            with mask_path.open("rb") as handle:
                masks.fromfile(handle, int(manifest["token_count"]))
            self.assertEqual(sum(masks), manifest["supervised_token_count"])
            self.assertGreater(sum(masks), 0)
            self.assertLess(sum(masks), len(masks))


if __name__ == "__main__":
    unittest.main()
