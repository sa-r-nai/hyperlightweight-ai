from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from generate_native_data import build_records, validate_records
from generate_native_corpus import build_pretraining_records
from train_native_tokenizer import extract_text


class NativeDataTests(unittest.TestCase):
    def test_generated_corpus_is_deterministic_unique_and_ascii(self) -> None:
        first = build_pretraining_records(target_count=500, seed=42)
        second = build_pretraining_records(target_count=500, seed=42)

        self.assertEqual(first, second)
        texts = [record["text"] for record in first]
        self.assertEqual(len(texts), 500)
        self.assertEqual(len(set(texts)), 500)
        self.assertTrue(all(text.isascii() for text in texts))

    def test_requested_count_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            build_pretraining_records(target_count=0)
        with self.assertRaises(ValueError):
            build_pretraining_records(target_count=100_000)

    def test_chat_corpus_is_unique_ascii_and_contains_multi_turn_dialogue(self) -> None:
        records = build_records(target_count=1000, seed=42)
        validate_records(records)
        conversation_keys = [
            tuple((message["role"], message["content"]) for message in record["messages"])
            for record in records
        ]

        self.assertEqual(len(records), 1000)
        self.assertEqual(len(set(conversation_keys)), 1000)
        self.assertGreater(
            sum(len(record["messages"]) > 3 for record in records),
            200,
        )
        self.assertTrue(
            all(
                message["content"].isascii()
                for record in records
                for message in record["messages"]
            )
        )

    def test_tokenizer_corpus_excludes_chat_metadata(self) -> None:
        record = build_records(target_count=30, seed=7)[-1]
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "chat.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="ascii")
            extracted = extract_text(path)

        self.assertIn(record["messages"][1]["content"], extracted)
        self.assertNotIn(record["id"], extracted)
        self.assertNotIn(record["source"], extracted)


if __name__ == "__main__":
    unittest.main()
