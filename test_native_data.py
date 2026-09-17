from __future__ import annotations

import unittest

from generate_native_corpus import build_pretraining_records


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


if __name__ == "__main__":
    unittest.main()
