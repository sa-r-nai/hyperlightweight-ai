"""Offline vocabulary/protocol tests using a manual byte alphabet, never training."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from native_600m.chat import generate_reply, parse_args as chat_args
from native_600m.tokenizer import NativeTokenizer, SPECIAL_TOKENS, parse_args, train_tokenizer

try:
    import tokenizers
except ImportError:
    tokenizers = None


@unittest.skipIf(tokenizers is None, "tokenizers is not installed")
class NativeTokenizerTests(unittest.TestCase):
    def setUp(self):
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers

        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "tokenizer.json"
        # Fixed, manually specified fixture: no train() / train_from_iterator().
        vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        for byte_symbol in sorted(pre_tokenizers.ByteLevel.alphabet()):
            vocabulary[byte_symbol] = len(vocabulary)
        backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
        backend.add_special_tokens(list(SPECIAL_TOKENS))
        backend.normalizer = normalizers.NFC()
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.save(str(self.path))
        self.tokenizer = NativeTokenizer.load(self.path)

    def test_unicode_round_trip_with_whitespace_and_unseen_bytes(self):
        text = "  안녕하세요\n한글과 123 🙂\t입력"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode("한글")), "한글")
        self.assertEqual(self.tokenizer.encode(""), [])

    def test_fixed_protocol_ids_and_byte_file_fingerprint(self):
        for expected, token in enumerate(SPECIAL_TOKENS):
            self.assertEqual(self.tokenizer._backend.token_to_id(token), expected)
        self.assertEqual(self.tokenizer.vocab_size, 263)
        self.assertEqual(self.tokenizer.fingerprint, hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_reserved_markers_are_rejected_in_content(self):
        for marker in SPECIAL_TOKENS:
            with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, "reserved"):
                self.tokenizer.encode("인용 " + marker)

    def test_documents_have_explicit_begin_and_end(self):
        self.assertEqual(self.tokenizer.encode_document("한국어"),
                         [1, *self.tokenizer.encode("한국어"), 2])

    def test_all_assistant_turns_and_only_their_content_are_supervised(self):
        messages = [
            {"role": "system", "content": "한국어 도우미"},
            {"role": "user", "content": "안녕"},
            {"role": "assistant", "content": "반가워요"},
            {"role": "user", "content": "뭐 해?"},
            {"role": "assistant", "content": "대화하고 있어요"},
        ]
        ids, mask = self.tokenizer.encode_chat(messages)
        expected = [*self.tokenizer.encode("반가워요"), 6,
                    *self.tokenizer.encode("대화하고 있어요"), 6]
        self.assertEqual([token for token, active in zip(ids, mask) if active], expected)
        self.assertEqual(len(ids), len(mask))
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[-1], 2)
        self.assertEqual(mask[-1], 0)

    def test_prompt_ends_with_assistant_and_has_no_supervised_response(self):
        messages = [{"role": "user", "content": "안녕"}]
        self.assertEqual(self.tokenizer.encode_prompt(messages, 100),
                         [1, 4, *self.tokenizer.encode("안녕"), 6, 5])

    def test_context_trims_complete_old_pairs_preserving_system_and_last_user(self):
        system = {"role": "system", "content": "한국어"}
        last = {"role": "user", "content": "오늘은?"}
        messages = [system, {"role": "user", "content": "어제는?"},
                    {"role": "assistant", "content": "좋았어요"}, last]
        minimal = self.tokenizer.encode_prompt([system, last], 100)
        self.assertEqual(self.tokenizer.encode_prompt(messages, len(minimal)), minimal)
        with self.assertRaisesRegex(ValueError, "context budget"):
            self.tokenizer.encode_prompt(messages, len(minimal) - 1)

    def test_malformed_turn_sequences_are_rejected(self):
        invalid = [[], [{"role": "system", "content": "한국어"}],
                   [{"role": "assistant", "content": "안녕"}],
                   [{"role": "user", "content": " "}],
                   [{"role": "user", "content": "안녕"}, {"role": "user", "content": "누구야"}]]
        for messages in invalid:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                self.tokenizer.encode_prompt(messages, 100)
        with self.assertRaises(ValueError):
            self.tokenizer.encode_chat([{"role": "user", "content": "안녕"}])

    def test_load_rejects_foreign_or_changed_protocol(self):
        definition = json.loads(self.path.read_text(encoding="utf-8"))
        definition["pre_tokenizer"]["add_prefix_space"] = True
        self.path.write_text(json.dumps(definition), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ByteLevel"):
            NativeTokenizer.load(self.path)

    def test_save_and_reload_preserve_ids_and_update_file_fingerprint(self):
        destination = self.path.parent / "saved.json"
        self.tokenizer.save(destination)
        restored = NativeTokenizer.load(destination)
        self.assertEqual(restored.encode("한국어 🙂"), self.tokenizer.encode("한국어 🙂"))
        self.assertEqual(restored.fingerprint, hashlib.sha256(destination.read_bytes()).hexdigest())

    def test_generation_uses_own_stop_tokens_and_cuts_at_turn_end(self):
        import torch

        class FakeModel:
            config = SimpleNamespace(max_seq_len=128, vocab_size=263)

            def parameters(self):
                return iter([torch.zeros(1)])

            def eval(self):
                return self

            def generate(inner, inputs, **kwargs):
                inner.options = kwargs
                reply = self.tokenizer.encode("안녕하세요") + [6] + self.tokenizer.encode("다른 화자")
                return torch.cat([inputs, torch.tensor([reply])], dim=1)

        model = FakeModel()
        reply = generate_reply(model, self.tokenizer, [{"role": "user", "content": "안녕"}], max_new_tokens=32)
        self.assertEqual(reply, "안녕하세요")
        self.assertEqual(model.options["eos_token_id"], (6, 2))
        self.assertEqual(model.options["forbidden_token_ids"], (0, 1, 3, 4, 5))

    def test_generation_rejects_mismatched_vocab_before_model_execution(self):
        model = SimpleNamespace(config=SimpleNamespace(max_seq_len=128, vocab_size=32000))
        with self.assertRaisesRegex(ValueError, "vocabulary sizes"):
            generate_reply(model, self.tokenizer, [{"role": "user", "content": "안녕"}], max_new_tokens=32)


class TokenizerCommandTests(unittest.TestCase):
    def test_training_is_guarded_before_inputs_are_opened_or_training_can_run(self):
        with patch("native_600m.runtime.require_studio", side_effect=RuntimeError("Studio only")) as guard:
            with self.assertRaisesRegex(RuntimeError, "Studio only"):
                train_tokenizer("nonexistent-input.jsonl", "not-written.json")
            guard.assert_called_once_with()

    def test_cli_configuration_is_dependency_free(self):
        args = parse_args(["--input", "pretrain.jsonl"])
        self.assertEqual(args.vocab_size, 32000)
        self.assertEqual(args.max_documents, 200000)
        self.assertEqual(chat_args([]).device, "cuda")
        self.assertEqual(chat_args(["--device", "cpu", "--message", "안녕"]).message, "안녕")


if __name__ == "__main__":
    unittest.main()
