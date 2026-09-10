from __future__ import annotations

import unittest

import torch

from native_200m import Native200MConfig, NativeCausalLM, SMOKE_CONFIG, estimate_parameter_count
from native_tokenizer import NativeTokenizer
from train_native_tokenizer import train_bpe


class NativeTokenizerTests(unittest.TestCase):
    def test_english_round_trip_and_bpe_compression(self) -> None:
        text = "English language models should tokenize repeated English words."
        tokenizer = train_bpe([text] * 20, target_vocab_size=256, min_frequency=2)
        token_ids = tokenizer.encode(text, normalize=False)
        self.assertEqual(tokenizer.decode(token_ids), text)
        self.assertLess(len(token_ids), len(text.encode("ascii")))

    def test_non_ascii_input_is_rejected(self) -> None:
        tokenizer = NativeTokenizer()
        with self.assertRaises(ValueError):
            tokenizer.encode("Non-ASCII input: \u00e9")

    def test_chat_protocol_has_no_external_template(self) -> None:
        tokenizer = NativeTokenizer()
        messages = [
            {"role": "system", "content": "Answer in English."},
            {"role": "user", "content": "What is a cache?"},
        ]
        prompt = tokenizer.encode_generation_prompt(messages)
        self.assertEqual(prompt[0], tokenizer.bos_token_id)
        self.assertEqual(prompt[-1], tokenizer.assistant_token_id)
        self.assertIn(tokenizer.system_token_id, prompt)
        self.assertIn(tokenizer.user_token_id, prompt)


class NativeModelTests(unittest.TestCase):
    def test_default_configuration_is_near_200m(self) -> None:
        count = estimate_parameter_count(Native200MConfig())
        self.assertGreater(count, 190_000_000)
        self.assertLess(count, 215_000_000)

    def test_causal_forward_and_loss(self) -> None:
        torch.manual_seed(7)
        model = NativeCausalLM(SMOKE_CONFIG)
        model.eval()
        input_ids = torch.randint(
            0,
            SMOKE_CONFIG.vocab_size,
            (2, 20),
            dtype=torch.long,
        )
        logits, loss = model(input_ids, input_ids)
        self.assertEqual(tuple(logits.shape), (2, 20, SMOKE_CONFIG.vocab_size))
        self.assertTrue(torch.isfinite(loss))

    def test_future_tokens_do_not_change_previous_logits(self) -> None:
        torch.manual_seed(11)
        model = NativeCausalLM(SMOKE_CONFIG)
        model.eval()
        prefix = torch.randint(0, SMOKE_CONFIG.vocab_size, (1, 8))
        first = torch.cat((prefix, torch.zeros((1, 3), dtype=torch.long)), dim=1)
        second = torch.cat((prefix, torch.ones((1, 3), dtype=torch.long)), dim=1)
        first_logits = model(first)
        second_logits = model(second)
        self.assertTrue(torch.allclose(first_logits[:, :8], second_logits[:, :8], atol=1e-5))

    def test_generation_keeps_batch_and_appends_tokens(self) -> None:
        torch.manual_seed(13)
        model = NativeCausalLM(SMOKE_CONFIG)
        prompt = torch.randint(0, SMOKE_CONFIG.vocab_size, (2, 7))
        result = model.generate(
            prompt,
            max_new_tokens=4,
            temperature=0.0,
            top_k=None,
            top_p=None,
        )
        self.assertEqual(tuple(result.shape), (2, 11))

    def test_generation_respects_english_output_tokens(self) -> None:
        torch.manual_seed(17)
        tokenizer = NativeTokenizer()
        model = NativeCausalLM(SMOKE_CONFIG)
        prompt = torch.tensor(
            [[1, 5, tokenizer.byte_offset + ord("H"), tokenizer.byte_offset + ord("i"), 6]]
        )
        allowed = tokenizer.english_output_token_ids()
        result = model.generate(
            prompt,
            max_new_tokens=8,
            temperature=0.0,
            top_k=None,
            top_p=None,
            allowed_token_ids=allowed,
        )
        generated = set(result[0, prompt.size(1) :].tolist())
        self.assertTrue(generated <= {tokenizer.eos_token_id, *allowed})


if __name__ == "__main__":
    unittest.main()
