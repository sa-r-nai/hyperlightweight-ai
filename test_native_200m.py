from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from native_200m import Native200MConfig, NativeCausalLM, SMOKE_CONFIG, estimate_parameter_count
from native_tokenizer import NativeTokenizer
from train_native_200m import PackedTextDataset, split_documents
from train_native_tokenizer import train_bpe


class ScriptedStopModel(NativeCausalLM):
    def __init__(self, stop_token_id: int) -> None:
        super().__init__(SMOKE_CONFIG)
        self.stop_token_id = stop_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if labels is not None:
            raise AssertionError("ScriptedStopModel is inference-only.")
        logits = torch.full(
            (input_ids.size(0), input_ids.size(1), self.config.vocab_size),
            -1e9,
            device=input_ids.device,
        )
        logits[:, -1, self.stop_token_id] = 0.0
        return logits


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

    def test_assistant_loss_mask_excludes_prompt_tokens(self) -> None:
        tokenizer = NativeTokenizer()
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]
        tokens, loss_mask = tokenizer.encode_chat_with_assistant_mask(messages)
        assistant_position = tokens.index(tokenizer.assistant_token_id)

        self.assertEqual(len(tokens), len(loss_mask))
        self.assertFalse(any(loss_mask[: assistant_position + 1]))
        self.assertTrue(all(loss_mask[assistant_position + 1 :]))


class NativeModelTests(unittest.TestCase):
    def test_default_configuration_is_near_200m(self) -> None:
        count = estimate_parameter_count(Native200MConfig())
        self.assertGreater(count, 190_000_000)
        self.assertLess(count, 215_000_000)

    def test_causal_forward_and_loss(self) -> None:
        torch.manual_seed(7)
        model = NativeCausalLM(SMOKE_CONFIG)
        model.eval()
        token_ids = torch.randint(
            0,
            SMOKE_CONFIG.vocab_size,
            (2, 21),
            dtype=torch.long,
        )
        input_ids = token_ids[:, :-1]
        labels = token_ids[:, 1:]
        logits, loss = model(input_ids, labels)
        self.assertEqual(tuple(logits.shape), (2, 20, SMOKE_CONFIG.vocab_size))
        self.assertTrue(torch.isfinite(loss))

    def test_training_targets_are_shifted_exactly_once(self) -> None:
        tokenizer = NativeTokenizer()
        dataset = PackedTextDataset(["abc"], tokenizer, seq_len=3)
        input_ids, labels = dataset[0]
        self.assertEqual(
            input_ids.tolist(),
            [
                tokenizer.bos_token_id,
                tokenizer.byte_offset + ord("a"),
                tokenizer.byte_offset + ord("b"),
            ],
        )
        self.assertEqual(
            labels.tolist(),
            [
                tokenizer.byte_offset + ord("a"),
                tokenizer.byte_offset + ord("b"),
                tokenizer.byte_offset + ord("c"),
            ],
        )

        config = Native200MConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=3,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
        )
        model = NativeCausalLM(config)
        logits, loss = model(input_ids.unsqueeze(0), labels.unsqueeze(0))
        expected = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_assistant_only_dataset_ignores_system_and_user_targets(self) -> None:
        tokenizer = NativeTokenizer()
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi."},
        ]
        encoded, mask = tokenizer.encode_chat_with_assistant_mask(
            messages,
            add_bos=False,
            add_eos=True,
        )
        stream = [tokenizer.bos_token_id, *encoded]
        stream_mask = [False, *mask]
        dataset = PackedTextDataset(
            [messages],
            tokenizer,
            seq_len=len(stream) - 1,
            assistant_only_loss=True,
        )
        input_ids, labels = dataset[0]
        expected_labels = torch.tensor(stream[1:], dtype=torch.long)
        expected_mask = torch.tensor(stream_mask[1:], dtype=torch.bool)
        expected_labels.masked_fill_(~expected_mask, -100)

        self.assertEqual(input_ids.tolist(), stream[:-1])
        self.assertEqual(labels.tolist(), expected_labels.tolist())
        self.assertGreater(int(labels.eq(-100).sum()), 0)
        self.assertGreater(int(labels.ne(-100).sum()), 0)

    def test_document_split_is_deterministic_and_disjoint(self) -> None:
        documents = [f"document-{index}" for index in range(20)]
        first = split_documents(documents, validation_ratio=0.2, seed=42)
        second = split_documents(documents, validation_ratio=0.2, seed=42)
        self.assertEqual(first, second)
        train_documents, validation_documents = first
        self.assertEqual(len(train_documents), 16)
        self.assertEqual(len(validation_documents), 4)
        self.assertTrue(set(train_documents).isdisjoint(validation_documents))

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

    def test_generation_stops_at_chat_turn_end(self) -> None:
        tokenizer = NativeTokenizer()
        model = ScriptedStopModel(tokenizer.turn_end_token_id)
        prompt = torch.tensor([[tokenizer.bos_token_id, tokenizer.user_token_id]])
        result = model.generate(
            prompt,
            max_new_tokens=8,
            temperature=0.0,
            top_k=None,
            top_p=None,
            allowed_token_ids=[tokenizer.byte_offset + ord("A")],
            additional_stop_token_ids=[tokenizer.turn_end_token_id],
        )

        self.assertEqual(result.size(1), prompt.size(1) + 1)
        self.assertEqual(int(result[0, -1]), tokenizer.turn_end_token_id)


if __name__ == "__main__":
    unittest.main()
