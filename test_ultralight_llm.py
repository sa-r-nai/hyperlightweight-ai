from __future__ import annotations

import unittest

import torch

from ultralight_llm import (
    ByteTokenizer,
    GenerationConfig,
    ModelConfig,
    TinyCausalLM,
    generate_text,
    sample_next_token,
)


class CountingTokenizer(ByteTokenizer):
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, text: str, **kwargs: bool) -> list[int]:
        self.encode_calls += 1
        return super().encode(text, **kwargs)


class ScriptedTinyCausalLM(TinyCausalLM):
    """Tiny model that emits predetermined tokens for inference-loop tests."""

    def __init__(self, token_steps: list[list[int]]) -> None:
        config = ModelConfig(
            max_seq_len=64,
            d_model=16,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ff=32,
        )
        super().__init__(config)
        self.token_steps = token_steps
        self.forward_calls = 0
        self.context_lengths: list[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if targets is not None:
            raise AssertionError("Scripted model is inference-only.")

        self.context_lengths.append(input_ids.size(1))
        step = min(self.forward_calls, len(self.token_steps) - 1)
        selected = self.token_steps[step]
        self.forward_calls += 1

        if len(selected) != input_ids.size(0):
            raise AssertionError("Each scripted step needs one token per batch item.")

        logits = torch.full(
            (input_ids.size(0), input_ids.size(1), self.config.vocab_size),
            -1e9,
            device=input_ids.device,
        )
        for batch_index, token_id in enumerate(selected):
            logits[batch_index, -1, token_id] = 0.0
        return logits


class InferenceFlowTests(unittest.TestCase):
    def test_prompt_is_tokenized_once_then_tokens_are_appended_until_eos(self) -> None:
        tokenizer = CountingTokenizer()
        token_a = tokenizer.byte_offset + ord("A")
        token_b = tokenizer.byte_offset + ord("B")
        model = ScriptedTinyCausalLM(
            [[token_a], [token_b], [tokenizer.eos_token_id]]
        )

        response = generate_text(
            model,
            tokenizer,
            "x",
            generation_config=GenerationConfig(
                max_new_tokens=10,
                temperature=0.0,
                top_k=None,
                top_p=None,
            ),
        )

        self.assertEqual(response, "AB")
        self.assertEqual(tokenizer.encode_calls, 1)
        self.assertEqual(model.forward_calls, 3)
        self.assertEqual(model.context_lengths, [2, 3, 4])

    def test_finished_batch_item_stays_finished_while_others_continue(self) -> None:
        eos = ByteTokenizer.eos_token_id
        token_a = ByteTokenizer.byte_offset + ord("A")
        token_b = ByteTokenizer.byte_offset + ord("B")
        model = ScriptedTinyCausalLM([[eos, token_a], [token_b, eos]])
        input_ids = torch.tensor(
            [
                [ByteTokenizer.bos_token_id],
                [ByteTokenizer.bos_token_id],
            ]
        )

        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            temperature=0.0,
            top_k=None,
            top_p=None,
            eos_token_id=eos,
        )

        self.assertEqual(generated[:, 1:].tolist(), [[eos, eos], [token_a, eos]])
        self.assertEqual(model.forward_calls, 2)

    def test_top_p_includes_the_token_that_crosses_the_threshold(self) -> None:
        torch.manual_seed(0)
        logits = torch.log(torch.tensor([[0.45, 0.35, 0.20]])).repeat(4096, 1)

        sampled = sample_next_token(
            logits,
            temperature=1.0,
            top_k=None,
            top_p=0.5,
        )

        sampled_ids = set(sampled.tolist())
        self.assertEqual(sampled_ids, {0, 1})


if __name__ == "__main__":
    unittest.main()
