"""Architecture checks using meta tensors and small inference-only examples."""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from native_600m.model import ModelConfig, NativeLM, RMSNorm, RotaryEmbedding, parameter_count


def tiny_config(**overrides) -> ModelConfig:
    return replace(ModelConfig(), vocab_size=19, max_seq_len=16, d_model=32,
                   n_layers=2, n_heads=4, n_kv_heads=2, d_ff=48, **overrides)


class ModelConfigurationTests(unittest.TestCase):
    def test_default_parameter_count_matches_meta_model(self):
        config = ModelConfig()
        self.assertEqual(parameter_count(config), 596_788_480)
        with torch.device("meta"):
            model = NativeLM(config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), parameter_count(config))
        self.assertTrue(all(parameter.device.type == "meta" for parameter in model.parameters()))
        self.assertIs(model.token_embedding.weight, model.lm_head.weight)

    def test_untied_parameter_count_matches_tiny_model(self):
        config = tiny_config(tie_embeddings=False)
        with torch.device("meta"):
            model = NativeLM(config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), parameter_count(config))
        self.assertIsNot(model.token_embedding.weight, model.lm_head.weight)

    def test_rejects_invalid_configuration(self):
        invalid = [dict(vocab_size=0), dict(max_seq_len=True), dict(d_model=31),
                   dict(n_layers=-1), dict(n_heads=3), dict(n_kv_heads=3),
                   dict(d_model=12, n_heads=4), dict(d_ff=0), dict(dropout=1),
                   dict(dropout=float("nan")), dict(rope_theta=1), dict(norm_eps=0),
                   dict(init_std=float("inf")), dict(tie_embeddings=1)]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(tiny_config(), **overrides)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = NativeLM(tiny_config()).eval()

    @torch.inference_mode()
    def test_logits_have_expected_shape_and_finite_values(self):
        logits, loss = self.model(torch.tensor([[1, 7, 9], [3, 4, 6]]))
        self.assertEqual(logits.shape, (2, 3, 19))
        self.assertIsNone(loss)
        self.assertTrue(torch.isfinite(logits).all())
        attention = self.model.layers[0].attention
        self.assertEqual(attention.q_proj.weight.shape, (32, 32))
        self.assertEqual(attention.k_proj.weight.shape, (16, 32))
        self.assertEqual(attention.v_proj.weight.shape, (16, 32))

    @torch.inference_mode()
    def test_future_tokens_cannot_change_past_predictions(self):
        first, _ = self.model(torch.tensor([[1, 5, 7, 8, 9]]))
        second, _ = self.model(torch.tensor([[1, 5, 7, 11, 12]]))
        torch.testing.assert_close(first[:, :3], second[:, :3], rtol=1e-5, atol=1e-6)
        prefix, _ = self.model(torch.tensor([[1, 5, 7]]))
        torch.testing.assert_close(first[:, :3], prefix, rtol=1e-5, atol=1e-6)

    @torch.inference_mode()
    def test_labels_are_aligned_and_ignored_tokens_do_not_enter_denominator(self):
        inputs = torch.tensor([[1, 3, 5, 7]])
        labels = torch.tensor([[3, -100, 7, 2]])
        logits, mean_loss = self.model(inputs, labels)
        expected_sum = F.cross_entropy(logits.reshape(-1, 19), labels.reshape(-1),
                                       ignore_index=-100, reduction="sum")
        _, sum_loss = self.model(inputs, labels, loss_reduction="sum")
        torch.testing.assert_close(sum_loss, expected_sum)
        torch.testing.assert_close(mean_loss, expected_sum / 3)

    @torch.inference_mode()
    def test_all_ignored_labels_return_finite_zero(self):
        inputs = torch.tensor([[1, 3, 5]])
        for reduction in ("mean", "sum"):
            _, loss = self.model(inputs, torch.full_like(inputs, -100), loss_reduction=reduction)
            self.assertEqual(loss.item(), 0)
            self.assertTrue(torch.isfinite(loss))

    @torch.inference_mode()
    def test_rope_is_identity_at_position_zero_and_preserves_norm(self):
        rope = RotaryEmbedding(8, 10_000)
        query = torch.randn(2, 4, 5, 8)
        key = torch.randn(2, 2, 5, 8)
        rotated_query, rotated_key = rope(query, key)
        torch.testing.assert_close(rotated_query[:, :, 0], query[:, :, 0])
        torch.testing.assert_close(rotated_key[:, :, 0], key[:, :, 0])
        torch.testing.assert_close(rotated_query.square().sum(-1), query.square().sum(-1))
        torch.testing.assert_close(rotated_key.square().sum(-1), key.square().sum(-1))
        self.assertFalse(torch.allclose(rotated_query[:, :, 1:], query[:, :, 1:]))

    @torch.inference_mode()
    def test_rms_norm_preserves_input_dtype(self):
        norm = RMSNorm(8, 1e-5)
        inputs = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        result = norm(inputs)
        self.assertEqual(result.dtype, inputs.dtype)
        self.assertTrue(torch.isfinite(result).all())

    def test_invalid_input_and_labels_are_rejected(self):
        for inputs in (torch.ones(3, dtype=torch.long), torch.ones(1, 0, dtype=torch.long),
                       torch.ones(0, 3, dtype=torch.long), torch.ones(1, 17, dtype=torch.long),
                       torch.ones(1, 3)):
            with self.subTest(shape=inputs.shape), self.assertRaises(ValueError):
                self.model(inputs)
        inputs = torch.tensor([[1, 3]])
        with self.assertRaises(ValueError):
            self.model(inputs, torch.tensor([[1]]))
        with self.assertRaises(ValueError):
            self.model(inputs, inputs.float())
        with self.assertRaises(ValueError):
            self.model(inputs, inputs, loss_reduction="invalid")

    def test_gradient_checkpointing_can_be_enabled_without_training(self):
        self.assertFalse(self.model.gradient_checkpointing)
        self.model.enable_gradient_checkpointing()
        self.assertTrue(self.model.gradient_checkpointing)
        with torch.inference_mode():
            logits, _ = self.model(torch.tensor([[1, 3]]))
        self.assertEqual(logits.shape, (1, 2, 19))


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.model = NativeLM(tiny_config())
        self.modes = []

    def scripted(self, next_tokens):
        def forward(inputs, labels=None, *, loss_reduction="mean"):
            self.modes.append(self.model.training)
            logits = torch.full((inputs.size(0), inputs.size(1), 19), -100.0)
            tokens = next(next_tokens)
            for row, token in enumerate(tokens):
                logits[row, -1, token] = 100.0
            return logits, None
        return forward

    def test_stops_on_eos_and_restores_previous_model_mode(self):
        scripted = self.scripted(iter([[7], [2], [9]]))
        with patch.object(self.model, "forward", side_effect=scripted) as forward:
            result = self.model.generate(torch.tensor([[1, 5]]), temperature=0, max_new_tokens=6)
        self.assertEqual(result.tolist(), [[1, 5, 7, 2]])
        self.assertEqual(forward.call_count, 2)
        self.assertEqual(self.modes, [False, False])
        self.assertTrue(self.model.training)

    def test_batch_rows_remain_ended_while_other_rows_finish(self):
        scripted = self.scripted(iter([[2, 7], [9, 2]]))
        with patch.object(self.model, "forward", side_effect=scripted):
            result = self.model.generate(torch.tensor([[1], [1]]), temperature=0)
        self.assertEqual(result.tolist(), [[1, 2, 2], [1, 7, 2]])

    def test_multiple_stop_ids_end_on_turn_boundary(self):
        scripted = self.scripted(iter([[7], [6], [9]]))
        with patch.object(self.model, "forward", side_effect=scripted) as forward:
            result = self.model.generate(torch.tensor([[1]]), temperature=0,
                                         eos_token_id=(6, 2))
        self.assertEqual(result.tolist(), [[1, 7, 6]])
        self.assertEqual(forward.call_count, 2)

    def test_stops_at_context_limit(self):
        scripted = self.scripted(iter([[7], [8], [9]]))
        prompt = torch.ones(1, 14, dtype=torch.long)
        with patch.object(self.model, "forward", side_effect=scripted) as forward:
            result = self.model.generate(prompt, temperature=0, max_new_tokens=10)
        self.assertEqual(result.shape, (1, 16))
        self.assertEqual(forward.call_count, 2)

    def test_zero_budget_and_full_context_do_not_call_forward(self):
        for prompt, budget in ((torch.tensor([[1]]), 0), (torch.ones(1, 16, dtype=torch.long), 7)):
            with patch.object(self.model, "forward") as forward:
                result = self.model.generate(prompt, max_new_tokens=budget)
            torch.testing.assert_close(result, prompt)
            forward.assert_not_called()

    def test_forbidden_ids_are_optional_and_honored(self):
        scripted = self.scripted(iter([[7]]))
        with patch.object(self.model, "forward", side_effect=scripted):
            result = self.model.generate(torch.tensor([[1]]), temperature=0,
                                         max_new_tokens=1, forbidden_token_ids=[7])
        self.assertNotEqual(result[0, -1].item(), 7)

    def test_top_k_and_nucleus_sampling_keep_scripted_top_token(self):
        scripted = self.scripted(iter([[7], [2]]))
        with patch.object(self.model, "forward", side_effect=scripted):
            result = self.model.generate(torch.tensor([[1]]), top_k=1, top_p=0.5)
        self.assertEqual(result.tolist(), [[1, 7, 2]])

    def test_invalid_sampling_options_fail_before_forward(self):
        invalid = [dict(max_new_tokens=-1), dict(temperature=-1), dict(temperature=float("nan")),
                   dict(top_k=-1), dict(top_p=0), dict(top_p=1.1), dict(repetition_penalty=0),
                   dict(eos_token_id=19), dict(forbidden_token_ids=[-1]),
                   dict(forbidden_token_ids=list(range(19)))]
        for options in invalid:
            with self.subTest(options=options), patch.object(self.model, "forward") as forward:
                with self.assertRaises(ValueError):
                    self.model.generate(torch.tensor([[1]]), **options)
                forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
