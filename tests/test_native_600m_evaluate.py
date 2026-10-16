"""Native evaluation contracts with mocked generation and no model training."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from native_600m.evaluate import (
    EVAL_CASES, GENERATION_DEFAULTS, EvalCase, case_signals, main,
    parse_args, response_sanity, run_evaluation, seeded_generate_reply,
)


class NativeEvaluationTests(unittest.TestCase):
    def test_thirteen_cases_cover_basic_language_and_multiple_turns(self):
        self.assertEqual(len(EVAL_CASES), 13)
        self.assertEqual(len({case.name for case in EVAL_CASES}), 13)
        self.assertGreater(len(next(case for case in EVAL_CASES if case.name == "memory_correction").prompts), 1)
        self.assertTrue(response_sanity("안녕하세요. 반가워요.")["passed"])

    def test_sanity_rejects_empty_wrong_language_repetition_and_native_markers(self):
        examples = {
            "": "empty_response", "hello there": "insufficient_hangul_ratio",
            "가" * 12: "repeated_character", "안녕<|assistant|>": "leaked_control_tokens",
            "안녕\ufffd": "replacement_character",
        }
        for text, reason in examples.items():
            with self.subTest(text=text):
                self.assertIn(reason, response_sanity(text)["reasons"])
        arithmetic = next(case for case in EVAL_CASES if case.name == "arithmetic")
        self.assertTrue(response_sanity("19", arithmetic.min_hangul_ratio)["passed"])
        self.assertTrue(all(case_signals(arithmetic, "19").values()))
        self.assertFalse(case_signals(arithmetic, "19개")["number_only"])

    def test_visible_instruction_checks(self):
        by_name = {case.name: case for case in EVAL_CASES}
        self.assertTrue(all(case_signals(by_name["three_items"], "- 물\n- 모자\n- 휴대전화").values()))
        self.assertFalse(case_signals(by_name["three_items"], "물과 모자")["exactly_three_bullet_lines"])
        self.assertFalse(case_signals(by_name["memory_correction"], "민수 아닌 지연")["old_name_absent"])
        self.assertFalse(case_signals(by_name["short_constraint"], "고마워\n정말")["single_line"])

    def test_seeded_generation_preserves_rng_scope_and_native_options(self):
        events = []

        @contextmanager
        def fork_rng():
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        fake_torch = SimpleNamespace(random=SimpleNamespace(fork_rng=fork_rng),
                                     manual_seed=lambda seed: events.append(seed))

        def reply(*args, **kwargs):
            events.append("generate")
            self.assertEqual(kwargs, {"max_new_tokens": 192, **GENERATION_DEFAULTS})
            return "안녕"

        with patch.dict(sys.modules, {"torch": fake_torch}), patch("native_600m.evaluate.generate_reply", side_effect=reply):
            self.assertEqual(seeded_generate_reply(None, None, [], max_new_tokens=192, seed=17), "안녕")
        self.assertEqual(events, ["enter", 17, "generate", "exit"])

    def test_multiturn_generation_receives_previous_real_responses(self):
        case = EvalCase("memory_correction", ("내 이름은 민수", "지연으로 정정", "내 이름은?"), (("지연",),))
        snapshots = []
        answers = iter(["민수님 반가워요", "지연님으로 기억할게요", "지연"])
        seeds = []

        def reply(model, tokenizer, messages, **options):
            snapshots.append([dict(message) for message in messages])
            seeds.append(options["seed"])
            return next(answers)

        with patch("native_600m.evaluate.EVAL_CASES", (case,)), patch("native_600m.evaluate.seeded_generate_reply", side_effect=reply):
            results = run_evaluation(None, None, {"system_prompt": "한국어", "seed": 42}, 64)
        self.assertTrue(results[0]["passed"])
        self.assertEqual(seeds, [42, 43, 44])
        self.assertEqual([message["role"] for message in snapshots[2]],
                         ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertEqual(snapshots[2][2]["content"], "민수님 반가워요")
        self.assertEqual(snapshots[2][4]["content"], "지연님으로 기억할게요")

    def test_empty_early_response_records_failure_without_corrupting_next_case(self):
        cases = (EvalCase("first", ("첫 질문", "후속 질문")), EvalCase("second", ("다음 사례",)))
        with patch("native_600m.evaluate.EVAL_CASES", cases), patch(
            "native_600m.evaluate.seeded_generate_reply", side_effect=["", "안녕하세요"]
        ) as reply:
            results = run_evaluation(None, None, {"system_prompt": "한국어", "seed": 7}, 64)
        self.assertEqual(reply.call_count, 2)
        self.assertFalse(results[0]["passed"])
        self.assertFalse(results[0]["checks"]["all_prompts_answered"])
        self.assertTrue(results[1]["passed"])
        self.assertEqual(results[1]["turns"][0]["seed"], 1007)

    def test_default_command_uses_native_checkpoint_and_explicit_cpu(self):
        args = parse_args([])
        self.assertEqual(args.checkpoint, Path("artifacts_native_600m/sft/best.pt"))
        self.assertEqual(args.max_new_tokens, 192)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(parse_args(["--device", "cpu"]).device, "cpu")

    def report_fixture(self, directory, *, passed=True, fingerprint="fixture-sha"):
        root = Path(directory)
        checkpoint = root / "fixture.pt"
        checkpoint.write_bytes(b"not model weights; mocked loader")
        report = root / "report.json"
        metadata = {"tokenizer_sha256": fingerprint, "stage": "sft", "config": {
            "vocab_size": 263, "max_seq_len": 128, "d_model": 16,
            "n_layers": 1, "n_heads": 4, "n_kv_heads": 2, "d_ff": 32,
        }}
        results = [{"name": "fixture", "passed": passed, "checks": {},
                    "turns": [{"prompt": "안녕", "response": "안녕하세요" if passed else "",
                               "sanity": {"passed": passed, "reasons": []}}]}]
        return checkpoint, report, metadata, results

    def test_report_counts_checkpoint_architecture_and_does_not_claim_quality(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            checkpoint, report, metadata, results = self.report_fixture(directory)
            with patch("native_600m.evaluate.load_config", return_value={"name": "NativeKoreanLM-600M", "system_prompt": "한국어", "seed": 42}), patch(
                "native_600m.evaluate.NativeTokenizer.load", return_value=SimpleNamespace(fingerprint="fixture-sha")
            ), patch("native_600m.evaluate.load_model", return_value=(object(), metadata)), patch(
                "native_600m.evaluate.run_evaluation", return_value=results
            ):
                main(["--checkpoint", str(checkpoint), "--report", str(report), "--device", "cpu"])
            saved = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(saved["parameter_count"], 6560)
        self.assertEqual(saved["model"], "NativeKoreanLM-600M")
        self.assertIsNone(saved["pretrained_base"])
        self.assertEqual(saved["tokenizer_sha256"], "fixture-sha")
        self.assertEqual(len(saved["checkpoint_sha256"]), 64)
        self.assertTrue(saved["heuristic_gate_passed"])
        self.assertFalse(saved["conversation_quality_verified"])
        self.assertIn("criteria", saved["human_rubric"])

    def test_failed_gate_writes_report_and_exits_unless_allow_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            checkpoint, report, metadata, results = self.report_fixture(directory, passed=False)
            with patch("native_600m.evaluate.load_config", return_value={"system_prompt": "한국어", "seed": 42}), patch(
                "native_600m.evaluate.NativeTokenizer.load", return_value=SimpleNamespace(fingerprint="fixture-sha")
            ), patch("native_600m.evaluate.load_model", return_value=(object(), metadata)), patch(
                "native_600m.evaluate.run_evaluation", return_value=results
            ):
                arguments = ["--checkpoint", str(checkpoint), "--report", str(report)]
                with self.assertRaises(SystemExit) as failure:
                    main(arguments)
                self.assertEqual(failure.exception.code, 1)
                self.assertFalse(json.loads(report.read_text(encoding="utf-8"))["heuristic_gate_passed"])
                main([*arguments, "--allow-fail"])

    def test_wrong_tokenizer_is_rejected_before_generation(self):
        with patch("native_600m.evaluate.load_config", return_value={"system_prompt": "한국어", "seed": 42}), patch(
            "native_600m.evaluate.NativeTokenizer.load", return_value=SimpleNamespace(fingerprint="different-sha")
        ), patch("native_600m.evaluate.load_model", return_value=(object(), {"tokenizer_sha256": "expected-sha"})), patch(
            "native_600m.evaluate.run_evaluation"
        ) as evaluation:
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                main([])
            evaluation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
