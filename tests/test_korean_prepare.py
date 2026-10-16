"""Dependency-free data preparation checks; no training or model downloads."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from korean_600m.config import sha256_file, write_json
from korean_600m.prepare import (
    DATASET_ID, DATASET_REVISION, RejectedRecord, buffered_shuffle, hangul_ratio,
    iter_source, local_source, normalize_messages, prepare_records, prompt_group,
    public_source, split_for_group, validate_prepared,
)


def example_config() -> dict:
    return {
        "system_prompt": "당신은 한국어 대화를 돕는 친절한 도우미입니다.",
        "seed": 42,
        "data": {"train_records": 5, "validation_records": 2,
                 "min_hangul_ratio": 0.7, "max_characters": 1000},
    }


def example_rows(count: int = 100) -> list[dict]:
    return [{"messages": [
        {"role": "user", "content": f"오늘 하루를 보내는 계획 {index}번을 함께 세워 줄래요?",
         "content_en": "The parallel English text must never be used."},
        {"role": "assistant", "content": "먼저 해야 할 일을 적고 가장 중요한 일부터 시작해 보세요."},
        {"role": "user", "content": "잠깐 쉬는 시간은 언제 넣으면 좋을까요?"},
        {"role": "assistant", "content": "집중한 뒤에는 짧게 쉬면서 물도 마시고 몸도 움직여 보세요."},
    ]} for index in range(count)]


def temporary_directory():
    # Keep generated fixtures inside the writable workspace in restricted runners.
    return tempfile.TemporaryDirectory(prefix=".prepare-test-", dir=Path(__file__).resolve().parent)


class KoreanPreparationTests(unittest.TestCase):
    def test_normalization_preserves_multiturn_korean_and_source_task(self) -> None:
        row = example_rows(1)[0]
        instruction = "사용자가 작성한 글의 의미를 유지하면서 더 짧게 다시 작성하세요."
        original_prompt = row["messages"][0]["content"]
        row["messages"].insert(0, {"role": "system", "content": instruction})
        messages = normalize_messages(row["messages"], example_config())
        self.assertEqual(messages[0]["content"], example_config()["system_prompt"])
        self.assertEqual(messages[1]["content"], f"작업 지침:\n{instruction}\n\n사용자 요청:\n{original_prompt}")
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user", "assistant"])
        self.assertTrue(all(set(message) == {"role", "content"} for message in messages))
        self.assertEqual(normalize_messages(messages, example_config()), messages)

    def test_fixed_system_does_not_add_a_duplicate_task_prefix(self) -> None:
        config = example_config()
        messages = [{"role": "system", "content": config["system_prompt"]}, *example_rows(1)[0]["messages"]]
        normalized = normalize_messages(messages, config)
        self.assertEqual(normalized[1]["content"], messages[1]["content"])
        self.assertEqual(normalize_messages(normalized, config), normalized)

    def test_source_system_obeys_korean_control_reasoning_and_length_filters(self) -> None:
        config = example_config()
        for instruction, reason in (
            ("Rewrite the provided text more concisely.", "insufficient_korean"),
            ("작성한 글을\x00 요약하세요.", "control_characters"),
            ("<think>먼저 생각한 과정을 보여 주세요.</think>", "template_or_reasoning"),
            ("사용자의 요청에 답하세요." * 100, "too_long"),
        ):
            with self.subTest(instruction=instruction):
                messages = [{"role": "system", "content": instruction}, *example_rows(1)[0]["messages"]]
                with self.assertRaises(RejectedRecord) as raised:
                    normalize_messages(messages, config)
                self.assertEqual(raised.exception.reason, reason)

    def test_distinct_source_instructions_form_distinct_prompt_groups(self) -> None:
        config = example_config()
        groups = []
        for instruction in ("입력한 글을 짧게 요약하세요.", "입력한 글의 맞춤법을 바르게 고쳐 주세요."):
            messages = [{"role": "system", "content": instruction}, *example_rows(1)[0]["messages"]]
            groups.append(prompt_group(normalize_messages(messages, config)))
        self.assertNotEqual(groups[0], groups[1])

    def test_rejects_english_bad_roles_controls_and_reasoning(self) -> None:
        config = example_config()
        for content, reason in (
            ("Hello, how are you today?", "insufficient_korean"),
            ("안녕하세요\x00 반가워요", "control_characters"),
            ("안녕하세요\u200b 반가워요", "control_characters"),
            ("<think>먼저 생각해 보겠습니다.</think> 반갑습니다.", "template_or_reasoning"),
            ("<|im_start|>assistant 안녕하세요", "template_or_reasoning"),
            ("Analysis: 문제를 순서대로 생각해 보겠습니다.", "template_or_reasoning"),
        ):
            with self.subTest(content=content):
                messages = [{"role": "user", "content": "인사해 주시겠어요?"},
                            {"role": "assistant", "content": content}]
                with self.assertRaises(RejectedRecord) as raised:
                    normalize_messages(messages, config)
                self.assertEqual(raised.exception.reason, reason)
        with self.assertRaises(RejectedRecord):
            normalize_messages([{"role": "assistant", "content": "안녕하세요"},
                                {"role": "user", "content": "반가워요"}], config)
        self.assertEqual(hangul_ratio("한국어 English"), 3 / 10)

    def test_long_conversations_are_rejected_before_writing(self) -> None:
        config = example_config()
        config["data"]["max_characters"] = 5
        with self.assertRaisesRegex(RejectedRecord, "too_long"):
            normalize_messages(example_rows(1)[0]["messages"], config)

    def test_group_key_normalizes_equivalent_first_prompts(self) -> None:
        messages = normalize_messages(example_rows(1)[0]["messages"], example_config())
        changed = copy.deepcopy(messages)
        changed[1]["content"] = "  " + changed[1]["content"].replace(" ", "\n") + "  "
        changed[-1]["content"] = "다른 답변을 사용해도 첫 질문의 그룹은 같아야 해요."
        self.assertEqual(prompt_group(messages), prompt_group(changed))
        group = prompt_group(messages)
        self.assertEqual(split_for_group(group, seed=42, train_records=5, validation_records=2),
                         split_for_group(group, seed=42, train_records=5, validation_records=2))

    def test_buffer_shuffle_is_deterministic_and_preserves_rows(self) -> None:
        first = list(buffered_shuffle(range(50), seed=7, buffer_size=5))
        self.assertEqual(first, list(buffered_shuffle(range(50), seed=7, buffer_size=5)))
        self.assertEqual(sorted(first), list(range(50)))
        self.assertNotEqual(first, list(range(50)))

    def test_exact_quotas_disjoint_groups_and_reproducible_hashes(self) -> None:
        config = example_config()
        rows = example_rows()
        rows.insert(1, copy.deepcopy(rows[0]))
        variation = copy.deepcopy(rows[0])
        variation["messages"][-1]["content"] = "기분 전환을 위한 산책도 함께 계획해 보세요."
        rows.insert(2, variation)
        with temporary_directory() as temporary:
            root = Path(temporary)
            manifest = prepare_records(rows, root / "first", config, source=public_source(), shuffle_buffer_size=3)
            second = prepare_records(rows, root / "second", config, source=public_source(), shuffle_buffer_size=3)
            self.assertEqual(manifest["outputs"], second["outputs"])
            self.assertEqual(manifest["outputs"]["train"]["records"], 5)
            self.assertEqual(manifest["outputs"]["validation"]["records"], 2)
            self.assertEqual(validate_prepared(root / "first", config), manifest)
            groups = {}
            for split in ("train", "validation"):
                records = [json.loads(line) for line in (root / "first" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
                groups[split] = {prompt_group(row["messages"]) for row in records}
            self.assertTrue(groups["train"].isdisjoint(groups["validation"]))
            self.assertGreater(manifest["statistics"]["rejections"].get("duplicate_record", 0)
                               + manifest["statistics"]["rejections"].get("duplicate_first_prompt", 0), 0)

    def test_hash_corruption_config_drift_and_source_drift_are_rejected(self) -> None:
        config = example_config()
        with temporary_directory() as temporary:
            root = Path(temporary)
            manifest = prepare_records(example_rows(), root, config, source=public_source(), shuffle_buffer_size=3)
            changed_config = copy.deepcopy(config)
            changed_config["seed"] += 1
            with self.assertRaisesRegex(ValueError, "configuration"):
                validate_prepared(root, changed_config)
            changed_manifest = copy.deepcopy(manifest)
            changed_manifest["source"]["revision"] = "0" * 40
            write_json(root / "manifest.json", changed_manifest)
            with self.assertRaisesRegex(ValueError, "pinned public"):
                validate_prepared(root, config)
            write_json(root / "manifest.json", manifest)
            with (root / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_prepared(root, config)

    def test_rehashed_duplicate_prompt_still_fails_validation(self) -> None:
        config = example_config()
        with temporary_directory() as temporary:
            root = Path(temporary)
            manifest = prepare_records(example_rows(), root, config, source=public_source(), shuffle_buffer_size=3)
            train_lines = (root / "train.jsonl").read_text(encoding="utf-8").splitlines()
            validation_lines = (root / "validation.jsonl").read_text(encoding="utf-8").splitlines()
            validation_lines[0] = train_lines[0]
            (root / "validation.jsonl").write_text("\n".join(validation_lines) + "\n", encoding="utf-8")
            manifest["outputs"]["validation"]["sha256"] = sha256_file(root / "validation.jsonl")
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "repeats a first user prompt"):
                validate_prepared(root, config)

    def test_insufficient_data_leaves_no_complete_manifest(self) -> None:
        with temporary_directory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "enough usable Korean"):
                prepare_records(example_rows(1), root, example_config(), source=public_source(),
                                max_scanned_records=1, shuffle_buffer_size=1)
            self.assertFalse((root / "manifest.json").exists())
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_public_stream_uses_pinned_revision_without_a_model_download(self) -> None:
        calls = []
        def mocked_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return iter(example_rows(1))
        with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=mocked_load_dataset)}):
            self.assertEqual(list(iter_source(public_source())), example_rows(1))
        self.assertEqual(calls, [((DATASET_ID,), {"revision": DATASET_REVISION, "split": "train", "streaming": True})])

    def test_local_source_requires_license_and_detects_source_changes(self) -> None:
        config = example_config()
        with temporary_directory() as temporary:
            root = Path(temporary)
            source_path = root / "source.jsonl"
            source_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in example_rows()) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "local-license"):
                local_source(source_path, None)
            source = local_source(source_path, "CC0-1.0")
            prepare_records(iter_source(source), root / "prepared", config, source=source, shuffle_buffer_size=3)
            validate_prepared(root / "prepared", config)
            with source_path.open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "local source"):
                validate_prepared(root / "prepared", config)


if __name__ == "__main__":
    unittest.main()
