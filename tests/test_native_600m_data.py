"""Small deterministic fixtures only; no dataset downloads or model training."""

from __future__ import annotations

from array import array
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from native_600m.config import config_fingerprint, sha256_file, write_json
from native_600m.prepare import (
    CHAT_DATASET_REVISION, PRETRAIN_DATASET_REVISION, RejectedRecord, buffered_shuffle,
    document_group, hangul_ratio, iter_source, local_source, normalize_messages,
    normalize_text, prepare_records, prompt_group, public_source, split_for_group, validate_prepared,
)
from native_600m.tokenize import SPLITS, tokenize_records, validate_tokenized


def example_config() -> dict:
    return {"seed": 42, "system_prompt": "당신은 한국어로 대화하는 친절한 도우미입니다.",
            "model": {"vocab_size": 300},
            "data": {"pretrain_train_documents": 5, "pretrain_validation_documents": 2,
                     "chat_train_records": 4, "chat_validation_records": 2,
                     "min_hangul_ratio": 0.7, "min_text_characters": 20, "max_text_characters": 1000}}


def example_documents(count=100) -> list[dict]:
    return [{"text": f"한국어 문서 {index}번입니다. 오늘은 자연스럽게 대화하는 방법을 함께 배워 봅니다.",
             "url": "https://example.test/attribution"} for index in range(count)]


def example_chats(count=100) -> list[dict]:
    return [{"messages": [{"role": "user", "content": f"오늘 하루를 보내는 계획 {index}번을 함께 세워 줄래요?",
                           "content_en": "This parallel translation must not be retained."},
                          {"role": "assistant", "content": "중요한 일을 먼저 적고 잠깐 쉬는 시간도 정해 보세요."},
                          {"role": "user", "content": "쉬는 동안에는 무엇을 하면 좋을까요?"},
                          {"role": "assistant", "content": "물을 마시거나 가볍게 몸을 움직여 보세요."}]} for index in range(count)]


class FakeTokenizer:
    vocab_size = 300
    fingerprint = "a" * 64

    def encode_document(self, text):
        return [1, 10, 11, 12, 2]

    def encode_chat(self, messages):
        return [1, 3, 10, 6, 4, 11, 6, 5, 12, 13, 6, 2], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]


class NativeDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".native-data-test-", dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = example_config()
        self.data_dir = self.root / "data"
        self.tokens_dir = self.root / "tokens"
        self.sources = {stage: public_source(stage) for stage in ("pretrain", "chat")}

    def prepare(self, documents=None, chats=None):
        return prepare_records(example_documents() if documents is None else documents,
                               example_chats() if chats is None else chats, self.data_dir, self.config,
                               sources=self.sources, shuffle_buffer_size=5)

    def test_sources_are_pinned_and_language_is_korean(self):
        self.assertEqual(len(PRETRAIN_DATASET_REVISION), 40)
        self.assertEqual(len(CHAT_DATASET_REVISION), 40)
        self.assertEqual(self.sources["pretrain"]["config"], "kor_Hang")
        self.assertEqual(self.sources["pretrain"]["license"], "ODC-By-1.0")
        self.assertEqual(self.sources["chat"]["license"], "Apache-2.0")
        self.assertNotIn("model", self.sources["pretrain"])

    def test_korean_normalization_and_quality_filters(self):
        text = "  한국어\t 문서를\r\n 함께   천천히 읽어 봅니다.  "
        self.assertEqual(normalize_text(text, self.config, document=True), "한국어 문서를\n 함께 천천히 읽어 봅니다.")
        self.assertEqual(hangul_ratio("한글 123!"), 1.0)
        self.assertEqual(hangul_ratio("123!"), 0.0)
        for content, reason in (("A long English source without Korean text.", "insufficient_korean"),
                                ("한국어\x00 문서를 읽어요.", "control_characters"),
                                ("<|assistant|> 한국어로 대답해요.", "template_or_reasoning"),
                                ("<think>생각을 먼저 표시합니다.</think>", "template_or_reasoning"),
                                ("짧아요.", "too_short"), ("한국어" * 400, "too_long")):
            with self.subTest(reason=reason), self.assertRaises(RejectedRecord) as raised:
                normalize_text(content, self.config, document=True)
            self.assertEqual(raised.exception.reason, reason)

    def test_task_instruction_preserved_and_english_parallel_text_removed(self):
        messages = example_chats(1)[0]["messages"]
        instruction = "사용자의 요청에 따라 하루 계획을 세워 주세요."
        messages.insert(0, {"role": "system", "content": instruction})
        normalized = normalize_messages(messages, self.config)
        self.assertEqual(normalized[0]["content"], self.config["system_prompt"])
        self.assertTrue(normalized[1]["content"].startswith(f"작업 지침:\n{instruction}\n\n사용자 요청:\n"))
        self.assertEqual(normalized, normalize_messages(normalized, self.config))
        self.assertTrue(all(set(message) == {"role", "content"} for message in normalized))
        self.assertNotIn("English", json.dumps(normalized))

    def test_invalid_roles_and_incomplete_chat_rejected(self):
        for messages in ([], [{"role": "user", "content": "안녕하세요?"}],
                         [{"role": "assistant", "content": "안녕하세요?"}, {"role": "user", "content": "반가워요."}]):
            with self.subTest(messages=messages), self.assertRaises(RejectedRecord):
                normalize_messages(messages, self.config)

    def test_grouping_whitespace_variants_and_split_are_deterministic(self):
        self.assertEqual(document_group("문서를 함께\n 읽어요."), document_group("문서를   함께 읽어요."))
        group = prompt_group(normalize_messages(example_chats(1)[0]["messages"], self.config))
        result = split_for_group(group, seed=42, train_records=4, validation_records=2)
        self.assertEqual(result, split_for_group(group, seed=42, train_records=4, validation_records=2))
        with self.assertRaises(ValueError):
            list(buffered_shuffle(range(2), seed=42, buffer_size=0))
        self.assertEqual(list(buffered_shuffle(range(10), seed=42, buffer_size=4)),
                         list(buffered_shuffle(range(10), seed=42, buffer_size=4)))

    def test_exact_quotas_deduplicate_and_audited_reuse(self):
        documents, chats = example_documents(), example_chats()
        # Repeated prompts with a different answer still form a single group.
        variant = copy.deepcopy(chats[0])
        variant["messages"][1]["content"] = "차근차근 하나씩 계획을 적으면 좋겠어요."
        result = self.prepare([documents[0]] * 10 + documents, [chats[0]] * 10 + [variant] + chats)
        self.assertEqual(result, validate_prepared(self.data_dir, self.config))
        self.assertEqual({name: result["outputs"][name]["records"] for name in SPLITS},
                         {"pretrain_train": 5, "pretrain_validation": 2, "chat_train": 4, "chat_validation": 2})
        self.assertGreater(result["statistics"]["pretrain"]["rejections"].get("duplicate_document", 0), 0)
        self.assertGreater(result["statistics"]["chat"]["rejections"].get("duplicate_first_prompt", 0), 0)

    def test_exhausted_stage_has_no_completion_manifest(self):
        self.prepare()
        with self.assertRaisesRegex(RuntimeError, "Insufficient usable Korean chat"):
            self.prepare(chats=[])
        self.assertFalse((self.data_dir / "manifest.json").exists())
        self.assertFalse(list(self.data_dir.glob("*.tmp")))

    def test_data_tampering_and_different_config_rejected(self):
        self.prepare()
        changed = copy.deepcopy(self.config)
        changed["seed"] = 43
        with self.assertRaisesRegex(ValueError, "different configuration"):
            validate_prepared(self.data_dir, changed)
        with (self.data_dir / "pretrain_train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_prepared(self.data_dir, self.config)

    def test_forged_cross_split_prompt_cannot_pass_updated_file_hash(self):
        manifest = self.prepare()
        train = self.data_dir / "chat_train.jsonl"
        validation = self.data_dir / "chat_validation.jsonl"
        lines = validation.read_text(encoding="utf-8").splitlines()
        lines[0] = train.read_text(encoding="utf-8").splitlines()[0]
        validation.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest["outputs"]["chat_validation"]["sha256"] = sha256_file(validation)
        write_json(self.data_dir / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "duplicate|deterministic"):
            validate_prepared(self.data_dir, self.config)

    def test_local_source_requires_license_and_tracks_content(self):
        path = self.root / "source.jsonl"
        path.write_text('{"text":"한국어"}\ninvalid\n\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "explicit source license"):
            local_source(path, None)
        source = local_source(path, "CC0-1.0")
        self.assertEqual(list(iter_source(source)), [{"text": "한국어"}, None])
        self.sources["pretrain"] = source
        self.prepare()
        path.write_text('{"text":"변경된 한국어"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "local source is missing or has changed"):
            validate_prepared(self.data_dir, self.config)

    def test_tokens_are_uint16_with_same_index_assistant_mask(self):
        self.prepare()
        result = tokenize_records(self.data_dir, FakeTokenizer(), self.tokens_dir, self.config)
        self.assertEqual(result, validate_tokenized(self.tokens_dir, self.config,
                                                  data_dir=self.data_dir, tokenizer=FakeTokenizer()))
        entry = result["outputs"]["chat_train"]
        self.assertEqual(entry["token_count"], 12 * 4)
        self.assertEqual(entry["supervised_tokens"], 3 * 4)
        values = array("H")
        values.frombytes((self.tokens_dir / entry["tokens_file"]).read_bytes())
        if sys.byteorder != "little":
            values.byteswap()
        self.assertEqual(list(values[:12]), FakeTokenizer().encode_chat(None)[0])
        masks = (self.tokens_dir / entry["mask_file"]).read_bytes()
        self.assertEqual(list(masks[:12]), FakeTokenizer().encode_chat(None)[1])

    def test_tokenizer_identity_and_token_file_corruption_rejected(self):
        self.prepare()
        result = tokenize_records(self.data_dir, FakeTokenizer(), self.tokens_dir, self.config)
        other = FakeTokenizer()
        other.fingerprint = "b" * 64
        with self.assertRaisesRegex(ValueError, "different tokenizer"):
            validate_tokenized(self.tokens_dir, self.config, tokenizer=other)
        path = self.tokens_dir / result["outputs"]["pretrain_train"]["tokens_file"]
        path.write_bytes(path.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, "Corrupt pretrain_train"):
            validate_tokenized(self.tokens_dir, self.config)

    def test_nonbinary_mask_rejected_even_with_updated_hash(self):
        self.prepare()
        result = tokenize_records(self.data_dir, FakeTokenizer(), self.tokens_dir, self.config)
        entry = result["outputs"]["chat_train"]
        path = self.tokens_dir / entry["mask_file"]
        data = bytearray(path.read_bytes())
        data[8] = 2
        path.write_bytes(data)
        entry["mask_sha256"] = sha256_file(path)
        write_json(self.tokens_dir / "manifest.json", result)
        with self.assertRaisesRegex(ValueError, "Nonbinary"):
            validate_tokenized(self.tokens_dir, self.config)

    def test_tokenization_failure_removes_previous_manifest(self):
        self.prepare()
        tokenize_records(self.data_dir, FakeTokenizer(), self.tokens_dir, self.config)
        class InvalidTokenizer(FakeTokenizer):
            def encode_chat(self, messages):
                return [1, 12, 2], [0, 0, 0]
        with self.assertRaisesRegex(ValueError, "supervised assistant targets"):
            tokenize_records(self.data_dir, InvalidTokenizer(), self.tokens_dir, self.config)
        self.assertFalse((self.tokens_dir / "manifest.json").exists())
        self.assertFalse(list(self.tokens_dir.glob("*.tmp")))

    def test_changed_source_manifest_invalidates_tokens(self):
        self.prepare()
        tokenize_records(self.data_dir, FakeTokenizer(), self.tokens_dir, self.config)
        path = self.data_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["statistics"]["annotation"] = "An updated provenance review."
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "source manifest changed"):
            validate_tokenized(self.tokens_dir, self.config, data_dir=self.data_dir)


if __name__ == "__main__":
    unittest.main()
