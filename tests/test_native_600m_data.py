"""Small deterministic fixtures only; no dataset downloads or model training."""

from __future__ import annotations

from array import array
import copy
from contextlib import redirect_stdout
import io
import importlib.util
import json
from itertools import count
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from native_600m.config import config_fingerprint, sha256_file, write_json
from native_600m.prepare import (
    CHAT_DATASET_REVISION, PRETRAIN_DATASET_REVISION, RejectedRecord, _normalized_records, buffered_shuffle,
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


class InvalidMaskTokenizer(FakeTokenizer):
    def encode_chat(self, messages):
        return [1, 12, 2], [0, 0, 0]


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

    def test_parallel_preparation_matches_serial_files_statistics_and_manifest(self):
        config = copy.deepcopy(self.config)
        config["data"].update(pretrain_train_documents=120, pretrain_validation_documents=40,
                              chat_train_records=100, chat_validation_records=40)
        documents, chats = [None, {"text": "An English document."}], [None, {"messages": []}]
        for document, chat in zip(example_documents(1000), example_chats(1000)):
            documents.extend([document, document])
            chats.extend([chat, chat])
        serial_dir, parallel_dir = self.root / "serial", self.root / "parallel"
        serial = prepare_records(documents, chats, serial_dir, config, sources=self.sources,
                                 shuffle_buffer_size=5, workers=1)
        parallel = prepare_records(documents, chats, parallel_dir, config, sources=self.sources,
                                   shuffle_buffer_size=5, workers=2)
        self.assertEqual(parallel, serial)
        for entry in serial["outputs"].values():
            self.assertEqual((parallel_dir / entry["file"]).read_bytes(), (serial_dir / entry["file"]).read_bytes())
        validate_prepared(parallel_dir, config)

    def test_parallel_validation_matches_serial_and_reports_progress(self):
        self.config["data"].update(pretrain_train_documents=1024, pretrain_validation_documents=256,
                                   chat_train_records=256, chat_validation_records=64)
        manifest = self.prepare(example_documents(5000), example_chats(2000))
        before = {path.name: path.read_bytes() for path in self.data_dir.iterdir()}
        serial = validate_prepared(self.data_dir, self.config, workers=1)
        output = io.StringIO()
        with redirect_stdout(output):
            parallel = validate_prepared(self.data_dir, self.config, workers=12)
        self.assertEqual(parallel, serial)
        self.assertEqual(parallel, manifest)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.data_dir.iterdir()})
        self.assertIn("[verify:data] hashing files=4 workers=4", output.getvalue())
        self.assertIn("[verify:pretrain_train] hash OK", output.getvalue())
        self.assertIn("[verify:pretrain_train] records=0/1,024 workers=12", output.getvalue())
        self.assertIn("[verify:pretrain_train] records=1,024/1,024 OK", output.getvalue())

    def test_parallel_validation_rejects_duplicates_across_batches(self):
        self.config["data"].update(pretrain_train_documents=130, pretrain_validation_documents=40)
        manifest = self.prepare(example_documents(1000), example_chats())
        path = self.data_dir / "pretrain_train.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[64] = lines[0]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest["outputs"]["pretrain_train"]["sha256"] = sha256_file(path)
        write_json(self.data_dir / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_prepared(self.data_dir, self.config, workers=2)

    def test_parallel_validation_rejects_invalid_rows_and_counts_with_updated_hashes(self):
        manifest = self.prepare()
        path = self.data_dir / "pretrain_train.jsonl"
        original = path.read_text(encoding="utf-8").splitlines()
        extra = json.loads(original[0])
        extra["extra"] = "unexpected"
        cases = [("invalid_json", ["invalid", *original[1:]]),
                 ("invalid_object", ["[]", *original[1:]]),
                 ("unnormalized", [json.dumps(extra, ensure_ascii=False), *original[1:]]),
                 ("rejected_text", [json.dumps({"text": "English"}), *original[1:]]),
                 ("wrong_count", original[:-1])]
        for label, lines in cases:
            with self.subTest(label=label):
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                manifest["outputs"]["pretrain_train"]["sha256"] = sha256_file(path)
                write_json(self.data_dir / "manifest.json", manifest)
                with self.assertRaises(ValueError):
                    validate_prepared(self.data_dir, self.config, workers=2)

    def test_parallel_validation_rejects_wrong_hash_before_parsing(self):
        self.prepare()
        (self.data_dir / "pretrain_train.jsonl").write_text("invalid", encoding="utf-8")
        with patch("native_600m.prepare.ProcessPoolExecutor") as pool:
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_prepared(self.data_dir, self.config, workers=2)
        pool.assert_not_called()

    def test_validation_rejects_invalid_worker_counts(self):
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "workers"):
                validate_prepared(self.data_dir, self.config, workers=workers)

    def test_parallel_stream_does_not_eagerly_read_the_corpus(self):
        consumed = 0

        def records():
            nonlocal consumed
            for index in count():
                consumed += 1
                if consumed > 1024:
                    raise AssertionError("The corpus was consumed eagerly.")
                yield {"text": f"한국어 문서 {index}번입니다. 오늘의 이야기를 함께 읽어 봅니다."}

        normalized = _normalized_records(records(), self.config, "pretrain", workers=2)
        try:
            serialized, group, reason = next(normalized)
            self.assertIn("0번", serialized)
            self.assertIsNotNone(group)
            self.assertIsNone(reason)
            self.assertLessEqual(consumed, 1024)
        finally:
            normalized.close()

    def test_parallel_failure_cleans_partial_outputs(self):
        with self.assertRaisesRegex(RuntimeError, "Insufficient usable Korean pretrain data"):
            prepare_records([None, {}, {"text": "English"}], example_chats(), self.data_dir, self.config,
                            sources=self.sources, shuffle_buffer_size=1, workers=2)
        self.assertFalse((self.data_dir / "manifest.json").exists())
        self.assertFalse(list(self.data_dir.glob("*.tmp")))

    def test_invalid_workers_are_rejected_before_data_is_written(self):
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "workers"):
                prepare_records([], [], self.data_dir, self.config, sources=self.sources, workers=workers)
        self.assertFalse(self.data_dir.exists())

    def test_public_sources_stream_without_a_platform_restriction(self):
        for stage, source in self.sources.items():
            with self.subTest(stage=stage):
                records = [{"text": "한국어 문서"}] if stage == "pretrain" else example_chats(1)
                load_dataset = Mock(return_value=iter(records))
                with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=load_dataset)}):
                    self.assertEqual(list(iter_source(source)), records)
                options = {"revision": source["revision"], "split": "train", "streaming": True}
                if stage == "pretrain":
                    options["name"] = "kor_Hang"
                load_dataset.assert_called_once_with(source["id"], **options)

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
        for workers in (1, 2):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "duplicate|deterministic"):
                validate_prepared(self.data_dir, self.config, workers=workers)

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
        result = tokenize_records(self.data_dir, FakeTokenizer(), self.tokens_dir, self.config, workers=2)
        self.assertEqual(result, validate_tokenized(self.tokens_dir, self.config,
                                                  data_dir=self.data_dir, tokenizer=FakeTokenizer(), workers=2))
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

    @unittest.skipIf(importlib.util.find_spec("tokenizers") is None, "tokenizers is not installed")
    def test_twelve_process_tokenization_matches_serial_native_tokens_and_masks(self):
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
        from native_600m.tokenizer import NativeTokenizer, SPECIAL_TOKENS

        vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        for symbol in sorted(pre_tokenizers.ByteLevel.alphabet()):
            vocabulary[symbol] = len(vocabulary)
        backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
        backend.add_special_tokens(list(SPECIAL_TOKENS))
        backend.normalizer = normalizers.NFC()
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        path = self.root / "tokenizer.json"
        backend.save(str(path))
        tokenizer = NativeTokenizer.load(path)
        self.config["data"].update(pretrain_train_documents=1024, pretrain_validation_documents=256,
                                   chat_train_records=1024, chat_validation_records=256)
        self.prepare(example_documents(5000), example_chats(5000))
        serial_dir, parallel_dir = self.root / "serial-tokens", self.root / "parallel-tokens"
        serial = tokenize_records(self.data_dir, tokenizer, serial_dir, self.config, workers=1)
        parallel = tokenize_records(self.data_dir, tokenizer, parallel_dir, self.config, workers=12)
        self.assertEqual(parallel, serial)
        for entry in parallel["outputs"].values():
            for kind in ("tokens", "mask"):
                filename = entry.get(kind + "_file")
                if filename:
                    self.assertEqual((parallel_dir / filename).read_bytes(), (serial_dir / filename).read_bytes())
                    self.assertEqual(sha256_file(parallel_dir / filename), entry[kind + "_sha256"])
        self.assertEqual(validate_tokenized(parallel_dir, self.config, tokenizer=tokenizer, workers=12), parallel)

    def test_parallel_tokenizer_failure_removes_manifest_and_temporary_files(self):
        self.prepare()
        tokenize_records(self.data_dir, FakeTokenizer(), self.tokens_dir, self.config)
        with self.assertRaisesRegex(ValueError, "supervised assistant targets"):
            tokenize_records(self.data_dir, InvalidMaskTokenizer(), self.tokens_dir, self.config, workers=2)
        self.assertFalse((self.tokens_dir / "manifest.json").exists())
        self.assertFalse(list(self.tokens_dir.glob("*.tmp")))

    def test_parallel_token_checks_reject_corruption_and_forged_mask_hash(self):
        self.prepare()
        manifest = tokenize_records(self.data_dir, FakeTokenizer(), self.tokens_dir, self.config)
        token_path = self.tokens_dir / manifest["outputs"]["pretrain_train"]["tokens_file"]
        original = token_path.read_bytes()
        token_path.write_bytes(original[:-2])
        with self.assertRaisesRegex(ValueError, "Corrupt pretrain_train"):
            validate_tokenized(self.tokens_dir, self.config, workers=12)
        token_path.write_bytes(original)
        entry = manifest["outputs"]["chat_train"]
        mask_path = self.tokens_dir / entry["mask_file"]
        content = bytearray(mask_path.read_bytes())
        content[8] = 2
        mask_path.write_bytes(content)
        entry["mask_sha256"] = sha256_file(mask_path)
        write_json(self.tokens_dir / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Nonbinary"):
            validate_tokenized(self.tokens_dir, self.config, workers=12)

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
