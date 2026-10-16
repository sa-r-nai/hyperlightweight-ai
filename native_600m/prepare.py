"""Stream and audit Korean pretraining text and supervised conversations.

Only datasets are downloaded. This module never imports a model or pretrained
tokenizer. FineWeb2 is attributed under ODC-By 1.0 and is also subject to the
CommonCrawl terms. The conversation publisher specifies Apache-2.0.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import islice
import json
from pathlib import Path
import random
import re
from typing import Iterable, Iterator, Mapping
import unicodedata

from .config import config_fingerprint, load_config, sha256_file, write_json

PRETRAIN_DATASET_ID = "HuggingFaceFW/fineweb-2"
PRETRAIN_DATASET_REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
CHAT_DATASET_ID = "lemon-mint/smol-koreantalk"
CHAT_DATASET_REVISION = "04eb12c1f999578d74dd87720bc853cedb7fa156"
MANIFEST_FORMAT = "native-600m-data-v1"
DEFAULT_OUTPUT_DIR = Path("artifacts_native_600m/data")
SHUFFLE_BUFFER_SIZE = 10_000
TEMPLATE_MARKERS = (
    "<|", "|>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
    "<start_of_turn>", "<end_of_turn>", "<think", "</think",
    "<analysis", "</analysis", "<reflection", "</reflection",
)
REASONING_HEADER = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:analysis|reasoning|chain[- ]of[- ]thought|thinking process)\s*:"
)


class RejectedRecord(ValueError):
    """A record rejected by deterministic normalization or quality filters."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def hangul_ratio(text: str) -> float:
    alphabetic = [character for character in text if character.isalpha()]
    if not alphabetic:
        return 0.0
    hangul = sum(
        "\uac00" <= character <= "\ud7a3" or "\u1100" <= character <= "\u11ff"
        or "\u3130" <= character <= "\u318f" or "\ua960" <= character <= "\ua97f"
        or "\ud7b0" <= character <= "\ud7ff" for character in alphabetic
    )
    return hangul / len(alphabetic)


def normalize_text(value: object, config: Mapping, *, document: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RejectedRecord("empty_text")
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"}
           and character not in "\n\r\t" for character in value):
        raise RejectedRecord("control_characters")
    if any(marker.casefold() in value.casefold() for marker in TEMPLATE_MARKERS):
        raise RejectedRecord("template_or_reasoning")
    if REASONING_HEADER.search(value):
        raise RejectedRecord("template_or_reasoning")
    value = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value).strip()
    if hangul_ratio(value) < config["data"]["min_hangul_ratio"]:
        raise RejectedRecord("insufficient_korean")
    if len(value) > config["data"]["max_text_characters"]:
        raise RejectedRecord("too_long")
    if document and len(value) < config["data"]["min_text_characters"]:
        raise RejectedRecord("too_short")
    return value


def normalize_messages(value: object, config: Mapping) -> list[dict[str, str]]:
    """Preserve task instructions while assigning one fixed assistant identity."""
    if not isinstance(value, list) or not value:
        raise RejectedRecord("invalid_messages")
    messages = list(value)
    fixed_system = str(config["system_prompt"]).strip()
    source_instruction = None
    if isinstance(messages[0], Mapping) and messages[0].get("role") == "system":
        content = messages.pop(0).get("content")
        if not isinstance(content, str) or content.strip() != fixed_system:
            source_instruction = normalize_text(content, config)
    if len(messages) < 2 or len(messages) % 2:
        raise RejectedRecord("invalid_roles")
    result = [{"role": "system", "content": fixed_system}]
    characters = 0
    for index, message in enumerate(messages):
        role = "user" if index % 2 == 0 else "assistant"
        if not isinstance(message, Mapping) or message.get("role") != role:
            raise RejectedRecord("invalid_roles")
        content = normalize_text(message.get("content"), config)
        if index == 0 and source_instruction is not None:
            content = normalize_text(f"작업 지침:\n{source_instruction}\n\n사용자 요청:\n{content}", config)
        characters += len(content)
        if characters > config["data"]["max_text_characters"]:
            raise RejectedRecord("too_long")
        result.append({"role": role, "content": content})
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def document_group(text: str) -> str:
    canonical = " ".join(unicodedata.normalize("NFC", text).split()).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prompt_group(messages: list[dict[str, str]]) -> str:
    return document_group(next(message["content"] for message in messages if message["role"] == "user"))


def split_for_group(group: str, *, seed: int, train_records: int, validation_records: int) -> str:
    bucket = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode("ascii")).digest()[:8], "big")
    threshold = validation_records * (1 << 64) // (train_records + validation_records)
    return "validation" if bucket < threshold else "train"


def buffered_shuffle(records: Iterable[object], *, seed: int, buffer_size: int) -> Iterator[object]:
    if type(buffer_size) is not int or buffer_size <= 0:
        raise ValueError("buffer_size must be a positive integer.")
    generator = random.Random(seed)
    buffer = []
    for record in records:
        if len(buffer) < buffer_size:
            buffer.append(record)
        else:
            index = generator.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = record
    generator.shuffle(buffer)
    yield from buffer


def public_source(stage: str) -> dict:
    if stage == "pretrain":
        return {"kind": "huggingface", "id": PRETRAIN_DATASET_ID,
                "revision": PRETRAIN_DATASET_REVISION, "config": "kor_Hang", "split": "train",
                "license": "ODC-By-1.0", "url": f"https://huggingface.co/datasets/{PRETRAIN_DATASET_ID}",
                "terms_url": "https://commoncrawl.org/terms-of-use"}
    if stage == "chat":
        return {"kind": "huggingface", "id": CHAT_DATASET_ID,
                "revision": CHAT_DATASET_REVISION, "split": "train", "license": "Apache-2.0",
                "url": f"https://huggingface.co/datasets/{CHAT_DATASET_ID}"}
    raise ValueError("stage must be pretrain or chat.")


def local_source(path: Path, license_name: str | None) -> dict:
    if not isinstance(license_name, str) or not license_name.strip():
        raise ValueError("Local JSONL requires an explicit source license or permission.")
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"kind": "local-jsonl", "id": "local-jsonl", "path": str(path),
            "revision": sha256_file(path), "license": license_name.strip(), "url": None}


def iter_source(source: Mapping) -> Iterator[object]:
    if source["kind"] == "local-jsonl":
        with Path(source["path"]).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield None
        return
    from .runtime import require_studio
    require_studio()
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Install requirements-native-600m.txt in Lightning AI Studio first.") from error
    kwargs = {"revision": source["revision"], "split": source["split"], "streaming": True}
    if "config" in source:
        kwargs["name"] = source["config"]
    yield from load_dataset(source["id"], **kwargs)


def quotas_for(config: Mapping, stage: str) -> dict[str, int]:
    suffix = "documents" if stage == "pretrain" else "records"
    result = {split: config["data"][f"{stage}_{split}_{suffix}"] for split in ("train", "validation")}
    if any(type(value) is not int or value <= 0 for value in result.values()):
        raise ValueError("Every data split quota must be a positive integer.")
    return result


def _prepare_stage(records: Iterable[object], output_dir: Path, config: dict, stage: str,
                   max_scanned_records: int, shuffle_buffer_size: int) -> tuple[dict, dict]:
    quotas = quotas_for(config, stage)
    paths = {split: output_dir / f"{stage}_{split}.jsonl" for split in quotas}
    temporaries = {split: path.with_suffix(".jsonl.tmp") for split, path in paths.items()}
    counts, rejected = Counter(), Counter()
    groups = set()
    scanned = 0
    try:
        with temporaries["train"].open("w", encoding="utf-8", newline="\n") as train, \
             temporaries["validation"].open("w", encoding="utf-8", newline="\n") as validation:
            handles = {"train": train, "validation": validation}
            for record in buffered_shuffle(islice(records, max_scanned_records), seed=config["seed"],
                                           buffer_size=shuffle_buffer_size):
                scanned += 1
                try:
                    if not isinstance(record, Mapping):
                        raise RejectedRecord("invalid_record")
                    if stage == "pretrain":
                        normalized = {"text": normalize_text(record.get("text"), config, document=True)}
                        group = document_group(normalized["text"])
                    else:
                        normalized = {"messages": normalize_messages(record.get("messages"), config)}
                        group = prompt_group(normalized["messages"])
                    if group in groups:
                        raise RejectedRecord("duplicate_document" if stage == "pretrain" else "duplicate_first_prompt")
                    groups.add(group)
                    split = split_for_group(group, seed=config["seed"], train_records=quotas["train"],
                                            validation_records=quotas["validation"])
                    if counts[split] >= quotas[split]:
                        raise RejectedRecord("split_quota_full")
                except RejectedRecord as error:
                    rejected[error.reason] += 1
                else:
                    handles[split].write(_canonical(normalized) + "\n")
                    counts[split] += 1
                if scanned % 10_000 == 0:
                    print(f"[prepare:{stage}] scanned={scanned:,} train={counts['train']:,} "
                          f"validation={counts['validation']:,}", flush=True)
                if all(counts[split] == quota for split, quota in quotas.items()):
                    break
        if any(counts[split] != quota for split, quota in quotas.items()):
            raise RuntimeError(f"Insufficient usable Korean {stage} data: train={counts['train']}/{quotas['train']}, "
                               f"validation={counts['validation']}/{quotas['validation']}, scanned={scanned}. "
                               "Supply more data, lower the configured quotas, or increase the scan limit.")
        for split, path in paths.items():
            temporaries[split].replace(path)
        outputs = {f"{stage}_{split}": {"file": path.name, "records": counts[split], "sha256": sha256_file(path)}
                   for split, path in paths.items()}
        statistics = {"scanned_records": scanned, "written_records": sum(counts.values()),
                      "rejected_records": sum(rejected.values()), "rejections": dict(rejected)}
        return outputs, statistics
    finally:
        for path in temporaries.values():
            path.unlink(missing_ok=True)


def prepare_records(pretrain_records: Iterable[object], chat_records: Iterable[object], output_dir: Path,
                    config: dict, *, sources: dict, max_pretrain_scanned: int = 20_000_000,
                    max_chat_scanned: int = 2_000_000, shuffle_buffer_size: int = SHUFFLE_BUFFER_SIZE) -> dict:
    """Publish a completion manifest only after both exact-size stages succeed."""
    for value in (max_pretrain_scanned, max_chat_scanned, shuffle_buffer_size):
        if type(value) is not int or value <= 0:
            raise ValueError("Scan limits and shuffle buffer size must be positive integers.")
    for stage in ("pretrain", "chat"):
        quotas_for(config, stage)
        _validate_source(sources.get(stage), stage)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    outputs, statistics = {}, {}
    for stage, records, limit in (("pretrain", pretrain_records, max_pretrain_scanned),
                                  ("chat", chat_records, max_chat_scanned)):
        entries, stats = _prepare_stage(records, output_dir, config, stage, limit, shuffle_buffer_size)
        outputs.update(entries)
        statistics[stage] = stats
    manifest = {"format": MANIFEST_FORMAT, "config_fingerprint": config_fingerprint(config), "sources": sources,
                "outputs": outputs, "statistics": statistics,
                "preparation": {"seed": config["seed"], "shuffle_buffer_size": shuffle_buffer_size,
                                "max_pretrain_scanned": max_pretrain_scanned, "max_chat_scanned": max_chat_scanned,
                                "split_method": "sha256-normalized-document-or-first-user-prompt-v1"}}
    write_json(manifest_path, manifest)
    return manifest


def _validate_source(source: object, stage: str) -> None:
    if not isinstance(source, dict):
        raise ValueError(f"Missing {stage} source identity.")
    if source.get("kind") == "huggingface":
        if source != public_source(stage):
            raise ValueError(f"The {stage} dataset does not match the pinned public source.")
    elif source.get("kind") == "local-jsonl":
        if not isinstance(source.get("license"), str) or not source["license"].strip():
            raise ValueError("A local source requires an explicit license.")
        if not re.fullmatch(r"[0-9a-f]{64}", str(source.get("revision", ""))):
            raise ValueError("A local source requires a SHA-256 identity.")
        path = Path(source.get("path", ""))
        if not path.is_file() or sha256_file(path) != source["revision"]:
            raise ValueError("The local source is missing or has changed.")
    else:
        raise ValueError("Unsupported data source kind.")


def validate_prepared(output_dir: Path, config: dict) -> dict:
    output_dir = Path(output_dir)
    path = output_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No complete data manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("Unsupported data manifest format.")
    if manifest.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError("Prepared data uses a different configuration.")
    for stage in ("pretrain", "chat"):
        _validate_source(manifest.get("sources", {}).get(stage), stage)
        quotas = quotas_for(config, stage)
        groups = set()
        for split, quota in quotas.items():
            name = f"{stage}_{split}"
            entry = manifest.get("outputs", {}).get(name)
            if not isinstance(entry, dict) or entry.get("file") != f"{name}.jsonl" or entry.get("records") != quota:
                raise ValueError(f"Invalid {name} output metadata.")
            data_path = output_dir / entry["file"]
            if not data_path.is_file() or sha256_file(data_path) != entry.get("sha256"):
                raise ValueError(f"Prepared {name} hash mismatch.")
            count = 0
            with data_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise ValueError("A prepared row must be an object.")
                    if stage == "pretrain":
                        normalized = {"text": normalize_text(row.get("text"), config, document=True)}
                        group = document_group(normalized["text"])
                    else:
                        normalized = {"messages": normalize_messages(row.get("messages"), config)}
                        group = prompt_group(normalized["messages"])
                    if row != normalized or group in groups:
                        raise ValueError(f"Unnormalized or duplicate {stage} data within or across splits.")
                    if split_for_group(group, seed=config["seed"], train_records=quotas["train"],
                                       validation_records=quotas["validation"]) != split:
                        raise ValueError(f"Incorrect deterministic {stage} split.")
                    groups.add(group)
                    count += 1
            if count != quota:
                raise ValueError(f"Prepared {name} row count mismatch.")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare independent Korean model data in Lightning AI Studio.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-pretrain-jsonl", type=Path)
    parser.add_argument("--local-chat-jsonl", type=Path)
    parser.add_argument("--pretrain-license")
    parser.add_argument("--chat-license")
    parser.add_argument("--max-pretrain-scanned", type=int, default=20_000_000)
    parser.add_argument("--max-chat-scanned", type=int, default=2_000_000)
    parser.add_argument("--reuse-prepared", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    sources = {}
    for stage in ("pretrain", "chat"):
        path, license_name = getattr(args, f"local_{stage}_jsonl"), getattr(args, f"{stage}_license")
        if license_name and path is None:
            raise ValueError(f"--{stage}-license requires --local-{stage}-jsonl.")
        sources[stage] = local_source(path, license_name) if path else public_source(stage)
    if args.reuse_prepared:
        manifest = validate_prepared(args.output_dir, config)
        if manifest["sources"] != sources:
            raise ValueError("Prepared data uses different sources; omit --reuse-prepared.")
        print(f"[prepare] Verified existing data: {args.output_dir}")
        return
    prepare_records(iter_source(sources["pretrain"]), iter_source(sources["chat"]), args.output_dir, config,
                    sources=sources, max_pretrain_scanned=args.max_pretrain_scanned,
                    max_chat_scanned=args.max_chat_scanned)
    print(f"[prepare] Wrote both Korean data stages: {args.output_dir}")


if __name__ == "__main__":
    main()
