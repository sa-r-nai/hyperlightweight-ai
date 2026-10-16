"""Prepare pinned Korean conversations without loading model weights.

Provenance: https://huggingface.co/datasets/lemon-mint/smol-koreantalk
The publisher labels this Korean translation of Smol-SmolTalk Apache-2.0.
Only ``messages[].role`` and Korean ``messages[].content`` are retained;
the parallel ``content_en`` text is never used as a training target.
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


DATASET_ID = "lemon-mint/smol-koreantalk"
DATASET_REVISION = "04eb12c1f999578d74dd87720bc853cedb7fa156"
DATASET_LICENSE = "Apache-2.0"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
MANIFEST_FORMAT = "korean-600m-data-v1"
DEFAULT_OUTPUT_DIR = Path("artifacts_korean_600m/data")
MAX_SCANNED_RECORDS = 2_000_000
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
    """A source row that cannot become a supervised conversation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def hangul_ratio(text: str) -> float:
    """Measure Hangul among alphabetic characters, excluding punctuation."""

    alphabetic = [character for character in text if character.isalpha()]
    if not alphabetic:
        return 0.0
    hangul = sum(
        "\uac00" <= character <= "\ud7a3"
        or "\u1100" <= character <= "\u11ff"
        or "\u3130" <= character <= "\u318f"
        or "\ua960" <= character <= "\ua97f"
        or "\ud7b0" <= character <= "\ud7ff"
        for character in alphabetic
    )
    return hangul / len(alphabetic)


def _normalize_content(content: object, config: Mapping) -> str:
    if not isinstance(content, str) or not content.strip():
        raise RejectedRecord("empty_content")
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        and character not in "\n\r\t"
        for character in content
    ):
        raise RejectedRecord("control_characters")
    lowered = content.casefold()
    if any(marker.casefold() in lowered for marker in TEMPLATE_MARKERS):
        raise RejectedRecord("template_or_reasoning")
    if REASONING_HEADER.search(content):
        raise RejectedRecord("template_or_reasoning")
    content = unicodedata.normalize("NFC", content)
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = re.sub(r"[ \t]+", " ", content).strip()
    if hangul_ratio(content) < config["data"]["min_hangul_ratio"]:
        raise RejectedRecord("insufficient_korean")
    if len(content) > config["data"]["max_characters"]:
        raise RejectedRecord("too_long")
    return content


def normalize_messages(value: object, config: Mapping) -> list[dict[str, str]]:
    """Validate Korean chat and retain source task instructions in the user turn.

    The fixed system instruction defines the assistant behavior. A source system
    message can also contain the task itself, so it becomes a labeled part of
    the first user request. Already normalized conversations are idempotent.
    """

    if not isinstance(value, list) or not value:
        raise RejectedRecord("invalid_messages")
    messages = list(value)
    fixed_system = str(config["system_prompt"]).strip()
    source_instruction: str | None = None
    if isinstance(messages[0], Mapping) and messages[0].get("role") == "system":
        source_system = messages[0].get("content")
        if not isinstance(source_system, str) or source_system.strip() != fixed_system:
            source_instruction = _normalize_content(source_system, config)
        messages = messages[1:]
    if len(messages) < 2 or len(messages) % 2:
        raise RejectedRecord("invalid_roles")
    result = [{"role": "system", "content": fixed_system}]
    characters = 0
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if not isinstance(message, Mapping) or message.get("role") != expected:
            raise RejectedRecord("invalid_roles")
        content = _normalize_content(message.get("content"), config)
        if index == 0 and source_instruction is not None:
            content = _normalize_content(
                f"작업 지침:\n{source_instruction}\n\n사용자 요청:\n{content}", config,
            )
        characters += len(content)
        if characters > config["data"]["max_characters"]:
            raise RejectedRecord("too_long")
        result.append({"role": expected, "content": content})
    return result


def prompt_group(messages: list[dict[str, str]]) -> str:
    """Group equivalent first user prompts before assigning a split."""

    first_prompt = next(message["content"] for message in messages if message["role"] == "user")
    canonical = " ".join(unicodedata.normalize("NFC", first_prompt).split()).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def split_for_group(group: str, *, seed: int, train_records: int, validation_records: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode("ascii")).digest()
    bucket = int.from_bytes(digest[:8], "big")
    threshold = (validation_records * (1 << 64)) // (train_records + validation_records)
    return "validation" if bucket < threshold else "train"


def buffered_shuffle(records: Iterable[object], *, seed: int, buffer_size: int) -> Iterator[object]:
    """Deterministically shuffle a stream with bounded memory."""

    if buffer_size <= 0:
        raise ValueError("buffer_size must be positive.")
    generator = random.Random(seed)
    buffer: list[object] = []
    for record in records:
        if len(buffer) < buffer_size:
            buffer.append(record)
            continue
        index = generator.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = record
    generator.shuffle(buffer)
    yield from buffer


def public_source() -> dict:
    return {
        "kind": "huggingface",
        "id": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": "train",
        "license": DATASET_LICENSE,
        "url": DATASET_URL,
    }


def local_source(path: Path, license_name: str | None) -> dict:
    if not isinstance(license_name, str) or not license_name.strip():
        raise ValueError("--local-jsonl requires an explicit --local-license.")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "kind": "local-jsonl",
        "id": "local-jsonl",
        "revision": sha256_file(path),
        "path": str(path),
        "license": license_name.strip(),
        "url": None,
    }


def iter_source(source: Mapping) -> Iterator[object]:
    """Read JSONL or lazily import the optional Hugging Face dependency."""

    if source["kind"] == "local-jsonl":
        with Path(source["path"]).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield None
        return
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Install requirements-korean-600m.txt in Lightning AI Studio first.") from error
    yield from load_dataset(
        source["id"],
        revision=source["revision"],
        split=source["split"],
        streaming=True,
    )


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def prepare_records(
    records: Iterable[object],
    output_dir: Path,
    config: dict,
    *,
    source: dict,
    train_records: int | None = None,
    validation_records: int | None = None,
    max_scanned_records: int = MAX_SCANNED_RECORDS,
    shuffle_buffer_size: int = SHUFFLE_BUFFER_SIZE,
) -> dict:
    """Write exact-size held-out splits, then publish an integrity manifest."""

    quotas = {
        "train": config["data"]["train_records"] if train_records is None else train_records,
        "validation": config["data"]["validation_records"] if validation_records is None else validation_records,
    }
    for name, value in (*quotas.items(), ("max_scanned_records", max_scanned_records),
                        ("shuffle_buffer_size", shuffle_buffer_size)):
        _positive_integer(name, value)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The manifest is the completion marker. Failed or interrupted new preparation
    # must never leave an apparently complete artifact from a previous request.
    manifest_path = output_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    paths = {name: output_dir / f"{name}.jsonl" for name in quotas}
    temporaries = {name: path.with_suffix(".jsonl.tmp") for name, path in paths.items()}
    seen_records: set[str] = set()
    seen_groups: set[str] = set()
    counts: Counter = Counter()
    rejected: Counter = Counter()
    scanned = 0
    try:
        with temporaries["train"].open("w", encoding="utf-8", newline="\n") as train_handle, \
             temporaries["validation"].open("w", encoding="utf-8", newline="\n") as validation_handle:
            handles = {"train": train_handle, "validation": validation_handle}
            bounded_records = islice(records, max_scanned_records)
            for record in buffered_shuffle(bounded_records, seed=config["seed"], buffer_size=shuffle_buffer_size):
                scanned += 1
                try:
                    if not isinstance(record, Mapping):
                        raise RejectedRecord("invalid_record")
                    messages = normalize_messages(record.get("messages"), config)
                except RejectedRecord as error:
                    rejected[error.reason] += 1
                else:
                    canonical = json.dumps({"messages": messages}, ensure_ascii=False, separators=(",", ":"))
                    record_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                    group = prompt_group(messages)
                    if record_hash in seen_records:
                        rejected["duplicate_record"] += 1
                    elif group in seen_groups:
                        rejected["duplicate_first_prompt"] += 1
                    else:
                        seen_records.add(record_hash)
                        seen_groups.add(group)
                        split = split_for_group(group, seed=config["seed"], train_records=quotas["train"],
                                                validation_records=quotas["validation"])
                        if counts[split] >= quotas[split]:
                            rejected["split_quota_full"] += 1
                        else:
                            handles[split].write(canonical + "\n")
                            counts[split] += 1
                if scanned % 10_000 == 0:
                    print(f"[prepare] scanned={scanned:,} train={counts['train']:,} "
                          f"validation={counts['validation']:,}", flush=True)
                if all(counts[name] == quota for name, quota in quotas.items()):
                    break
                if scanned >= max_scanned_records:
                    break
        if any(counts[name] != quota for name, quota in quotas.items()):
            raise RuntimeError(
                "The source did not provide enough usable Korean conversations within the scan limit: "
                f"train={counts['train']}/{quotas['train']}, "
                f"validation={counts['validation']}/{quotas['validation']}, scanned={scanned}. "
                "Use more local data, lower the explicit record counts, or increase --max-scanned-records."
            )
        for name, path in paths.items():
            temporaries[name].replace(path)
        manifest = {
            "format": MANIFEST_FORMAT,
            "source": source,
            "config_fingerprint": config_fingerprint(config),
            "preparation": {
                "seed": config["seed"],
                "train_records": quotas["train"],
                "validation_records": quotas["validation"],
                "max_scanned_records": max_scanned_records,
                "shuffle_buffer_size": shuffle_buffer_size,
                "split_method": "sha256-first-user-prompt-v1",
                "deduplication": "exact-conversation-and-normalized-first-user-prompt",
            },
            "outputs": {
                name: {"file": path.name, "records": counts[name], "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
            "statistics": {"scanned_records": scanned, "written_records": sum(counts.values()),
                           "rejected_records": sum(rejected.values()), "rejections": dict(rejected)},
        }
        write_json(manifest_path, manifest)
        return manifest
    finally:
        for path in temporaries.values():
            path.unlink(missing_ok=True)


def validate_prepared(output_dir: Path, config: dict) -> dict:
    """Verify the completion marker, configuration, provenance, rows and hashes."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No complete preparation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("Unsupported prepared-data manifest format.")
    if manifest.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError("Prepared data uses a different configuration; prepare it again.")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("Prepared data has no source identity.")
    if source.get("kind") == "huggingface":
        if source != public_source():
            raise ValueError("Prepared data does not match the pinned public dataset identity.")
    elif source.get("kind") == "local-jsonl":
        if not source.get("license") or not re.fullmatch(r"[0-9a-f]{64}", source.get("revision", "")):
            raise ValueError("Local data requires a source hash and explicit license.")
        source_path = Path(source.get("path", ""))
        if not source_path.is_file() or sha256_file(source_path) != source["revision"]:
            raise ValueError("The local source data is missing or has changed.")
    else:
        raise ValueError("Unsupported prepared-data source kind.")
    preparation = manifest.get("preparation")
    outputs = manifest.get("outputs")
    if not isinstance(preparation, dict) or not isinstance(outputs, dict):
        raise ValueError("Prepared data is missing its output or preparation metadata.")
    all_groups: set[str] = set()
    for name in ("train", "validation"):
        entry = outputs.get(name)
        if not isinstance(entry, dict) or entry.get("file") != f"{name}.jsonl":
            raise ValueError(f"Invalid {name} output entry.")
        expected_count = preparation.get(f"{name}_records")
        _positive_integer(f"{name}_records", expected_count)
        if entry.get("records") != expected_count:
            raise ValueError(f"The {name} record count differs from the preparation request.")
        path = output_dir / f"{name}.jsonl"
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"Prepared {name} data hash mismatch; prepare it again.")
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                messages = normalize_messages(row.get("messages"), config)
                if row != {"messages": messages}:
                    raise ValueError(f"Prepared {name} row has not been normalized.")
                group = prompt_group(messages)
                if group in all_groups:
                    raise ValueError("Prepared data repeats a first user prompt within or across splits.")
                if split_for_group(group, seed=config["seed"],
                                   train_records=preparation["train_records"],
                                   validation_records=preparation["validation_records"]) != name:
                    raise ValueError("Prepared data has an incorrect deterministic group split.")
                all_groups.add(group)
                count += 1
        if count != expected_count:
            raise ValueError(f"Prepared {name} file does not contain its declared number of rows.")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pinned Korean conversations for Lightning AI Studio.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-jsonl", type=Path, help="Optional UTF-8 JSONL containing messages arrays.")
    parser.add_argument("--local-license", help="License identifier or explicit permission for the local source.")
    parser.add_argument("--train-records", type=int)
    parser.add_argument("--validation-records", type=int)
    parser.add_argument("--max-scanned-records", type=int, default=MAX_SCANNED_RECORDS)
    parser.add_argument("--reuse-prepared", action="store_true", help="Verify and reuse complete, matching artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.local_license and args.local_jsonl is None:
        raise ValueError("--local-license requires --local-jsonl.")
    source = local_source(args.local_jsonl, args.local_license) if args.local_jsonl else public_source()
    requested = {
        "train_records": config["data"]["train_records"] if args.train_records is None else args.train_records,
        "validation_records": config["data"]["validation_records"] if args.validation_records is None else args.validation_records,
    }
    for name, value in (*requested.items(), ("max_scanned_records", args.max_scanned_records)):
        _positive_integer(name, value)
    if args.reuse_prepared:
        manifest = validate_prepared(args.output_dir, config)
        if manifest["source"] != source:
            raise ValueError("Prepared data uses another source; rerun without --reuse-prepared.")
        if any(manifest["preparation"][key] != value for key, value in requested.items()):
            raise ValueError("Prepared record counts differ; rerun without --reuse-prepared.")
        print(f"[prepare] Verified existing data in {args.output_dir}")
        return
    manifest = prepare_records(
        iter_source(source), args.output_dir, config, source=source,
        train_records=requested["train_records"], validation_records=requested["validation_records"],
        max_scanned_records=args.max_scanned_records,
    )
    print(f"[prepare] Wrote {manifest['statistics']['written_records']:,} conversations to {args.output_dir}")


if __name__ == "__main__":
    main()
