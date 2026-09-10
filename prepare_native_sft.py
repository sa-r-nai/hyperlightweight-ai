"""Validate, deduplicate, split, and record provenance for chat data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(paths: Iterable[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                validate_record(record, f"{path}:{line_number}")
                records.append(record)
    return records


def validate_record(record: dict, location: str) -> None:
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        raise ValueError(f"Missing messages array: {location}")
    messages = record["messages"]
    if len(messages) < 3 or messages[0].get("role") != "system":
        raise ValueError(f"Conversation does not start with a system message: {location}")
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Empty message content: {location}")
        try:
            content.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError(f"Non-ASCII text in English-only data: {location}") from error
    expected_role = "user"
    for message in messages[1:]:
        if message.get("role") != expected_role:
            raise ValueError(f"Message roles do not alternate: {location}")
        expected_role = "assistant" if expected_role == "user" else "user"
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"Conversation does not end with an assistant: {location}")


def conversation_key(record: dict) -> tuple[tuple[str, str], ...]:
    return tuple(
        (message["role"], message["content"].strip())
        for message in record["messages"]
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and split English SFT data.")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("sft_data"))
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("validation-ratio must be between zero and one.")

    input_paths = list(
        dict.fromkeys(args.input or [Path("data/native_sft_seed.jsonl")])
    )
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(map(str, missing)))

    source_records = load_records(input_paths)
    unique_records: list[dict] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for record in source_records:
        key = conversation_key(record)
        if key not in seen:
            seen.add(key)
            unique_records.append(record)

    random.Random(args.seed).shuffle(unique_records)
    validation_count = max(1, round(len(unique_records) * args.validation_ratio))
    validation_records = unique_records[:validation_count]
    train_records = unique_records[validation_count:]
    train_path = args.output_dir / "native_sft_train.jsonl"
    validation_path = args.output_dir / "native_sft_validation.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(validation_path, validation_records)

    manifest = {
        "format": "native-english-messages-jsonl-v1",
        "random_seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "record_counts": {
            "input": len(source_records),
            "unique": len(unique_records),
            "train": len(train_records),
            "validation": len(validation_records),
        },
        "sources": [
            {
                "path": str(path),
                "sha256": sha256(path),
                "license": "self-authored" if "native_sft_seed" in path.name else "recorded-by-caller",
            }
            for path in input_paths
        ],
        "outputs": [
            {"path": str(train_path), "sha256": sha256(train_path)},
            {"path": str(validation_path), "sha256": sha256(validation_path)},
        ],
        "model_relationship": (
            "These records are independent training material. No pretrained "
            "model weights or external model tokenizer are included."
        ),
    }
    manifest_path = args.output_dir / "native_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="ascii",
    )
    print(f"[info] Prepared {len(unique_records)} unique records.")
    print(f"[info] Training data: {train_path}")
    print(f"[info] Validation data: {validation_path}")
    print(f"[info] Provenance manifest: {manifest_path}")


if __name__ == "__main__":
    main()
