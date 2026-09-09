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
        raise ValueError(f"메시지 배열이 없습니다: {location}")
    messages = record["messages"]
    if len(messages) < 3 or messages[0].get("role") != "system":
        raise ValueError(f"system을 포함한 대화가 아닙니다: {location}")
    expected_role = "user"
    for message in messages[1:]:
        if message.get("role") != expected_role:
            raise ValueError(f"역할이 번갈아 나오지 않습니다: {location}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"빈 메시지가 있습니다: {location}")
        expected_role = "assistant" if expected_role == "user" else "user"
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"assistant로 끝나지 않습니다: {location}")


def conversation_key(record: dict) -> tuple[tuple[str, str], ...]:
    return tuple(
        (message["role"], message["content"].strip())
        for message in record["messages"]
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="NativeByteLM SFT 데이터 준비")
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
        raise ValueError("validation-ratio는 0과 1 사이여야 합니다.")

    input_paths = list(
        dict.fromkeys(args.input or [Path("data/native_sft_seed.jsonl")])
    )
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("입력 파일이 없습니다: " + ", ".join(map(str, missing)))

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
        "format": "nativebytelm-messages-jsonl-v1",
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
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[정보] 중복 제거 후 {len(unique_records)}건을 준비했습니다.")
    print(f"[정보] 학습 데이터: {train_path}")
    print(f"[정보] 검증 데이터: {validation_path}")
    print(f"[정보] 출처 매니페스트: {manifest_path}")


if __name__ == "__main__":
    main()
