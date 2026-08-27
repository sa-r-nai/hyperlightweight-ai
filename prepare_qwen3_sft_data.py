"""Normalize local and external Korean instruction data to chat messages.

The output is suitable as an SFT input artifact, but no fine-tuning is run by
this script. The official Qwen3 checkpoint remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


SEED_DATA_PATH = Path("data/korean_conversation_sft_seed.jsonl")
EXTERNAL_DATA_PATH = Path(
    "sft_data/external/CarrotAI-ko-instruction-dataset/instruction_korean.json"
)
TRAIN_OUTPUT_PATH = Path("sft_data/qwen3_korean_sft_train.jsonl")
VALIDATION_OUTPUT_PATH = Path("sft_data/qwen3_korean_sft_validation.jsonl")
MANIFEST_PATH = Path("sft_data/MANIFEST.json")
RANDOM_SEED = 42
VALIDATION_RATIO = 0.02
SYSTEM_PROMPT = (
    "너는 친절하고 정확한 한국어 도우미다. 질문의 핵심을 먼저 파악하고 "
    "자연스러운 한국어로 유용하게 답한다. 확실하지 않은 사실은 추측하지 않는다."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_seed_records() -> list[dict]:
    records: list[dict] = []
    with SEED_DATA_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                record["license"] = "self-authored"
                records.append(record)
    return records


def load_external_records() -> list[dict]:
    source_records = json.loads(EXTERNAL_DATA_PATH.read_text(encoding="utf-8"))
    records: list[dict] = []
    for index, source in enumerate(source_records, start=1):
        instruction = source.get("instruction")
        output = source.get("output")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        if not isinstance(output, str) or not output.strip():
            continue
        records.append(
            {
                "id": f"carrotai-ko-instruction-{index:05d}",
                "category": "general_instruction",
                "source": "CarrotAI/ko-instruction-dataset",
                "license": "Apache-2.0",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction.strip()},
                    {"role": "assistant", "content": output.strip()},
                ],
            }
        )
    return records


def conversation_key(record: dict) -> tuple[str, ...]:
    return tuple(
        f"{message['role']}\0{message['content'].strip()}"
        for message in record["messages"]
        if message["role"] != "system"
    )


def validate_record(record: dict) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError(f"Invalid messages in {record.get('id')}")
    if messages[0].get("role") != "system":
        raise ValueError(f"First role must be system in {record.get('id')}")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"Last role must be assistant in {record.get('id')}")
    expected = "user"
    for message in messages[1:]:
        if message.get("role") != expected:
            raise ValueError(f"Roles do not alternate in {record.get('id')}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"Empty message in {record.get('id')}")
        expected = "assistant" if expected == "user" else "user"


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    missing = [path for path in (SEED_DATA_PATH, EXTERNAL_DATA_PATH) if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing source data: {joined}")

    combined = load_external_records() + load_seed_records()
    unique: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for record in combined:
        validate_record(record)
        key = conversation_key(record)
        if key not in seen:
            seen.add(key)
            unique.append(record)

    random.Random(RANDOM_SEED).shuffle(unique)
    validation_count = max(1, round(len(unique) * VALIDATION_RATIO))
    validation_records = unique[:validation_count]
    train_records = unique[validation_count:]

    write_jsonl(TRAIN_OUTPUT_PATH, train_records)
    write_jsonl(VALIDATION_OUTPUT_PATH, validation_records)

    manifest = {
        "format": "messages-jsonl",
        "random_seed": RANDOM_SEED,
        "validation_ratio": VALIDATION_RATIO,
        "record_counts": {
            "combined_before_deduplication": len(combined),
            "unique": len(unique),
            "train": len(train_records),
            "validation": len(validation_records),
        },
        "sources": [
            {
                "name": "self-authored Korean conversation seed",
                "path": str(SEED_DATA_PATH),
                "sha256": sha256(SEED_DATA_PATH),
                "records": len(load_seed_records()),
            },
            {
                "name": "CarrotAI/ko-instruction-dataset",
                "path": str(EXTERNAL_DATA_PATH),
                "url": "https://huggingface.co/datasets/CarrotAI/ko-instruction-dataset",
                "declared_license": "Apache-2.0",
                "sha256": sha256(EXTERNAL_DATA_PATH),
                "records": len(load_external_records()),
            },
        ],
        "outputs": [
            {"path": str(TRAIN_OUTPUT_PATH), "sha256": sha256(TRAIN_OUTPUT_PATH)},
            {"path": str(VALIDATION_OUTPUT_PATH), "sha256": sha256(VALIDATION_OUTPUT_PATH)},
        ],
        "checkpoint_relationship": (
            "These records are supplemental SFT data and were not used to create "
            "the official Qwen3-0.6B checkpoint stored by this project."
        ),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Unique records: {len(unique)}")
    print(f"Train records: {len(train_records)} -> {TRAIN_OUTPUT_PATH}")
    print(f"Validation records: {len(validation_records)} -> {VALIDATION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
