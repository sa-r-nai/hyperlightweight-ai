"""Shared readers for native pretraining and chat records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Mapping, Sequence


TrainingRecord = str | Sequence[Mapping[str, str]]


def read_json_record(line: str) -> TrainingRecord | None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    instruction = record.get("instruction")
    output = record.get("output")
    if (
        isinstance(instruction, str)
        and instruction.strip()
        and isinstance(output, str)
        and output.strip()
    ):
        return [
            {"role": "user", "content": instruction.strip()},
            {"role": "assistant", "content": output.strip()},
        ]
    for key in ("text", "content", "prompt", "completion"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    normalized_messages: list[dict[str, str]] = []
    aliases = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = aliases.get(str(message.get("role", "unknown")).lower())
        content = message.get("content")
        if role and isinstance(content, str) and content.strip():
            normalized_messages.append({"role": role, "content": content.strip()})
    return normalized_messages or None


def iter_documents(data_path: Path) -> Iterator[TrainingRecord]:
    paths = [data_path] if data_path.is_file() else sorted(data_path.rglob("*"))
    supported = {".txt", ".md", ".json", ".jsonl"}
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        if path.name.lower() in {"manifest.json", "readme.info"}:
            continue
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="ascii", errors="strict").strip()
            if text:
                yield text
            continue
        if path.suffix.lower() == ".json":
            raw_records = json.loads(path.read_text(encoding="ascii", errors="strict"))
            if isinstance(raw_records, dict):
                raw_records = [raw_records]
            if isinstance(raw_records, list):
                for raw_record in raw_records:
                    if isinstance(raw_record, dict):
                        record = read_json_record(json.dumps(raw_record, ensure_ascii=True))
                        if record is not None:
                            yield record
            continue
        with path.open("r", encoding="ascii", errors="strict") as handle:
            for line in handle:
                record = read_json_record(line)
                if record is not None:
                    yield record


__all__ = ["TrainingRecord", "iter_documents", "read_json_record"]
