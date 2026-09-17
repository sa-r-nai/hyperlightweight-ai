"""Stream and normalize public English pretraining and chat datasets.

The default sources are pinned revisions of FineWeb-Edu and Smol-SmolTalk.
Large source shards are streamed instead of downloaded into the repository.
Only the normalized JSONL outputs and a provenance manifest are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping


FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FINEWEB_LICENSE = "ODC-By-1.0; subject to Common Crawl Terms of Use"

CHAT_DATASET = "HuggingFaceTB/smol-smoltalk"
CHAT_REVISION = "2ca82d2fe63c017af8d4cb30e81e95cb25516851"
CHAT_LICENSE = "Apache-2.0"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful English assistant. Answer accurately, clearly, and "
    "concisely. State uncertainty instead of inventing facts."
)

PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_ascii_text(value: object, *, min_length: int = 1) -> str | None:
    """Return readable ASCII English text or ``None`` for unsuitable input."""

    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.translate(PUNCTUATION_TRANSLATION)
    printable = sum(character.isascii() and (character.isprintable() or character in "\n\t") for character in raw)
    if printable / max(1, len(raw)) < 0.80:
        return None
    text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < min_length or not re.search(r"[A-Za-z]", text):
        return None
    return text


def normalize_messages(value: object, *, max_characters: int) -> list[dict[str, str]] | None:
    """Normalize one chat while preserving a strict alternating role protocol."""

    if not isinstance(value, list):
        return None
    aliases = {"human": "user", "gpt": "assistant", "bot": "assistant"}
    messages: list[dict[str, str]] = []
    for raw_message in value:
        if not isinstance(raw_message, Mapping):
            return None
        raw_role = str(raw_message.get("role", "")).lower()
        role = aliases.get(raw_role, raw_role)
        if role not in {"system", "user", "assistant"}:
            return None
        content = normalize_ascii_text(raw_message.get("content"), min_length=1)
        if content is None:
            return None
        messages.append({"role": role, "content": content})

    if not messages:
        return None
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    if any(message["role"] == "system" for message in messages[1:]):
        return None
    expected = "user"
    for message in messages[1:]:
        if message["role"] != expected:
            return None
        expected = "assistant" if expected == "user" else "user"
    if messages[-1]["role"] != "assistant":
        return None
    if sum(len(message["content"]) for message in messages) > max_characters:
        return None
    return messages


def _write_record(handle, record: dict) -> int:
    line = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
    handle.write(line + "\n")
    return len(line)


def write_pretraining_stream(
    rows: Iterable[Mapping[str, object]],
    *,
    train_path: Path,
    validation_path: Path,
    train_count: int,
    validation_count: int,
    min_characters: int,
    max_characters: int,
) -> dict[str, int]:
    """Normalize and write a large pretraining stream without retaining it in RAM."""

    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    train_written = validation_written = rejected = 0
    train_characters = validation_characters = 0
    seen: set[bytes] = set()
    with (
        train_path.open("w", encoding="ascii", newline="\n") as train_handle,
        validation_path.open("w", encoding="ascii", newline="\n") as validation_handle,
    ):
        for row in rows:
            text = normalize_ascii_text(row.get("text"), min_length=min_characters)
            if text is None or len(text) > max_characters:
                rejected += 1
                continue
            key = hashlib.blake2b(text.encode("ascii"), digest_size=16).digest()
            if key in seen:
                rejected += 1
                continue
            seen.add(key)
            record = {"text": text}
            if validation_written < validation_count:
                validation_characters += _write_record(validation_handle, record)
                validation_written += 1
            elif train_written < train_count:
                train_characters += _write_record(train_handle, record)
                train_written += 1
            if train_written == train_count and validation_written == validation_count:
                break
    if train_written != train_count or validation_written != validation_count:
        raise RuntimeError(
            "FineWeb stream ended before the requested number of usable records "
            f"was reached ({train_written} train, {validation_written} validation)."
        )
    return {
        "train_records": train_written,
        "validation_records": validation_written,
        "train_characters": train_characters,
        "validation_characters": validation_characters,
        "rejected_records": rejected,
    }


def write_chat_stream(
    rows: Iterable[Mapping[str, object]],
    *,
    train_path: Path,
    validation_path: Path,
    train_count: int,
    validation_count: int,
    max_characters: int,
) -> dict[str, int]:
    """Normalize and write a large chat stream without retaining it in RAM."""

    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    train_written = validation_written = rejected = 0
    train_characters = validation_characters = 0
    seen: set[bytes] = set()
    with (
        train_path.open("w", encoding="ascii", newline="\n") as train_handle,
        validation_path.open("w", encoding="ascii", newline="\n") as validation_handle,
    ):
        for row in rows:
            messages = normalize_messages(row.get("messages"), max_characters=max_characters)
            if messages is None:
                rejected += 1
                continue
            canonical = json.dumps(messages, ensure_ascii=True, separators=(",", ":"))
            key = hashlib.blake2b(canonical.encode("ascii"), digest_size=16).digest()
            if key in seen:
                rejected += 1
                continue
            seen.add(key)
            record = {
                "source": str(row.get("source", "smol-smoltalk")),
                "messages": messages,
            }
            if validation_written < validation_count:
                validation_characters += _write_record(validation_handle, record)
                validation_written += 1
            elif train_written < train_count:
                train_characters += _write_record(train_handle, record)
                train_written += 1
            if train_written == train_count and validation_written == validation_count:
                break
    if train_written != train_count or validation_written != validation_count:
        raise RuntimeError(
            "Smol-SmolTalk stream ended before the requested number of usable "
            f"records was reached ({train_written} train, {validation_written} validation)."
        )
    return {
        "train_records": train_written,
        "validation_records": validation_written,
        "train_characters": train_characters,
        "validation_characters": validation_characters,
        "rejected_records": rejected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream public English data for native training.")
    parser.add_argument("--output-dir", type=Path, default=Path("public_data"))
    parser.add_argument("--fineweb-train-records", type=int, default=1_000_000)
    parser.add_argument("--fineweb-validation-records", type=int, default=10_000)
    parser.add_argument("--chat-train-records", type=int, default=400_000)
    parser.add_argument("--chat-validation-records", type=int, default=5_000)
    parser.add_argument("--min-pretraining-characters", type=int, default=200)
    parser.add_argument("--max-pretraining-characters", type=int, default=100_000)
    parser.add_argument("--max-chat-characters", type=int, default=24_000)
    parser.add_argument("--shuffle-buffer", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    numeric = {
        "fineweb train records": args.fineweb_train_records,
        "fineweb validation records": args.fineweb_validation_records,
        "chat train records": args.chat_train_records,
        "chat validation records": args.chat_validation_records,
        "shuffle buffer": args.shuffle_buffer,
    }
    if any(value <= 0 for value in numeric.values()):
        raise ValueError("All record counts and --shuffle-buffer must be positive.")

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The datasets package is required. Run: python -m pip install -r requirements-native.txt"
        ) from error

    fineweb = load_dataset(
        FINEWEB_DATASET,
        FINEWEB_CONFIG,
        split="train",
        streaming=True,
        revision=FINEWEB_REVISION,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    chats = load_dataset(
        CHAT_DATASET,
        split="train",
        streaming=True,
        revision=CHAT_REVISION,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    pretrain_train_path = args.output_dir / "pretraining_train.jsonl"
    pretrain_validation_path = args.output_dir / "pretraining_validation.jsonl"
    chat_train_path = args.output_dir / "chat_train.jsonl"
    chat_validation_path = args.output_dir / "chat_validation.jsonl"
    pretrain_stats = write_pretraining_stream(
        fineweb,
        train_path=pretrain_train_path,
        validation_path=pretrain_validation_path,
        train_count=args.fineweb_train_records,
        validation_count=args.fineweb_validation_records,
        min_characters=args.min_pretraining_characters,
        max_characters=args.max_pretraining_characters,
    )
    chat_stats = write_chat_stream(
        chats,
        train_path=chat_train_path,
        validation_path=chat_validation_path,
        train_count=args.chat_train_records,
        validation_count=args.chat_validation_records,
        max_characters=args.max_chat_characters,
    )

    outputs: dict[str, dict[str, object]] = {}
    for name, path, count, characters in (
        ("pretraining_train", pretrain_train_path, pretrain_stats["train_records"], pretrain_stats["train_characters"]),
        ("pretraining_validation", pretrain_validation_path, pretrain_stats["validation_records"], pretrain_stats["validation_characters"]),
        ("chat_train", chat_train_path, chat_stats["train_records"], chat_stats["train_characters"]),
        ("chat_validation", chat_validation_path, chat_stats["validation_records"], chat_stats["validation_characters"]),
    ):
        outputs[name] = {
            "path": path.as_posix(),
            "records": count,
            "jsonl_characters": characters,
            "sha256": sha256(path),
        }

    manifest = {
        "format": "native-public-data-v1",
        "seed": args.seed,
        "sources": {
            "pretraining": {
                "dataset": FINEWEB_DATASET,
                "config": FINEWEB_CONFIG,
                "revision": FINEWEB_REVISION,
                "license": FINEWEB_LICENSE,
                "rejected_records": pretrain_stats["rejected_records"],
            },
            "chat": {
                "dataset": CHAT_DATASET,
                "revision": CHAT_REVISION,
                "license": CHAT_LICENSE,
                "rejected_records": chat_stats["rejected_records"],
            },
        },
        "outputs": outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(
        "[info] Public data prepared: "
        f"{args.fineweb_train_records:,} pretraining and "
        f"{args.chat_train_records:,} chat training records."
    )
    print(f"[info] Provenance manifest: {manifest_path}")


if __name__ == "__main__":
    main()
