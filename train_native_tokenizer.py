"""Train the repository's English BPE tokenizer from local text files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from native_tokenizer import NativeTokenizer


SUPPORTED_SUFFIXES = {".txt", ".md", ".json", ".jsonl"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for source in inputs:
        candidates = [source] if source.is_file() else sorted(source.rglob("*"))
        for path in candidates:
            if (
                path not in seen
                and path.is_file()
                and path.suffix.lower() in SUPPORTED_SUFFIXES
                and path.name.lower() not in {"manifest.json", "readme.info"}
            ):
                seen.add(path)
                paths.append(path)
    return paths


def extract_text(path: Path) -> str:
    """Return all training text from one file (convenience API for small files)."""

    return "\n".join(iter_texts(path))


def _record_texts(value: object) -> Iterator[str]:
    values: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            messages = value.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict):
                        visit(message.get("content"))
                return
            text_keys = ("text", "content", "instruction", "output", "prompt", "completion")
            matched = False
            for key in text_keys:
                if key in value:
                    visit(value[key])
                    matched = True
            if not matched:
                ignored_metadata = {"id", "category", "source", "license", "role"}
                for key, item in value.items():
                    if key not in ignored_metadata:
                        visit(item)

    visit(value)
    yield from values


def iter_texts(path: Path) -> Iterator[str]:
    """Yield records incrementally so multi-gigabyte JSONL stays memory bounded."""

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="ascii", errors="strict") as handle:
            for line in handle:
                if line.strip():
                    yield from _record_texts(json.loads(line))
        return
    if path.suffix.lower() == ".json":
        yield from _record_texts(json.loads(path.read_text(encoding="ascii")))
        return
    with path.open("r", encoding="ascii", errors="strict") as handle:
        block: list[str] = []
        block_characters = 0
        for line in handle:
            block.append(line)
            block_characters += len(line)
            if block_characters >= 1_000_000:
                yield "".join(block)
                block = []
                block_characters = 0
        if block:
            yield "".join(block)


def limited_texts(
    paths: Iterable[Path],
    *,
    max_documents: int | None,
    max_characters: int | None,
    max_documents_per_file: int | None = None,
    max_characters_per_file: int | None = None,
) -> Iterator[str]:
    documents = 0
    characters = 0
    for path in paths:
        file_documents = 0
        file_characters = 0
        for text in iter_texts(path):
            if max_documents is not None and documents >= max_documents:
                return
            if max_documents_per_file is not None and file_documents >= max_documents_per_file:
                break
            remaining = None if max_characters is None else max_characters - characters
            file_remaining = (
                None
                if max_characters_per_file is None
                else max_characters_per_file - file_characters
            )
            if remaining is not None and remaining <= 0:
                return
            if file_remaining is not None and file_remaining <= 0:
                break
            allowed = len(text)
            if remaining is not None:
                allowed = min(allowed, remaining)
            if file_remaining is not None:
                allowed = min(allowed, file_remaining)
            text = text[:allowed]
            if text:
                yield text
                documents += 1
                characters += len(text)
                file_documents += 1
                file_characters += len(text)


def merge_pair(tokens: tuple[int, ...], pair: tuple[int, int], merged_id: int) -> tuple[int, ...]:
    result: list[int] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
            result.append(merged_id)
            index += 2
        else:
            result.append(tokens[index])
            index += 1
    return tuple(result)


def train_bpe(texts: Iterable[str], target_vocab_size: int, min_frequency: int) -> NativeTokenizer:
    if target_vocab_size < NativeTokenizer.base_vocab_size:
        raise ValueError("Target vocabulary is smaller than the ASCII base vocabulary.")
    piece_counts: Counter[tuple[int, ...]] = Counter()
    for text in texts:
        normalized = NativeTokenizer.normalize(text)
        for piece in NativeTokenizer.split_pieces(normalized):
            piece_counts[tuple(NativeTokenizer.byte_offset + value for value in piece.encode("ascii"))] += 1

    merges: list[tuple[int, int]] = []
    while NativeTokenizer.base_vocab_size + len(merges) < target_vocab_size:
        pair_counts: Counter[tuple[int, int]] = Counter()
        for tokens, frequency in piece_counts.items():
            pair_counts.update({pair: count * frequency for pair, count in Counter(zip(tokens, tokens[1:])).items()})
        if not pair_counts:
            break
        pair, frequency = min(pair_counts.items(), key=lambda item: (-item[1], item[0]))
        if frequency < min_frequency:
            break
        merged_id = NativeTokenizer.base_vocab_size + len(merges)
        merges.append(pair)
        updated: Counter[tuple[int, ...]] = Counter()
        for tokens, count in piece_counts.items():
            updated[merge_pair(tokens, pair, merged_id)] += count
        piece_counts = updated
    return NativeTokenizer(merges)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an English-only tokenizer from local data.")
    parser.add_argument("--input", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=Path("tokenizer/native_english_bpe.json"))
    parser.add_argument("--target-vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--max-documents",
        type=int,
        help="Optional tokenizer-training record cap for very large corpora.",
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        help="Optional tokenizer-training character cap for very large corpora.",
    )
    parser.add_argument("--max-documents-per-file", type=int)
    parser.add_argument("--max-characters-per-file", type=int)
    args = parser.parse_args()

    paths = iter_paths(args.input or [Path("data")])
    if not paths:
        raise FileNotFoundError("No tokenizer training files were found.")
    tokenizer = train_bpe(
        limited_texts(
            paths,
            max_documents=args.max_documents,
            max_characters=args.max_characters,
            max_documents_per_file=args.max_documents_per_file,
            max_characters_per_file=args.max_characters_per_file,
        ),
        target_vocab_size=args.target_vocab_size,
        min_frequency=args.min_frequency,
    )
    tokenizer.save(
        args.output,
        metadata={
            "target_vocab_size": args.target_vocab_size,
            "min_frequency": args.min_frequency,
            "max_documents": args.max_documents,
            "max_characters": args.max_characters,
            "max_documents_per_file": args.max_documents_per_file,
            "max_characters_per_file": args.max_characters_per_file,
            "sources": [{"path": path.as_posix(), "sha256": sha256(path)} for path in paths],
        },
    )
    print(f"[info] Wrote {tokenizer.vocab_size}-token English BPE model to {args.output}")


if __name__ == "__main__":
    main()
