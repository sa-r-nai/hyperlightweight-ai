"""Train the repository's English BPE tokenizer from local text files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

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
    for source in inputs:
        candidates = [source] if source.is_file() else source.rglob("*")
        paths.extend(
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
    return sorted(set(paths))


def extract_text(path: Path) -> str:
    raw = path.read_text(encoding="ascii")
    if path.suffix.lower() not in {".json", ".jsonl"}:
        return raw
    values: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    if path.suffix.lower() == ".jsonl":
        for line in raw.splitlines():
            if line.strip():
                visit(json.loads(line))
    else:
        visit(json.loads(raw))
    return "\n".join(values)


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
    args = parser.parse_args()

    paths = iter_paths(args.input or [Path("data")])
    if not paths:
        raise FileNotFoundError("No tokenizer training files were found.")
    tokenizer = train_bpe(
        (extract_text(path) for path in paths),
        target_vocab_size=args.target_vocab_size,
        min_frequency=args.min_frequency,
    )
    tokenizer.save(
        args.output,
        metadata={
            "target_vocab_size": args.target_vocab_size,
            "min_frequency": args.min_frequency,
            "sources": [{"path": str(path), "sha256": sha256(path)} for path in paths],
        },
    )
    print(f"[info] Wrote {tokenizer.vocab_size}-token English BPE model to {args.output}")


if __name__ == "__main__":
    main()
