"""Tokenize native text/chat JSONL into memory-mappable binary files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from array import array
from pathlib import Path
from typing import BinaryIO, Iterable

from native_data import TrainingRecord, iter_documents
from native_tokenizer import NativeTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_uint16(handle: BinaryIO, values: Iterable[int]) -> int:
    chunk = array("H", values)
    if chunk.itemsize != 2:
        raise RuntimeError("This platform does not provide 16-bit unsigned short arrays.")
    if sys.byteorder != "little":
        chunk.byteswap()
    chunk.tofile(handle)
    return len(chunk)


def write_uint8(handle: BinaryIO, values: Iterable[bool]) -> int:
    chunk = array("B", (1 if value else 0 for value in values))
    chunk.tofile(handle)
    return len(chunk)


def encode_record(
    record: TrainingRecord,
    tokenizer: NativeTokenizer,
    assistant_only_loss: bool,
) -> tuple[list[int], list[bool] | None]:
    if isinstance(record, str):
        if assistant_only_loss:
            raise ValueError("--assistant-only-loss requires chat records, not plain text.")
        tokens = tokenizer.encode(record, add_eos=True)
        return tokens, None
    if assistant_only_loss:
        return tokenizer.encode_chat_with_assistant_mask(
            record,
            add_bos=False,
            add_eos=True,
        )
    return tokenizer.encode_chat(record, add_bos=False, add_eos=True), None


def tokenize_to_binary(
    documents: Iterable[TrainingRecord],
    tokenizer: NativeTokenizer,
    *,
    output_manifest: Path,
    assistant_only_loss: bool,
    source: Path | None = None,
) -> dict[str, object]:
    if tokenizer.vocab_size > 65_536:
        raise ValueError("Binary uint16 format supports at most 65,536 tokenizer entries.")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    stem = output_manifest.stem
    tokens_path = output_manifest.with_name(f"{stem}.tokens.bin")
    mask_path = output_manifest.with_name(f"{stem}.mask.bin")
    temp_tokens = tokens_path.with_suffix(tokens_path.suffix + ".tmp")
    temp_mask = mask_path.with_suffix(mask_path.suffix + ".tmp")

    token_count = 0
    supervised_token_count = 0
    record_count = 0
    try:
        with temp_tokens.open("wb") as token_handle:
            mask_context = temp_mask.open("wb") if assistant_only_loss else None
            try:
                token_count += write_uint16(token_handle, [tokenizer.bos_token_id])
                if mask_context is not None:
                    write_uint8(mask_context, [False])
                for record in documents:
                    tokens, mask = encode_record(record, tokenizer, assistant_only_loss)
                    token_count += write_uint16(token_handle, tokens)
                    if mask_context is not None:
                        if mask is None or len(mask) != len(tokens):
                            raise RuntimeError("Token and assistant-loss mask lengths differ.")
                        write_uint8(mask_context, mask)
                        supervised_token_count += sum(mask)
                    else:
                        supervised_token_count += len(tokens)
                    record_count += 1
                    if record_count % 10_000 == 0:
                        print(
                            f"[info] Tokenized {record_count:,} records / "
                            f"{token_count:,} tokens"
                        )
            finally:
                if mask_context is not None:
                    mask_context.close()
        if record_count == 0 or token_count < 2:
            raise RuntimeError("No usable records were tokenized.")
        if assistant_only_loss and supervised_token_count == 0:
            raise RuntimeError("Chat data did not contain any assistant targets.")
        temp_tokens.replace(tokens_path)
        if assistant_only_loss:
            temp_mask.replace(mask_path)
    except BaseException:
        temp_tokens.unlink(missing_ok=True)
        temp_mask.unlink(missing_ok=True)
        raise

    manifest: dict[str, object] = {
        "format": "native-token-binary-v1",
        "dtype": "uint16-le",
        "tokens_file": tokens_path.name,
        "mask_file": mask_path.name if assistant_only_loss else None,
        "token_count": token_count,
        "supervised_token_count": supervised_token_count,
        "record_count": record_count,
        "assistant_only_loss": assistant_only_loss,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "tokens_sha256": sha256(tokens_path),
    }
    if assistant_only_loss:
        manifest["mask_sha256"] = sha256(mask_path)
    if source is not None:
        manifest["source"] = {
            "path": source.as_posix(),
            "sha256": sha256(source) if source.is_file() else None,
        }
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create native memory-mapped token data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizer/native_english_bpe.json"),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON manifest path.")
    parser.add_argument("--assistant-only-loss", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    tokenizer = NativeTokenizer.load(args.tokenizer)
    manifest = tokenize_to_binary(
        iter_documents(args.input),
        tokenizer,
        output_manifest=args.output,
        assistant_only_loss=args.assistant_only_loss,
        source=args.input,
    )
    print(
        f"[info] Wrote {manifest['token_count']:,} tokens from "
        f"{manifest['record_count']:,} records to {args.output}"
    )


if __name__ == "__main__":
    main()
