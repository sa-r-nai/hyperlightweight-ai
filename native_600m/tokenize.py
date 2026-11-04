"""Stream prepared records into little-endian uint16 token artifacts."""

from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import sys
import time
from typing import Mapping

from .config import DEFAULT_CPU_WORKERS, config_fingerprint, load_config, sha256_file, write_json
from .prepare import _batches, _ordered_batches, _prepared_file_digest, public_source, validate_prepared

MANIFEST_FORMAT = "native-600m-tokens-v1"
DEFAULT_OUTPUT_DIR = Path("artifacts_native_600m/tokens")
SPLITS = ("pretrain_train", "pretrain_validation", "chat_train", "chat_validation")
_WORKER_TOKENIZER = None


@dataclass
class EncodedBatch:
    tokens: bytes
    mask: bytes
    records: int
    token_count: int
    supervised_tokens: int


def _init_tokenization_worker(tokenizer) -> None:
    from .tokenizer import configure_tokenizer_threads

    # One Rust thread per process avoids nested worker pools.
    configure_tokenizer_threads(1)
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = tokenizer


def _encode_lines(lines: list[str], is_chat: bool, tokenizer=None) -> EncodedBatch:
    tokenizer = _WORKER_TOKENIZER if tokenizer is None else tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenization worker has no tokenizer.")
    rows = [json.loads(line) for line in lines]
    if is_chat:
        conversations = [row["messages"] for row in rows]
        encode_batch = getattr(tokenizer, "encode_chats_batch", None)
        encoded = encode_batch(conversations) if encode_batch else [tokenizer.encode_chat(row) for row in conversations]
    else:
        texts = [row["text"] for row in rows]
        encode_batch = getattr(tokenizer, "encode_documents_batch", None)
        encoded = encode_batch(texts) if encode_batch else [tokenizer.encode_document(text) for text in texts]
    if len(encoded) != len(rows):
        raise ValueError("Tokenizer batch output does not match its input record count.")
    tokens, masks = bytearray(), bytearray()
    token_count = supervised_tokens = 0
    vocab_size = tokenizer.vocab_size
    for item in encoded:
        if is_chat:
            ids, mask = item
            if not isinstance(mask, (list, tuple)) or len(mask) != len(ids):
                raise ValueError("Chat token and assistant-mask lengths differ.")
            if any(type(value) is not int or value not in (0, 1) for value in mask):
                raise ValueError("Assistant masks must contain only integer zeroes and ones.")
            if not any(mask) or mask[0] != 0:
                raise ValueError("A chat record needs supervised assistant targets after its first token.")
            masks.extend(mask)
            supervised_tokens += sum(mask)
        else:
            ids = item
            supervised_tokens += len(ids) - 1
        tokens.extend(_token_bytes(ids, vocab_size))
        token_count += len(ids)
    return EncodedBatch(bytes(tokens), bytes(masks), len(rows), token_count, supervised_tokens)


def _token_bytes(ids: object, vocab_size: int) -> bytes:
    if not isinstance(ids, (list, tuple)) or len(ids) < 2:
        raise ValueError("Every encoded record must contain at least two tokens.")
    if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
        raise ValueError("The tokenizer emitted an invalid token ID.")
    values = array("H", ids)
    if values.itemsize != 2:
        raise RuntimeError("This platform has no 16-bit unsigned short array.")
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def tokenize_records(data_dir: Path, tokenizer: object, output_dir: Path, config: dict, *, workers: int = 1) -> dict:
    """Encode bounded batches in CPU processes, writing results in source order.

    Chat masks select the token at the same index. Training shifts these masks
    with its target labels. BOS/EOS and explicit turn boundaries separate
    conversations even when training windows cross record boundaries.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer.")
    if workers == 1:
        from .tokenizer import configure_tokenizer_threads
        configure_tokenizer_threads(1)
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    source_manifest = validate_prepared(data_dir, config, workers=workers)
    vocab_size = tokenizer.vocab_size
    if type(vocab_size) is not int or not 263 <= vocab_size <= 65535:
        raise ValueError("The independent tokenizer vocabulary must fit uint16.")
    if vocab_size > config["model"]["vocab_size"]:
        raise ValueError("The tokenizer vocabulary exceeds the model embedding table.")
    if not re.fullmatch(r"[0-9a-f]{64}", str(tokenizer.fingerprint)):
        raise ValueError("The tokenizer requires a SHA-256 fingerprint.")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    temporaries = []
    executor = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                    initializer=_init_tokenization_worker, initargs=(tokenizer,))
                if workers > 1 else nullcontext(None))
    try:
        with executor as pool:
            return _write_tokenized_splits(pool, workers, data_dir, tokenizer, output_dir, config,
                                           source_manifest, temporaries)
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)


def _write_tokenized_splits(pool, workers, data_dir, tokenizer, output_dir, config, source_manifest, temporaries):
    outputs = {}
    for name in SPLITS:
        token_path = output_dir / f"{name}.bin"
        token_tmp = token_path.with_suffix(".bin.tmp")
        temporaries.append(token_tmp)
        is_chat = name.startswith("chat_")
        mask_path = output_dir / f"{name}.mask.bin" if is_chat else None
        mask_tmp = mask_path.with_suffix(".bin.tmp") if mask_path else None
        if mask_tmp:
            temporaries.append(mask_tmp)
        token_count = supervised_tokens = records = last_report = 0
        token_digest, mask_digest = hashlib.sha256(), hashlib.sha256()
        started = time.perf_counter()
        total_records = source_manifest["outputs"][name]["records"]
        print(f"[tokenize:{name}] records=0/{total_records:,} workers={workers}", flush=True)
        mask_handle = mask_tmp.open("wb") if mask_tmp else None
        try:
            with (data_dir / source_manifest["outputs"][name]["file"]).open("r", encoding="utf-8") as source, \
                 token_tmp.open("wb") as target:
                results = _ordered_batches(pool, _encode_lines, _batches(source), workers,
                                           is_chat, tokenizer if pool is None else None)
                try:
                    for batch in results:
                        target.write(batch.tokens)
                        token_digest.update(batch.tokens)
                        if mask_handle:
                            mask_handle.write(batch.mask)
                            mask_digest.update(batch.mask)
                        token_count += batch.token_count
                        supervised_tokens += batch.supervised_tokens
                        records += batch.records
                        if records - last_report >= 10_000:
                            elapsed = max(time.perf_counter() - started, 1e-6)
                            print(f"[tokenize:{name}] records={records:,}/{total_records:,} "
                                  f"tokens={token_count:,} records/s={records / elapsed:.0f}", flush=True)
                            last_report = records
                finally:
                    results.close()
        finally:
            if mask_handle:
                mask_handle.close()
        if records != source_manifest["outputs"][name]["records"] or token_count < 2:
            raise ValueError(f"Incomplete tokenization for {name}.")
        token_tmp.replace(token_path)
        entry = {"tokens_file": token_path.name, "tokens_sha256": token_digest.hexdigest(),
                 "token_count": token_count, "supervised_tokens": supervised_tokens, "records": records}
        if mask_tmp:
            mask_tmp.replace(mask_path)
            entry.update({"mask_file": mask_path.name, "mask_sha256": mask_digest.hexdigest()})
        outputs[name] = entry
        print(f"[tokenize:{name}] records={records:,}/{total_records:,} tokens={token_count:,} OK", flush=True)
    manifest = {"format": MANIFEST_FORMAT, "token_dtype": "<u2", "mask_dtype": "u1",
                "config_fingerprint": config_fingerprint(config),
                "tokenizer_fingerprint": tokenizer.fingerprint, "vocab_size": tokenizer.vocab_size,
                "source_manifest_sha256": sha256_file(data_dir / "manifest.json"), "outputs": outputs,
                "data_provenance": source_manifest,
                "chat_mask_alignment": "same-index-target-token"}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _verify_token_files(root: Path, name: str, entry: dict) -> None:
    count = entry["token_count"]
    token_path = root / entry["tokens_file"]
    if (not token_path.is_file() or token_path.stat().st_size != 2 * count
            or _prepared_file_digest(token_path, f"tokens:{name}") != entry.get("tokens_sha256")):
        raise ValueError(f"Corrupt {name} token file.")
    if name.startswith("chat_"):
        mask_path = root / entry["mask_file"]
        if (not mask_path.is_file() or mask_path.stat().st_size != count
                or _prepared_file_digest(mask_path, f"mask:{name}") != entry.get("mask_sha256")):
            raise ValueError(f"Corrupt {name} assistant mask.")
        mask_sum = 0
        with mask_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if max(chunk, default=0) > 1:
                    raise ValueError(f"Nonbinary {name} assistant mask.")
                mask_sum += sum(chunk)
        if mask_sum != entry["supervised_tokens"]:
            raise ValueError(f"Incorrect {name} assistant target count.")


def validate_tokenized(output_dir: Path, config: dict, *, data_dir: Path | None = None,
                       tokenizer: object | None = None, workers: int = 1) -> dict:
    """Check artifact identity and byte sizes before reusing token streams."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer.")
    output_dir = Path(output_dir)
    path = output_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No complete token manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT or manifest.get("token_dtype") != "<u2" \
            or manifest.get("mask_dtype") != "u1" or manifest.get("chat_mask_alignment") != "same-index-target-token":
        raise ValueError("Unsupported token artifact schema.")
    if manifest.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError("Token artifacts use a different configuration.")
    provenance = manifest.get("data_provenance")
    if (not isinstance(provenance, dict) or provenance.get("format") != "native-600m-data-v1"
            or provenance.get("config_fingerprint") != config_fingerprint(config)):
        raise ValueError("Token artifacts are missing their prepared-data provenance.")
    for stage in ("pretrain", "chat"):
        source = provenance.get("sources", {}).get(stage)
        if not isinstance(source, dict):
            raise ValueError("Data provenance is missing a source identity.")
        if source.get("kind") == "huggingface":
            if source != public_source(stage):
                raise ValueError("Data provenance differs from the pinned public source.")
        elif source.get("kind") == "local-jsonl":
            if not source.get("license") or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("revision", ""))):
                raise ValueError("Local data provenance requires its source hash and permission.")
        else:
            raise ValueError("Unknown data provenance source.")
    for key in ("tokenizer_fingerprint", "source_manifest_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(key, ""))):
            raise ValueError(f"Invalid token artifact {key}.")
    vocab_size = manifest.get("vocab_size")
    if type(vocab_size) is not int or not 263 <= vocab_size <= config["model"]["vocab_size"] <= 65535:
        raise ValueError("Invalid token vocabulary size.")
    if tokenizer is not None and (manifest["tokenizer_fingerprint"] != tokenizer.fingerprint
                                  or vocab_size != tokenizer.vocab_size):
        raise ValueError("Token artifacts were encoded with a different tokenizer.")
    if data_dir is not None:
        validate_prepared(data_dir, config, workers=workers)
        if sha256_file(Path(data_dir) / "manifest.json") != manifest["source_manifest_sha256"]:
            raise ValueError("The prepared source manifest changed after tokenization.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(SPLITS):
        raise ValueError("Token artifacts require all four data splits.")
    for name in SPLITS:
        entry = outputs[name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Invalid {name} token metadata.")
        count, supervised, records = (entry.get(key) for key in ("token_count", "supervised_tokens", "records"))
        if type(count) is not int or count < 2 or type(supervised) is not int or not 0 < supervised <= count:
            raise ValueError(f"Invalid {name} token counts.")
        stage, split = name.split("_", 1)
        quota_key = f"{stage}_{split}_{'documents' if stage == 'pretrain' else 'records'}"
        if type(records) is not int or records != config["data"][quota_key]:
            raise ValueError(f"Invalid {name} record count.")
        source_entry = provenance.get("outputs", {}).get(name, {})
        if (source_entry.get("records") != records or source_entry.get("file") != f"{name}.jsonl"
                or not re.fullmatch(r"[0-9a-f]{64}", str(source_entry.get("sha256", "")))):
            raise ValueError(f"Invalid {name} source provenance.")
        if entry.get("tokens_file") != f"{name}.bin":
            raise ValueError(f"Invalid {name} token filename.")
        if stage == "chat":
            if entry.get("mask_file") != f"{name}.mask.bin":
                raise ValueError(f"Invalid {name} mask filename.")
        elif "mask_file" in entry or "mask_sha256" in entry:
            raise ValueError("Pretraining must not use an assistant-only mask.")
    print(f"[verify:tokens] splits={len(SPLITS)} workers={min(workers, len(SPLITS))}", flush=True)
    if workers == 1:
        for name in SPLITS:
            _verify_token_files(output_dir, name, outputs[name])
            print(f"[verify:tokens:{name}] OK", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(SPLITS)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            checks = {pool.submit(_verify_token_files, output_dir, name, outputs[name]): name for name in SPLITS}
            for future in as_completed(checks):
                future.result()
                print(f"[verify:tokens:{checks[future]}] OK", flush=True)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode independent Korean model data.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts_native_600m/data"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts_native_600m/tokenizer/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_CPU_WORKERS,
                        help="CPU processes for data validation, tokenization, and token-file validation")
    parser.add_argument("--reuse-prepared", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    from .tokenizer import NativeTokenizer
    tokenizer = NativeTokenizer.load(args.tokenizer)
    if args.reuse_prepared:
        validate_tokenized(args.output_dir, config, data_dir=args.data_dir, tokenizer=tokenizer, workers=args.workers)
        print(f"[tokenize] Verified existing tokens: {args.output_dir}")
    else:
        tokenize_records(args.data_dir, tokenizer, args.output_dir, config, workers=args.workers)
        print(f"[tokenize] Wrote all four token streams: {args.output_dir}")


if __name__ == "__main__":
    main()
