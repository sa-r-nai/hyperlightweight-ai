"""Run the reproducible public-data pretraining and chat-SFT pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

from native_tokenizer import NativeTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path) -> None:
    print("[pipeline] " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_unless_prepared(
    command: list[str],
    *,
    cwd: Path,
    expected: Path,
    reuse_prepared: bool,
) -> None:
    if reuse_prepared and expected.is_file():
        print(f"[pipeline] Reusing {expected}", flush=True)
        return
    run(command, cwd=cwd)


def token_count(manifest: Path) -> int:
    payload = json.loads(manifest.read_text(encoding="ascii"))
    if payload.get("format") != "native-token-binary-v1":
        raise ValueError(f"Invalid token manifest: {manifest}")
    return int(payload["token_count"])


def validate_public_manifest(
    path: Path,
    args: argparse.Namespace,
    *,
    verify_hashes: bool,
) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if payload.get("format") != "native-public-data-v1":
        raise ValueError(f"Invalid public-data manifest: {path}")
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Public-data manifest has no output table.")
    expected = {
        "pretraining_train": args.fineweb_train_records,
        "pretraining_validation": args.fineweb_validation_records,
        "chat_train": args.chat_train_records,
        "chat_validation": args.chat_validation_records,
    }
    for name, count in expected.items():
        entry = outputs.get(name)
        if not isinstance(entry, dict) or entry.get("records") != count:
            raise ValueError(
                f"Prepared {name} does not match the requested count. "
                "Rerun without --reuse-prepared."
            )
        output_path = Path(str(entry.get("path", "")))
        if not output_path.is_absolute():
            output_path = path.parent.parent / output_path
        if not output_path.is_file():
            raise FileNotFoundError(output_path)
        if verify_hashes and sha256(output_path) != entry.get("sha256"):
            raise ValueError(
                f"Prepared data hash mismatch: {output_path}. "
                "Rerun without --reuse-prepared."
            )
    return outputs


def validate_token_manifest(
    manifest_path: Path,
    *,
    expected_source_sha256: str,
    tokenizer_fingerprint: str,
    verify_hashes: bool,
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="ascii"))
    if payload.get("format") != "native-token-binary-v1":
        raise ValueError(f"Invalid token manifest: {manifest_path}")
    if payload.get("tokenizer_fingerprint") != tokenizer_fingerprint:
        raise ValueError(
            f"{manifest_path} uses another tokenizer. Rerun without --reuse-prepared."
        )
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("sha256") != expected_source_sha256:
        raise ValueError(
            f"{manifest_path} does not match its source data. "
            "Rerun without --reuse-prepared."
        )
    tokens_file = manifest_path.parent / str(payload.get("tokens_file", ""))
    token_count_value = int(payload.get("token_count", 0))
    if not tokens_file.is_file() or tokens_file.stat().st_size != token_count_value * 2:
        raise ValueError(f"Token binary is missing or truncated: {tokens_file}")
    if verify_hashes and sha256(tokens_file) != payload.get("tokens_sha256"):
        raise ValueError(f"Token binary hash mismatch: {tokens_file}")
    mask_file = payload.get("mask_file")
    if isinstance(mask_file, str):
        mask_path = manifest_path.parent / mask_file
        if not mask_path.is_file() or mask_path.stat().st_size != token_count_value:
            raise ValueError(f"Mask binary is missing or truncated: {mask_path}")
        if verify_hashes and sha256(mask_path) != payload.get("mask_sha256"):
            raise ValueError(f"Mask binary hash mismatch: {mask_path}")


def steps_for_epochs(
    tokens: int,
    *,
    seq_len: int,
    batch_size: int,
    grad_accumulation: int,
    epochs: float,
) -> int:
    tokens_per_step = seq_len * batch_size * grad_accumulation
    return max(1, math.ceil(tokens * epochs / tokens_per_step))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare public data, pretrain NativeEnglishLM, SFT it, and evaluate chat."
    )
    parser.add_argument("--work-dir", type=Path, default=Path("."))
    parser.add_argument("--fineweb-train-records", type=int, default=1_000_000)
    parser.add_argument("--fineweb-validation-records", type=int, default=10_000)
    parser.add_argument("--chat-train-records", type=int, default=400_000)
    parser.add_argument("--chat-validation-records", type=int, default=5_000)
    parser.add_argument("--pretrain-epochs", type=float, default=4.0)
    parser.add_argument("--sft-epochs", type=float, default=2.0)
    parser.add_argument("--pretrain-seq-len", type=int, default=2048)
    parser.add_argument("--sft-seq-len", type=int, default=1024)
    parser.add_argument("--pretrain-batch-size", type=int, default=1)
    parser.add_argument("--pretrain-grad-accumulation", type=int, default=16)
    parser.add_argument("--sft-batch-size", type=int, default=2)
    parser.add_argument("--sft-grad-accumulation", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--reuse-prepared",
        action="store_true",
        help="Reuse existing public-data and token manifests; training still resumes or runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.work_dir.resolve()
    python = sys.executable
    tokenizer = root / "tokenizer" / "native_english_bpe.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(tokenizer)
    for name in (
        "fineweb_train_records",
        "fineweb_validation_records",
        "chat_train_records",
        "chat_validation_records",
        "pretrain_seq_len",
        "sft_seq_len",
        "pretrain_batch_size",
        "pretrain_grad_accumulation",
        "sft_batch_size",
        "sft_grad_accumulation",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.pretrain_epochs <= 0 or args.sft_epochs <= 0:
        raise ValueError("Epoch counts must be positive.")

    public_dir = root / "public_data"
    token_dir = root / "tokenized_data"
    public_manifest = public_dir / "manifest.json"
    run_unless_prepared(
        [
            python,
            "prepare_public_data.py",
            "--output-dir",
            str(public_dir),
            "--fineweb-train-records",
            str(args.fineweb_train_records),
            "--fineweb-validation-records",
            str(args.fineweb_validation_records),
            "--chat-train-records",
            str(args.chat_train_records),
            "--chat-validation-records",
            str(args.chat_validation_records),
        ],
        cwd=root,
        expected=public_manifest,
        reuse_prepared=args.reuse_prepared,
    )
    public_outputs = validate_public_manifest(
        public_manifest,
        args,
        verify_hashes=args.reuse_prepared,
    )
    tokenizer_fingerprint = NativeTokenizer.load(tokenizer).fingerprint()

    token_jobs = (
        ("pretraining_train", False),
        ("pretraining_validation", False),
        ("chat_train", True),
        ("chat_validation", True),
    )
    token_manifests: dict[str, Path] = {}
    for name, assistant_only in token_jobs:
        output = token_dir / f"{name}.json"
        command = [
            python,
            "tokenize_native_data.py",
            "--input",
            str(public_dir / f"{name}.jsonl"),
            "--tokenizer",
            str(tokenizer),
            "--output",
            str(output),
        ]
        if assistant_only:
            command.append("--assistant-only-loss")
        run_unless_prepared(
            command,
            cwd=root,
            expected=output,
            reuse_prepared=args.reuse_prepared,
        )
        validate_token_manifest(
            output,
            expected_source_sha256=str(public_outputs[name]["sha256"]),
            tokenizer_fingerprint=tokenizer_fingerprint,
            verify_hashes=args.reuse_prepared,
        )
        token_manifests[name] = output

    pretrain_steps = steps_for_epochs(
        token_count(token_manifests["pretraining_train"]),
        seq_len=args.pretrain_seq_len,
        batch_size=args.pretrain_batch_size,
        grad_accumulation=args.pretrain_grad_accumulation,
        epochs=args.pretrain_epochs,
    )
    pretrain_eval_every = max(250, pretrain_steps // 20)
    pretrain_warmup = min(1000, max(10, round(pretrain_steps * 0.03)))
    base_dir = root / "checkpoints_native_200m"
    base_last = base_dir / "last.pt"
    pretrain_command = [
        python,
        "train_native_200m.py",
        "--device",
        "cuda",
        "--tokenized-data",
        str(token_manifests["pretraining_train"]),
        "--validation-tokenized-data",
        str(token_manifests["pretraining_validation"]),
        "--tokenizer",
        str(tokenizer),
        "--seq-len",
        str(args.pretrain_seq_len),
        "--batch-size",
        str(args.pretrain_batch_size),
        "--grad-accumulation",
        str(args.pretrain_grad_accumulation),
        "--grad-checkpointing",
        "--num-workers",
        str(args.num_workers),
        "--warmup-steps",
        str(pretrain_warmup),
        "--max-steps",
        str(pretrain_steps),
        "--eval-every",
        str(pretrain_eval_every),
        "--checkpoint-every",
        str(pretrain_eval_every),
        "--output-dir",
        str(base_dir),
    ]
    if base_last.is_file():
        pretrain_command.extend(("--resume", str(base_last)))
    run(pretrain_command, cwd=root)

    base_best = base_dir / "best.pt"
    if not base_best.is_file():
        raise RuntimeError("Pretraining did not create best.pt.")
    sft_steps = steps_for_epochs(
        token_count(token_manifests["chat_train"]),
        seq_len=args.sft_seq_len,
        batch_size=args.sft_batch_size,
        grad_accumulation=args.sft_grad_accumulation,
        epochs=args.sft_epochs,
    )
    sft_eval_every = max(100, sft_steps // 20)
    sft_warmup = min(500, max(10, round(sft_steps * 0.03)))
    chat_dir = root / "checkpoints_native_200m_chat"
    chat_last = chat_dir / "last.pt"
    sft_command = [
        python,
        "train_native_200m.py",
        "--device",
        "cuda",
        "--tokenized-data",
        str(token_manifests["chat_train"]),
        "--validation-tokenized-data",
        str(token_manifests["chat_validation"]),
        "--tokenizer",
        str(tokenizer),
        "--assistant-only-loss",
        "--seq-len",
        str(args.sft_seq_len),
        "--batch-size",
        str(args.sft_batch_size),
        "--grad-accumulation",
        str(args.sft_grad_accumulation),
        "--grad-checkpointing",
        "--num-workers",
        str(args.num_workers),
        "--lr",
        "1e-4",
        "--min-lr",
        "1e-5",
        "--warmup-steps",
        str(sft_warmup),
        "--max-steps",
        str(sft_steps),
        "--eval-every",
        str(sft_eval_every),
        "--checkpoint-every",
        str(sft_eval_every),
        "--output-dir",
        str(chat_dir),
    ]
    if chat_last.is_file():
        sft_command.extend(("--resume", str(chat_last)))
    else:
        sft_command.extend(("--init-from", str(base_best)))
    run(sft_command, cwd=root)

    chat_best = chat_dir / "best.pt"
    if not chat_best.is_file():
        raise RuntimeError("Chat SFT did not create best.pt.")
    run(
        [
            python,
            "evaluate_native_chat.py",
            "--checkpoint",
            str(chat_best),
            "--tokenizer",
            str(tokenizer),
            "--device",
            "cuda",
        ],
        cwd=root,
    )


if __name__ == "__main__":
    main()
