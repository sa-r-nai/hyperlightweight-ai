"""Workflow for a Korean 300M model with independent random weights."""

from __future__ import annotations

import argparse
import shlex
import signal
import subprocess
import sys
from pathlib import Path

from native_600m.config import DEFAULT_CONFIG, DEFAULT_CPU_WORKERS, ROOT, load_config
from native_600m.runtime import require_cuda


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts_native_300m")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare data, tokenizer and tokens on CPU")
    parser.add_argument("--prepare-workers", "--cpu-workers", dest="prepare_workers", type=int,
                        default=DEFAULT_CPU_WORKERS, help="CPU workers for preparation, tokenization, and validation")
    parser.add_argument("--gpu-profile", choices=("auto", "balanced", "h100", "h200", "rtx-pro-6000"), default="auto")
    parser.add_argument("--compile-model", action="store_true", help="Compile training model calls with TorchInductor")
    parser.add_argument("--plan", action="store_true", help="Print commands without downloading or training")
    parser.add_argument("--local-pretrain-jsonl", type=Path)
    parser.add_argument("--local-chat-jsonl", type=Path)
    parser.add_argument("--pretrain-license")
    parser.add_argument("--chat-license")
    return parser.parse_args(argv)


def build_commands(args):
    if args.prepare_workers < 1:
        raise ValueError("--prepare-workers must be positive.")
    config = load_config(args.config)
    root = args.artifacts_dir.resolve()
    common = ["--config", str(args.config.resolve())]
    data, tokens = root / "data", root / "tokens"
    tokenizer = root / "tokenizer" / "tokenizer.json"
    prepare = [sys.executable, "-m", "native_600m.prepare", *common, "--output-dir", str(data),
               "--workers", str(args.prepare_workers)]
    for name in ("local_pretrain_jsonl", "local_chat_jsonl", "pretrain_license", "chat_license"):
        value = getattr(args, name)
        if value is not None:
            prepare.extend(["--" + name.replace("_", "-"), str(value.resolve()) if isinstance(value, Path) else value])
    tokenizer_command = [sys.executable, "-m", "native_600m.tokenizer", "--input", str(data / "pretrain_train.jsonl"),
                         "--output", str(tokenizer), "--vocab-size", str(config["tokenizer"]["vocab_size"]),
                         "--max-documents", str(config["tokenizer"]["max_documents"]),
                         "--workers", str(args.prepare_workers)]
    tokenize = [sys.executable, "-m", "native_600m.tokenize", *common,
                "--data-dir", str(data), "--tokenizer", str(tokenizer), "--output-dir", str(tokens),
                "--workers", str(args.prepare_workers)]
    if args.reuse_prepared:
        for command in (prepare, tokenizer_command, tokenize):
            command.append("--reuse-prepared")
    commands = [prepare, tokenizer_command, tokenize]
    if args.prepare_only:
        return commands
    train_common = [*common, "--tokens-dir", str(tokens), "--tokenizer", str(tokenizer),
                    "--gpu-profile", args.gpu_profile, "--cpu-workers", str(args.prepare_workers)]
    if args.compile_model:
        train_common.append("--compile-model")
    commands.extend([
        [sys.executable, "-m", "native_600m.train", *train_common, "--stage", "pretrain",
         "--output-dir", str(root / "pretrain")],
        [sys.executable, "-m", "native_600m.train", *train_common, "--stage", "sft",
         "--init-from", str(root / "pretrain" / "best.pt"), "--output-dir", str(root / "sft")],
        [sys.executable, "-m", "native_600m.export", "--checkpoint", str(root / "sft" / "best.pt"),
         "--tokenizer", str(tokenizer), "--output-dir", str(root / "model")],
        [sys.executable, "-m", "native_600m.evaluate", *common, "--checkpoint", str(root / "sft" / "best.pt"),
         "--tokenizer", str(tokenizer), "--device", "cuda", "--report", str(root / "evaluation.json")],
    ])
    return commands


def run_command(command: list[str], *, cwd: str | Path = ROOT) -> None:
    """Run a stage, verifying completed data after a preparation SIGABRT."""
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as error:
        if (error.returncode != -signal.SIGABRT or command[1:3] != ["-m", "native_600m.prepare"]
                or "--reuse-prepared" in command):
            raise
        from native_600m.prepare import parse_args as prepare_args

        args = prepare_args(command[3:])
        output_dir = args.output_dir if args.output_dir.is_absolute() else Path(cwd) / args.output_dir
        if not (output_dir / "manifest.json").is_file():
            raise
        # A native streaming reader may abort during interpreter shutdown even
        # after publishing complete data. A fresh process checks all artifacts
        # without loading the dataset reader. Never accept the marker alone.
        print("[prepare] SIGABRT after data publication; verifying saved artifacts in a fresh process.", flush=True)
        subprocess.run([*command, "--reuse-prepared"], cwd=cwd, check=True)
        print("[prepare] Saved artifacts passed verification; continuing the workflow.", flush=True)


def main():
    args = parse_args()
    commands = build_commands(args)
    if not args.plan and not args.prepare_only:
        require_cuda()
    for command in commands:
        print("[native] " + shlex.join(command), flush=True)
        if not args.plan:
            run_command(command)
    if not args.plan:
        print(f"[complete] Native artifacts: {args.artifacts_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
