"""Cloud-only workflow for a Korean 600M model with independent random weights."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from native_600m.config import DEFAULT_CONFIG, ROOT, load_config
from native_600m.runtime import require_studio, require_studio_cuda


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts_native_600m")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare data, tokenizer and tokens in a CPU Studio")
    parser.add_argument("--gpu-profile", choices=("auto", "balanced", "h200"), default="auto")
    parser.add_argument("--plan", action="store_true", help="Print commands without downloading or training")
    parser.add_argument("--local-pretrain-jsonl", type=Path)
    parser.add_argument("--local-chat-jsonl", type=Path)
    parser.add_argument("--pretrain-license")
    parser.add_argument("--chat-license")
    return parser.parse_args(argv)


def build_commands(args):
    config = load_config(args.config)
    root = args.artifacts_dir.resolve()
    common = ["--config", str(args.config.resolve())]
    data, tokens = root / "data", root / "tokens"
    tokenizer = root / "tokenizer" / "tokenizer.json"
    prepare = [sys.executable, "-m", "native_600m.prepare", *common, "--output-dir", str(data)]
    for name in ("local_pretrain_jsonl", "local_chat_jsonl", "pretrain_license", "chat_license"):
        value = getattr(args, name)
        if value is not None:
            prepare.extend(["--" + name.replace("_", "-"), str(value.resolve()) if isinstance(value, Path) else value])
    tokenizer_command = [sys.executable, "-m", "native_600m.tokenizer", "--input", str(data / "pretrain_train.jsonl"),
                         "--output", str(tokenizer), "--vocab-size", str(config["tokenizer"]["vocab_size"]),
                         "--max-documents", str(config["tokenizer"]["max_documents"])]
    tokenize = [sys.executable, "-m", "native_600m.tokenize", *common,
                "--data-dir", str(data), "--tokenizer", str(tokenizer), "--output-dir", str(tokens)]
    if args.reuse_prepared:
        for command in (prepare, tokenizer_command, tokenize):
            command.append("--reuse-prepared")
    commands = [prepare, tokenizer_command, tokenize]
    if args.prepare_only:
        return commands
    train_common = [*common, "--tokens-dir", str(tokens), "--tokenizer", str(tokenizer),
                    "--gpu-profile", args.gpu_profile]
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


def main():
    args = parse_args()
    commands = build_commands(args)
    if not args.plan:
        require_studio() if args.prepare_only else require_studio_cuda()
    for command in commands:
        print("[native] " + shlex.join(command), flush=True)
        if not args.plan:
            subprocess.run(command, cwd=ROOT, check=True)
    if not args.plan:
        print(f"[complete] Native artifacts: {args.artifacts_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
