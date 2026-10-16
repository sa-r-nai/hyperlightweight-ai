"""Prepare, fine-tune, compare, and export Korean Qwen3-0.6B in Lightning Studio."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from korean_600m.config import DEFAULT_CONFIG, ROOT, load_config
from korean_600m.runtime import require_studio_cuda


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts_korean_600m")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--local-jsonl", type=Path)
    parser.add_argument("--local-license")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--plan", action="store_true", help="Print commands without downloads, model loading, or training")
    return parser.parse_args(argv)


def build_commands(args) -> list[list[str]]:
    config_path = str(args.config.resolve())
    artifacts = args.artifacts_dir.resolve()
    common = ["--config", config_path]
    prepare = [sys.executable, "-m", "korean_600m.prepare", *common,
               "--output-dir", str(artifacts / "data")]
    if args.reuse_prepared:
        prepare.append("--reuse-prepared")
    if args.local_jsonl:
        prepare.extend(["--local-jsonl", str(args.local_jsonl.resolve())])
    if args.local_license:
        prepare.extend(["--local-license", args.local_license])
    return [
        prepare,
        [sys.executable, "-m", "korean_600m.evaluate", *common, "--base-model", "--device", "cuda",
         "--report", str(artifacts / "evaluation_base.json"), "--allow-fail"],
        [sys.executable, "-m", "korean_600m.train", *common,
         "--data-dir", str(artifacts / "data"), "--output-dir", str(artifacts / "training"),
         "--adapter-dir", str(artifacts / "adapter"), "--max-steps", str(args.max_steps)],
        [sys.executable, "-m", "korean_600m.export", *common,
         "--adapter", str(artifacts / "adapter"), "--output", str(artifacts / "merged"),
         "--device", "cuda", "--reuse-existing"],
        [sys.executable, "-m", "korean_600m.evaluate", *common,
         "--adapter", str(artifacts / "adapter"), "--device", "cuda",
         "--report", str(artifacts / "evaluation_tuned.json")],
    ]


def main() -> None:
    args = parse_args()
    load_config(args.config)
    if args.max_steps != -1 and args.max_steps <= 0:
        raise ValueError("--max-steps must be -1 or positive.")
    commands = build_commands(args)
    if not args.plan:
        require_studio_cuda()
    for command in commands:
        print("[studio] " + shlex.join(command), flush=True)
        if not args.plan:
            subprocess.run(command, cwd=ROOT, check=True)
    if not args.plan:
        print(f"[complete] Model and reports: {args.artifacts_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
