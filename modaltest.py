"""Prepare on CPU and train NativeKoreanLM-600M on a Modal A100-80GB."""

from __future__ import annotations

from pathlib import Path
import shlex

import modal


LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/workspace"
VOLUME_ROOT = "/training"
ARTIFACTS_DIR = Path(VOLUME_ROOT) / "artifacts_native_600m"
TIMEOUT_SECONDS = 60 * 60

app = modal.App("native-korean-600m")
volume = modal.Volume.from_name("native-korean-600m", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .pip_install_from_requirements(str(LOCAL_ROOT / "requirements-native-600m.txt"))
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": REMOTE_ROOT, "HF_HOME": f"{VOLUME_ROOT}/huggingface",
          "RAYON_NUM_THREADS": "24", "TOKENIZERS_PARALLELISM": "true",
          "OMP_NUM_THREADS": "24", "MKL_NUM_THREADS": "24"})
    .workdir(REMOTE_ROOT)
    .add_local_dir(LOCAL_ROOT / "native_600m", f"{REMOTE_ROOT}/native_600m", ignore=["__pycache__", "*.pyc"])
    .add_local_dir(LOCAL_ROOT / "configs", f"{REMOTE_ROOT}/configs")
    .add_local_file(LOCAL_ROOT / "run_native_600m.py", f"{REMOTE_ROOT}/run_native_600m.py")
)


def _workflow_commands(artifacts_dir: Path) -> list[list[str]]:
    from run_native_600m import build_commands, parse_args

    args = parse_args(["--artifacts-dir", str(artifacts_dir), "--gpu-profile", "balanced", "--cpu-workers", "24"])
    return build_commands(args)


def _preparation_commands(artifacts_dir: Path) -> list[list[str]]:
    commands = _workflow_commands(artifacts_dir)[:3]
    markers = (
        artifacts_dir / "data" / "manifest.json",
        artifacts_dir / "tokenizer" / "tokenizer.json",
        artifacts_dir / "tokens" / "manifest.json",
    )
    for command, marker in zip(commands, markers):
        if marker.is_file():
            # Each existing stage validates its hashes and configuration before reuse.
            command.append("--reuse-prepared")
    return commands


def _training_commands(artifacts_dir: Path, *, smoke: bool = False) -> list[list[str]]:
    commands = _workflow_commands(artifacts_dir)[3:]
    if smoke:
        command = commands[0]
        command[command.index("--output-dir") + 1] = str(
            artifacts_dir.with_name(artifacts_dir.name + "_smoke") / "pretrain"
        )
        command.extend(["--seq-len", "128", "--batch-size", "1", "--grad-accumulation", "1", "--max-steps", "2"])
        return [command]
    return commands


def _run_command(command: list[str]) -> None:
    from run_native_600m import run_command

    print("[modal] " + shlex.join(command), flush=True)
    try:
        run_command(command, cwd=REMOTE_ROOT)
    finally:
        # Preserve completed stages, checkpoints, and evaluation reports on failure.
        volume.commit()


@app.function(
    image=image,
    cpu=24,
    memory=65536,
    volumes={VOLUME_ROOT: volume},
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def prepare_data() -> None:
    """Download, prepare, and tokenize data without allocating a GPU."""
    volume.reload()
    for command in _preparation_commands(ARTIFACTS_DIR):
        _run_command(command)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=24,
    memory=65536,
    volumes={VOLUME_ROOT: volume},
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def train(smoke: bool = False) -> None:
    """Run pretraining, SFT, export, and evaluation once, without automatic retries."""
    volume.reload()
    for command in _training_commands(ARTIFACTS_DIR, smoke=smoke):
        _run_command(command)
    print(f"[complete] Artifacts: {ARTIFACTS_DIR}{'_smoke' if smoke else ''}", flush=True)


@app.local_entrypoint()
def main(prepare_only: bool = False, smoke: bool = False) -> None:
    """Prepare data, then launch training; --smoke runs only two GPU steps."""
    if prepare_only and smoke:
        raise ValueError("Use either --prepare-only or --smoke.")
    prepare_data.remote()
    if not prepare_only:
        train.remote(smoke=smoke)
