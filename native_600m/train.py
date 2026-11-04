"""Train the independent Korean model from random weights, then perform full SFT."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

from .checkpoint import read_checkpoint, save_checkpoint
from .config import DEFAULT_CONFIG, DEFAULT_CPU_WORKERS, config_fingerprint, load_config, sha256_file, write_json
from .runtime import require_cuda
from .training_data import TokenWindows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("pretrain", "sft"), required=True)
    parser.add_argument("--tokens-dir", type=Path, default=Path("artifacts_native_600m/tokens"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts_native_600m/tokenizer/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-from", type=Path, help="Required pretrained native checkpoint for SFT only")
    parser.add_argument("--max-steps", type=int, help="Override budget for a short GPU execution check")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accumulation", type=int)
    parser.add_argument("--gpu-profile", choices=("auto", "balanced", "h200", "rtx-pro-6000"), default="auto")
    parser.add_argument("--checkpoint-seconds", type=float, default=300,
                        help="Save last.pt at a completed step at least this often, in addition to step-based saves")
    parser.add_argument("--cpu-workers", type=int, default=DEFAULT_CPU_WORKERS,
                        help="CPU processes for token-artifact validation before training")
    return parser.parse_args(argv)


def learning_rate(step: int, total: int, peak: float, minimum: float, warmup: int) -> float:
    warmup = min(warmup, max(0, total - 1))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * min(1.0, progress)))


def stage_options(config, args, *, gpu_name="", gpu_memory_gb=0):
    options = dict(config[args.stage])
    large_gpu = ("H200" in gpu_name.upper() or "RTX PRO 6000" in gpu_name.upper()) and gpu_memory_gb >= 80
    if args.gpu_profile in ("h200", "rtx-pro-6000") or (args.gpu_profile == "auto" and large_gpu):
        options.update(batch_size=8, grad_accumulation=8 if args.stage == "pretrain" else 4)
    for name in ("seq_len", "batch_size", "grad_accumulation"):
        value = getattr(args, name)
        if value is not None:
            if value <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive.")
            options[name] = value
    if options["seq_len"] > config["model"]["max_seq_len"]:
        raise ValueError("Batch sequence exceeds model context.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds <= 0:
        raise ValueError("--checkpoint-seconds must be positive and finite.")
    return options


def resolve_run_identity(requested, saved):
    """Retain the run identity while allowing equivalent microbatch regrouping."""
    message = "Run configuration or data changed. Use a new --output-dir."
    if not isinstance(saved, dict) or not isinstance(saved.get("options"), dict):
        raise ValueError(message)
    previous, current = saved["options"], requested["options"]
    grouping = ("batch_size", "grad_accumulation")
    if any(type(options.get(name)) is not int or options[name] <= 0
           for options in (previous, current) for name in grouping):
        raise ValueError(message)
    candidate = {**requested, "options": {**current, **{name: previous[name] for name in grouping}}}
    # All hashes, precision, schedule, validation settings, and the total sample
    # count per optimizer step must still match. Keep the original identity so
    # existing last.pt/best.pt and completion markers remain valid after a crash.
    if candidate != saved or previous["batch_size"] * previous["grad_accumulation"] != (
        current["batch_size"] * current["grad_accumulation"]
    ):
        raise ValueError(message)
    return saved


def checkpoint_due(step, *, start_step, max_steps, interval_steps, elapsed_seconds, interval_seconds, evaluated=False):
    """Save at the first completed step, periodic boundaries, and completion."""
    return (step == start_step + 1 or step == max_steps or evaluated
            or step % interval_steps == 0 or elapsed_seconds >= interval_seconds)


def checkpoint_activations(args, *, gpu_name="", gpu_memory_gb=0):
    """Use RTX PRO 6000 memory to retain activations instead of recomputing them."""
    full_rtx = "RTX PRO 6000" in gpu_name.upper() and gpu_memory_gb >= 80
    return not (args.gpu_profile in ("auto", "rtx-pro-6000") and full_rtx)


def evaluate(model, dataset, *, batch_size, max_batches, device, dtype, max_windows=None):
    import torch

    was_training = model.training
    model.eval()
    total_loss = total_targets = 0
    limit = min(len(dataset), batch_size * max_batches)
    if max_windows is not None:
        limit = min(limit, max_windows)
    with torch.inference_mode():
        for first in range(0, limit, batch_size):
            rows = [dataset[index] for index in range(first, min(first + batch_size, limit))]
            x = torch.stack([row[0] for row in rows]).to(device)
            y = torch.stack([row[1] for row in rows]).to(device)
            with torch.autocast("cuda", dtype=dtype):
                _, loss = model(x, y, loss_reduction="sum")
            total_loss += float(loss)
            total_targets += int((y != -100).sum())
    model.train(was_training)
    if total_targets == 0:
        raise ValueError("Validation has no supervised targets.")
    return total_loss / total_targets


def train(args):
    # Check CUDA availability before creating model weights or starting training.
    torch = require_cuda()
    from .model import ModelConfig, NativeLM, parameter_count
    from .tokenizer import NativeTokenizer
    from .tokenize import validate_tokenized

    config = load_config(args.config)
    gpu_name = torch.cuda.get_device_name()
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    options = stage_options(config, args, gpu_name=gpu_name, gpu_memory_gb=gpu_memory_gb)
    run_file = args.output_dir / "run_manifest.json"
    saved_run = json.loads(run_file.read_text(encoding="utf-8")) if run_file.is_file() else None
    tokenizer = NativeTokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != config["model"]["vocab_size"]:
        raise ValueError("Tokenizer vocabulary must match the native model configuration.")
    manifest = validate_tokenized(args.tokens_dir, config, tokenizer=tokenizer, workers=args.cpu_workers)
    stage_prefix = "pretrain" if args.stage == "pretrain" else "chat"
    train_data = TokenWindows(args.tokens_dir, manifest["outputs"][f"{stage_prefix}_train"], options["seq_len"])
    validation = TokenWindows(args.tokens_dir, manifest["outputs"][f"{stage_prefix}_validation"], options["seq_len"])
    examples_per_step = options["batch_size"] * options["grad_accumulation"]
    input_tokens_per_step = examples_per_step * options["seq_len"]
    max_steps = args.max_steps or (
        math.ceil(options["target_tokens"] / input_tokens_per_step) if args.stage == "pretrain"
        else math.ceil(len(train_data) * options["epochs"] / examples_per_step)
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    identity = {
        "config_sha256": config_fingerprint(config), "stage": args.stage,
        "token_manifest_sha256": sha256_file(args.tokens_dir / "manifest.json"),
        "tokenizer_sha256": tokenizer.fingerprint, "options": options,
        "max_steps": max_steps, "precision": str(dtype),
        "init_checkpoint_sha256": sha256_file(args.init_from) if args.init_from else None,
    }
    if args.stage == "sft" and args.init_from is None:
        raise ValueError("SFT requires --init-from pointing to the native pretraining best.pt.")
    if args.stage == "pretrain" and args.init_from is not None:
        raise ValueError("Pretraining starts with random weights; --init-from is reserved for SFT.")
    if saved_run is not None:
        requested_identity = identity
        identity = resolve_run_identity(requested_identity, saved_run.get("identity"))
        previous = saved_run.get("execution_options", identity["options"])
        if previous != options:
            print(f"[resume] batch_size={previous['batch_size']} -> {options['batch_size']} "
                  f"grad_accumulation={previous['grad_accumulation']} -> {options['grad_accumulation']} "
                  f"effective_batch={examples_per_step}; run identity and sample addressing unchanged.", flush=True)
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory has unrelated files. Use an empty --output-dir.")
    validation_window_budget = identity["options"]["batch_size"] * identity["options"]["max_eval_batches"]
    completion = args.output_dir / "completed.json"
    if completion.is_file():
        marker = json.loads(completion.read_text(encoding="utf-8"))
        if marker.get("identity") != identity:
            raise ValueError("Completion marker belongs to another run.")
        for filename in ("best.pt", "last.pt"):
            if sha256_file(args.output_dir / filename) != marker.get("hashes", {}).get(filename):
                raise ValueError("Completed checkpoint is missing or changed.")
        print(f"[complete] Verified completed {args.stage}: {args.output_dir}", flush=True)
        return

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = torch.cuda.get_device_capability()[0] >= 8
    model_config = ModelConfig(**config["model"])
    model = NativeLM(model_config).to("cuda")
    recompute_activations = checkpoint_activations(args, gpu_name=gpu_name, gpu_memory_gb=gpu_memory_gb)
    if recompute_activations:
        model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"], betas=(0.9, 0.95),
                                  weight_decay=0.1, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    step = 0
    best_loss = None
    last_path = args.output_dir / "last.pt"
    if last_path.is_file():
        checkpoint = read_checkpoint(last_path)
        if checkpoint.get("run_identity") != identity:
            raise ValueError("Resume checkpoint has a different run identity.")
        required = ("optimizer", "scaler", "torch_rng", "cuda_rng", "python_rng", "step")
        if any(key not in checkpoint for key in required):
            raise ValueError("Checkpoint lacks complete optimizer/scaler/RNG state.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        random.setstate(checkpoint["python_rng"])
        step = checkpoint["step"]
        best_loss = checkpoint.get("best_loss")
        del checkpoint
        print(f"[resume] Restored weights, optimizer, scaler, and RNG at step={step}.", flush=True)
    elif args.init_from:
        initial = read_checkpoint(args.init_from)
        if initial["config"] != asdict(model_config) or initial.get("tokenizer_sha256") != tokenizer.fingerprint:
            raise ValueError("Initial native checkpoint architecture or tokenizer does not match.")
        if initial.get("stage") != "pretrain":
            raise ValueError("SFT must initialize from this native model's pretraining checkpoint.")
        model.load_state_dict(initial["model"])
        del initial
    elif saved_run is not None:
        print("[resume] No last.pt found; starting at step=0. Unsaved progress cannot be restored.", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_file, {"identity": identity, "model_parameters": parameter_count(model_config),
                          "workflow_config": config,
                          "execution_options": options, "validation_window_budget": validation_window_budget,
                          "checkpoint_seconds": args.checkpoint_seconds,
                          "training_windows": len(train_data), "validation_windows": len(validation),
                          "input_tokens_per_step": input_tokens_per_step, "max_steps": max_steps,
                          "planned_corpus_passes": max_steps * examples_per_step / len(train_data),
                          "source_manifest": manifest})
    print(f"[train] stage={args.stage} parameters={parameter_count(model_config):,} "
          f"steps={step}/{max_steps} input_tokens_per_step={input_tokens_per_step:,} "
          f"corpus_passes={max_steps * examples_per_step / len(train_data):.2f}", flush=True)
    print(f"[train] gpu={gpu_name} vram={gpu_memory_gb:.1f}GiB precision={dtype} "
          f"batch_size={options['batch_size']} grad_accumulation={options['grad_accumulation']} "
          f"activation_checkpointing={recompute_activations}", flush=True)
    print(f"[checkpoint] first_step=True interval_seconds={args.checkpoint_seconds:g} "
          f"interval_steps={options['checkpoint_every']}", flush=True)

    def metadata():
        return {"stage": args.stage, "step": step, "best_loss": best_loss, "run_identity": identity,
                "execution_options": options, "validation_window_budget": validation_window_budget,
                "tokenizer_sha256": tokenizer.fingerprint, "workflow_config": config,
                "training_provenance": {"workflow_config": config, "identity": identity,
                                        "token_manifest": manifest}}

    started = time.perf_counter()
    last_checkpoint_at = started
    start_step = step
    overflow_retries = 0
    model.train()
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        microbatches = [train_data.batch((step * options["grad_accumulation"] + micro) * options["batch_size"],
                                        options["batch_size"], config["seed"])
                        for micro in range(options["grad_accumulation"])]
        target_count = sum(int((y != -100).sum()) for _, y in microbatches)
        if target_count == 0:
            raise ValueError("Training step has no supervised targets.")
        lr = learning_rate(step, max_steps, options["lr"], options["min_lr"], options["warmup_steps"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        for x, y in microbatches:
            x, y = x.to("cuda"), y.to("cuda")
            with torch.autocast("cuda", dtype=dtype):
                _, summed_loss = model(x, y, loss_reduction="sum")
                loss = summed_loss / target_count
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss; the previous atomic checkpoint is preserved.")
            scaler.scale(loss).backward()
            loss_sum += float(summed_loss.detach())
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        if not bool(torch.isfinite(grad_norm)):
            if dtype != torch.float16:
                raise FloatingPointError("Nonfinite BF16 gradients; previous checkpoint is preserved.")
            # unscale_ has recorded overflow. Reduce the FP16 loss scale without
            # advancing the optimizer, schedule, or deterministic sample cursor.
            scaler.update()
            overflow_retries += 1
            if overflow_retries > 8:
                raise FloatingPointError("Repeated FP16 gradient overflow; inspect the corpus and learning rate.")
            print(f"[overflow] retrying step={step + 1} scale={scaler.get_scale()}", flush=True)
            continue
        scaler.step(optimizer)
        scaler.update()
        overflow_retries = 0
        step += 1
        if step == start_step + 1 or step % 10 == 0:
            tokens_per_second = (step - start_step) * input_tokens_per_step / max(1e-6, time.perf_counter() - started)
            print(f"[train] step={step}/{max_steps} loss={loss_sum / target_count:.4f} "
                  f"lr={lr:.3e} tokens/s={tokens_per_second:.0f}", flush=True)
        evaluated = best_loss is None or step % options["eval_every"] == 0 or step == max_steps
        if evaluated:
            validation_loss = evaluate(model, validation, batch_size=options["batch_size"],
                                       max_batches=math.ceil(validation_window_budget / options["batch_size"]),
                                       max_windows=validation_window_budget, device="cuda", dtype=dtype)
            if not math.isfinite(validation_loss):
                raise FloatingPointError("Nonfinite validation loss.")
            print(f"[validation] step={step} loss={validation_loss:.4f}", flush=True)
            if best_loss is None or validation_loss < best_loss:
                best_loss = validation_loss
                save_checkpoint(args.output_dir / "best.pt", model, **metadata())
        if checkpoint_due(step, start_step=start_step, max_steps=max_steps,
                          interval_steps=options["checkpoint_every"],
                          elapsed_seconds=time.perf_counter() - last_checkpoint_at,
                          interval_seconds=args.checkpoint_seconds, evaluated=evaluated):
            save_checkpoint(last_path, model, **metadata(), optimizer=optimizer.state_dict(),
                            scaler=scaler.state_dict(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate())
            last_checkpoint_at = time.perf_counter()
            print(f"[checkpoint] step={step} saved={last_path}", flush=True)
    write_json(completion, {"identity": identity, "best_validation_loss": best_loss,
                            "hashes": {name: sha256_file(args.output_dir / name) for name in ("best.pt", "last.pt")}})
    print(f"[complete] Native {args.stage}: {args.output_dir}", flush=True)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
