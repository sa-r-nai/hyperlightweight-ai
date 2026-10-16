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
from .config import DEFAULT_CONFIG, config_fingerprint, load_config, sha256_file, write_json
from .runtime import require_studio_cuda
from .training_data import TokenWindows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("pretrain", "sft"), required=True)
    parser.add_argument("--tokens-dir", type=Path, default=Path("artifacts_native_600m/tokens"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts_native_600m/tokenizer/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-from", type=Path, help="Required pretrained native checkpoint for SFT only")
    parser.add_argument("--max-steps", type=int, help="Override budget for a Studio-only execution check")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accumulation", type=int)
    parser.add_argument("--gpu-profile", choices=("auto", "balanced", "h200"), default="auto")
    return parser.parse_args(argv)


def learning_rate(step: int, total: int, peak: float, minimum: float, warmup: int) -> float:
    warmup = min(warmup, max(0, total - 1))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * min(1.0, progress)))


def stage_options(config, args, *, gpu_name="", gpu_memory_gb=0):
    options = dict(config[args.stage])
    if args.gpu_profile == "h200" or (args.gpu_profile == "auto" and "H200" in gpu_name.upper() and gpu_memory_gb >= 80):
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
    return options


def evaluate(model, dataset, *, batch_size, max_batches, device, dtype):
    import torch

    was_training = model.training
    model.eval()
    total_loss = total_targets = 0
    with torch.inference_mode():
        for first in range(0, min(len(dataset), batch_size * max_batches), batch_size):
            rows = [dataset[index] for index in range(first, min(first + batch_size, len(dataset)))]
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
    # All weight creation and all training remain behind the Studio guard.
    torch = require_studio_cuda()
    from .model import ModelConfig, NativeLM, parameter_count
    from .tokenizer import NativeTokenizer
    from .tokenize import validate_tokenized

    config = load_config(args.config)
    options = stage_options(config, args, gpu_name=torch.cuda.get_device_name(),
                            gpu_memory_gb=torch.cuda.get_device_properties(0).total_memory / 1024**3)
    tokenizer = NativeTokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != config["model"]["vocab_size"]:
        raise ValueError("Tokenizer vocabulary must match the native model configuration.")
    manifest = validate_tokenized(args.tokens_dir, config, tokenizer=tokenizer)
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
    run_file = args.output_dir / "run_manifest.json"
    if run_file.is_file():
        if json.loads(run_file.read_text(encoding="utf-8")).get("identity") != identity:
            raise ValueError("Run configuration or data changed. Use a new --output-dir.")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory has unrelated files. Use an empty --output-dir.")
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
    elif args.init_from:
        initial = read_checkpoint(args.init_from)
        if initial["config"] != asdict(model_config) or initial.get("tokenizer_sha256") != tokenizer.fingerprint:
            raise ValueError("Initial native checkpoint architecture or tokenizer does not match.")
        if initial.get("stage") != "pretrain":
            raise ValueError("SFT must initialize from this native model's pretraining checkpoint.")
        model.load_state_dict(initial["model"])
        del initial

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_file, {"identity": identity, "model_parameters": parameter_count(model_config),
                          "workflow_config": config,
                          "training_windows": len(train_data), "validation_windows": len(validation),
                          "input_tokens_per_step": input_tokens_per_step, "max_steps": max_steps,
                          "planned_corpus_passes": max_steps * examples_per_step / len(train_data),
                          "source_manifest": manifest})
    print(f"[train] stage={args.stage} parameters={parameter_count(model_config):,} "
          f"steps={step}/{max_steps} input_tokens_per_step={input_tokens_per_step:,} "
          f"corpus_passes={max_steps * examples_per_step / len(train_data):.2f}", flush=True)

    def metadata():
        return {"stage": args.stage, "step": step, "best_loss": best_loss, "run_identity": identity,
                "tokenizer_sha256": tokenizer.fingerprint, "workflow_config": config,
                "training_provenance": {"workflow_config": config, "identity": identity,
                                        "token_manifest": manifest}}

    started = time.perf_counter()
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
        if step % options["eval_every"] == 0 or step == max_steps:
            validation_loss = evaluate(model, validation, batch_size=options["batch_size"],
                                       max_batches=options["max_eval_batches"], device="cuda", dtype=dtype)
            if not math.isfinite(validation_loss):
                raise FloatingPointError("Nonfinite validation loss.")
            print(f"[validation] step={step} loss={validation_loss:.4f}", flush=True)
            if best_loss is None or validation_loss < best_loss:
                best_loss = validation_loss
                save_checkpoint(args.output_dir / "best.pt", model, **metadata())
        if step % options["checkpoint_every"] == 0 or step % options["eval_every"] == 0 or step == max_steps:
            save_checkpoint(last_path, model, **metadata(), optimizer=optimizer.state_dict(),
                            scaler=scaler.state_dict(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate())
    write_json(completion, {"identity": identity, "best_validation_loss": best_loss,
                            "hashes": {name: sha256_file(args.output_dir / name) for name in ("best.pt", "last.pt")}})
    print(f"[complete] Native {args.stage}: {args.output_dir}", flush=True)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
