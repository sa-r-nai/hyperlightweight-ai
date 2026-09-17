"""Train NativeEnglishLM-200M from random initialization.

The default execution target is CUDA.  CPU training is available only when
the caller explicitly passes ``--device cpu``; an unavailable GPU never causes
an accidental silent fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from native_200m import (
    Native200MConfig,
    NativeCausalLM,
    SMOKE_CONFIG,
    save_checkpoint,
)
from native_tokenizer import NativeTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NativeEnglishLM-200M from scratch.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--validation-data",
        type=Path,
        help="Optional separate validation file or directory.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.02,
        help="Document-level validation split used when --validation-data is omitted.",
    )
    parser.add_argument(
        "--validation-seq-len",
        type=int,
        default=256,
        help="Sequence length used for validation batches.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizer/native_english_bpe.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints_native_200m"),
    )
    parser.add_argument(
        "--preset",
        choices=("native-200m", "smoke"),
        default="native-200m",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument(
        "--assistant-only-loss",
        action="store_true",
        help="For chat records, compute loss only on assistant responses and endings.",
    )
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-from",
        type=Path,
        help="Load model weights only and start a fresh optimizer/schedule, for example for SFT.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace) -> None:
    if args.resume is not None and args.init_from is not None:
        raise ValueError("--resume and --init-from cannot be used together.")
    positive_values = {
        "--seq-len": args.seq_len,
        "--batch-size": args.batch_size,
        "--grad-accumulation": args.grad_accumulation,
        "--max-steps": args.max_steps,
        "--checkpoint-every": args.checkpoint_every,
        "--eval-every": args.eval_every,
        "--log-every": args.log_every,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.seq_len < 2 or args.validation_seq_len < 2:
        raise ValueError("Training and validation sequence lengths must be at least two.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must not be negative.")
    if args.lr <= 0.0 or not 0.0 <= args.min_lr <= args.lr:
        raise ValueError("Learning rates must satisfy 0 <= --min-lr <= --lr and --lr > 0.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must not be negative.")


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Pass --device cpu to train on the CPU.")
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_amp_dtype(device: torch.device, requested: str) -> torch.dtype | None:
    if requested == "fp32" or device.type == "cpu":
        return None
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The current CUDA device does not support bfloat16.")
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


TrainingRecord = str | Sequence[Mapping[str, str]]


def _read_json_record(path: Path, line: str) -> TrainingRecord | None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(record, dict):
        instruction = record.get("instruction")
        output = record.get("output")
        if (
            isinstance(instruction, str)
            and instruction.strip()
            and isinstance(output, str)
            and output.strip()
        ):
            return [
                {"role": "user", "content": instruction.strip()},
                {"role": "assistant", "content": output.strip()},
            ]
        for key in ("text", "content", "prompt", "completion"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        messages = record.get("messages")
        if isinstance(messages, list):
            normalized_messages: list[dict[str, str]] = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = {
                    "human": "user",
                    "user": "user",
                    "gpt": "assistant",
                    "assistant": "assistant",
                    "system": "system",
                }.get(str(message.get("role", "unknown")))
                content = message.get("content")
                if role and isinstance(content, str) and content.strip():
                    normalized_messages.append(
                        {"role": str(role), "content": content.strip()}
                    )
            if normalized_messages:
                return normalized_messages
    return None


def iter_documents(data_path: Path) -> Iterator[TrainingRecord]:
    paths = [data_path] if data_path.is_file() else sorted(data_path.rglob("*"))
    supported = {".txt", ".md", ".json", ".jsonl"}
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        if path.name.lower() in {"manifest.json", "readme.info"}:
            continue
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="ascii", errors="strict").strip()
            if text:
                yield text
            continue
        if path.suffix.lower() == ".json":
            raw_records = json.loads(path.read_text(encoding="ascii", errors="strict"))
            if isinstance(raw_records, dict):
                raw_records = [raw_records]
            if isinstance(raw_records, list):
                for raw_record in raw_records:
                    if isinstance(raw_record, dict):
                        if record := _read_json_record(
                            path,
                            json.dumps(raw_record, ensure_ascii=True),
                        ):
                            yield record
            continue
        with path.open("r", encoding="ascii", errors="strict") as handle:
            for line in handle:
                if record := _read_json_record(path, line):
                    yield record


class PackedTextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Pack documents into fixed-length next-token-prediction examples."""

    def __init__(
        self,
        documents: Iterable[TrainingRecord],
        tokenizer: NativeTokenizer,
        seq_len: int,
        assistant_only_loss: bool = False,
    ):
        if seq_len < 2:
            raise ValueError("seq_len must be at least two.")
        tokens: list[int] = [tokenizer.bos_token_id]
        loss_mask: list[bool] = [False]
        for document in documents:
            if isinstance(document, str):
                encoded = tokenizer.encode(document)
                tokens.extend(encoded)
                loss_mask.extend([True] * len(encoded))
                tokens.append(tokenizer.eos_token_id)
                loss_mask.append(True)
            else:
                if assistant_only_loss:
                    encoded, encoded_mask = tokenizer.encode_chat_with_assistant_mask(
                        document,
                        add_bos=False,
                        add_eos=True,
                    )
                else:
                    encoded = tokenizer.encode_chat(
                        document,
                        add_bos=False,
                        add_eos=True,
                    )
                    encoded_mask = [True] * len(encoded)
                tokens.extend(encoded)
                loss_mask.extend(encoded_mask)
        if len(tokens) < seq_len + 1:
            raise ValueError(
                f"The dataset needs at least {seq_len + 1} tokens but contains "
                f"only {len(tokens)}."
            )
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.loss_mask = torch.tensor(loss_mask, dtype=torch.bool)
        self.token_count = len(tokens)
        self.supervised_token_count = int(self.loss_mask.sum())
        self.seq_len = seq_len
        possible_examples = (len(self.tokens) - 1) // seq_len
        self.example_starts = [
            index * seq_len
            for index in range(possible_examples)
            if bool(self.loss_mask[index * seq_len + 1 : (index + 1) * seq_len + 1].any())
        ]

    def __len__(self) -> int:
        return len(self.example_starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.example_starts[index]
        values = self.tokens[start : start + self.seq_len + 1]
        labels = values[1:].clone()
        target_mask = self.loss_mask[start + 1 : start + self.seq_len + 1]
        labels.masked_fill_(~target_mask, -100)
        return values[:-1], labels


def split_documents(
    documents: Sequence[TrainingRecord],
    validation_ratio: float,
    seed: int,
) -> tuple[list[TrainingRecord], list[TrainingRecord]]:
    """Create a deterministic document-level train/validation split."""

    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be at least zero and less than one.")
    items = list(documents)
    if validation_ratio == 0.0:
        return items, []
    if len(items) < 2:
        raise ValueError(
            "At least two documents are required for an automatic validation split. "
            "Provide --validation-data or pass --validation-ratio 0."
        )
    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(len(items) * validation_ratio))
    validation_count = min(validation_count, len(items) - 1)
    validation_indices = set(indices[:validation_count])
    train_documents = [item for index, item in enumerate(items) if index not in validation_indices]
    validation_documents = [item for index, item in enumerate(items) if index in validation_indices]
    return train_documents, validation_documents


@torch.no_grad()
def evaluate(
    model: NativeCausalLM,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, amp_dtype):
            _, loss = model(input_ids, labels)
        total_loss += float(loss.item())
        batches += 1
    model.train()
    if batches == 0:
        raise RuntimeError("The validation dataset did not produce any batches.")
    return total_loss / batches


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def make_scaler(device: torch.device, amp_dtype: torch.dtype | None):
    enabled = device.type == "cuda" and amp_dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    enabled = amp_dtype is not None and device.type == "cuda"
    return torch.autocast(
        device_type=device.type,
        dtype=amp_dtype or torch.float32,
        enabled=enabled,
    )


def load_or_create_model(
    args: argparse.Namespace,
    tokenizer: NativeTokenizer,
) -> tuple[NativeCausalLM, int, float | None]:
    checkpoint_path = args.resume or args.init_from
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError("The checkpoint is invalid.")
        saved_config = checkpoint.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError("The checkpoint does not contain a model configuration.")
        config = Native200MConfig(**saved_config)
        if config.vocab_size != tokenizer.vocab_size:
            raise ValueError("The checkpoint and tokenizer vocabulary sizes do not match.")
        if args.seq_len > config.max_seq_len:
            raise ValueError(
                f"--seq-len {args.seq_len} exceeds the checkpoint context length "
                f"{config.max_seq_len}."
            )
        saved_fingerprint = checkpoint.get("tokenizer_fingerprint")
        if saved_fingerprint and saved_fingerprint != tokenizer.fingerprint():
            raise ValueError("The checkpoint was created with a different tokenizer.")
        model = NativeCausalLM(config)
        model.load_state_dict(checkpoint["model"])
        if args.resume is not None:
            return model, int(checkpoint.get("step", 0)), checkpoint.get("best_loss")
        return model, 0, None

    if args.preset == "smoke":
        config = replace(
            SMOKE_CONFIG,
            vocab_size=tokenizer.vocab_size,
            max_seq_len=args.seq_len,
        )
    else:
        config = Native200MConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=args.seq_len,
        )

    return NativeCausalLM(config), 0, None


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(device, args.dtype)
    tokenizer = NativeTokenizer.load(args.tokenizer)
    documents = list(iter_documents(args.data))
    if not documents:
        if args.require_real_data:
            raise RuntimeError("No training data was found.")
        documents = [
            "This is smoke-test data. Real training requires enough licensed English text."
        ] * 256
        print("[warning] No data was found; using repeated smoke-test text.")

    if args.validation_data is not None:
        validation_documents = list(iter_documents(args.validation_data))
        if not validation_documents:
            raise RuntimeError("No validation data was found.")
        train_documents = documents
    else:
        train_documents, validation_documents = split_documents(
            documents,
            args.validation_ratio,
            args.seed,
        )

    dataset = PackedTextDataset(
        train_documents,
        tokenizer,
        args.seq_len,
        assistant_only_loss=args.assistant_only_loss,
    )
    if len(dataset) == 0:
        raise RuntimeError("The training data did not produce any supervised examples.")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError("The dataset did not produce a complete batch.")

    validation_loader: DataLoader | None = None
    validation_dataset: PackedTextDataset | None = None
    if validation_documents:
        validation_seq_len = min(args.validation_seq_len, args.seq_len)
        validation_dataset = PackedTextDataset(
            validation_documents,
            tokenizer,
            validation_seq_len,
            assistant_only_loss=args.assistant_only_loss,
        )
        if len(validation_dataset) == 0:
            raise RuntimeError("The validation data did not produce any supervised examples.")
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    model, resume_step, best_loss = load_or_create_model(args, tokenizer)
    model.to(device)
    if args.grad_checkpointing:
        model.enable_gradient_checkpointing()
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        args.warmup_steps,
        args.max_steps,
        args.min_lr / args.lr,
    )
    scaler = make_scaler(device, amp_dtype)

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])

    model.train()
    model.zero_grad(set_to_none=True)
    step = resume_step
    data_iterator: Iterator[tuple[torch.Tensor, torch.Tensor]] = iter(loader)
    started_at = time.perf_counter()
    running_loss = 0.0

    print(
        f"[info] Training started. device={device}, "
        f"dtype={amp_dtype or torch.float32}, train_documents={len(train_documents)}, "
        f"train_tokens={dataset.token_count}, supervised_tokens={dataset.supervised_token_count}, "
        f"assistant_only_loss={args.assistant_only_loss}, batches={len(loader)}, "
        f"validation_documents={len(validation_documents)}, "
        f"validation_batches={len(validation_loader) if validation_loader else 0}"
    )
    if args.preset == "native-200m" and dataset.token_count < 100_000_000:
        print(
            "[warning] This corpus is far too small for a useful 200M model. "
            "Treat this as a pipeline test and supply a much larger licensed corpus "
            "before a production run."
        )

    while step < args.max_steps:
        for _ in range(args.grad_accumulation):
            try:
                input_ids, labels = next(data_iterator)
            except StopIteration:
                data_iterator = iter(loader)
                input_ids, labels = next(data_iterator)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_context(device, amp_dtype):
                _, loss = model(input_ids, labels)
                scaled_loss = loss / args.grad_accumulation
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach())

        scaler.unscale_(optimizer)
        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1

        average_loss = running_loss / args.grad_accumulation
        running_loss = 0.0
        if step % args.log_every == 0 or step == 1:
            elapsed = max(1e-6, time.perf_counter() - started_at)
            tokens = step * args.grad_accumulation * args.batch_size * args.seq_len
            print(
                f"[info] step={step}/{args.max_steps} "
                f"train_loss={average_loss:.4f} "
                f"best_validation_loss={best_loss if best_loss is not None else 'n/a'} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"tokens/s={tokens / elapsed:.1f}"
            )

        should_evaluate = (
            validation_loader is not None
            and (step % args.eval_every == 0 or step == args.max_steps)
        )
        if should_evaluate:
            validation_loss = evaluate(model, validation_loader, device, amp_dtype)
            print(f"[info] step={step} validation_loss={validation_loss:.4f}")
            if best_loss is None or validation_loss < best_loss:
                best_loss = validation_loss
                save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_loss=best_loss,
                    tokenizer_fingerprint=tokenizer.fingerprint(),
                )
                print(f"[info] Saved new best checkpoint to {args.output_dir / 'best.pt'}")

        should_checkpoint = step % args.checkpoint_every == 0 or step == args.max_steps
        if should_checkpoint:
            if validation_loader is None and (best_loss is None or average_loss < best_loss):
                best_loss = average_loss
                save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_loss=best_loss,
                    tokenizer_fingerprint=tokenizer.fingerprint(),
                )
            save_checkpoint(
                args.output_dir / "last.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_loss=best_loss,
                tokenizer_fingerprint=tokenizer.fingerprint(),
            )
            print(f"[info] Saved checkpoint to {args.output_dir}")


if __name__ == "__main__":
    train(parse_args())
