"""Train NativeByteLM-500M from random initialization.

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

from native_500m import (
    Native500MConfig,
    NativeCausalLM,
    SMOKE_CONFIG,
    save_checkpoint,
)
from native_tokenizer import NativeTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NativeByteLM-500M을 처음부터 학습합니다."
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints_native_500m"),
    )
    parser.add_argument(
        "--preset",
        choices=("native-500m", "smoke"),
        default="native-500m",
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
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA를 사용할 수 없습니다. CPU 학습을 원하면 --device cpu를 "
                "명시적으로 사용해 주세요."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_amp_dtype(device: torch.device, requested: str) -> torch.dtype | None:
    if requested == "fp32" or device.type == "cpu":
        return None
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("현재 CUDA 장치가 bfloat16을 지원하지 않습니다.")
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
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                yield text
            continue
        if path.suffix.lower() == ".json":
            raw_records = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(raw_records, dict):
                raw_records = [raw_records]
            if isinstance(raw_records, list):
                for raw_record in raw_records:
                    if isinstance(raw_record, dict):
                        if record := _read_json_record(
                            path,
                            json.dumps(raw_record, ensure_ascii=False),
                        ):
                            yield record
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
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
    ):
        if seq_len < 2:
            raise ValueError("seq_len은 2 이상이어야 합니다.")
        tokens: list[int] = [tokenizer.bos_token_id]
        for document in documents:
            if isinstance(document, str):
                tokens.extend(tokenizer.encode(document))
                tokens.append(tokenizer.eos_token_id)
            else:
                tokens.extend(
                    tokenizer.encode_chat(
                        document,
                        add_bos=False,
                        add_eos=True,
                    )
                )
        if len(tokens) < seq_len + 1:
            raise ValueError(
                f"학습 토큰이 부족합니다. 최소 {seq_len + 1}개가 필요하고 "
                f"현재 {len(tokens)}개입니다."
            )
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.examples = (len(self.tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.seq_len
        values = self.tokens[start : start + self.seq_len + 1]
        return values[:-1], values[1:]


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


def load_or_create_model(args: argparse.Namespace) -> tuple[NativeCausalLM, int, float | None]:
    if args.preset == "smoke":
        config = replace(SMOKE_CONFIG, max_seq_len=args.seq_len)
    else:
        config = Native500MConfig(max_seq_len=args.seq_len)

    model = NativeCausalLM(config)
    if args.resume is None:
        return model, 0, None

    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("재개할 체크포인트 형식이 올바르지 않습니다.")
    model.load_state_dict(checkpoint["model"])
    return model, int(checkpoint.get("step", 0)), checkpoint.get("best_loss")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(device, args.dtype)
    tokenizer = NativeTokenizer()
    documents = list(iter_documents(args.data))
    if not documents:
        if args.require_real_data:
            raise RuntimeError("실제 학습 데이터가 없습니다.")
        documents = [
            "초기 점검용 데이터입니다. 실제 학습에서는 충분한 라이선스 확인 데이터가 필요합니다."
        ]
        print("[주의] 데이터가 없어 점검용 문장으로 실행합니다.")

    dataset = PackedTextDataset(documents, tokenizer, args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError("배치가 하나도 만들어지지 않았습니다.")

    model, resume_step, best_loss = load_or_create_model(args)
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
        f"[정보] 학습을 시작합니다. device={device}, "
        f"dtype={amp_dtype or torch.float32}, documents={len(documents)}, "
        f"batches={len(loader)}"
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
        best_loss = average_loss if best_loss is None else min(best_loss, average_loss)
        if step % args.log_every == 0 or step == 1:
            elapsed = max(1e-6, time.perf_counter() - started_at)
            tokens = step * args.grad_accumulation * args.batch_size * args.seq_len
            print(
                f"[정보] step={step}/{args.max_steps} "
                f"loss={average_loss:.4f} best={best_loss:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"tokens/s={tokens / elapsed:.1f}"
            )
        if step % args.checkpoint_every == 0 or step == args.max_steps:
            save_checkpoint(
                args.output_dir / "last.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_loss=best_loss,
            )
            if average_loss <= best_loss:
                save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_loss=best_loss,
                )
            print(f"[정보] 체크포인트를 저장했습니다: {args.output_dir}")


if __name__ == "__main__":
    train(parse_args())
