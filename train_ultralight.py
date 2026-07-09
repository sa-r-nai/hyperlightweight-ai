"""Train the ultra-light decoder-only model on local text data.

Usage example:
  python train_ultralight.py --data ./data --preset 600m --seq-len 512 --batch-size 2

Input data:
- plain text files (`.txt`, `.md`)
- JSONL files with a `text` field

Training objective:
- next-token prediction over packed text streams
- the model file already handles the causal shift internally
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from ultralight_llm import (
    ByteTokenizer,
    MODEL_PRESETS,
    TextPreprocessor,
    TinyCausalLM,
    estimate_parameter_count,
)


def synthetic_documents() -> list[str]:
    """Small multilingual corpus for CPU-only smoke training."""

    base_docs = [
        "This compact language model predicts the next token in a sequence.",
        "The training loop includes preprocessing tokenization inference and stopping rules.",
        "A tiny decoder only transformer can still learn short repeated patterns.",
        "Byte level tokenization keeps the vocabulary very small and simple.",
        "The goal is to learn next-token prediction from short text sequences.",
        "Synthetic text is useful for smoke testing when no local dataset is available.",
        "Gradient descent updates the weights step by step.",
        "Small models train faster on a CPU and are easier to debug.",
        "Decoder-only transformer, causal mask, and rotary embeddings are used.",
        "Checkpoints allow training to resume from the latest saved step.",
        "Prompt in response out prompt in response out.",
        "short context short answer short context short answer.",
    ]
    return base_docs * 200


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_documents(path: Path) -> Iterator[str]:
    if path.is_file():
        yield from read_document_file(path)
        return

    for file_path in sorted(path.rglob("*")):
        if file_path.suffix.lower() in {".txt", ".md", ".jsonl"}:
            yield from read_document_file(file_path)


def read_document_file(path: Path) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            yield text
        return

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text")
                if isinstance(text, str) and text.strip():
                    yield text


def build_token_stream(documents: Iterable[str], tokenizer: ByteTokenizer) -> List[int]:
    token_ids: List[int] = []
    for doc in documents:
        cleaned = TextPreprocessor.normalize(doc)
        if not cleaned:
            continue
        token_ids.extend(tokenizer.encode(cleaned, add_eos=True))
    return token_ids


def split_stream(token_ids: Sequence[int], val_ratio: float) -> tuple[List[int], List[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if len(token_ids) < 1024:
        raise ValueError("Not enough tokens. Add more training text.")

    split_idx = max(1, int(len(token_ids) * (1.0 - val_ratio)))
    split_idx = min(split_idx, len(token_ids) - 1)
    train_tokens = list(token_ids[:split_idx])
    val_tokens = list(token_ids[split_idx:])
    return train_tokens, val_tokens


class PackedTokenDataset(Dataset):
    """Pack a long token stream into fixed-size training examples."""

    def __init__(self, token_ids: Sequence[int], seq_len: int) -> None:
        self.tokens = torch.tensor(list(token_ids), dtype=torch.long)
        self.seq_len = seq_len
        self.chunk_len = seq_len + 1
        self.num_examples = max(0, (len(self.tokens) - 1) // seq_len)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.chunk_len]
        if chunk.numel() < self.chunk_len:
            pad_value = ByteTokenizer.eos_token_id
            padded = torch.full((self.chunk_len,), pad_value, dtype=torch.long)
            padded[: chunk.numel()] = chunk
            chunk = padded
        x = chunk[:-1].contiguous()
        y = x.clone()
        return x, y


def collate_batch(batch: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def evaluate(
    model: TinyCausalLM,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                _, loss = model(input_ids, targets)
            total_loss += float(loss.item())
            total_batches += 1

    return total_loss / max(1, total_batches)


def save_checkpoint(
    path: Path,
    model: TinyCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    step: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "model_config": asdict(model.config),
            "step": step,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: TinyCausalLM,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: torch.device | None = None,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if device is not None:
            move_optimizer_state_to_device(optimizer, device)
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None and scaler.is_enabled():
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    step = int(checkpoint.get("step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return step, best_val_loss


def load_model_weights(path: Path, model: TinyCausalLM) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ultra-light decoder-only model.")
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Text file or directory of text files. If omitted, synthetic text is used.",
    )
    parser.add_argument("--preset", type=str, default="600m", choices=sorted(MODEL_PRESETS.keys()))
    parser.add_argument("--seq-len", type=int, default=512, help="Training sequence length.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--init-from", type=str, default="", help="Load only model weights from a checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="Optional step limit for quick experiments.")
    parser.add_argument(
        "--require-real-data",
        action="store_true",
        help="Fail instead of falling back to the built-in synthetic corpus.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    tokenizer = ByteTokenizer()
    documents: list[str]
    if args.data:
        data_path = Path(args.data)
        documents = list(iter_documents(data_path))
    else:
        documents = []

    if not documents:
        if args.require_real_data:
            raise ValueError(
                "No real training text was found. Add .txt/.md/.jsonl files under --data and try again."
            )
        documents = synthetic_documents()
        print("No real data found. Using built-in synthetic corpus.")
    else:
        print(f"Loaded {len(documents)} real documents from {data_path}.")

    token_stream = build_token_stream(documents, tokenizer)
    train_tokens, val_tokens = split_stream(token_stream, args.val_ratio)

    train_dataset = PackedTokenDataset(train_tokens, args.seq_len)
    val_dataset = PackedTokenDataset(val_tokens, args.seq_len)

    if len(train_dataset) == 0:
        raise ValueError("Training split is too small for the requested sequence length.")
    if len(val_dataset) == 0:
        raise ValueError("Validation split is too small for the requested sequence length.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    model = TinyCausalLM(MODEL_PRESETS[args.preset]).to(device)
    if args.init_from and not args.resume:
        load_model_weights(Path(args.init_from), model)
        print(f"Initialized model weights from {args.init_from}.")
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    total_updates = max(1, math.ceil(len(train_loader) / args.grad_accum) * args.epochs)
    warmup_steps = min(args.warmup_steps, total_updates)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_updates - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        start_step, best_val_loss = load_checkpoint(
            Path(args.resume),
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
        model = model.to(device)

    print(f"Preset: {args.preset}")
    print(f"Estimated parameters: {estimate_parameter_count(MODEL_PRESETS[args.preset]) / 1e6:.1f}M")
    print(f"Device: {device}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    global_step = start_step
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        window_batches = 0
        accum_batches = 0

        for _, (input_ids, targets) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(input_ids, targets)
                loss = loss / args.grad_accum

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += float(loss.item()) * args.grad_accum
            window_batches += 1
            accum_batches += 1

            if accum_batches == args.grad_accum:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(
                        f"epoch={epoch + 1} step={global_step} "
                        f"loss={running_loss / max(1, window_batches):.4f} lr={current_lr:.2e}"
                    )
                    running_loss = 0.0
                    window_batches = 0
                accum_batches = 0

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if accum_batches > 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        val_loss = evaluate(model, val_loader, device, use_amp, amp_dtype)
        val_ppl = math.exp(min(20.0, val_loss))
        print(f"epoch={epoch + 1} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = save_dir / "best.pt"
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, global_step, best_val_loss)
            print(f"saved best checkpoint -> {best_path}")

        last_path = save_dir / "last.pt"
        save_checkpoint(last_path, model, optimizer, scheduler, scaler, global_step, best_val_loss)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
