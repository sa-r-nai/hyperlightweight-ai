"""Interactive runner for a NativeEnglishLM checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from native_200m import load_checkpoint
from native_tokenizer import NativeTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NativeEnglishLM chat runner")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizer/native_english_bpe.json"),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--message", type=str)
    parser.add_argument(
        "--system",
        default=(
            "You are a helpful English assistant. Answer accurately, clearly, "
            "and concisely. Say when you do not know something."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Pass --device cpu to run on the CPU.")
        return torch.device("cuda")
    return torch.device("cpu")


def generate_reply(
    model,
    tokenizer: NativeTokenizer,
    messages: list[dict[str, str]],
    device: torch.device,
    args: argparse.Namespace,
) -> str:
    prompt = tokenizer.encode_generation_prompt(messages)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        allowed_token_ids=tokenizer.english_output_token_ids(),
    )[0]
    new_tokens = output_ids[len(prompt) :].tolist()
    return tokenizer.decode(new_tokens).strip()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    if device.type == "cuda":
        inference_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        model.to(device=device, dtype=inference_dtype)
    else:
        model.to(device)
    model.eval()
    tokenizer = NativeTokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("The checkpoint and tokenizer vocabulary sizes do not match.")
    saved_fingerprint = checkpoint.get("tokenizer_fingerprint")
    if saved_fingerprint and saved_fingerprint != tokenizer.fingerprint():
        raise ValueError("The checkpoint was created with a different tokenizer.")
    messages = [{"role": "system", "content": args.system}]

    if args.message:
        messages.append({"role": "user", "content": args.message})
        print(generate_reply(model, tokenizer, messages, device, args))
        return

    print("[info] Chat started. Type /exit to quit.")
    while True:
        try:
            user_text = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/exit":
            break
        messages.append({"role": "user", "content": user_text})
        answer = generate_reply(model, tokenizer, messages, device, args)
        print(f"assistant> {answer}")
        messages.append({"role": "assistant", "content": answer})
        # Keep the newest turns while always preserving the system instruction.
        while len(tokenizer.encode_generation_prompt(messages)) > model.config.max_seq_len:
            if len(messages) <= 3:
                break
            del messages[1:3]


if __name__ == "__main__":
    main()
