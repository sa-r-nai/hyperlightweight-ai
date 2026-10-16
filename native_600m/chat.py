"""Inference for the independently initialized native Korean model."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

from .tokenizer import DEFAULT_TOKENIZER, NativeTokenizer


def generate_reply(model, tokenizer: NativeTokenizer, messages: Sequence[Mapping[str, str]], *,
                   max_new_tokens: int = 256, temperature: float = 0.8,
                   top_k: int = 40, top_p: float = 0.9,
                   repetition_penalty: float = 1.1) -> str:
    """Generate a single assistant turn using only the native vocabulary/protocol."""
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer.")
    context_length = model.config.max_seq_len
    if max_new_tokens >= context_length:
        raise ValueError("max_new_tokens must leave space for the prompt within max_seq_len.")
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model and tokenizer vocabulary sizes differ.")
    prompt = tokenizer.encode_prompt(messages, max_tokens=context_length - max_new_tokens)

    import torch

    device = next(model.parameters()).device
    inputs = torch.tensor([prompt], dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=(tokenizer.turn_end_token_id, tokenizer.eos_token_id),
            forbidden_token_ids=tuple(token_id for token_id in tokenizer.special_token_ids
                                      if token_id not in (tokenizer.turn_end_token_id, tokenizer.eos_token_id)),
        )
    generated = output[0, len(prompt):].tolist()
    for index, token_id in enumerate(generated):
        if token_id in (tokenizer.turn_end_token_id, tokenizer.eos_token_id):
            generated = generated[:index]
            break
    return tokenizer.decode(generated).strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a trained, independent native Korean 600M model.")
    parser.add_argument("--checkpoint", default="artifacts_native_600m/sft/best.pt")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--config", help="Native workflow configuration JSON.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--message", help="Generate one answer and exit; omit for interactive chat.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    from .checkpoint import load_model
    from .config import load_config

    tokenizer = NativeTokenizer.load(args.tokenizer)
    model, metadata = load_model(args.checkpoint, device=args.device)
    if metadata.get("tokenizer_sha256") != tokenizer.fingerprint:
        raise ValueError("Checkpoint tokenizer fingerprint differs from the supplied tokenizer file.")
    config = load_config(args.config) if args.config else (metadata.get("workflow_config") or load_config())
    history = [{"role": "system", "content": config["system_prompt"]}]
    options = {
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_k": args.top_k, "top_p": args.top_p, "repetition_penalty": args.repetition_penalty,
    }
    if args.message is not None:
        print(generate_reply(model, tokenizer, [*history, {"role": "user", "content": args.message}], **options))
        return
    print("한국어 대화를 시작합니다. /reset: 대화 초기화, /exit: 종료")
    while True:
        try:
            message = input("사용자> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if message == "/exit":
            return
        if message == "/reset":
            history = history[:1]
            print("대화가 초기화되었습니다.")
            continue
        if not message:
            continue
        pending = [*history, {"role": "user", "content": message}]
        try:
            answer = generate_reply(model, tokenizer, pending, **options)
        except ValueError as error:
            print(f"입력을 처리할 수 없습니다: {error}")
            continue
        print(f"도우미> {answer}")
        # Do not retain empty output as a malformed turn in the next prompt.
        if answer:
            history = [*pending, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
