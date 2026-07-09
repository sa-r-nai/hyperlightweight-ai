"""Simple terminal chat app for the tiny language model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ultralight_llm import ByteTokenizer, GenerationConfig, MODEL_PRESETS, TinyCausalLM, generate_text


DEFAULT_SYSTEM_PROMPT = (
    "너는 친절하고 정확한 한국어 도우미다. "
    "모르면 모른다고 말하고, 답은 짧은 문단으로 분명하게 설명한다."
)

STOP_MARKERS = [
    "\n사용자:",
    "\n도우미:",
    "\n시스템:",
    "\nUser:",
    "\nAssistant:",
    "\nSystem:",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the tiny language model in the terminal.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path, for example ./checkpoints_5m_realish/best.pt")
    parser.add_argument("--preset", type=str, default="5m", choices=sorted(MODEL_PRESETS.keys()))
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--history-turns", type=int, default=6)
    parser.add_argument("--message", type=str, default="", help="Generate a single reply and exit.")
    return parser.parse_args()


def safe_console_text(text: str) -> str:
    text = text.replace("\ufffd", "?")
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def trim_response(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    for marker in STOP_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    return cleaned.strip()


def build_prompt(system_prompt: str, history: list[tuple[str, str]], user_message: str, history_turns: int) -> str:
    parts = [f"시스템: {system_prompt.strip()}"]
    for role, content in history[-history_turns:]:
        speaker = "사용자" if role == "user" else "도우미"
        parts.append(f"{speaker}: {content.strip()}")
    parts.append(f"사용자: {user_message.strip()}")
    parts.append("도우미:")
    return "\n".join(parts)


def load_model(checkpoint_path: Path, preset: str) -> TinyCausalLM:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = TinyCausalLM(MODEL_PRESETS[preset])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def reply(
    model: TinyCausalLM,
    tokenizer: ByteTokenizer,
    system_prompt: str,
    history: list[tuple[str, str]],
    user_message: str,
    args: argparse.Namespace,
) -> str:
    prompt = build_prompt(system_prompt, history, user_message, args.history_turns)
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    output = generate_text(model, tokenizer, prompt, generation_config=config)
    return trim_response(output)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except OSError:
            pass

    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    model = load_model(checkpoint_path, args.preset)
    tokenizer = ByteTokenizer()
    history: list[tuple[str, str]] = []

    if args.message:
        answer = reply(model, tokenizer, args.system_prompt, history, args.message, args)
        print(safe_console_text(answer))
        return

    print(safe_console_text("터미널 채팅을 시작합니다. '/reset'은 대화 초기화, '/quit'은 종료입니다."))
    while True:
        try:
            user_message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_message:
            continue
        if user_message == "/quit":
            break
        if user_message == "/reset":
            history.clear()
            print(safe_console_text("대화 기록을 초기화했습니다."))
            continue

        answer = reply(model, tokenizer, args.system_prompt, history, user_message, args)
        history.append(("user", user_message))
        history.append(("assistant", answer))
        print(f"bot> {safe_console_text(answer)}")


if __name__ == "__main__":
    main()
