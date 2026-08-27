"""GPU-first terminal chat for the standard Qwen3-0.6B PyTorch checkpoint."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "checkpoints_qwen3_06b_pytorch"
DEFAULT_SYSTEM_PROMPT = (
    "너는 친절하고 정확한 한국어 도우미다. 사용자의 의도를 먼저 파악하고 자연스러운 한국어로 답한다. "
    "확실하지 않은 사실은 추측하지 말고 모른다고 밝힌다. 일반적인 질문에는 핵심부터 간결하게 답한다."
)
REQUIRED_MODEL_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with Qwen3-0.6B through Python, PyTorch, and Transformers."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Execution device. CUDA is required by default; CPU is explicit opt-in.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--message", default="", help="Generate one reply and exit.")
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--threads", type=int, default=6, help="PyTorch thread count in CPU mode.")
    return parser.parse_args()


def validate_model_directory(model_path: Path) -> Path:
    resolved = model_path.expanduser().resolve()
    missing = sorted(name for name in REQUIRED_MODEL_FILES if not (resolved / name).is_file())
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing model files in {resolved}: {names}. "
            "Run download_qwen3_06b_pytorch_checkpoint.ps1 first."
        )
    return resolved


def load_runtime(model_path: Path, device: str, threads: int) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Transformers are required. Install packages from "
            "requirements-qwen3.txt in the Python environment you will use."
        ) from exc

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA mode is the default, but no CUDA GPU is available. "
                "The program will not fall back to CPU. Use --device cpu explicitly if intended."
            )
        runtime_device = torch.device("cuda")
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.set_float32_matmul_precision("high")
    else:
        runtime_device = torch.device("cpu")
        model_dtype = torch.float32
        torch.set_num_threads(max(1, threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=model_dtype,
    )
    model.to(runtime_device)
    model.eval()
    return torch, tokenizer, model


def build_messages(
    system_prompt: str,
    history: list[tuple[str, str]],
    user_message: str,
    history_turns: int,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt.strip()}]
    recent_history = history[-history_turns * 2 :] if history_turns > 0 else []
    for role, content in recent_history:
        messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": user_message.strip()})
    return messages


def clean_response(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def generate_reply(
    torch: Any,
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    input_length = inputs["input_ids"].shape[-1]

    generation_options: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_options.update(
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    else:
        generation_options["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_options)

    response_ids = output_ids[0, input_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return clean_response(response)


def safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def main() -> None:
    args = parse_args()
    model_path = validate_model_directory(args.model)
    torch, tokenizer, model = load_runtime(model_path, args.device, args.threads)
    history: list[tuple[str, str]] = []

    if args.message:
        messages = build_messages(args.system_prompt, history, args.message, args.history_turns)
        print(safe_console_text(generate_reply(torch, tokenizer, model, messages, args)))
        return

    print(
        safe_console_text(
            f"Python/PyTorch {args.device.upper()} 채팅을 시작합니다. "
            "'/reset'은 초기화, '/quit'은 종료입니다."
        )
    )
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

        messages = build_messages(args.system_prompt, history, user_message, args.history_turns)
        answer = generate_reply(torch, tokenizer, model, messages, args)
        history.append(("user", user_message))
        history.append(("assistant", answer))
        print(f"bot> {safe_console_text(answer)}")


if __name__ == "__main__":
    main()
