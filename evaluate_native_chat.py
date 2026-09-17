"""Run a deterministic basic-conversation quality gate on a trained checkpoint."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import torch

from chat_native_200m import generate_reply, resolve_device
from native_200m import load_checkpoint
from native_tokenizer import NativeTokenizer


SYSTEM_PROMPT = (
    "You are a helpful English assistant. Answer accurately, clearly, and "
    "concisely. State uncertainty instead of inventing facts."
)

EVAL_CASES = [
    ("greeting", "Hello! How are you today?", ("hello", "help", "ready", "good")),
    ("capabilities", "What kinds of tasks can you help me with?", ("help", "explain", "plan", "write", "code")),
    ("clarification", "Can you make this better?", ("what", "which", "goal", "audience", "working")),
    ("cache", "Explain what a cache is in simple words.", ("cache", "temporary", "data", "faster")),
    ("backup", "How is a backup different from synchronization?", ("backup", "copy", "recover", "synchron")),
    ("debugging", "My program crashes. What should I check first?", ("error", "input", "reproduce", "small", "test")),
    ("planning", "I have too many tasks. What should I do first?", ("task", "priority", "deadline", "start", "impact")),
    ("study", "Give me a small study plan for this evening.", ("study", "review", "minutes", "recall", "topic")),
    ("writing", "Write a polite sentence asking for the report by Friday.", ("please", "report", "friday")),
    ("math", "What is 15 percent of 200?", ("30",)),
    ("python", "What does sum(range(1, 5)) return?", ("10",)),
    ("uncertainty", "Can you guarantee what will happen next year?", ("cannot", "uncertain", "evidence", "guarantee")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate basic NativeEnglishLM chat behavior.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizer/native_english_bpe.json"),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--min-pass-rate", type=float, default=0.75)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def response_is_well_formed(response: str) -> tuple[bool, str]:
    words = re.findall(r"[A-Za-z0-9']+", response)
    if len(words) < 3:
        return False, "response is too short"
    if len(response) > 1200:
        return False, "response is too long"
    if "\ufffd" in response:
        return False, "response contains a replacement character"
    lowered_words = [word.lower() for word in words]
    if len(lowered_words) >= 8 and len(set(lowered_words)) / len(lowered_words) < 0.25:
        return False, "response has excessive word repetition"
    trigrams = list(zip(lowered_words, lowered_words[1:], lowered_words[2:]))
    if trigrams and len(set(trigrams)) / len(trigrams) < 0.45:
        return False, "response has excessive phrase repetition"
    return True, "ok"


def main() -> None:
    args = parse_args()
    if not 0.0 < args.min_pass_rate <= 1.0:
        raise ValueError("--min-pass-rate must be greater than zero and at most one.")
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    tokenizer = NativeTokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("The checkpoint and tokenizer vocabulary sizes do not match.")
    saved_fingerprint = checkpoint.get("tokenizer_fingerprint")
    if saved_fingerprint and saved_fingerprint != tokenizer.fingerprint():
        raise ValueError("The checkpoint was created with a different tokenizer.")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model.to(device=device, dtype=dtype)
    else:
        model.to(device)
    model.eval()
    generation_args = SimpleNamespace(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.05,
    )

    passed = 0
    for name, prompt, expected_any in EVAL_CASES:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = generate_reply(model, tokenizer, messages, device, generation_args)
        well_formed, reason = response_is_well_formed(response)
        lowered = response.lower()
        relevant = any(keyword in lowered for keyword in expected_any)
        case_passed = well_formed and relevant
        passed += int(case_passed)
        status = "PASS" if case_passed else "FAIL"
        detail = reason if not well_formed else "missing expected concept"
        if case_passed:
            detail = "ok"
        print(f"[{status}] {name}: {detail}")
        print(f"  prompt: {prompt}")
        print(f"  response: {response}")

    pass_rate = passed / len(EVAL_CASES)
    print(f"[summary] passed={passed}/{len(EVAL_CASES)} pass_rate={pass_rate:.1%}")
    if pass_rate < args.min_pass_rate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
