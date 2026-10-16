"""Qwen chat formatting with loss restricted to a complete assistant response.

Each assistant turn becomes a separate example. Generation and SFT share the
same non-thinking prefix. Older complete user/assistant pairs may be removed;
current messages and responses are never silently truncated.
"""

from __future__ import annotations


def build_prompt(tokenizer, messages: list[dict[str, str]], max_input_tokens: int) -> list[int]:
    if max_input_tokens <= 0:
        raise ValueError("The response leaves no room for a prompt.")
    if (len(messages) < 2 or messages[0]["role"] != "system"
            or messages[-1]["role"] != "user"):
        raise ValueError("A prompt must contain a system message and end with a user turn.")
    for index, message in enumerate(messages[1:]):
        if message["role"] != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("Prompt roles must alternate user and assistant.")
    retained = list(messages)
    while True:
        tokens = tokenizer.apply_chat_template(
            retained, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        if len(tokens) <= max_input_tokens:
            return tokens
        if len(retained) <= 2:
            raise ValueError("The system message and latest user turn exceed the context budget.")
        retained = [retained[0], *retained[3:]]


def encode_conversation(tokenizer, messages: list[dict[str, str]], max_length: int) -> list[dict]:
    """Return turn-level SFT examples; skip responses that cannot fit intact."""
    examples = []
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an end-of-turn token.")
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        answer = tokenizer.encode(message["content"], add_special_tokens=False)
        answer.append(tokenizer.eos_token_id)
        try:
            prompt = build_prompt(tokenizer, messages[:index], max_length - len(answer))
        except ValueError:
            continue
        input_ids = prompt + answer
        examples.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt) + answer,
        })
    return examples


class AssistantCollator:
    """Right-pad context tokens while masking all padding labels."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        longest = max(len(row["input_ids"]) for row in features)
        return {
            key: torch.tensor([
                row[key] + [pad] * (longest - len(row[key])) for row in features
            ], dtype=torch.long)
            for key, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))
        }
