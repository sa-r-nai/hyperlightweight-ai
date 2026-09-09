"""Self-contained UTF-8 byte tokenizer for NativeByteLM.

The tokenizer deliberately has no dependency on a pretrained vocabulary or a
third-party chat template.  Every non-special token is one byte from the
UTF-8 representation of the normalized input.  This makes the model able to
round-trip Korean, emoji, source code, and arbitrary Unicode text without an
external vocabulary file.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Mapping, Sequence


class NativeTokenizer:
    """A deterministic byte-level tokenizer with an explicit chat protocol."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    system_token_id = 4
    user_token_id = 5
    assistant_token_id = 6
    turn_end_token_id = 7
    byte_offset = 8
    vocab_size = byte_offset + 256

    role_token_ids = {
        "system": system_token_id,
        "user": user_token_id,
        "assistant": assistant_token_id,
    }

    _whitespace = re.compile(r"[ \t\f\v]+")

    @staticmethod
    def normalize(text: str) -> str:
        """Apply only representation-safe normalization."""

        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = NativeTokenizer._whitespace.sub(" ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        normalize: bool = True,
    ) -> list[int]:
        if normalize:
            text = self.normalize(text)
        tokens = [self.byte_offset + value for value in text.encode("utf-8")]
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        byte_values = bytearray()
        for token_id in token_ids:
            if self.byte_offset <= token_id < self.vocab_size:
                byte_values.append(token_id - self.byte_offset)
            elif not skip_special_tokens and token_id == self.unk_token_id:
                byte_values.extend("�".encode("utf-8"))
        return byte_values.decode("utf-8", errors="replace")

    def encode_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """Encode messages using the repository's own role-token protocol."""

        result: list[int] = [self.bos_token_id] if add_bos else []
        for message_index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content")
            if role not in self.role_token_ids:
                raise ValueError(f"지원하지 않는 메시지 역할입니다: {role!r}")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("메시지 내용은 비어 있지 않은 문자열이어야 합니다.")

            result.append(self.role_token_ids[role])
            result.extend(self.encode(content))
            if message_index != len(messages) - 1 or add_eos:
                result.append(self.turn_end_token_id)

        if add_eos:
            result.append(self.eos_token_id)
        return result

    def encode_generation_prompt(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> list[int]:
        """Encode a conversation and leave the assistant role open."""

        if not messages:
            raise ValueError("대화에는 최소 한 개의 메시지가 필요합니다.")
        tokens = self.encode_chat(messages, add_bos=True, add_eos=False)
        if messages[-1].get("role") != "assistant":
            tokens.extend((self.turn_end_token_id, self.assistant_token_id))
        return tokens

    def special_token_name(self, token_id: int) -> str | None:
        names = {
            self.pad_token_id: "<|pad|>",
            self.bos_token_id: "<|bos|>",
            self.eos_token_id: "<|eos|>",
            self.unk_token_id: "<|unk|>",
            self.system_token_id: "<|system|>",
            self.user_token_id: "<|user|>",
            self.assistant_token_id: "<|assistant|>",
            self.turn_end_token_id: "<|turn_end|>",
        }
        return names.get(token_id)


__all__ = ["NativeTokenizer"]
