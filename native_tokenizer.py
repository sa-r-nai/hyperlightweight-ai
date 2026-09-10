"""Self-contained English ASCII byte-pair tokenizer.

The tokenizer starts from 128 ASCII byte symbols and applies merge rules
trained only from caller-provided English text. It does not load a pretrained
vocabulary, a third-party tokenizer, or a model-specific chat template.
"""

from __future__ import annotations

import json
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class NativeTokenizer:
    """Deterministic English BPE with an explicit chat protocol."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    system_token_id = 4
    user_token_id = 5
    assistant_token_id = 6
    turn_end_token_id = 7
    byte_offset = 8
    alphabet_size = 128
    base_vocab_size = byte_offset + alphabet_size

    role_token_ids = {
        "system": system_token_id,
        "user": user_token_id,
        "assistant": assistant_token_id,
    }

    _whitespace = re.compile(r"[ \t\f\v]+")
    _pieces = re.compile(r"[A-Za-z]+|[0-9]+|[ \t\n]+|[^A-Za-z0-9 \t\n]+")

    def __init__(self, merges: Sequence[Sequence[int]] | None = None) -> None:
        self.merges = [tuple(map(int, pair)) for pair in (merges or [])]
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.token_bytes: list[bytes] = [b""] * self.base_vocab_size
        for value in range(self.alphabet_size):
            self.token_bytes[self.byte_offset + value] = bytes((value,))
        for left, right in self.merges:
            if left >= len(self.token_bytes) or right >= len(self.token_bytes):
                raise ValueError("A tokenizer merge references an unknown token ID.")
            self.token_bytes.append(self.token_bytes[left] + self.token_bytes[right])

    @property
    def vocab_size(self) -> int:
        return len(self.token_bytes)

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize layout while keeping the accepted language explicit."""

        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = NativeTokenizer._whitespace.sub(" ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
        try:
            normalized.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("The English tokenizer accepts ASCII text only.") from error
        return normalized

    @classmethod
    def split_pieces(cls, text: str) -> list[str]:
        return cls._pieces.findall(text)

    def _encode_piece(self, piece: str) -> list[int]:
        tokens = [self.byte_offset + value for value in piece.encode("ascii")]
        while len(tokens) > 1:
            candidates = (
                (self.merge_ranks[pair], index)
                for index, pair in enumerate(zip(tokens, tokens[1:]))
                if pair in self.merge_ranks
            )
            best = min(candidates, default=None)
            if best is None:
                break
            rank, _ = best
            pair = self.merges[rank]
            merged_id = self.base_vocab_size + rank
            updated: list[int] = []
            index = 0
            while index < len(tokens):
                if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
                    updated.append(merged_id)
                    index += 2
                else:
                    updated.append(tokens[index])
                    index += 1
            tokens = updated
        return tokens

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
        else:
            try:
                text.encode("ascii")
            except UnicodeEncodeError as error:
                raise ValueError("The English tokenizer accepts ASCII text only.") from error
        tokens = [token for piece in self.split_pieces(text) for token in self._encode_piece(piece)]
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        values = bytearray()
        for token_id in token_ids:
            if self.byte_offset <= token_id < self.vocab_size:
                values.extend(self.token_bytes[token_id])
            elif not skip_special_tokens and token_id == self.unk_token_id:
                values.extend(b"?")
        return values.decode("ascii", errors="strict")

    def encode_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        result = [self.bos_token_id] if add_bos else []
        for message_index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content")
            if role not in self.role_token_ids:
                raise ValueError(f"Unsupported message role: {role!r}")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Message content must be a non-empty string.")
            result.append(self.role_token_ids[role])
            result.extend(self.encode(content))
            if message_index != len(messages) - 1 or add_eos:
                result.append(self.turn_end_token_id)
        if add_eos:
            result.append(self.eos_token_id)
        return result

    def encode_generation_prompt(self, messages: Sequence[Mapping[str, str]]) -> list[int]:
        if not messages:
            raise ValueError("A conversation must contain at least one message.")
        tokens = self.encode_chat(messages, add_bos=True, add_eos=False)
        if messages[-1].get("role") != "assistant":
            tokens.extend((self.turn_end_token_id, self.assistant_token_id))
        return tokens

    def english_output_token_ids(self) -> list[int]:
        allowed_bytes = {9, 10, *range(32, 127)}
        return [
            token_id
            for token_id in range(self.byte_offset, self.vocab_size)
            if self.token_bytes[token_id] and set(self.token_bytes[token_id]) <= allowed_bytes
        ]

    def fingerprint(self) -> str:
        canonical = json.dumps(self.merges, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()

    def save(self, path: str | Path, metadata: Mapping[str, object] | None = None) -> None:
        payload = {
            "format": "native-english-bpe-v1",
            "language": "en",
            "encoding": "ascii",
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
            "metadata": dict(metadata or {}),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")

    @classmethod
    def load(cls, path: str | Path) -> "NativeTokenizer":
        payload = json.loads(Path(path).read_text(encoding="ascii"))
        if payload.get("format") != "native-english-bpe-v1":
            raise ValueError("Unsupported tokenizer format.")
        tokenizer = cls(payload.get("merges", []))
        if payload.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError("Tokenizer vocabulary size does not match its merge table.")
        return tokenizer


__all__ = ["NativeTokenizer"]
