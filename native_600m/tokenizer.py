"""Independent byte-level BPE vocabulary and native conversation protocol.

Tokenizer corpus training is restricted to Lightning AI Studio. Importing this
module, loading a tokenizer, and encoding text never train a tokenizer or model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


SPECIAL_TOKENS = (
    "<|pad|>", "<|bos|>", "<|eos|>", "<|system|>",
    "<|user|>", "<|assistant|>", "<|turn_end|>",
)
SPECIAL_TOKEN_IDS = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
DEFAULT_TOKENIZER = "artifacts_native_600m/tokenizer/tokenizer.json"
_ROLE_IDS = {"system": 3, "user": 4, "assistant": 5}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeTokenizer:
    """A fresh BPE tokenizer with explicit role tokens, independent of any model."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    system_token_id = 3
    user_token_id = 4
    assistant_token_id = 5
    turn_end_token_id = 6
    special_token_ids = tuple(range(len(SPECIAL_TOKENS)))

    def __init__(self, backend, path: Path):
        self._backend = backend
        self.path = path
        self.fingerprint = _sha256(path)
        self._validate()

    def _validate(self) -> None:
        from tokenizers.pre_tokenizers import ByteLevel

        definition = json.loads(self._backend.to_str())
        if definition.get("model", {}).get("type") != "BPE":
            raise ValueError("The native tokenizer must use a BPE model.")
        if (definition.get("normalizer") or {}).get("type") != "NFC":
            raise ValueError("The native tokenizer must use Unicode NFC normalization.")
        pre_tokenizer = definition.get("pre_tokenizer") or {}
        if pre_tokenizer.get("type") != "ByteLevel" or pre_tokenizer.get("add_prefix_space") is not False:
            raise ValueError("The native tokenizer requires ByteLevel(add_prefix_space=False).")
        if (definition.get("decoder") or {}).get("type") != "ByteLevel":
            raise ValueError("The native tokenizer requires the ByteLevel decoder.")
        if definition.get("post_processor") is not None:
            raise ValueError("Automatic post-processing is incompatible with the native chat protocol.")
        for token, expected in SPECIAL_TOKEN_IDS.items():
            if self._backend.token_to_id(token) != expected:
                raise ValueError(f"Special token {token} must have ID {expected}.")
        registered = {entry["content"] for entry in definition.get("added_tokens", []) if entry["special"]}
        if registered != set(SPECIAL_TOKENS):
            raise ValueError("Only the native protocol tokens may be registered as special tokens.")
        vocabulary = self._backend.get_vocab()
        if not set(ByteLevel.alphabet()).issubset(vocabulary):
            raise ValueError("The native tokenizer must contain all 256 byte-level symbols.")
        if sorted(vocabulary.values()) != list(range(len(vocabulary))):
            raise ValueError("Tokenizer IDs must be unique and contiguous.")
        self._backend.no_truncation()
        self._backend.no_padding()

    @classmethod
    def load(cls, path: str | Path = DEFAULT_TOKENIZER) -> "NativeTokenizer":
        from tokenizers import Tokenizer

        resolved = Path(path)
        return cls(Tokenizer.from_file(str(resolved)), resolved)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._backend.save(str(destination))
        self.path = destination
        self.fingerprint = _sha256(destination)

    @property
    def vocab_size(self) -> int:
        return self._backend.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be text.")
        for token in SPECIAL_TOKENS:
            if token in text:
                raise ValueError(f"Text contains reserved protocol marker {token}.")
        return self._backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._backend.decode(list(ids), skip_special_tokens=True)

    def encode_document(self, text: str) -> list[int]:
        return [self.bos_token_id, *self.encode(text), self.eos_token_id]

    @staticmethod
    def _messages(messages: Sequence[Mapping[str, str]], *, training: bool) -> list[dict[str, str]]:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
            raise ValueError("messages must be a nonempty sequence of role/content objects.")
        checked: list[dict[str, str]] = []
        expected = "user"
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise ValueError("Each message must contain a role and content.")
            role, content = message.get("role"), message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Each message must contain nonempty text.")
            if role == "system" and index == 0:
                checked.append({"role": role, "content": content})
                continue
            if role != expected:
                raise ValueError("Messages require an optional initial system message and alternating user/assistant turns.")
            checked.append({"role": role, "content": content})
            expected = "assistant" if role == "user" else "user"
        last_role = "assistant" if training else "user"
        if checked[-1]["role"] != last_role:
            raise ValueError(f"{'Training conversations' if training else 'Prompts'} must end with a {last_role} message.")
        return checked

    def _encode_turns(self, messages: Sequence[Mapping[str, str]]) -> tuple[list[int], list[int]]:
        ids, mask = [self.bos_token_id], [0]
        for message in messages:
            content = self.encode(message["content"])
            supervised = int(message["role"] == "assistant")
            turn = [_ROLE_IDS[message["role"]], *content, self.turn_end_token_id]
            ids.extend(turn)
            mask.extend([0, *([supervised] * (len(content) + 1))])
        return ids, mask

    def encode_prompt(self, messages: Sequence[Mapping[str, str]], max_tokens: int) -> list[int]:
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer.")
        checked = self._messages(messages, training=False)
        # Encode every supplied turn before trimming, so reserved-marker input
        # is rejected consistently even when the old turn leaves the context.
        encoded = []
        for message in checked:
            encoded.append([_ROLE_IDS[message["role"]], *self.encode(message["content"]), self.turn_end_token_id])
        has_system = checked[0]["role"] == "system"
        system = encoded[:1] if has_system else []
        conversation = encoded[1:] if has_system else encoded
        while True:
            result = [self.bos_token_id]
            for turn in [*system, *conversation]:
                result.extend(turn)
            result.append(self.assistant_token_id)
            if len(result) <= max_tokens:
                return result
            if len(conversation) <= 1:
                raise ValueError("The system prompt and latest user message exceed the context budget; shorten the message.")
            # Preserve complete turns and the latest user message.
            conversation = conversation[2:]

    def encode_chat(self, messages: Sequence[Mapping[str, str]]) -> tuple[list[int], list[int]]:
        checked = self._messages(messages, training=True)
        ids, mask = self._encode_turns(checked)
        ids.append(self.eos_token_id)
        mask.append(0)
        return ids, mask


def train_tokenizer(input_path: str | Path, output_path: str | Path, *, vocab_size: int = 32000,
                    max_documents: int = 200000) -> dict:
    """Train a new vocabulary from UTF-8 JSONL text records inside Studio only."""
    from .runtime import require_studio

    require_studio()
    if type(vocab_size) is not int or vocab_size < 264:
        raise ValueError("vocab_size must be at least 264 (256 bytes, 7 markers, and a learned merge).")
    if type(max_documents) is not int or max_documents < 1:
        raise ValueError("max_documents must be positive.")
    from tokenizers import AddedToken, Tokenizer, __version__, decoders, models, normalizers, pre_tokenizers, trainers

    source = Path(input_path)
    destination = Path(output_path)
    if destination.exists():
        raise ValueError("Tokenizer already exists. Use --reuse-prepared or choose a new output path.")
    source_hash = _sha256(source)
    stats = {"documents": 0, "characters": 0}

    def records():
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {source}:{line_number}.") from error
                text = record.get("text") if isinstance(record, dict) else None
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Expected a nonempty text field at {source}:{line_number}.")
                if any(marker in text for marker in SPECIAL_TOKENS):
                    raise ValueError(f"Reserved chat marker in tokenizer corpus at {source}:{line_number}.")
                stats["documents"] += 1
                stats["characters"] += len(text)
                yield text
                if stats["documents"] >= max_documents:
                    break

    backend = Tokenizer(models.BPE())
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        special_tokens=[AddedToken(token, normalized=False, special=True) for token in SPECIAL_TOKENS],
    )
    backend.train_from_iterator(records(), trainer=trainer)
    if not stats["documents"] or backend.get_vocab_size() != vocab_size:
        raise ValueError(
            f"Corpus yielded vocabulary size {backend.get_vocab_size()}, below requested {vocab_size}. "
            "Provide more varied Korean pretraining text or increase --max-documents. No tokenizer was saved."
        )
    if _sha256(source) != source_hash:
        raise ValueError("Tokenizer source changed during training; retry with an immutable source file.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    backend.save(str(temporary))
    tokenizer = NativeTokenizer.load(temporary)
    temporary.replace(destination)
    manifest = {
        "format_version": 1,
        "initialization": "new_byte_level_bpe",
        "tokenizers_version": __version__,
        "input": str(source),
        "input_sha256": source_hash,
        "max_documents": max_documents,
        **stats,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": tokenizer.fingerprint,
        "special_tokens": SPECIAL_TOKEN_IDS,
        "normalization": "NFC",
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def validate_prepared_tokenizer(input_path: str | Path, output_path: str | Path, *,
                                vocab_size: int, max_documents: int) -> dict:
    source, destination = Path(input_path), Path(output_path)
    manifest = json.loads(destination.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    tokenizer = NativeTokenizer.load(destination)
    expected = {"initialization": "new_byte_level_bpe", "input_sha256": _sha256(source),
                "max_documents": max_documents, "vocab_size": vocab_size,
                "tokenizer_sha256": tokenizer.fingerprint}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Prepared tokenizer differs from the requested source or configuration.")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an independent Korean BPE tokenizer in Lightning AI Studio.")
    parser.add_argument("--input", required=True, help="UTF-8 JSONL file containing text records.")
    parser.add_argument("--output", default=DEFAULT_TOKENIZER)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-documents", type=int, default=200000)
    parser.add_argument("--reuse-prepared", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.reuse_prepared:
        result = validate_prepared_tokenizer(args.input, args.output, vocab_size=args.vocab_size,
                                            max_documents=args.max_documents)
    else:
        result = train_tokenizer(args.input, args.output, vocab_size=args.vocab_size, max_documents=args.max_documents)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
