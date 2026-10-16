"""Run Korean conversations with the pinned base model or a trained LoRA adapter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, load_config
from .protocol import build_prompt


DEFAULT_ADAPTER = Path("artifacts_korean_600m/adapter")
PROVENANCE_FILE = "korean_600m_provenance.json"


def validate_adapter(adapter: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Reject an adapter produced against a different or unrecorded base revision."""
    adapter = Path(adapter)
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Adapter configuration not found: {config_path}. "
            "Train in Lightning AI Studio first, or pass --base-model."
        )
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    if metadata.get("base_model_name_or_path") != config["model_name"]:
        raise ValueError("Adapter base_model_name_or_path does not match the configured base model.")
    if metadata.get("revision") != config["model_revision"]:
        raise ValueError("Adapter revision is missing or differs from the pinned base revision.")
    if metadata.get("peft_type") != "LORA":
        raise ValueError("This pipeline requires a LoRA adapter.")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Safe adapter weights not found: {adapter / 'adapter_model.safetensors'}")
    return metadata


def _validate_merged(merged: Path, config: dict[str, Any]) -> None:
    if not merged.is_dir():
        raise FileNotFoundError(f"Merged model directory not found: {merged}")
    provenance_path = merged / PROVENANCE_FILE
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Merged model provenance not found: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (provenance.get("base_model") != config["model_name"]
            or provenance.get("base_revision") != config["model_revision"]):
        raise ValueError("Merged model provenance does not match the configured base and revision.")


def load_chat_model(
    config: dict[str, Any], *, adapter: Path | None, merged: Path | None, device: str
) -> tuple[Any, Any]:
    """Load weights only when explicitly invoked; CUDA never falls back to CPU."""
    if adapter is not None and merged is not None:
        raise ValueError("Choose either an adapter or a merged model.")
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be 'cuda' or 'cpu'.")
    if adapter is not None:
        validate_adapter(adapter, config)
    if merged is not None:
        _validate_merged(Path(merged), config)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-korean-600m.txt in Lightning AI Studio first.") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run in a GPU Studio, or explicitly use --device cpu for inference.")
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16

    model_source = str(merged) if merged is not None else config["model_name"]
    source_args: dict[str, Any] = {"trust_remote_code": False}
    if merged is not None:
        source_args["local_files_only"] = True
    else:
        source_args["revision"] = config["model_revision"]
    tokenizer_source = model_source
    tokenizer_args = dict(source_args)
    if adapter is not None and (Path(adapter) / "tokenizer_config.json").is_file():
        tokenizer_source = str(adapter)
        tokenizer_args = {"trust_remote_code": False, "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_args)
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=dtype,
        use_safetensors=True,
        **source_args,
    )
    if adapter is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Install PEFT from requirements-korean-600m.txt in Lightning AI Studio.") from exc
        model = PeftModel.from_pretrained(
            model, str(adapter), is_trainable=False, local_files_only=True,
            use_safetensors=True, autocast_adapter_dtype=False,
        )
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer


def generate_reply(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_seq_length: int,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """Reserve output space before pruning complete old conversation turns."""
    if max_new_tokens < 1 or max_new_tokens >= max_seq_length:
        raise ValueError("max_new_tokens must be positive and smaller than max_seq_length.")
    if do_sample and (not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be finite and positive when sampling.")
    prompt_ids = build_prompt(tokenizer, messages, max_seq_length - max_new_tokens)
    import torch

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    generation_args: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": 1.05,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_args.update(temperature=temperature, top_p=0.8, top_k=20)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **generation_args
        )
    return tokenizer.decode(output[0, len(prompt_ids):].tolist(), skip_special_tokens=True).strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the Korean 600M pipeline (loads weights when run).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER, help=f"Default: {DEFAULT_ADAPTER}")
    source.add_argument("--merged", type=Path, help="Standalone model created by korean_600m.export.")
    source.add_argument("--base-model", action="store_true", help="Compare the untouched pinned base.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--message", help="Generate one answer instead of starting interactive chat.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    adapter = None if (args.base_model or args.merged is not None) else (args.adapter or DEFAULT_ADAPTER)
    model, tokenizer = load_chat_model(config, adapter=adapter, merged=args.merged, device=args.device)
    messages = [{"role": "system", "content": config["system_prompt"]}]

    def answer(user_text: str) -> str:
        messages.append({"role": "user", "content": user_text})
        try:
            reply = generate_reply(
                model, tokenizer, messages, max_seq_length=config["max_seq_length"],
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            )
        except Exception:
            messages.pop()
            raise
        messages.append({"role": "assistant", "content": reply})
        return reply

    if args.message is not None:
        print(answer(args.message))
        return
    print("한국어 대화를 시작합니다. /reset: 대화 초기화, /exit: 종료")
    while True:
        try:
            text = input("사용자> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text == "/exit":
            return
        if text == "/reset":
            messages[:] = messages[:1]
            print("대화를 초기화했습니다.")
        elif text:
            try:
                print(f"도우미> {answer(text)}")
            except ValueError as exc:
                print(f"입력 오류: {exc}")


if __name__ == "__main__":
    main()
