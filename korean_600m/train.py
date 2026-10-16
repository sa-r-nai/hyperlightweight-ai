"""Run Korean assistant-only LoRA SFT in a single-GPU Lightning AI Studio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, config_fingerprint, load_config, sha256_file, write_json
from .protocol import AssistantCollator, encode_conversation
from .runtime import require_studio_cuda


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts_korean_600m/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_korean_600m/training"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("artifacts_korean_600m/adapter"))
    parser.add_argument("--resume", default="auto", help="auto, none, or a checkpoint directory")
    parser.add_argument("--max-steps", type=int, default=-1, help="Studio-only short run; -1 uses configured epochs")
    return parser.parse_args(argv)


def resolve_resume(output_dir: Path, requested: str, *, require_scaler: bool = False) -> Path | None:
    if requested == "none":
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("Use an empty --output-dir for a new run; existing training state is preserved.")
        return None
    if requested != "auto":
        path = Path(requested).resolve()
        if path.parent != output_dir.resolve():
            raise ValueError("The resume checkpoint must belong to --output-dir.")
        candidates = [path]
    else:
        candidates = sorted(
            (p for p in output_dir.glob("checkpoint-*") if p.name.removeprefix("checkpoint-").isdigit()),
            key=lambda p: int(p.name.removeprefix("checkpoint-")), reverse=True,
        )
    for candidate in candidates:
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json",
                    "adapter_model.safetensors", "rng_state.pth")
        if require_scaler:
            required += ("scaler.pt",)
        if all((candidate / filename).is_file() for filename in required):
            return candidate
    if candidates:
        raise ValueError("No complete resumable checkpoint found; retain the directory and choose a new output path.")
    if requested != "auto":
        raise FileNotFoundError(requested)
    return None


def verify_completed_run(path: Path, identity: dict, adapter_dir: Path) -> bool:
    if not path.is_file():
        return False
    completed = json.loads(path.read_text(encoding="utf-8"))
    if completed.get("identity") != identity or completed.get("adapter_dir") != str(adapter_dir.resolve()):
        raise ValueError("Completed run belongs to a different configuration or adapter directory.")
    hashes = completed.get("adapter_files")
    if not isinstance(hashes, dict) or set(hashes) != {
        "adapter_model.safetensors", "adapter_config.json", "korean_600m_provenance.json",
        "tokenizer.json", "tokenizer_config.json",
    }:
        raise ValueError("Completed run has an incomplete adapter file manifest.")
    for name, expected in hashes.items():
        file = adapter_dir / name
        if not file.is_file() or sha256_file(file) != expected:
            raise ValueError(f"Completed adapter is missing or changed: {file}")
    return True


def tokenize_split(dataset, tokenizer, max_length: int):
    def encode_batch(batch):
        rows = [row for messages in batch["messages"]
                for row in encode_conversation(tokenizer, messages, max_length)]
        return {key: [row[key] for row in rows] for key in ("input_ids", "attention_mask", "labels")}

    result = dataset.map(encode_batch, batched=True, batch_size=100,
                         remove_columns=dataset.column_names, desc="Encoding complete assistant turns")
    if len(result) == 0:
        raise ValueError("No complete assistant responses fit the context length. Review the corpus/configuration.")
    return result


def train(args: argparse.Namespace) -> None:
    # This check precedes imports, network operations, model creation, and optimizer setup.
    torch = require_studio_cuda()
    config = load_config(args.config)
    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    precision = "bf16" if bf16 else "fp16"
    if args.max_steps != -1 and args.max_steps <= 0:
        raise ValueError("--max-steps must be -1 or positive.")
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    from .prepare import validate_prepared

    validate_prepared(args.data_dir, config)
    resume = resolve_resume(args.output_dir, args.resume, require_scaler=not bf16)
    run_identity = {
        "format": "korean-600m-run-v1",
        "config_sha256": config_fingerprint(config),
        "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "max_steps": args.max_steps,
        "precision": precision,
    }
    run_file = args.output_dir / "run_manifest.json"
    if run_file.exists():
        previous = json.loads(run_file.read_text(encoding="utf-8"))
        if previous.get("identity") != run_identity:
            raise ValueError("Training configuration/data changed. Choose a new --output-dir and --adapter-dir.")
    elif resume is not None:
        raise ValueError("Checkpoint has no run_manifest.json; its configuration cannot be verified.")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory contains unrelated files. Choose an empty directory.")

    adapter_provenance = args.adapter_dir / "korean_600m_provenance.json"
    if args.adapter_dir.exists() and any(args.adapter_dir.iterdir()):
        if not adapter_provenance.is_file():
            raise ValueError("Adapter directory contains unrelated files; choose another --adapter-dir.")
        previous_adapter = json.loads(adapter_provenance.read_text(encoding="utf-8"))
        if previous_adapter.get("run_identity") != run_identity:
            raise ValueError("Adapter directory belongs to another run; choose another --adapter-dir.")

    if args.resume == "auto" and verify_completed_run(
        args.output_dir / "completed.json", run_identity, args.adapter_dir,
    ):
        print(f"[complete] Verified completed training and adapter: {args.adapter_dir}", flush=True)
        return

    set_seed(config["seed"])
    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], revision=config["model_revision"], trust_remote_code=False,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The pinned tokenizer must define pad and EOS tokens.")
    splits = load_dataset("json", data_files={
        "train": str(args.data_dir / "train.jsonl"),
        "validation": str(args.data_dir / "validation.jsonl"),
    })
    encoded = {key: tokenize_split(dataset, tokenizer, config["max_seq_length"])
               for key, dataset in splits.items()}
    token_stats = {}
    for key, dataset in encoded.items():
        input_tokens = supervised_tokens = 0
        for row in dataset:
            input_tokens += len(row["input_ids"])
            supervised_tokens += sum(label != -100 for label in row["labels"])
        original_turns = sum(sum(message["role"] == "assistant" for message in row["messages"])
                             for row in splits[key])
        token_stats[key] = {"examples": len(dataset), "input_tokens": input_tokens,
                            "supervised_tokens": supervised_tokens,
                            "skipped_oversized_turns": original_turns - len(dataset)}
    print(json.dumps({"tokenization": token_stats}, ensure_ascii=False), flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], revision=config["model_revision"], torch_dtype=dtype,
        attn_implementation="sdpa", trust_remote_code=False, use_safetensors=True,
    )
    unique_parameters = sum(parameter.numel() for parameter in model.parameters())
    if not 590_000_000 <= unique_parameters <= 610_000_000:
        raise ValueError(f"Unexpected unique base parameter count: {unique_parameters:,}")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, bias="none", revision=config["model_revision"], **config["lora"],
    ))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    train_options = dict(config["training"])
    # Evaluate/save at least once for short Studio checks as well as full runs.
    if args.max_steps > 0:
        interval = min(args.max_steps, train_options["eval_steps"])
        train_options.update(eval_steps=interval, save_steps=interval)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), **train_options, max_steps=args.max_steps,
        lr_scheduler_type="cosine", max_grad_norm=1.0,
        bf16=bf16, fp16=not bf16, tf32=torch.cuda.get_device_capability()[0] >= 8,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", save_strategy="steps", logging_strategy="steps",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_safetensors=True, save_only_model=False, prediction_loss_only=True,
        optim="adamw_torch", report_to="none", seed=config["seed"], data_seed=config["seed"],
        remove_unused_columns=False, label_names=["labels"],
        dataloader_pin_memory=True,
    )
    class EvaluateFinalStep(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            if state.global_step >= state.max_steps:
                control.should_evaluate = True
                control.should_save = True
            return control

    trainer = Trainer(
        model=model, args=training_args, train_dataset=encoded["train"], eval_dataset=encoded["validation"],
        data_collator=AssistantCollator(tokenizer.pad_token_id), processing_class=tokenizer,
        callbacks=[EvaluateFinalStep()],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_file, {"identity": run_identity, "config": config, "tokenization": token_stats,
                          "unique_base_parameters": unique_parameters})
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    # The Trainer has reloaded its best held-out checkpoint here.
    trainer.save_model(str(args.adapter_dir))
    tokenizer.save_pretrained(args.adapter_dir)
    trainer.save_state()
    metrics = {**result.metrics, **trainer.evaluate()}
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(adapter_provenance, {
        "format": "korean-600m-adapter-v1", "run_identity": run_identity,
        "base_model": config["model_name"], "base_revision": config["model_revision"],
        "unique_base_parameters": unique_parameters,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_loss": trainer.state.best_metric,
        "config": config, "data_manifest": json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8")),
        "metrics": metrics,
        "quality_status": "Requires generated conversation evaluation and human review.",
    })
    write_json(args.output_dir / "completed.json", {
        "identity": run_identity, "adapter_dir": str(args.adapter_dir.resolve()),
        "adapter_files": {name: sha256_file(args.adapter_dir / name) for name in (
            "adapter_model.safetensors", "adapter_config.json", "korean_600m_provenance.json",
            "tokenizer.json", "tokenizer_config.json",
        )},
    })
    print(f"[complete] Best held-out adapter saved to {args.adapter_dir}", flush=True)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
