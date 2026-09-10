"""Create a small, self-authored English seed corpus for smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a helpful English assistant. Answer accurately, clearly, and "
    "concisely. State uncertainty instead of inventing facts."
)

PASSAGES = [
    ("Planning", "A useful plan separates the goal, completion criteria, sequence, and review points. Start with the smallest verifiable result, check it, and then add the next step."),
    ("Debugging", "Debugging narrows observable differences. Fix the reproduction steps, record the expected and actual results, reduce the case, and inspect the first intermediate value that differs."),
    ("Data quality", "Training data should be checked for duplicates, broken encoding, contradictions, unclear provenance, and license restrictions. Record every transformation so problems remain traceable."),
    ("Testing", "Unit tests provide fast feedback about small behaviors. Integration tests reveal failures between components. Both are needed to change software quickly without losing confidence."),
    ("Networks", "Investigate network problems from physical connectivity through addressing, name resolution, routing, and application responses. Change one variable at a time so each result is meaningful."),
    ("Learning", "Active recall is more effective than repeated reading. Study briefly, reconstruct the main ideas without notes, and revisit difficult material after increasing intervals."),
    ("Writing", "Clear technical writing states the decision first and then gives evidence, constraints, and exceptions. Short paragraphs and reproducible examples make review easier."),
    ("Language models", "A causal language model learns to predict the next token from previous tokens. Training loss measures fit to training examples, while validation loss estimates performance on unseen data."),
]

SINGLE_TURN = [
    ("Definition", "Explain a cache in one short paragraph.", "A cache is a temporary copy of data stored closer to where it will be used, making repeated access faster. It needs an expiration or refresh policy because the original data can change."),
    ("Definition", "What is the difference between backup and synchronization?", "A backup is a separate copy intended for recovery. Synchronization keeps current files aligned across locations and may propagate deletions, so synchronization alone is not a backup."),
    ("Math", "What is the price after a 15 percent discount on 120 dollars?", "The discount is 120 times 0.15, which is 18 dollars. The final price is 102 dollars."),
    ("Debugging", "Python reports ModuleNotFoundError. What should I check?", "Verify that the package is installed in the same Python environment that runs the program. Check the interpreter path and inspect the package with python -m pip show."),
    ("Debugging", "My program runs but returns the wrong result.", "Reduce the input to the smallest reproducible example. Record expected and actual values, then inspect intermediate results until you find the first divergence."),
    ("Writing", "Write a short message saying I will be ten minutes late.", "I am sorry, but my travel has been delayed and I expect to arrive about ten minutes late. I will join as soon as I arrive."),
    ("Planning", "How should I start learning a programming language?", "Learn the basic syntax, build a small input-output program, add one practical feature such as file access, and then practice testing and packaging."),
    ("Safety", "Should I immediately run a file downloaded from the internet?", "No. Verify the source and file type, scan it, inspect the requested permissions, and use an isolated environment when the source is uncertain."),
    ("Learning", "Why do I forget material soon after studying it?", "Reading creates familiarity but not reliable recall. Close the material, retrieve the main ideas from memory, and repeat that exercise after increasing intervals."),
    ("Networks", "The internet fails on only one device. What should I do?", "Reconnect that device, verify its address and DNS settings, and then inspect its firewall and network driver. If other devices work, the router connection is probably not the main cause."),
    ("Coding", "How should I name a function?", "Use a short verb phrase that describes the observable action. Prefer a specific name such as validate_email over vague names such as process_data."),
    ("Summary", "Summarize this: Small verified changes reveal problems early and reduce rework.", "Small, verified changes catch problems early and reduce rework."),
    ("Honesty", "What should you do when you do not know an answer?", "I should say that I am uncertain, identify what information is missing, and suggest a reliable way to verify it instead of inventing an answer."),
    ("Git", "What does a Git commit do?", "A commit records a named snapshot of staged changes and links it to the repository history. It does not automatically upload the change to a remote server."),
    ("Files", "Why should large model checkpoints stay out of Git?", "Model checkpoints are large binary artifacts that change as a whole. Object storage or an artifact service handles them more efficiently than a normal Git repository."),
]

MULTI_TURN = [
    ("Planning", [("user", "I have too many tasks. What should I do first?"), ("assistant", "Order them by deadline, impact, and whether they block other people. Choose one high-impact task as the next action."), ("user", "Three tasks are equally urgent."), ("assistant", "Estimate their duration, start with the shortest blocking task, and tell affected people early if any deadline is at risk.")]),
    ("Debugging", [("user", "The codebase is too large to debug."), ("assistant", "Start at the first observable failure and isolate the function that produces it with a fixed input."), ("user", "That function has many dependencies."), ("assistant", "Replace dependencies with small controlled substitutes one at a time. The first replacement that removes the failure narrows the cause.")]),
    ("Learning", [("user", "I forget new vocabulary quickly."), ("assistant", "Hide the definition and retrieve it from memory instead of rereading it."), ("user", "How many words should I study each day?"), ("assistant", "Begin with about ten new words and spend more time reviewing older words. Adjust the number after measuring recall for a week.")]),
]


def make_record(record_id: str, category: str, turns: list[tuple[str, str]]) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": role, "content": content} for role, content in turns)
    return {
        "id": record_id,
        "category": category,
        "source": "self_authored_seed",
        "license": "self-authored",
        "messages": messages,
    }


def build_records() -> list[dict]:
    records = [
        make_record(
            f"native-single-{index:04d}",
            category,
            [("user", user), ("assistant", assistant)],
        )
        for index, (category, user, assistant) in enumerate(SINGLE_TURN, start=1)
    ]
    records.extend(
        make_record(f"native-multi-{index:04d}", category, turns)
        for index, (category, turns) in enumerate(MULTI_TURN, start=1)
    )
    return records


def validate_records(records: list[dict]) -> None:
    seen_ids: set[str] = set()
    for record in records:
        record_id = record["id"]
        if record_id in seen_ids:
            raise ValueError(f"Duplicate record ID: {record_id}")
        seen_ids.add(record_id)
        messages = record["messages"]
        if not messages or messages[0]["role"] != "system":
            raise ValueError(f"Missing system message: {record_id}")
        if messages[-1]["role"] != "assistant":
            raise ValueError(f"Conversation does not end with an assistant: {record_id}")
        for message in messages:
            if message["role"] not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported role in record: {record_id}")
            if not message["content"].strip():
                raise ValueError(f"Empty message in record: {record_id}")
            message["content"].encode("ascii")


def write_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_records()
    validate_records(records)
    sft_path = output_dir / "native_sft_seed.jsonl"
    with sft_path.open("w", encoding="ascii", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    pretraining_path = output_dir / "native_pretraining_seed.txt"
    pretraining_path.write_text(
        "\n\n".join(f"# {title}\n\n{text}" for title, text in PASSAGES) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(f"[info] Generated {len(records)} English SFT records: {sft_path}")
    print(f"[info] Generated {len(PASSAGES)} English passages: {pretraining_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-authored English seed generator")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    write_outputs(args.output_dir)


if __name__ == "__main__":
    main()
