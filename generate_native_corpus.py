"""Build a deterministic, self-authored ASCII English training corpus."""

from __future__ import annotations

import json
import random
from itertools import product
from pathlib import Path


# name, definition, concrete example, common caution, related concept
TOPIC_GUIDES = [
    ("algorithms", "step-by-step procedures for solving a defined problem", "sorting a list of names into alphabetical order", "an elegant method may still be too slow for large inputs", "A data structure organizes information, while an algorithm operates on it."),
    ("application programming interfaces", "documented boundaries that let software components exchange requests and results", "a weather application requesting a forecast from a server", "callers must handle errors, timeouts, and version changes", "A user interface serves people, while an API serves other software."),
    ("backups", "independent copies kept so lost or damaged data can be restored", "saving an encrypted copy of project files in another location", "a backup that is never tested may fail when recovery is needed", "Synchronization keeps current copies aligned, but a backup preserves recovery points."),
    ("caching", "keeping reusable data close to where it is consumed", "storing a recent database result in memory", "stale entries need an expiration or invalidation rule", "A cache improves access speed, while permanent storage preserves data."),
    ("databases", "systems that store, organize, and retrieve persistent information", "recording customers and orders in related tables", "poor constraints allow incomplete or contradictory records", "A file stores bytes, while a database adds structured querying and consistency rules."),
    ("debugging", "finding the cause of a difference between expected and actual behavior", "reducing a failing request to the smallest reproducible input", "changing several variables at once hides which change mattered", "Testing detects failures, while debugging explains and removes their cause."),
    ("encryption", "transforming readable data so only holders of the correct key can recover it", "encrypting a laptop drive before travel", "encryption does not replace access control or backups", "Encoding changes representation, while encryption protects confidentiality with a key."),
    ("Git", "a version-control system that records snapshots and relationships between changes", "committing a tested bug fix on a feature branch", "a local commit is not automatically uploaded to a remote", "Git records version history, while a hosting service shares repositories."),
    ("HTTP", "a request-response protocol used to exchange web resources", "a browser sending a GET request for a page", "successful transport does not guarantee a semantically correct response", "HTTP defines messages, while DNS maps names to network addresses."),
    ("machine learning", "estimating model parameters from examples to improve predictions", "classifying support messages by intent", "training accuracy alone can hide overfitting", "Rules are written directly, while learned models infer patterns from data."),
    ("model validation", "measuring behavior on held-out examples that were not used for updates", "tracking validation loss after each training interval", "reusing the test set for tuning makes the final estimate optimistic", "Training loss guides updates, while validation loss estimates generalization."),
    ("networks", "connected systems that exchange data according to shared protocols", "a laptop sending a request through a router to a server", "a failure can occur at addressing, routing, naming, or application layers", "A network moves data, while an application decides what the data means."),
    ("probability", "a mathematical language for describing uncertainty", "estimating the chance of drawing a red card from a shuffled deck", "possible outcomes must be defined before a probability is interpreted", "Probability models uncertainty, while statistics learns from observed samples."),
    ("project planning", "turning a goal into ordered work, dependencies, owners, and checkpoints", "splitting a release into implementation, testing, and rollout tasks", "a plan without completion criteria makes progress difficult to verify", "A goal names the result, while a plan describes how to reach it."),
    ("queues", "ordered collections where items wait to be processed", "placing print jobs in arrival order", "unbounded arrival rates create growing delay and memory use", "A stack removes the newest item first, while a queue normally removes the oldest."),
    ("software testing", "checking observable behavior against explicit expectations", "verifying that an empty input returns a helpful error", "tests that depend on unstable external state become unreliable", "A unit test isolates a small behavior, while an integration test checks components together."),
    ("technical writing", "communicating decisions, evidence, constraints, and actions clearly", "writing a reproducible bug report", "undefined terms and hidden assumptions force readers to guess", "A summary shortens content, while a specification defines required behavior."),
    ("web security", "reducing application risk through validation, authentication, authorization, and safe defaults", "rejecting an invalid upload before storing it", "client-side checks alone cannot protect a server", "Authentication identifies a user, while authorization controls permitted actions."),
]

AUDIENCES = (
    "a high school student",
    "a first-year university student",
    "a beginner programmer",
    "a new team member",
    "a project manager",
    "a curious non-specialist",
)
TASK_KINDS = ("explain", "example", "compare", "caution", "plan")


def topic_prompt(kind: str, name: str, audience: str) -> str:
    return {
        "explain": f"Explain {name} clearly for {audience}.",
        "example": f"Define {name} and give one concrete example for {audience}.",
        "compare": f"Compare {name} with a related concept for {audience}.",
        "caution": f"Describe a common mistake involving {name} and how to avoid it.",
        "plan": f"Give {audience} a short plan for learning {name}.",
    }[kind]


def topic_answer(kind: str, guide: tuple[str, str, str, str, str], audience: str) -> str:
    name, definition, example, caution, related = guide
    return {
        "explain": f"The term {name} refers to {definition}. For {audience}, connect the definition to a visible input, action, and result.",
        "example": f"The term {name} refers to {definition}. A concrete example is {example}. The example shows what the concept changes in practice.",
        "compare": f"The term {name} refers to {definition}. {related}",
        "caution": f"A common mistake is assuming that {caution}. Avoid it by stating the expected result, testing a small case, and recording the evidence.",
        "plan": f"Start with this definition: {definition}. Next, study {example}. Then build a small example, test its limits, and explain the result in your own words.",
    }[kind]


def build_topic_texts() -> list[str]:
    return [
        f"Instruction: {topic_prompt(kind, guide[0], audience)}\n"
        f"Response: {topic_answer(kind, guide, audience)}"
        for guide, audience, kind in product(TOPIC_GUIDES, AUDIENCES, TASK_KINDS)
    ]


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def build_arithmetic_texts() -> list[str]:
    records: list[str] = []
    operations = ("addition", "subtraction", "multiplication", "division")
    for index in range(800):
        operation = operations[index % len(operations)]
        left = 12 + (index * 37) % 987
        right = 2 + (index * 19) % 97
        if operation == "addition":
            expression, answer, explanation = f"{left} + {right}", left + right, "Add the two quantities."
        elif operation == "subtraction":
            expression, answer, explanation = f"{left + right} - {right}", left, "Subtract the second quantity from the first."
        elif operation == "multiplication":
            expression, answer, explanation = f"{left} * {right}", left * right, "Multiply the number of groups by the size of each group."
        else:
            expression, answer, explanation = f"{left * right} / {right}", left, "Divide the total into equal groups."
        records.append(
            f"Instruction: Calculate {expression} and explain the operation briefly.\n"
            f"Response: {explanation} The result is {answer}."
        )

    percentages = (5, 10, 12, 15, 20, 25, 30, 40)
    for index in range(600):
        percentage = percentages[index % len(percentages)]
        base = 20 + (index * 13) % 980
        result = base * percentage / 100
        records.append(
            f"Instruction: What is {percentage} percent of {base}? Show the calculation.\n"
            f"Response: Multiply {base} by {percentage / 100:g}. The result is {format_number(float(result))}."
        )

    for index in range(600):
        first = 10 + index
        second = 20 + (index * 3) % 700
        third = 30 + (index * 7) % 900
        average = (first + second + third) / 3
        records.append(
            f"Instruction: Find the average of {first}, {second}, and {third}.\n"
            f"Response: Their sum is {first + second + third}. Divide by 3 to get {format_number(float(average))}."
        )

    for index in range(600):
        hours = 2 + index % 11
        rate = 15 + (index * 7) % 86
        total = hours * rate
        records.append(
            f"Instruction: A process handles {rate} items per hour for {hours} hours. How many items does it handle?\n"
            f"Response: Multiply the rate by the time: {rate} * {hours} = {total} items."
        )
    return records


def build_code_texts() -> list[str]:
    records: list[str] = []
    for index in range(600):
        start = index + 1
        step = 1 + index % 4
        length = 3 + (index * 7) % 20
        stop = start + step * length
        values = list(range(start, stop, step))
        records.append(
            "Instruction: What does this Python code print? Explain briefly.\n"
            f"Code:\nvalues = list(range({start}, {stop}, {step}))\nprint(sum(values))\n"
            f"Response: The range contains {values}. Their sum is {sum(values)}, so the code prints {sum(values)}."
        )

    operations = ("length", "first character", "last character", "uppercase")
    words = ("cache", "network", "database", "testing", "planning", "security", "program", "function")
    for index in range(400):
        value = f"{words[index % len(words)]}{index:03d}"
        operation = operations[index % len(operations)]
        if operation == "length":
            expression, result = "len(text)", str(len(value))
        elif operation == "first character":
            expression, result = "text[0]", value[0]
        elif operation == "last character":
            expression, result = "text[-1]", value[-1]
        else:
            expression, result = "text.upper()", value.upper()
        records.append(
            "Instruction: Determine the output of the Python expression.\n"
            f"Code:\ntext = \"{value}\"\nprint({expression})\n"
            f"Response: The {operation} result is {result}, so the code prints {result}."
        )

    for index in range(400):
        value = index - 200
        threshold = (index * 13) % 101 - 50
        expected = "above" if value > threshold else "not above"
        records.append(
            "Instruction: What does this Python condition print?\n"
            f"Code:\nvalue = {value}\nthreshold = {threshold}\n"
            "print(\"above\" if value > threshold else \"not above\")\n"
            f"Response: The comparison {value} > {threshold} is {str(value > threshold)}. The code prints {expected}."
        )
    return records


def build_writing_texts(seed: int) -> list[str]:
    actions = ("review", "update", "confirm", "prepare", "check", "publish", "archive", "compare", "schedule", "send")
    objects = ("project plan", "test report", "budget draft", "meeting notes", "release checklist", "support summary", "design proposal", "risk register", "training outline", "status update")
    deadlines = ("by noon", "before Tuesday", "this afternoon", "before the meeting", "by the end of the week", "when you have verified the figures")
    tones = ("formal", "friendly", "concise", "direct", "reassuring", "neutral")
    combinations = list(product(actions, objects, deadlines, tones))
    random.Random(seed + 1).shuffle(combinations)
    records: list[str] = []
    for action, item, deadline, tone in combinations[:600]:
        source = f"Could you {action} the {item} {deadline}?"
        response = {
            "formal": f"Please {action} the {item} {deadline}, and let me know when it is complete.",
            "friendly": f"When you have a moment, could you {action} the {item} {deadline}? Thanks!",
            "concise": f"Please {action} the {item} {deadline}.",
            "direct": f"{action.title()} the {item} {deadline} and confirm completion.",
            "reassuring": f"Please {action} the {item} {deadline}; a careful first pass is enough.",
            "neutral": f"Please {action} the {item} {deadline} and share the result.",
        }[tone]
        records.append(
            f"Instruction: Rewrite the sentence in a {tone} tone without changing its meaning.\n"
            f"Sentence: {source}\nResponse: {response}"
        )
    return records


def build_pretraining_records(target_count: int = 5000, seed: int = 42) -> list[dict[str, str]]:
    texts = build_topic_texts() + build_arithmetic_texts() + build_code_texts() + build_writing_texts(seed)
    unique_texts = list(dict.fromkeys(texts))
    random.Random(seed).shuffle(unique_texts)
    if target_count <= 0 or target_count > len(unique_texts):
        raise ValueError(f"target_count must be between 1 and {len(unique_texts)}.")
    return [{"text": text} for text in unique_texts[:target_count]]


def write_pretraining_corpus(path: str | Path, target_count: int = 5000, seed: int = 42) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = build_pretraining_records(target_count=target_count, seed=seed)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"[info] Generated {len(records)} English pretraining records: {path}")


if __name__ == "__main__":
    write_pretraining_corpus("data/english_training_corpus.jsonl")
