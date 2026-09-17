"""Generate deterministic, self-authored English chat conversations."""

from __future__ import annotations

import random
from itertools import product

from generate_native_corpus import (
    AUDIENCES,
    TOPIC_GUIDES,
    build_pretraining_records,
    topic_answer,
    topic_prompt,
)


SYSTEM_PROMPT = (
    "You are a helpful English assistant. Answer accurately, clearly, and "
    "concisely. State uncertainty instead of inventing facts."
)

# subject, user request, general guidance, concrete first step, caution
EVERYDAY_SCENARIOS = [
    ("an overloaded task list", "I have too many tasks and cannot decide where to start.", "Sort the tasks by deadline, impact, and whether they block someone else.", "Choose one small blocking task and work on it for twenty focused minutes.", "Do not spend the whole session reorganizing the list instead of doing the work."),
    ("a study schedule", "I need a study schedule that I can actually follow.", "Use short sessions, active recall, and planned review instead of one long session.", "Choose one topic and schedule a twenty-five minute recall session today.", "Leave spare time so one difficult topic does not break the entire schedule."),
    ("a difficult email", "I need to write a polite email about a delay.", "State the delay early, give the new expected time, and keep the explanation brief.", "Write one sentence with the new delivery time before adding any background.", "Avoid promises you cannot verify."),
    ("a program crash", "My program crashes and I do not know how to debug it.", "Capture the exact error, input, environment, and shortest reproduction steps.", "Run the smallest failing case and inspect the first unexpected intermediate value.", "Change one variable at a time so the evidence remains useful."),
    ("a meeting", "I have to prepare for a meeting but the agenda is unclear.", "Define the decision, required evidence, participants, and desired next action.", "Write the one decision that must be made during the meeting.", "Do not fill the agenda with updates that can be read asynchronously."),
    ("organizing files", "My project files are disorganized and hard to find.", "Group files by purpose, use consistent names, and separate source files from generated artifacts.", "Create one clear top-level folder for source material and move only verified files into it.", "Keep a backup before a large reorganization."),
    ("learning Python", "I want to learn Python but I keep jumping between tutorials.", "Choose one course, practice each concept in a tiny program, and review errors.", "Build a program that reads input, transforms it, and prints a result.", "Avoid collecting resources faster than you complete exercises."),
    ("a missed deadline", "I missed a deadline and need to tell my team.", "Acknowledge the miss, describe the current state, give a realistic new estimate, and name any risk.", "Send a short update now with the next verifiable milestone.", "Do not hide uncertainty behind an overly precise promise."),
    ("a difficult conversation", "I need to discuss a problem with a teammate without starting an argument.", "Describe the observable behavior, its impact, and the change you are requesting.", "Write one neutral sentence about what happened without guessing the other person's motive.", "Discuss the behavior rather than attacking the person."),
    ("a presentation", "I am nervous about an upcoming presentation.", "Practice the opening, the three main points, and the final request instead of memorizing every word.", "Say the first minute aloud twice while timing yourself.", "Do not add new material immediately before presenting."),
    ("a job interview", "I have a job interview and do not know how to prepare.", "Prepare concise examples that show the situation, your action, and the measurable result.", "Choose one project and write three facts about your personal contribution.", "Do not claim work that you cannot explain in detail."),
    ("a simple budget", "I want a simple monthly budget.", "Separate fixed costs, essential variable costs, savings, and optional spending.", "List the last month of recurring expenses before setting targets.", "Use actual transactions rather than optimistic guesses."),
    ("a healthy routine", "I want a healthier daily routine but large plans never last.", "Change one repeatable behavior at a time and make the cue and duration specific.", "Pick a ten-minute activity and attach it to an existing daily cue.", "Seek qualified medical advice for symptoms or individual health restrictions."),
    ("packing for travel", "I always forget something when I pack for a trip.", "Build the list around documents, medication, clothing, communication, and destination needs.", "Place required documents and medication in one visible location first.", "Check current carrier and destination rules instead of relying on memory."),
    ("meal planning", "I need an easy meal plan for a busy week.", "Choose a few repeatable meals with shared ingredients and one flexible backup option.", "Select two proteins, two vegetables, and one base ingredient for the first three days.", "Account for allergies and food-safety requirements."),
    ("a reading habit", "I want to read more but I lose focus quickly.", "Use a small daily target, remove distractions, and note one idea after each session.", "Read five pages at the same time today and write one sentence about them.", "Increase the target only after the small habit is consistent."),
    ("creative block", "I am stuck and cannot begin a creative project.", "Reduce the first draft to an intentionally rough, time-limited experiment.", "Make three imperfect versions in fifteen minutes without editing them.", "Do not judge an early sketch by final-product standards."),
    ("a customer complaint", "A customer is angry and I need to reply calmly.", "Acknowledge the specific problem, explain the next action, and provide a realistic update time.", "Restate the issue in one neutral sentence before proposing a remedy.", "Do not promise a refund or deadline unless you have authority to do so."),
    ("project scope", "My project keeps growing and I cannot finish it.", "Define the smallest complete outcome and move optional features into a later list.", "Write one sentence describing what the first release must accomplish.", "Treat every added requirement as a tradeoff in time, cost, or quality."),
    ("a failed test", "A test fails only sometimes.", "Record timing, order, shared state, randomness, and external dependencies.", "Run the single test repeatedly with a fixed seed and isolated temporary state.", "Do not hide the failure with retries before identifying the cause."),
    ("a network problem", "One computer cannot reach a service that works for everyone else.", "Compare connectivity, address, DNS, route, firewall, and application settings on that computer.", "Check whether the hostname resolves to the expected address.", "Change one layer at a time and record each result."),
    ("asking for clarification", "I received a vague request and do not know what result is expected.", "Ask about the desired outcome, audience, deadline, format, and acceptance criteria.", "Reply with your current interpretation and ask the requester to confirm it.", "Do not begin expensive work based only on an ambiguous phrase."),
    ("giving feedback", "I need to give useful feedback on a draft.", "Identify the goal, mention what already works, and suggest a small number of specific changes.", "Choose the single revision that would most improve the reader's understanding.", "Separate personal preference from an actual requirement."),
    ("making a decision", "I keep comparing options and cannot make a decision.", "Choose a few important criteria, identify reversible choices, and set a decision time.", "Write the top three criteria and score each option with the same evidence.", "Do not treat minor differences as if they are equally important."),
]

CONSTRAINTS = (
    "I only have fifteen minutes right now.",
    "I am a beginner.",
    "I keep getting interrupted.",
    "I need a low-cost approach.",
    "I want the first step to be very small.",
    "I have already tried once and it did not work.",
    "I need to explain the plan to another person.",
    "I feel overwhelmed by long instructions.",
    "I would like a checklist.",
    "I need a result by the end of the week.",
    "I am not sure what information is missing.",
    "I want a practical answer without jargon.",
)

FOLLOW_UPS = (
    "What should I do first?",
    "Can you make that more specific?",
    "What common mistake should I avoid?",
    "How can I tell whether it worked?",
    "Can you turn that into a short checklist?",
    "What should I do if the first step fails?",
)


def make_record(record_id: str, category: str, messages: list[dict[str, str]]) -> dict:
    return {
        "id": record_id,
        "category": category,
        "source": "self_authored_synthetic",
        "license": "self-authored",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
    }


def split_instruction(text: str) -> tuple[str, str]:
    marker = "\nResponse: "
    if not text.startswith("Instruction: ") or marker not in text:
        raise ValueError("Generated pretraining record does not use instruction format.")
    instruction, response = text.split(marker, maxsplit=1)
    return instruction.removeprefix("Instruction: ").strip(), response.strip()


def build_single_turn_pool(seed: int) -> list[tuple[str, list[dict[str, str]]]]:
    pool = []
    for record in build_pretraining_records(target_count=5000, seed=seed):
        user, assistant = split_instruction(record["text"])
        pool.append(
            (
                "instruction",
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
            )
        )
    return pool


def build_topic_followup_pool() -> list[tuple[str, list[dict[str, str]]]]:
    pool = []
    followup_kinds = ("example", "compare", "caution", "plan")
    for guide, audience, followup_kind in product(TOPIC_GUIDES, AUDIENCES, followup_kinds):
        pool.append(
            (
                "topic_followup",
                [
                    {"role": "user", "content": topic_prompt("explain", guide[0], audience)},
                    {"role": "assistant", "content": topic_answer("explain", guide, audience)},
                    {"role": "user", "content": topic_prompt(followup_kind, guide[0], audience)},
                    {"role": "assistant", "content": topic_answer(followup_kind, guide, audience)},
                ],
            )
        )
    return pool


def followup_answer(
    followup: str,
    first_step: str,
    caution: str,
) -> str:
    return {
        "What should I do first?": first_step,
        "Can you make that more specific?": f"Use this concrete first action: {first_step}",
        "What common mistake should I avoid?": caution,
        "How can I tell whether it worked?": "Define one visible result before you begin, then compare the actual result with that expectation and record what changed.",
        "Can you turn that into a short checklist?": f"1. State the desired result. 2. {first_step} 3. Check the result. 4. Record the next action.",
        "What should I do if the first step fails?": f"Record what happened, reduce the step further, and check the first assumption. Also remember: {caution}",
    }[followup]


def build_everyday_pool() -> list[tuple[str, list[dict[str, str]]]]:
    pool = []
    for scenario, constraint, followup in product(EVERYDAY_SCENARIOS, CONSTRAINTS, FOLLOW_UPS):
        subject, request, guidance, first_step, caution = scenario
        pool.append(
            (
                "everyday_conversation",
                [
                    {"role": "user", "content": f"{request} {constraint}"},
                    {"role": "assistant", "content": f"{guidance} {first_step}"},
                    {"role": "user", "content": followup},
                    {"role": "assistant", "content": followup_answer(followup, first_step, caution)},
                ],
            )
        )
    return pool


def build_generated_chat_records(target_count: int, seed: int = 42) -> list[dict]:
    """Return a stratified mix of instruction and multi-turn conversations."""

    if target_count <= 0:
        return []
    rng = random.Random(seed)
    single_turn = build_single_turn_pool(seed)
    topic_followups = build_topic_followup_pool()
    everyday = build_everyday_pool()
    for pool in (single_turn, topic_followups, everyday):
        rng.shuffle(pool)

    single_count = min(len(single_turn), round(target_count * 0.65))
    topic_count = min(len(topic_followups), round(target_count * 0.10))
    everyday_count = min(len(everyday), target_count - single_count - topic_count)
    selected = (
        single_turn[:single_count]
        + topic_followups[:topic_count]
        + everyday[:everyday_count]
    )
    remaining = target_count - len(selected)
    if remaining:
        unused = (
            single_turn[single_count:]
            + topic_followups[topic_count:]
            + everyday[everyday_count:]
        )
        rng.shuffle(unused)
        selected.extend(unused[:remaining])
    if len(selected) != target_count:
        raise ValueError("Requested more chat records than the generator can provide.")
    rng.shuffle(selected)
    return [
        make_record(f"native-chat-{index:05d}", category, messages)
        for index, (category, messages) in enumerate(selected, start=1)
    ]


__all__ = ["build_generated_chat_records"]
