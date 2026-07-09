"""Generate local instruction-style and chat-style corpora for training.

This creates a self-authored synthetic dataset. It is useful when no
externally collected corpus is available, but it should not be described
as real-world data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


EN_OUTPUT_PATH = Path("data/generated_corpus.jsonl")
KO_OUTPUT_PATH = Path("data/generated_korean_chat_corpus.jsonl")
MIX_OUTPUT_PATH = Path("data/generated_bilingual_chat_corpus.jsonl")
SEED = 7


TOPICS = [
    ("artificial intelligence", "systems that learn patterns from data and use them to make predictions or generate text"),
    ("databases", "tools that store structured information and answer queries reliably"),
    ("web security", "practices that reduce risk by validating input and protecting sessions"),
    ("time management", "methods for choosing the most important task and protecting focus time"),
    ("climate", "long-term weather patterns influenced by oceans, land, and human activity"),
    ("customer support", "clear communication that solves the user's problem with calm and accurate guidance"),
    ("testing", "checking software behavior so that regressions are caught early"),
    ("machine learning", "optimizing model parameters so that loss decreases on useful examples"),
    ("writing", "turning rough ideas into clear sentences with structure and purpose"),
    ("product design", "shaping interfaces so users can complete goals with less confusion"),
    ("networks", "systems that move data between machines using shared protocols"),
    ("studying", "reviewing information over time so understanding becomes durable"),
]

AUDIENCES = [
    "a middle school student",
    "a first-year university student",
    "a busy office worker",
    "a beginner programmer",
    "a new team member",
    "someone who feels overwhelmed",
]

TONES = [
    "calm",
    "practical",
    "friendly",
    "concise",
    "teacherly",
]

TASKS = [
    "Explain the topic simply.",
    "Give a short definition and one example.",
    "Compare it with a related idea.",
    "List three common mistakes and how to avoid them.",
    "Provide a step-by-step plan for getting started.",
]

REWRITE_STYLES = [
    "more formal",
    "more casual",
    "shorter",
    "clearer for a beginner",
    "more persuasive",
]

SHORT_PARAGRAPHS = [
    "A small model can still be useful when latency and cost matter more than raw capability.",
    "A checklist is valuable because it removes the need to remember routine steps under pressure.",
    "Good feedback is specific, kind, and tied to behavior that can actually change.",
    "Learning becomes more durable when practice is spaced over time instead of crammed into one session.",
    "A clear interface reduces hesitation by making the next action obvious.",
    "Reliable systems usually come from many small safeguards rather than one giant clever trick.",
]

CLASS_LABELS = [
    ("The customer says the app crashes every time they try to upload a file.", "bug report"),
    ("The writer thanks the reader and confirms the meeting for Tuesday morning.", "confirmation"),
    ("The note asks whether a cheaper plan is available for students.", "pricing question"),
    ("The message contains angry language and demands a refund today.", "urgent complaint"),
    ("The paragraph describes steps for boiling pasta and making a quick sauce.", "how-to text"),
]

KO_TOPICS = [
    ("인공지능", "데이터의 패턴을 학습해 예측하거나 문장을 생성하는 기술"),
    ("데이터베이스", "구조화된 정보를 저장하고 빠르게 조회하는 도구"),
    ("웹 보안", "입력 검증과 권한 관리로 위험을 줄이는 실천 방법"),
    ("시간 관리", "중요한 일을 먼저 끝내도록 돕는 선택과 집중의 방법"),
    ("기계 학습", "손실을 줄이도록 모델의 가중치를 조정하는 과정"),
    ("글쓰기", "생각을 분명한 문장과 구조로 정리하는 작업"),
    ("테스트", "변경 후에도 프로그램이 의도대로 동작하는지 확인하는 과정"),
    ("네트워크", "장치 사이에서 규칙에 따라 데이터를 주고받는 체계"),
]

KO_AUDIENCES = [
    "초등 고학년 학생",
    "중학생",
    "처음 배우는 대학생",
    "바쁜 직장인",
    "처음 개발을 시작한 사람",
    "새로 합류한 팀원",
]

KO_TASKS = [
    "쉽게 설명해 줘.",
    "짧은 정의와 예시 하나를 알려 줘.",
    "비슷한 개념과 차이를 설명해 줘.",
    "자주 하는 실수 세 가지와 예방법을 알려 줘.",
    "처음 시작하는 사람을 위한 순서를 알려 줘.",
]

KO_TONES = [
    "차분한",
    "친절한",
    "실용적인",
    "격려하는",
    "간결한",
]

KO_DIALOGUE_PROMPTS = [
    ("오늘 해야 할 일이 너무 많은데 어디서부터 시작해야 할까?", "가장 불안을 줄여 주는 일 하나만 먼저 고르세요. 그 일을 이십 분만 집중해서 끝낸 뒤에 다음 일을 보세요."),
    ("버그 수정이 뭐야? 쉽게 설명해 줘.", "버그 수정은 프로그램의 실수를 찾아 고쳐서 예상한 대로 동작하게 만드는 일입니다."),
    ("체크포인트를 왜 저장해?", "체크포인트를 저장하면 학습이 멈춰도 이어서 시작할 수 있고, 가장 성능이 좋은 지점을 다시 불러올 수 있습니다."),
    ("요약과 바꿔쓰기의 차이가 뭐야?", "요약은 핵심만 짧게 남기는 것이고, 바꿔쓰기는 같은 뜻을 다른 표현으로 다시 말하는 것입니다."),
    ("이번 주 공부를 더 잘하려면 어떻게 해야 해?", "짧게 나눠 반복하고, 읽기만 하지 말고 스스로 떠올려 보는 연습을 하세요."),
    ("긴장을 좀 줄이고 싶어.", "한 번에 모든 문제를 해결하려고 하지 말고, 지금 바로 할 수 있는 작은 행동 하나에만 집중해 보세요."),
    ("파이썬 함수가 뭐야?", "함수는 여러 번 사용할 동작을 이름으로 묶어 둔 것입니다. 같은 코드를 반복해서 쓰지 않게 도와줍니다."),
    ("회의 전에 뭘 준비하면 좋아?", "목표, 결정이 필요한 항목, 필요한 자료 세 가지만 먼저 정리하면 회의가 훨씬 선명해집니다."),
]

KO_REWRITE_SAMPLES = [
    "보고서를 오늘 안에 보내 주세요. 내일 전에 검토해야 합니다.",
    "지금은 통화가 어렵지만 저녁까지 메모를 보내 드릴 수 있습니다.",
    "기능은 동작하지만 설정 화면이 초보자에게는 아직 복잡합니다.",
    "업데이트 감사합니다. 계획에 동의하고 점심 이후에 바로 시작하겠습니다.",
]

KO_REWRITE_STYLES = [
    "더 공손하게",
    "더 짧게",
    "더 쉽게",
    "더 자신감 있게",
]

KO_SHORT_PARAGRAPHS = [
    "작은 모델은 성능이 아주 높지 않아도 속도와 비용이 중요할 때 충분히 가치가 있다.",
    "체크리스트는 익숙한 작업도 빠뜨리지 않게 만들어서 긴장한 상황에서 특히 도움이 된다.",
    "좋은 피드백은 사람을 공격하지 않고 바꿀 수 있는 행동을 구체적으로 짚어 준다.",
    "공부는 한 번에 오래 하기보다 여러 날에 나눠 반복할 때 더 오래 기억된다.",
    "명확한 화면은 사용자가 다음에 무엇을 해야 할지 바로 알게 해 준다.",
]

KO_LABELS = [
    ("앱에서 파일을 올릴 때마다 바로 종료된다고 사용자가 말한다.", "버그 제보"),
    ("화요일 오전 회의를 확인하고 참석하겠다고 답한다.", "일정 확인"),
    ("학생 할인 요금제가 있는지 묻는다.", "가격 문의"),
    ("환불을 당장 처리해 달라며 강한 불만을 표현한다.", "긴급 불만"),
    ("파스타를 삶고 소스를 만드는 순서를 설명한다.", "방법 설명"),
]


def add_record(records: list[dict[str, str]], text: str) -> None:
    cleaned = text.strip()
    if cleaned:
        records.append({"text": cleaned})


def build_explanations(records: list[dict[str, str]]) -> None:
    for topic, meaning in TOPICS:
        for audience in AUDIENCES:
            for task in TASKS:
                response = (
                    f"{topic.title()} means {meaning}. "
                    f"For {audience}, the key idea is to start with one simple use case, "
                    f"notice the main tradeoff, and practice with a concrete example. "
                    f"A helpful first example is to describe where this topic appears in daily work."
                )
                text = (
                    f"Instruction: {task} The topic is {topic}. The audience is {audience}. Use a calm tone.\n"
                    f"Response: {response}"
                )
                add_record(records, text)


def build_rewrites(records: list[dict[str, str]]) -> None:
    samples = [
        "Please send the report soon because we need to review it before tomorrow.",
        "I cannot join the call today, but I can send written notes by the evening.",
        "The feature works, although the settings page is still confusing for new users.",
        "Thanks for the update. I agree with the plan and will start after lunch.",
    ]
    for sample in samples:
        for style in REWRITE_STYLES:
            rewritten = {
                "more formal": f"I would appreciate it if you could send the report promptly, as it must be reviewed before tomorrow.",
                "more casual": f"Please send the report soon. We need to look at it before tomorrow.",
                "shorter": f"Please send the report today so we can review it before tomorrow.",
                "clearer for a beginner": f"Please send the report today. Our team needs time to read it before tomorrow.",
                "more persuasive": f"Please send the report today so the team has enough time to review it well before tomorrow.",
            }[style]
            text = (
                f"Instruction: Rewrite the following sentence to be {style}.\n"
                f"Sentence: {sample}\n"
                f"Response: {rewritten}"
            )
            add_record(records, text)


def build_summaries(records: list[dict[str, str]]) -> None:
    for paragraph in SHORT_PARAGRAPHS:
        summary = paragraph.split(" because ")[0].split(" when ")[0]
        text = (
            "Instruction: Summarize the paragraph in one sentence.\n"
            f"Paragraph: {paragraph}\n"
            f"Response: {summary}."
        )
        add_record(records, text)


def build_classification(records: list[dict[str, str]]) -> None:
    for text_body, label in CLASS_LABELS:
        text = (
            "Instruction: Read the text and choose the best label.\n"
            f"Text: {text_body}\n"
            f"Response: {label}"
        )
        add_record(records, text)


def build_dialogues(records: list[dict[str, str]]) -> None:
    dialogue_prompts = [
        ("I feel behind on everything and I do not know where to start.", "Pick one task that will reduce the most stress today, work on it for twenty focused minutes, and ignore the rest until that timer ends."),
        ("Can you explain what a bug fix is in simple words?", "A bug fix is a change that removes a mistake in the program so the software behaves the way people expect."),
        ("Why do people save checkpoints during training?", "A checkpoint stores the model state so training can continue later without losing progress."),
        ("What is the difference between a summary and a paraphrase?", "A summary keeps only the main ideas, while a paraphrase restates the same ideas in different words with similar detail."),
        ("How can I study better this week?", "Review a little every day, quiz yourself from memory, and keep the sessions short enough that you can repeat them tomorrow."),
    ]
    for user_text, assistant_text in dialogue_prompts:
        for tone in TONES:
            response = f"{assistant_text} The tone should feel {tone}, steady, and easy to follow."
            text = f"User: {user_text}\nAssistant: {response}"
            add_record(records, text)


def build_reasoning(records: list[dict[str, str]]) -> None:
    scenarios = [
        ("A model is overfitting the training set", "Reduce complexity, add more varied examples, and measure performance on validation data."),
        ("Users abandon a signup form halfway through", "Shorten the form, clarify the benefits, and remove fields that are not necessary at the start."),
        ("A server becomes slow only at peak hours", "Check load, queueing, and expensive operations that scale poorly under heavy traffic."),
        ("A student keeps rereading notes but remembers very little", "Replace passive review with recall practice and spaced repetition."),
    ]
    for problem, answer in scenarios:
        text = (
            "Instruction: Read the problem and suggest a likely cause with a practical next step.\n"
            f"Problem: {problem}\n"
            f"Response: {answer}"
        )
        add_record(records, text)


def build_longform(records: list[dict[str, str]]) -> None:
    for topic, meaning in TOPICS:
        paragraph = (
            f"{topic.title()} matters because it helps people make better decisions with limited time and attention. "
            f"In practice, it is useful to begin with a small repeatable workflow instead of an ambitious plan. "
            f"The core idea is {meaning}. "
            f"A beginner often improves faster by practicing one concrete example, reviewing mistakes, and keeping notes in plain language."
        )
        add_record(records, paragraph)


def build_korean_explanations(records: list[dict[str, str]]) -> None:
    for topic, meaning in KO_TOPICS:
        for audience in KO_AUDIENCES:
            for task in KO_TASKS:
                response = (
                    f"{topic}은 {meaning}입니다. "
                    f"{audience}에게는 먼저 아주 작은 예시 하나를 잡고, 왜 필요한지와 어떤 장단점이 있는지 함께 보는 방식이 가장 쉽습니다. "
                    f"처음에는 실제 생활이나 학교, 업무에서 어디에 쓰이는지 연결해서 이해하면 오래 기억됩니다."
                )
                text = (
                    f"지시: {task} 주제는 {topic}이고 대상은 {audience}이야. 차분한 한국어로 답해.\n"
                    f"응답: {response}"
                )
                add_record(records, text)


def build_korean_rewrites(records: list[dict[str, str]]) -> None:
    rewrite_map = {
        "더 공손하게": "보고서를 오늘 안에 보내 주시면 감사하겠습니다. 내일 전에 검토가 필요합니다.",
        "더 짧게": "보고서를 오늘 보내 주세요. 내일 전에 검토해야 합니다.",
        "더 쉽게": "보고서를 오늘 보내 주세요. 우리 팀이 내일 전에 읽어야 합니다.",
        "더 자신감 있게": "보고서를 오늘 보내 주세요. 그러면 내일 전에 충분히 검토할 수 있습니다.",
    }
    for sample in KO_REWRITE_SAMPLES:
        for style in KO_REWRITE_STYLES:
            text = (
                f"지시: 다음 문장을 {style} 바꿔 써 줘.\n"
                f"문장: {sample}\n"
                f"응답: {rewrite_map[style]}"
            )
            add_record(records, text)


def build_korean_summaries(records: list[dict[str, str]]) -> None:
    for paragraph in KO_SHORT_PARAGRAPHS:
        summary = paragraph.split("그래서")[0].split("특히")[0].strip()
        text = (
            "지시: 다음 문단을 한 문장으로 요약해 줘.\n"
            f"문단: {paragraph}\n"
            f"응답: {summary}"
        )
        add_record(records, text)


def build_korean_classification(records: list[dict[str, str]]) -> None:
    for text_body, label in KO_LABELS:
        text = (
            "지시: 아래 문장을 읽고 가장 알맞은 분류를 골라 줘.\n"
            f"문장: {text_body}\n"
            f"응답: {label}"
        )
        add_record(records, text)


def build_korean_chat(records: list[dict[str, str]]) -> None:
    for user_text, assistant_text in KO_DIALOGUE_PROMPTS:
        for tone in KO_TONES:
            response = (
                f"{assistant_text} "
                f"말투는 {tone} 느낌으로, 부담을 주지 않으면서 바로 실천할 수 있게 설명합니다."
            )
            text = (
                "시스템: 너는 친절하고 정확한 한국어 도우미다. 모르면 모른다고 말하고, "
                "답은 간결하지만 충분히 도움이 되게 한다.\n"
                f"사용자: {user_text}\n"
                f"도우미: {response}"
            )
            add_record(records, text)


def build_korean_multi_turn_chat(records: list[dict[str, str]]) -> None:
    conversations = [
        (
            "시스템: 너는 친절하고 정확한 한국어 도우미다.\n"
            "사용자: 오늘 하루 계획을 못 세우겠어.\n"
            "도우미: 가장 중요한 일 하나만 먼저 고르고, 그 일을 끝낼 시간을 짧게 잡아 보세요.\n"
            "사용자: 그 다음에는 어떻게 해?\n"
            "도우미: 첫 작업이 끝나면 잠깐 쉬고, 두 번째로 중요한 일을 같은 방식으로 이어가면 됩니다."
        ),
        (
            "시스템: 너는 친절하고 정확한 한국어 도우미다.\n"
            "사용자: 코드를 읽을 때 너무 막막해.\n"
            "도우미: 함수 이름과 입력, 출력부터 먼저 보고 큰 흐름을 잡아 보세요.\n"
            "사용자: 그래도 이해가 안 되면?\n"
            "도우미: 작은 예시를 직접 넣어 보면서 값이 어떻게 바뀌는지 따라가면 훨씬 쉬워집니다."
        ),
        (
            "시스템: 너는 친절하고 정확한 한국어 도우미다.\n"
            "사용자: 발표 전에 너무 긴장돼.\n"
            "도우미: 완벽하게 하려는 마음을 잠시 내려놓고, 첫 문장만 또렷하게 말하는 데 집중해 보세요.\n"
            "사용자: 첫 문장 다음에 막히면?\n"
            "도우미: 미리 준비한 핵심 키워드 세 개를 천천히 떠올리면 흐름을 다시 잡을 수 있습니다."
        ),
    ]
    for conversation in conversations:
        add_record(records, conversation)


def build_korean_longform(records: list[dict[str, str]]) -> None:
    for topic, meaning in KO_TOPICS:
        paragraph = (
            f"{topic}은 사람들이 더 나은 판단을 하도록 돕는 중요한 주제입니다. "
            f"핵심은 {meaning}이라는 점입니다. "
            f"처음 배울 때는 너무 큰 목표를 잡기보다 작은 예시를 반복하면서 감을 익히는 편이 좋습니다. "
            f"한 번 설명을 읽고 끝내지 말고, 스스로 다시 말해 보거나 간단한 사례에 적용해 보면 이해가 훨씬 깊어집니다."
        )
        add_record(records, paragraph)


def write_records(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    random.seed(SEED)
    english_records: list[dict[str, str]] = []
    korean_records: list[dict[str, str]] = []

    build_explanations(english_records)
    build_rewrites(english_records)
    build_summaries(english_records)
    build_classification(english_records)
    build_dialogues(english_records)
    build_reasoning(english_records)
    build_longform(english_records)

    build_korean_explanations(korean_records)
    build_korean_rewrites(korean_records)
    build_korean_summaries(korean_records)
    build_korean_classification(korean_records)
    build_korean_chat(korean_records)
    build_korean_multi_turn_chat(korean_records)
    build_korean_longform(korean_records)

    random.shuffle(english_records)
    random.shuffle(korean_records)

    mixed_records = english_records + korean_records
    random.shuffle(mixed_records)

    write_records(EN_OUTPUT_PATH, english_records)
    write_records(KO_OUTPUT_PATH, korean_records)
    write_records(MIX_OUTPUT_PATH, mixed_records)

    print(f"wrote {len(english_records)} records to {EN_OUTPUT_PATH}")
    print(f"wrote {len(korean_records)} records to {KO_OUTPUT_PATH}")
    print(f"wrote {len(mixed_records)} records to {MIX_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
