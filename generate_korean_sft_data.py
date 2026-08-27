"""Create a small, self-authored Korean conversation SFT seed dataset.

This dataset is supplemental material for future fine-tuning or evaluation.
It is not the pretraining/post-training corpus used by Qwen3-0.6B and must
not be presented as the source of the downloaded Qwen checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT_PATH = Path("data/korean_conversation_sft_seed.jsonl")
SYSTEM_PROMPT = (
    "너는 친절하고 정확한 한국어 도우미다. 사용자의 의도를 먼저 파악하고, "
    "확실하지 않은 내용은 추측하지 않으며, 필요한 설명과 실행 가능한 다음 단계를 간결하게 제공한다."
)


CONCEPTS = [
    ("인공지능", "데이터에서 패턴을 찾아 예측이나 생성을 수행하는 기술", "사진 속 물체를 구분하거나 질문에 답하는 서비스"),
    ("기계 학습", "예시 데이터의 오차가 줄어들도록 모델의 가중치를 조정하는 방법", "스팸 메일과 정상 메일을 학습해 새 메일을 분류하는 기능"),
    ("언어 모델", "앞에 나온 토큰을 바탕으로 다음 토큰의 확률을 계산하는 모델", "입력한 질문에 이어질 답변을 한 토큰씩 생성하는 챗봇"),
    ("데이터베이스", "정보를 일정한 구조로 저장하고 필요할 때 빠르게 조회하는 시스템", "쇼핑몰에서 회원과 주문 내역을 관리하는 저장소"),
    ("API", "서로 다른 프로그램이 정해진 형식으로 기능과 데이터를 주고받는 접점", "날씨 앱이 서버에 오늘의 예보를 요청하는 통로"),
    ("캐시", "자주 쓰는 결과를 가까운 곳에 잠시 저장해 다음 요청을 빠르게 처리하는 방식", "웹 브라우저가 이미지 파일을 보관해 재방문 속도를 높이는 기능"),
    ("암호화", "허가된 사람만 읽을 수 있도록 데이터를 변환하는 기술", "메신저가 전송 중인 대화 내용을 보호하는 기능"),
    ("버전 관리", "파일의 변경 이력과 여러 사람의 작업을 체계적으로 관리하는 방법", "Git으로 코드 수정 내역을 기록하고 이전 상태를 확인하는 작업"),
    ("단위 테스트", "함수나 작은 기능이 예상대로 동작하는지 자동으로 확인하는 검사", "두 수를 더하는 함수가 올바른 결과를 내는지 확인하는 코드"),
    ("클라우드", "인터넷을 통해 서버·저장소·소프트웨어 자원을 필요할 때 사용하는 방식", "직접 서버를 구매하지 않고 온라인 서버를 빌려 웹사이트를 운영하는 것"),
    ("백업", "원본이 손상되거나 사라질 때 복구할 수 있도록 별도 사본을 보관하는 것", "중요한 문서를 외장 저장장치와 온라인 저장소에 함께 복사하는 일"),
    ("확률", "가능한 여러 결과 가운데 특정 결과가 일어날 가능성을 수치로 표현한 것", "동전을 던졌을 때 앞면이 나올 가능성을 2분의 1로 나타내는 것"),
]


PLANS = [
    ("시험 공부", "시험까지 일주일 남았고 하루 두 시간만 쓸 수 있어", ["시험 범위를 과목별로 나누기", "취약한 단원부터 50분씩 복습하기", "매일 마지막 20분에 문제로 확인하기"]),
    ("발표 준비", "사흘 뒤 10분 발표인데 아직 자료가 없어", ["핵심 메시지를 한 문장으로 정하기", "도입·근거·결론 세 부분으로 슬라이드 구성하기", "시간을 재며 두 번 이상 소리 내어 연습하기"]),
    ("이사 준비", "주말 하루 동안 짐을 정리해야 해", ["버릴 것·기부할 것·가져갈 것을 구분하기", "방별로 상자와 라벨 준비하기", "당장 필요한 물건은 별도 가방에 넣기"]),
    ("운동 시작", "운동을 거의 안 했고 무릎에 부담을 주고 싶지 않아", ["통증 여부를 먼저 확인하기", "걷기와 가벼운 근력 운동을 짧게 시작하기", "주당 횟수와 시간을 조금씩 늘리기"]),
    ("파이썬 공부", "코딩 경험이 없고 평일에는 30분만 가능해", ["변수·조건문·반복문을 작은 예제로 익히기", "매일 한 문제를 직접 실행해 보기", "주말에 한 주의 코드를 다시 작성해 보기"]),
    ("집중 시간 확보", "알림과 메신저 때문에 업무가 계속 끊겨", ["가장 중요한 작업 하나를 먼저 정하기", "25~50분 동안 알림을 끄기", "집중 시간이 끝난 뒤 메시지를 한꺼번에 확인하기"]),
    ("여행 짐 꾸리기", "2박 3일 일정이고 비가 올 수 있어", ["날씨와 일정에 맞춰 옷 수를 정하기", "우산과 방수 가능한 겉옷 챙기기", "신분증·충전기·약을 마지막에 다시 확인하기"]),
    ("회의 정리", "회의가 길고 결정 사항이 자주 누락돼", ["회의 전에 결정할 질문을 공유하기", "논의 중 결정·담당자·기한을 바로 기록하기", "종료 후 짧은 회의록을 참석자에게 확인받기"]),
]


SINGLE_TURN = [
    ("casual", "안녕! 오늘 기분이 좀 가라앉아 있어.", "안녕하세요. 그런 날도 있어요. 지금 가장 마음에 걸리는 일이 무엇인지 이야기해도 좋고, 잠시 다른 가벼운 주제로 쉬어 가도 괜찮습니다."),
    ("casual", "비 오는 날 집에서 뭐 하면 좋을까?", "따뜻한 음료를 준비하고 미뤄 둔 영화나 책을 보는 건 어때요? 몸을 움직이고 싶다면 10분 정도 스트레칭한 뒤 방 한 곳만 가볍게 정리해도 기분 전환이 됩니다."),
    ("casual", "오늘 저녁 메뉴를 못 고르겠어.", "원하는 느낌부터 고르면 쉬워요. 든든한 음식이면 덮밥이나 찌개, 가벼운 음식이면 국수나 샐러드가 무난합니다. 집에 있는 재료를 알려 주면 그 안에서 골라 드릴게요."),
    ("clarification", "그거 다시 해 줘.", "어떤 내용을 다시 해야 하는지 현재 대화만으로는 알기 어렵습니다. 다시 작성할 문장이나 작업 내용을 붙여 주시면 바로 도와드릴게요."),
    ("uncertainty", "내일 주식이 무조건 오를 종목을 알려 줘.", "내일 반드시 오를 종목은 누구도 확실하게 알 수 없습니다. 투자 판단 전에는 손실 가능성, 투자 기간, 기업 공시와 재무 상태를 함께 확인해야 합니다."),
    ("privacy", "여기에 주민등록번호를 적어도 돼?", "적지 않는 것이 안전합니다. 주민등록번호·비밀번호·인증번호 같은 민감한 정보는 가리고, 문제 해결에 필요한 비식별 정보만 공유해 주세요."),
    ("health", "머리가 갑자기 너무 아프고 말이 잘 안 나와.", "갑작스러운 심한 두통과 말하기 어려움은 응급 신호일 수 있습니다. 지금 바로 119에 연락하거나 주변 사람에게 응급실 이동을 요청하세요. 혼자 운전하지 마세요."),
    ("writing", "'확인 부탁드립니다'를 조금 더 부드럽게 바꿔 줘.", "시간 되실 때 확인해 주시면 감사하겠습니다."),
    ("writing", "회의에 늦는다고 팀에 보낼 짧은 메시지를 써 줘.", "죄송합니다. 이동이 지연되어 회의에 약 10분 늦을 것 같습니다. 도착하는 대로 바로 참여하겠습니다."),
    ("writing", "면접 일정에 참석한다고 답장하는 메일을 써 줘.", "안녕하세요. 안내해 주신 면접 일정을 확인했습니다. 해당 시간에 참석하겠습니다. 기회를 주셔서 감사합니다."),
    ("troubleshooting", "컴퓨터가 갑자기 인터넷에 연결되지 않아.", "먼저 다른 기기도 같은 와이파이에 연결되지 않는지 확인하세요. 모두 안 되면 공유기 전원을 10초 정도 껐다 켜고, 한 기기만 안 되면 와이파이를 껐다 켠 뒤 네트워크를 다시 연결해 보세요."),
    ("troubleshooting", "파이썬에서 ModuleNotFoundError가 나와.", "오류에 나온 모듈이 현재 실행 중인 Python 환경에 설치되어 있는지 확인하세요. `python -m pip show 모듈명`으로 확인하고, 실행 Python과 설치에 사용한 Python이 같은지도 점검해야 합니다."),
    ("troubleshooting", "휴대폰 저장 공간이 부족하다고 나와.", "설정의 저장 공간 메뉴에서 큰 파일과 사용하지 않는 앱부터 확인하세요. 사진은 백업 여부를 확인한 뒤 정리하고, 메신저의 오래된 첨부파일과 다운로드 폴더도 살펴보는 것이 좋습니다."),
    ("comparison", "백업이랑 동기화는 같은 거야?", "다릅니다. 동기화는 여러 기기의 현재 상태를 맞추는 기능이라 한쪽에서 삭제하면 다른 쪽에서도 지워질 수 있습니다. 백업은 문제가 생겼을 때 과거 상태를 복구하기 위한 별도 사본입니다."),
    ("math", "12만 원에서 15% 할인하면 얼마야?", "할인액은 120,000원 × 0.15 = 18,000원입니다. 따라서 결제 금액은 120,000원 − 18,000원 = 102,000원입니다."),
    ("summary", "핵심만 말해 줘: 작업을 시작하기 전에 목표와 완료 기준을 정하면 불필요한 수정이 줄고 결과를 검토하기도 쉬워진다.", "작업 전에 목표와 완료 기준을 정하면 재작업을 줄이고 검토가 쉬워집니다."),
]


MULTI_TURN = [
    (
        "planning",
        [
            ("user", "할 일이 너무 많아서 뭘 먼저 해야 할지 모르겠어."),
            ("assistant", "오늘 반드시 끝내야 하는 일과 미뤄도 되는 일을 먼저 나눠 보세요. 반드시 끝낼 일 중 영향이 가장 큰 하나를 첫 작업으로 잡으면 됩니다."),
            ("user", "급한 일이 세 개나 있어."),
            ("assistant", "각 일의 마감 시각과 끝내는 데 필요한 시간을 적어 주세요. 마감이 가장 빠르면서 다른 사람의 작업을 막고 있는 일부터 처리하고, 나머지는 예상 완료 시각을 관계자에게 알려 두는 것이 좋습니다."),
        ],
    ),
    (
        "learning",
        [
            ("user", "영어 단어를 외워도 금방 잊어버려."),
            ("assistant", "한 번에 오래 보기보다 간격을 두고 여러 번 떠올리는 방식이 효과적입니다. 단어를 보고 뜻을 읽는 것보다 뜻을 가리고 직접 회상해 보세요."),
            ("user", "하루에 몇 개가 적당할까?"),
            ("assistant", "처음에는 새 단어 10개 정도로 시작해 보세요. 새 단어보다 이전 단어 복습에 더 많은 시간을 쓰고, 일주일 뒤 기억률을 보고 수를 조절하면 됩니다."),
        ],
    ),
    (
        "debugging",
        [
            ("user", "코드는 실행되는데 결과가 예상과 달라."),
            ("assistant", "입력값, 예상 결과, 실제 결과를 가장 작은 예제로 줄여 보세요. 그런 다음 중간 변수 값을 출력하거나 디버거로 따라가면 처음 달라지는 지점을 찾을 수 있습니다."),
            ("user", "전체 코드가 너무 길어."),
            ("assistant", "문제가 의심되는 함수 하나만 떼어 내고 고정된 입력을 넣어 보세요. 외부 파일이나 네트워크 호출은 임시 값으로 바꾸면 원인을 훨씬 좁히기 쉽습니다."),
        ],
    ),
    (
        "conversation",
        [
            ("user", "요즘 잠들기 전에 계속 휴대폰을 보게 돼."),
            ("assistant", "의지만으로 참기보다 환경을 바꾸는 편이 쉽습니다. 잠들기 30분 전에 충전기를 침대에서 먼 곳에 두고, 대신 짧게 읽을 책이나 음악을 준비해 보세요."),
            ("user", "알람 때문에 휴대폰은 방에 있어야 해."),
            ("assistant", "그렇다면 침대에서 손이 닿지 않는 책상에 두고 방해 금지 모드를 예약해 보세요. 알람 소리는 미리 확인하고 화면은 흑백 모드로 바꾸면 유혹을 조금 줄일 수 있습니다."),
        ],
    ),
]


def make_record(record_id: str, category: str, turns: list[tuple[str, str]]) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": role, "content": content} for role, content in turns)
    return {
        "id": record_id,
        "category": category,
        "source": "self_authored_synthetic",
        "messages": messages,
    }


def build_records() -> list[dict]:
    records: list[dict] = []

    for index, (category, user, assistant) in enumerate(SINGLE_TURN, start=1):
        records.append(
            make_record(
                f"ko-single-{index:04d}",
                category,
                [("user", user), ("assistant", assistant)],
            )
        )

    concept_prompts = [
        "{name}이 무엇인지 초보자도 이해할 수 있게 설명해 줘.",
        "{name}을 한 문장으로 정의하고 실제 예시도 하나 들어 줘.",
        "{name}을 처음 배우는 사람이 기억해야 할 핵심은 뭐야?",
    ]
    for concept_index, (name, definition, example) in enumerate(CONCEPTS, start=1):
        answers = [
            f"{name}은 {definition}입니다. 예를 들어 {example}이 있습니다.",
            f"{name}은 {definition}이며, 실제 사례로는 {example}을 들 수 있습니다.",
            f"핵심은 {name}이 {definition}이라는 점입니다. 처음에는 {example}처럼 익숙한 사례와 연결해서 이해하면 좋습니다.",
        ]
        for prompt_index, (prompt, answer) in enumerate(zip(concept_prompts, answers), start=1):
            records.append(
                make_record(
                    f"ko-concept-{concept_index:03d}-{prompt_index}",
                    "explanation",
                    [("user", prompt.format(name=name)), ("assistant", answer)],
                )
            )

    for plan_index, (goal, constraint, steps) in enumerate(PLANS, start=1):
        numbered = " ".join(f"{index}. {step}." for index, step in enumerate(steps, start=1))
        records.append(
            make_record(
                f"ko-plan-{plan_index:03d}-1",
                "planning",
                [
                    ("user", f"{goal} 계획을 세워 줘. {constraint}."),
                    ("assistant", f"다음 순서로 진행해 보세요. {numbered} 처음부터 완벽하게 하려 하지 말고 첫 단계가 끝난 뒤 계획을 조정하세요."),
                ],
            )
        )
        records.append(
            make_record(
                f"ko-plan-{plan_index:03d}-2",
                "planning",
                [
                    ("user", f"{constraint}. 그래도 {goal}을 시작할 수 있을까?"),
                    ("assistant", f"가능합니다. 우선 {steps[0]}부터 시작하세요. 다음으로 {steps[1]}, 마지막으로 {steps[2]} 순서가 현실적입니다."),
                ],
            )
        )

    for index, (category, turns) in enumerate(MULTI_TURN, start=1):
        records.append(make_record(f"ko-multiturn-{index:04d}", category, turns))

    return records


def validate_records(records: list[dict]) -> None:
    ids: set[str] = set()
    for record in records:
        if record["id"] in ids:
            raise ValueError(f"Duplicate id: {record['id']}")
        ids.add(record["id"])

        messages = record["messages"]
        if len(messages) < 3 or messages[0]["role"] != "system":
            raise ValueError(f"Invalid message sequence: {record['id']}")
        if messages[-1]["role"] != "assistant":
            raise ValueError(f"Conversation must end with assistant: {record['id']}")
        for message in messages:
            if message["role"] not in {"system", "user", "assistant"}:
                raise ValueError(f"Invalid role in {record['id']}")
            if not message["content"].strip():
                raise ValueError(f"Empty message in {record['id']}")


def main() -> None:
    records = build_records()
    validate_records(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
