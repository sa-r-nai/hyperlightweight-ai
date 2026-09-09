"""Create a small, self-authored seed corpus for NativeByteLM experiments.

This is intentionally a seed generator, not a claim that a few local records
are enough to pretrain a 500M model.  It provides reproducible Korean/English
text and supervised conversations for smoke tests and an initial data audit.
Larger corpora should be added only with a recorded source, license, and
content-quality review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "너는 정확하고 정직한 다국어 도우미다. 질문의 핵심을 먼저 파악하고 "
    "필요한 만큼만 설명한다. 모르는 사실은 모른다고 말하고 확인되지 않은 "
    "출처나 결과를 꾸며 내지 않는다."
)


PASSAGES = [
    (
        "계획 세우기",
        "좋은 계획은 목표, 완료 조건, 순서, 점검 시점을 분리해서 적습니다. "
        "목표만 크고 완료 조건이 없으면 작업이 끝났는지 판단하기 어렵습니다. "
        "처음에는 가장 작은 검증 가능한 결과를 정하고, 결과를 확인한 뒤 다음 단계를 추가하는 편이 안전합니다.",
    ),
    (
        "디버깅",
        "디버깅은 추측을 늘리는 일이 아니라 관찰 가능한 차이를 좁히는 과정입니다. "
        "재현 절차를 고정하고 입력과 기대 결과를 기록한 다음, 가장 작은 예제로 줄여야 합니다. "
        "첫 번째로 기대와 달라지는 중간값을 찾으면 원인 후보를 빠르게 줄일 수 있습니다.",
    ),
    (
        "데이터 품질",
        "학습 데이터는 양만으로 평가할 수 없습니다. 중복, 깨진 인코딩, 서로 모순되는 답, "
        "출처와 라이선스가 불명확한 문장을 분리해야 합니다. 데이터의 출처와 전처리 규칙을 "
        "함께 기록하면 나중에 문제가 발생했을 때 원인을 추적할 수 있습니다.",
    ),
    (
        "소프트웨어 테스트",
        "테스트는 구현 세부사항보다 외부에서 관찰되는 약속을 확인해야 합니다. "
        "작은 단위 테스트는 빠른 피드백을 주고, 통합 테스트는 여러 구성요소의 연결 문제를 찾습니다. "
        "두 종류를 함께 두면 변경 속도와 신뢰성을 동시에 유지할 수 있습니다.",
    ),
    (
        "네트워크",
        "네트워크 문제를 확인할 때는 물리 연결, 주소 설정, 이름 해석, 전송 경로, "
        "애플리케이션 응답 순서로 범위를 넓히는 것이 좋습니다. 한 번에 여러 설정을 바꾸면 "
        "무엇이 문제를 해결했는지 알 수 없으므로 한 번에 한 가지 가설만 검증해야 합니다.",
    ),
    (
        "학습 방법",
        "새로운 개념은 읽기만 할 때보다 직접 설명하거나 작은 예제를 만들 때 오래 기억됩니다. "
        "짧게 공부한 뒤 빈 종이에 핵심을 재구성하고, 막힌 부분만 다시 확인하는 회상 연습이 효율적입니다.",
    ),
    (
        "한국어 글쓰기",
        "한국어 기술 문서는 결론을 먼저 쓰고, 그 결론을 뒷받침하는 조건과 예외를 뒤에 배치하면 읽기 쉽습니다. "
        "한 문장에 너무 많은 원인을 넣지 말고, 용어를 처음 사용할 때 짧게 정의하는 것이 좋습니다.",
    ),
    (
        "English writing",
        "Clear technical writing states the decision first, then explains the evidence and the limits. "
        "Short paragraphs, explicit assumptions, and reproducible examples make a document easier to review.",
    ),
    (
        "Unicode text",
        "Unicode gives text systems a common representation for many writing systems. UTF-8 represents a text stream "
        "with one to four bytes per code point, so a byte-level tokenizer can preserve unseen characters without a "
        "language-specific vocabulary file.",
    ),
    (
        "모델 학습",
        "언어 모델의 사전 학습은 앞의 토큰으로 다음 토큰을 예측하도록 가중치를 조정하는 과정입니다. "
        "검증 손실은 학습에 직접 사용하지 않은 데이터를 얼마나 잘 예측하는지 확인하는 지표입니다. "
        "검증 손실이 계속 오르면 데이터 순서, 학습률, 과적합 여부를 점검해야 합니다.",
    ),
]


SINGLE_TURN = [
    ("정의", "캐시가 무엇인지 초보자에게 설명해 줘.", "캐시는 자주 쓰는 데이터를 가까운 곳에 임시로 보관해 다시 가져오는 비용을 줄이는 방법입니다. 대신 원본이 바뀌었을 때 오래된 값이 남을 수 있어 만료나 갱신 규칙이 필요합니다."),
    ("정의", "백업과 동기화의 차이를 알려 줘.", "백업은 복구를 위한 별도 사본이고, 동기화는 여러 위치의 현재 상태를 맞추는 기능입니다. 동기화 중 삭제가 함께 전파될 수 있으므로 동기화만으로 백업을 대신할 수는 없습니다."),
    ("수학", "12만 원에서 15퍼센트 할인된 가격은 얼마야?", "할인액은 120,000 × 0.15 = 18,000원입니다. 따라서 결제 금액은 102,000원입니다."),
    ("디버깅", "파이썬에서 ModuleNotFoundError가 나와.", "오류에 나온 모듈이 현재 실행 중인 Python 환경에 설치되어 있는지 확인하세요. `python -m pip show 모듈명`으로 확인하고, 설치에 사용한 Python과 실행하는 Python이 같은지도 점검해야 합니다."),
    ("디버깅", "코드는 실행되는데 결과가 예상과 달라.", "입력값, 기대 결과, 실제 결과를 가장 작은 예제로 줄여 보세요. 중간 변수 값을 기록해 처음으로 기대와 달라지는 지점을 찾으면 원인을 좁히기 쉽습니다."),
    ("작성", "회의에 늦는다고 팀에 보낼 짧은 메시지를 써 줘.", "죄송합니다. 이동이 지연되어 회의에 약 10분 늦을 것 같습니다. 도착하는 대로 바로 참여하겠습니다."),
    ("작성", "확인 부탁드립니다를 더 부드럽게 바꿔 줘.", "시간 되실 때 확인해 주시면 감사하겠습니다."),
    ("계획", "새로운 프로그래밍 언어를 공부하는 순서를 알려 줘.", "기본문법을 짧게 익힌 뒤 작은 입력·출력 프로그램을 만들어 보세요. 다음으로 파일이나 네트워크처럼 실제로 필요한 기능을 하나 붙이고, 마지막에 테스트와 패키징을 연습하면 됩니다."),
    ("안전", "온라인에서 받은 파일을 바로 실행해도 돼?", "출처와 파일 확장자를 먼저 확인하고, 백신 검사와 격리된 환경을 거치는 편이 안전합니다. 실행 전에 필요한 권한을 확인하고, 출처가 불분명한 실행 파일은 열지 않는 것이 좋습니다."),
    ("학습", "공부한 내용을 금방 잊어버려.", "읽은 뒤 책을 덮고 핵심을 직접 설명해 보세요. 다음 날과 며칠 뒤에 다시 떠올리는 간격 반복을 사용하면 단순 재독보다 기억을 확인하기 쉽습니다."),
    ("네트워크", "인터넷이 한 기기에서만 안 돼.", "다른 기기도 같은 문제가 있는지 먼저 확인하세요. 한 기기만 안 되면 네트워크 연결을 다시 맺고 주소 설정과 DNS를 확인한 뒤, 그래도 안 되면 해당 기기의 방화벽과 드라이버를 점검하세요."),
    ("코딩", "함수 이름을 어떻게 정하면 좋아?", "함수가 하는 일을 짧은 동사구로 표현하세요. 입력과 반환값이 이름에 드러나면 좋고, 지나치게 넓은 `process`나 `data` 같은 이름은 구체적인 동작으로 바꾸는 편이 낫습니다."),
    ("요약", "핵심만 요약해 줘: 작은 단위로 검증하면 문제를 빨리 찾고 재작업을 줄일 수 있다.", "작게 나눠 검증하면 문제를 빨리 찾고 재작업을 줄일 수 있습니다."),
    ("English", "Explain a cache in one short paragraph.", "A cache is a temporary copy of data kept closer to the next reader so repeated access is faster. It needs an invalidation or refresh rule because the original data can change."),
    ("English", "How should I start debugging an unexpected result?", "Record the input, expected result, and actual result. Reduce the case to the smallest reproducible example, then inspect the first intermediate value that differs from the expectation."),
]


MULTI_TURN = [
    (
        "계획",
        [
            ("user", "할 일이 너무 많아서 뭘 먼저 해야 할지 모르겠어."),
            ("assistant", "마감 시각과 다른 작업을 막고 있는지를 기준으로 먼저 정렬해 보세요. 가장 영향이 큰 한 가지를 첫 작업으로 정하고, 나머지는 예상 완료 시각을 적어 두면 됩니다."),
            ("user", "급한 일이 세 개나 있어."),
            ("assistant", "각 작업에 필요한 시간을 추정한 뒤 마감이 가장 빠르고 다른 사람을 막는 일부터 처리하세요. 동시에 끝낼 수 없다면 우선순위와 지연 가능성을 관계자에게 먼저 알려야 합니다."),
        ],
    ),
    (
        "디버깅",
        [
            ("user", "전체 코드가 너무 길어서 어디부터 봐야 할지 모르겠어."),
            ("assistant", "오류가 관찰되는 함수 하나만 떼어 내고 고정된 입력을 넣어 보세요. 파일, 네트워크 같은 외부 요인은 임시 값으로 바꾸면 원인을 더 빨리 좁힐 수 있습니다."),
            ("user", "그 함수도 다른 함수가 많이 불러."),
            ("assistant", "호출되는 함수 중 결과에 영향을 주는 것만 작은 가짜 구현으로 바꿔 보세요. 문제가 사라지면 바꾼 의존성 중 하나가 원인이고, 그대로면 현재 함수의 입력과 분기부터 확인하면 됩니다."),
        ],
    ),
    (
        "학습",
        [
            ("user", "새 단어를 외워도 금방 잊어버려."),
            ("assistant", "단어와 뜻을 계속 읽기보다 뜻을 가리고 직접 떠올려 보세요. 짧은 간격을 두고 여러 번 회상하는 방식이 기억을 확인하는 데 도움이 됩니다."),
            ("user", "하루에 몇 개가 적당할까?"),
            ("assistant", "처음에는 새 단어 10개 안팎으로 시작하고, 새 단어보다 이전 단어 복습에 더 많은 시간을 쓰세요. 일주일 뒤 기억률을 보고 수를 조절하면 됩니다."),
        ],
    ),
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
    records: list[dict] = []
    for index, (category, user, assistant) in enumerate(SINGLE_TURN, start=1):
        records.append(
            make_record(
                f"native-single-{index:04d}",
                category,
                [("user", user), ("assistant", assistant)],
            )
        )
    for index, (category, turns) in enumerate(MULTI_TURN, start=1):
        records.append(make_record(f"native-multi-{index:04d}", category, turns))
    return records


def validate_records(records: list[dict]) -> None:
    seen_ids: set[str] = set()
    for record in records:
        record_id = record["id"]
        if record_id in seen_ids:
            raise ValueError(f"중복 ID가 있습니다: {record_id}")
        seen_ids.add(record_id)
        messages = record["messages"]
        if not messages or messages[0]["role"] != "system":
            raise ValueError(f"system 메시지가 없습니다: {record_id}")
        if messages[-1]["role"] != "assistant":
            raise ValueError(f"assistant로 끝나지 않습니다: {record_id}")
        for message in messages:
            if message["role"] not in {"system", "user", "assistant"}:
                raise ValueError(f"지원하지 않는 역할입니다: {record_id}")
            if not message["content"].strip():
                raise ValueError(f"빈 메시지가 있습니다: {record_id}")


def write_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_records()
    validate_records(records)

    sft_path = output_dir / "native_sft_seed.jsonl"
    with sft_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    pretraining_path = output_dir / "native_pretraining_seed.txt"
    sections = [f"# {title}\n\n{text}" for title, text in PASSAGES]
    pretraining_path.write_text(
        "\n\n".join(sections) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"[정보] SFT 시드 {len(records)}건을 생성했습니다: {sft_path}")
    print(f"[정보] 사전 학습 시드 {len(PASSAGES)}개 단락을 생성했습니다: {pretraining_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="자체 작성 학습 데이터 시드 생성기")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    write_outputs(args.output_dir)


if __name__ == "__main__":
    main()
