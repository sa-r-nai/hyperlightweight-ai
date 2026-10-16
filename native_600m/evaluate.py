"""Transparent conversation sanity checks for native checkpoints and vocabulary."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any

from .chat import generate_reply
from .checkpoint import load_model
from .config import DEFAULT_CONFIG, load_config, sha256_file, write_json
from .tokenizer import DEFAULT_TOKENIZER, SPECIAL_TOKENS, NativeTokenizer


@dataclass(frozen=True)
class EvalCase:
    name: str
    prompts: tuple[str, ...]
    required_groups: tuple[tuple[str, ...], ...] = ()
    min_hangul_ratio: float = 0.75


EVAL_CASES = (
    EvalCase("greeting", ("안녕! 처음 만났으니 짧게 인사해 줘.",), (("안녕", "반가"),)),
    EvalCase("memory_correction", (
        "내 이름은 민수야. 기억해 줘.",
        "아까 이름을 잘못 말했어. 내 이름은 지연이야.",
        "내 이름이 뭐라고 했지? 이름만 답해 줘.",
    ), (("지연",),)),
    EvalCase("preference_memory", (
        "나는 매운 음식을 못 먹어. 점심 추천할 때 참고해 줘.",
        "점심으로 매운 떡볶이와 순한 우동 중 하나를 골라 줘. 이유도 한 문장으로 말해 줘.",
    ), (("우동",), ("맵", "매운", "순한", "안 매"))),
    EvalCase("three_items", ("산책 갈 때 챙길 물건을 정확히 세 개만, 각 줄에 '- '로 시작해서 적어 줘.",)),
    EvalCase("short_constraint", ("친구에게 고맙다는 말을 30자 이내 한 문장으로 써 줘.",), (("고마", "감사"),)),
    EvalCase("ambiguity", ("그거 좀 바꿔 줘.",), (("무엇", "어떤", "어떻게", "뭘", "구체", "알려", "내용"),)),
    EvalCase("polite_rewrite", ("'자료 빨리 보내'를 공손한 부탁 한 문장으로 바꿔 줘.",), (("자료",), ("주세요", "주실", "부탁", "주시겠"))),
    EvalCase("small_plan", ("오늘 저녁 공부할 시간이 30분이야. 복습과 문제 풀이를 포함한 간단한 계획을 세워 줘.",), (("복습",), ("문제",))),
    EvalCase("uncertainty", ("내가 지금 손에 들고 있는 물건이 뭔지 알려 줘. 사진이나 설명은 없어.",), (("알 수 없", "알기 어렵", "모르", "확인할 수 없", "설명", "사진"),)),
    EvalCase("arithmetic", ("사과 12개가 있고 7개를 더 받았어. 모두 몇 개야? 숫자만 답해 줘.",), (("19",),), min_hangul_ratio=0.0),
    EvalCase("followup", (
        "초록색 컵은 책상 위에, 파란색 컵은 서랍 안에 있어.",
        "파란색 컵은 어디에 있어?",
    ), (("서랍",),)),
    EvalCase("summarize", ("다음을 한 문장으로 요약해 줘: 오늘 비가 와서 소풍을 취소했다. 대신 교실에서 보드게임을 했다.",), (("비",), ("보드게임",))),
    EvalCase("supportive_reply", ("오늘 하던 일을 망쳐서 속상해. 짧게 위로하고, 지금 할 수 있는 작은 행동 하나만 제안해 줘.",), (("속상", "괜찮", "힘들", "마음", "쉬", "실수"),)),
)

HEURISTIC_LIMITS = (
    "자동 검사는 한글 비율, 반복, 일부 표현과 형식만 확인합니다. "
    "동의어 때문에 정답을 실패로 판정하거나 핵심 단어만 포함한 부적절한 답을 통과시킬 수 있습니다. "
    "이 결과는 정상적인 대화 품질을 입증하지 않으며 별도 사람 평가가 필요합니다."
)
HUMAN_RUBRIC = {
    "scale": "각 항목을 1~5점으로 평가하고 실패 예시를 기록합니다.",
    "criteria": [
        "한국어 자연스러움과 문맥에 맞는 말투",
        "사용자 의도 및 길이·형식 지시 준수",
        "여러 차례 대화에서 정보 기억과 정정 반영",
        "모호한 요청에 대한 적절한 확인 질문",
        "알 수 없는 사실을 꾸며내지 않는 태도",
        "반복이나 대화 단절 없이 도움이 되는 응답",
    ],
    "procedure": "학습에 사용하지 않은 새 대화 30개 이상을 평가하고 모델 버전·생성 조건·원문과 실패 사례를 보관합니다.",
}
GENERATION_DEFAULTS = {"temperature": 0.8, "top_k": 40, "top_p": 0.9, "repetition_penalty": 1.1}


def response_sanity(response: str, min_hangul_ratio: float = 0.75) -> dict[str, Any]:
    text = response.strip()
    letters = [character for character in text if character.isalpha()]
    hangul = sum("\uac00" <= character <= "\ud7a3" or "\u1100" <= character <= "\u11ff"
                 or "\u3130" <= character <= "\u318f" for character in letters)
    ratio = hangul / len(letters) if letters else 0.0
    reasons = []
    if not text:
        reasons.append("empty_response")
    if "\ufffd" in text:
        reasons.append("replacement_character")
    if any(marker in text for marker in SPECIAL_TOKENS):
        reasons.append("leaked_control_tokens")
    if len(text) > 2000:
        reasons.append("excessive_length")
    if ratio < min_hangul_ratio:
        reasons.append("insufficient_hangul_ratio")
    words = re.findall(r"[\w]+", text, flags=re.UNICODE)
    triples = list(zip(words, words[1:], words[2:]))
    if len(triples) >= 6 and max(Counter(triples).values()) >= 4:
        reasons.append("repeated_word_sequence")
    compact = re.sub(r"\s", "", text)
    if re.search(r"(.)\1{11,}", compact):
        reasons.append("repeated_character")
    if len(compact) >= 60:
        grams = [compact[index:index + 12] for index in range(len(compact) - 11)]
        if max(Counter(grams).values(), default=0) >= 5:
            reasons.append("repeated_character_sequence")
    return {"passed": not reasons, "hangul_ratio": round(ratio, 4), "reasons": reasons}


def case_signals(case: EvalCase, response: str) -> dict[str, bool]:
    """Expose simple lexical/format checks without treating them as semantic grades."""
    checks = {f"required_group_{index + 1}": any(word in response for word in group)
              for index, group in enumerate(case.required_groups)}
    if case.name == "memory_correction":
        checks["old_name_absent"] = "민수" not in response
    elif case.name == "three_items":
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        checks["exactly_three_bullet_lines"] = len(lines) == 3 and all(line.startswith("- ") for line in lines)
    elif case.name == "short_constraint":
        checks["at_most_30_characters"] = len(response.strip()) <= 30
        checks["single_line"] = "\n" not in response.strip()
    elif case.name == "arithmetic":
        checks["number_only"] = response.strip() == "19"
    return checks


def seeded_generate_reply(model, tokenizer, messages: list[dict[str, str]], *,
                          max_new_tokens: int, seed: int) -> str:
    """Preserve surrounding CPU/CUDA RNG state while seeding each generated turn."""
    import torch

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return generate_reply(model, tokenizer, messages, max_new_tokens=max_new_tokens, **GENERATION_DEFAULTS)


def run_evaluation(model, tokenizer, config: dict, max_new_tokens: int = 192) -> list[dict]:
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    results = []
    for case_index, case in enumerate(EVAL_CASES):
        messages = [{"role": "system", "content": config["system_prompt"]}]
        turns = []
        for turn_index, prompt in enumerate(case.prompts):
            messages.append({"role": "user", "content": prompt})
            seed = (config["seed"] + 1000 * case_index + turn_index) % (2**63)
            response = seeded_generate_reply(model, tokenizer, messages, max_new_tokens=max_new_tokens, seed=seed)
            ratio_threshold = case.min_hangul_ratio if turn_index == len(case.prompts) - 1 else 0.75
            turns.append({"prompt": prompt, "response": response, "seed": seed,
                          "sanity": response_sanity(response, ratio_threshold)})
            # Empty or marker-bearing output cannot become a valid native
            # protocol message. Record the failure and stop this case cleanly.
            if not response.strip() or any(marker in response for marker in SPECIAL_TOKENS):
                break
            messages.append({"role": "assistant", "content": response})
        completed = len(turns) == len(case.prompts)
        signals = case_signals(case, turns[-1]["response"] if completed else "")
        signals["all_prompts_answered"] = completed
        passed = all(turn["sanity"]["passed"] for turn in turns) and all(signals.values())
        results.append({"name": case.name, "passed": passed, "checks": signals, "turns": turns})
        print(f"[{'PASS' if passed else 'FAIL'}] {case.name}")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate native Korean model conversations with visible sanity heuristics.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts_native_600m/sft/best.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path(DEFAULT_TOKENIZER))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--report", type=Path, default=Path("artifacts_native_600m/evaluation.json"))
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--min-pass-rate", type=float, default=0.8)
    parser.add_argument("--allow-fail", action="store_true", help="Save failures without a nonzero exit status.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not 0 < args.min_pass_rate <= 1:
        raise ValueError("--min-pass-rate must be greater than zero and at most one.")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")
    config = load_config(args.config)
    tokenizer = NativeTokenizer.load(args.tokenizer)
    model, metadata = load_model(args.checkpoint, device=args.device)
    if metadata.get("tokenizer_sha256") != tokenizer.fingerprint:
        raise ValueError("Checkpoint tokenizer fingerprint differs from the supplied tokenizer file.")
    from .model import ModelConfig, parameter_count

    checkpoint_config = ModelConfig(**metadata["config"])
    results = run_evaluation(model, tokenizer, config, args.max_new_tokens)
    passed = sum(result["passed"] for result in results)
    pass_rate = passed / len(results)
    sane = all(turn["sanity"]["passed"] for result in results for turn in result["turns"])
    gate_passed = sane and pass_rate >= args.min_pass_rate
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": config.get("name", "NativeKoreanLM-600M"),
        "pretrained_base": None,
        "parameter_count": parameter_count(checkpoint_config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint / "manifest.json" if args.checkpoint.is_dir() else args.checkpoint),
        "checkpoint_stage": metadata.get("stage"), "tokenizer_sha256": tokenizer.fingerprint,
        "model_config": metadata["config"], "device": args.device,
        "generation": {**GENERATION_DEFAULTS, "max_new_tokens": args.max_new_tokens,
                       "seed": config["seed"], "seed_schedule": "(seed + 1000 * case_index + turn_index) modulo 2**63",
                       "reproducibility_note": "Seeded sampling does not guarantee identical results across hardware or library versions."},
        "passed_cases": passed, "total_cases": len(results), "pass_rate": pass_rate,
        "min_pass_rate": args.min_pass_rate, "all_turns_sanity_passed": sane,
        "heuristic_gate_passed": gate_passed, "conversation_quality_verified": False,
        "heuristic_limits": HEURISTIC_LIMITS, "human_rubric": HUMAN_RUBRIC, "cases": results,
    }
    write_json(args.report, report)
    print(f"{passed}/{len(results)} cases passed ({pass_rate:.1%}). Report: {args.report}")
    print(HEURISTIC_LIMITS)
    if not gate_passed and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
