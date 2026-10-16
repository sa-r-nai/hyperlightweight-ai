# 한국어 대화 모델 600M 학습 안내

> 이전 Qwen 추가 학습 경로의 문서입니다. 현재 기본 경로는
> [자체 모델 안내](native_600m.md)이며 `run_lightning_600m.py`도 자체 모델을 실행합니다.
> 이 문서의 과거 파이프라인 명령은 `run_qwen_600m_legacy.py`로 변경되었습니다.

## 모델 및 기본 설정

이미 대화 학습을 마친 Qwen3-0.6B를 한국어로 추가 학습합니다.
많은 지식을 새로 주입하기보다 요청 이해, 대화 맥락, 자연스러운 응답을 목표로 합니다.

| 항목 | 설정 |
|---|---|
| 모델 | `Qwen/Qwen3-0.6B` |
| 고정 revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| 고유 파라미터 | 596,049,920개, 약 596M |
| 블록 / hidden size | 28 / 1,024 |
| 토크나이저 | 기본 모델의 Unicode 토크나이저 |
| 학습·추론 문맥 | 1,024토큰 |
| LoRA | rank 32, alpha 64, dropout 0.05 |
| 학습 대상 | attention·MLP projection의 LoRA |
| 손실 | 도우미 응답과 종료 토큰만 사용 |
| 학습률 / 에포크 | 1e-4 / 2 |
| 배치 / 누적 | 1 / 16 |
| 검증·체크포인트 | 200 optimizer step 및 마지막 step |
| 정밀도 | 하드웨어 BF16 지원 시 BF16, 그 외 FP16 |

입력 임베딩과 출력 헤드의 공유 가중치를 한 번만 세는 파라미터 수입니다.
저장 텐서의 합계에는 공유 임베딩이 중복 포함될 수 있습니다. 코드에서 실제 고유
파라미터를 확인하고 기록합니다. LoRA 병합 후에도 같은 크기의 모델입니다.

한국어 전용은 학습 데이터와 기본 응답 언어를 의미합니다. 기존 다국어
토크나이저를 유지하며 한국어 유지율은 생성 평가에서 확인합니다.

## Studio 환경

Lightning AI에서 Python 3.10~3.12, 단일 CUDA GPU의 Studio를 준비하고 저장소를
복제하거나 업로드하십시오. 코드와 결과는 영구 저장 공간에 두십시오.
여러 GPU가 있다면 `CUDA_VISIBLE_DEVICES=0`으로 하나만 노출하십시오.

```bash
cd /teamspace/studios/this_studio/hyperlightweight-ai
python -m pip install -r requirements-korean-600m.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python run_lightning_600m.py
```

`cd`는 실제 저장소 경로로 변경하십시오. 학습 코드는 `/teamspace/studios` 및
CUDA를 먼저 검사하고 실패하면 다운로드와 학습을 시작하지 않습니다.
16GB 이상의 GPU 메모리를 시작점으로 권장합니다. 메모리가 부족하면 설정 파일의
`max_seq_length`를 768 또는 512로 낮추고 새로운 결과 디렉터리를 사용하십시오.
기본 배치는 이미 1입니다. 누적 횟수만 줄이면 microbatch 활성값 메모리는 줄지 않습니다.
학습 시간은 준비 후 토큰 수와 GPU 처리 속도로 산정하십시오.

## 재개 및 설정 변경

```bash
python run_lightning_600m.py --reuse-prepared
```

동일한 설정·데이터·정밀도의 완성 `checkpoint-*`를 자동 선택합니다.
최신 체크포인트가 불완전하면 이전 완성본을 탐색합니다. FP16에서는 `scaler.pt`도
필요합니다. 과거 실험의 `best.pt`는 읽지 않습니다. 데이터나 설정 또는 BF16/FP16이
바뀌면 새 결과 디렉터리를 사용하십시오.

```bash
python run_lightning_600m.py \
  --config configs/korean_600m.json \
  --artifacts-dir artifacts_korean_600m_v2
```

완료 후 재실행 시 병합 결과는 어댑터·설정·결과 파일 해시가 일치할 때 재사용합니다.
원본 대화 평가의 실패는 학습을 중단하지 않습니다. 추가 학습 모델이 평가 기준에
미달하면 보고서를 저장하고 종료 코드 1을 반환합니다. 이때 어댑터와 병합 모델은
이미 저장되어 있습니다.

## Studio에서 짧은 실행 검증

아래 명령은 **Studio GPU에서 실제 2 step 학습**을 수행합니다.
대화 품질을 평가하는 학습량은 아니며 결과를 별도 저장합니다.

```bash
python -m korean_600m.prepare \
  --train-records 128 --validation-records 32 \
  --output-dir artifacts_korean_600m_smoke/data
python -m korean_600m.train \
  --data-dir artifacts_korean_600m_smoke/data \
  --output-dir artifacts_korean_600m_smoke/training \
  --adapter-dir artifacts_korean_600m_smoke/adapter \
  --max-steps 2
```

코드만 확인하려면 `python run_lightning_600m.py --plan`과 README의 단위 검사를 사용하십시오.
이 두 명령은 학습을 실행하지 않습니다.

## 데이터

기본 소스는 `lemon-mint/smol-koreantalk`, revision은
`04eb12c1f999578d74dd87720bc853cedb7fa156`입니다. 배포자는 소형 모델용
Smol-SmolTalk의 한국어 번역본이며 Apache-2.0이라고 명시합니다.
`messages[].content`만 사용하고 병렬 영어 필드 `content_en`은 제외합니다.

- 고정 seed의 버퍼 셔플과 스트리밍으로 데이터를 준비합니다.
- 한국어 비율, 발화 순서, 길이, 제어 토큰을 검사합니다.
- 원본 시스템의 작업 지침은 첫 사용자 요청에 보존합니다.
- 첫 사용자 요청을 정규화하여 중복을 제거하고 학습·검증을 분리합니다.
- 출처·라이선스·revision·기록 수·SHA-256을 명세에 저장합니다.

문자 비율 검사는 의미 품질 평가를 대신하지 않습니다. 다른 표현으로 된 유사
문제까지 완전히 제거하지는 않습니다. 각 도우미 발화는 별도 예시가 됩니다.
문맥이 길면 오래된 완성 대화 쌍부터 제거합니다. 최신 요청과 응답이 여전히
너무 길면 해당 예시를 제외하고 개수를 기록합니다. 응답 자체는 잘라 학습하지 않습니다.

자체 데이터는 다음 UTF-8 JSONL 형식을 사용하십시오.

```json
{"messages":[{"role":"user","content":"내 이름은 지연이야."},{"role":"assistant","content":"반가워요, 지연님."},{"role":"user","content":"내 이름을 기억해?"},{"role":"assistant","content":"지연님이라고 말씀하셨어요."}]}
```

자체 데이터도 설정된 기록 수를 충족해야 합니다. 작은 데이터셋은 기본 설정을
복사한 뒤 `data.train_records`, `data.validation_records`를 줄여 사용하십시오.
소스 사용 허가 또는 라이선스를 함께 기록합니다.

```bash
python run_lightning_600m.py \
  --config configs/my_korean_600m.json \
  --local-jsonl /teamspace/studios/this_studio/my_conversations.jsonl \
  --local-license '자체 작성, 학습 및 배포 허용' \
  --artifacts-dir artifacts_korean_600m_custom
```

## 평가 및 병합

학습 전후에 같은 사례와 시드가 고정된 생성 조건을 사용합니다.
13개 사례는 인사, 이름 정정, 선호 기억, 항목 작성, 짧은 감사, 확인 질문,
공손한 문장, 계획, 불확실성, 산수, 위치 기억, 요약, 위로를 포함합니다.
모델의 실제 답변을 다음 발화의 문맥으로 사용합니다.

```bash
python -m korean_600m.evaluate --base-model --device cuda \
  --report artifacts_korean_600m/evaluation_base.json --allow-fail
python -m korean_600m.evaluate --device cuda \
  --report artifacts_korean_600m/evaluation_tuned.json
```

모든 발화의 언어·반복 검사와 사례 통과율 80% 이상이 기본 기준입니다.
키워드 검사의 오탐과 누락이 있으므로 보고서 원문을 확인하십시오. 별도의 새 대화
30개 이상을 원본 모델과 비교하여 자연스러움, 지시 준수, 맥락 기억, 확인 질문,
정직성, 반복을 사람이 평가하도록 기준을 포함했습니다.
현재 저장소에는 추가 학습 결과나 대화 품질 실측값이 없습니다.

독립 Safetensors 모델을 수동 저장할 수도 있습니다.

```bash
python -m korean_600m.export --device cuda \
  --adapter artifacts_korean_600m/adapter \
  --output artifacts_korean_600m/merged
```

병합 결과에 모델, 토크나이저, 학습·데이터 출처 명세가 저장됩니다.
배포 시 기본 모델과 데이터의 라이선스 및 출처 고지를 함께 유지하십시오.

## 근거 자료

- [Qwen 공식 소개: 한국어 지원 및 라이선스](https://qwenlm.github.io/blog/qwen3/)
- [고정 모델 설정](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json)
- [고정 대화 템플릿](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json)
- [한국어 데이터](https://huggingface.co/datasets/lemon-mint/smol-koreantalk)
- [Transformers 4.57.1 API](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer)
- [PEFT LoRA](https://huggingface.co/docs/peft/v0.17.0/package_reference/lora)
- [Lightning Studio 영구 저장 환경](https://lightning.ai/docs/overview/ai-studio/)
