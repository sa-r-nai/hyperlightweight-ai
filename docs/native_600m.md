# 자체 한국어 300M 모델 학습

## 독립 모델의 범위

`native_600m/model.py`에서 Transformer를 직접 정의합니다. 외부 모델의 가중치,
모델 설정, 토크나이저 어휘, 대화 템플릿을 불러오지 않습니다. 무작위 가중치와
한국어 데이터로 새로 학습한 BPE에서 시작합니다. 범용 라이브러리와 공개 학습
데이터는 사용합니다. 대화 데이터에 다른 모델이 생성한 예시가 포함되어 있어도
메인 모델의 구조와 가중치는 독립적입니다.

기본 구조는 24개 블록, hidden 1,024, query heads 16, KV heads 4, FFN 2,816,
RoPE, RMSNorm, SwiGLU, 공유 입출력 임베딩입니다. 어휘 32,000개일 때 고유
파라미터는 **303,350,784개**입니다. 문맥 길이는 2,048토큰입니다.
생성은 현재 전체 문맥을 다시 계산하며 KV cache 최적화는 적용하지 않았습니다.

## H100, RTX PRO 6000, H200 및 GPU 설정

RTX PRO 6000 Blackwell의 명목 메모리는 96GB, H200은 141GB입니다.
실행 환경에서 할당된 실제 메모리는 학습 시작 로그에서 확인하십시오.
자동 프로필은 H100에서 전체 메모리가 75GiB 이상이거나 RTX PRO 6000·H200에서
80GiB 이상일 때 선택합니다. 작은 MIG 할당에는 기본 프로필을 사용합니다.

| 프로필 | 사전학습 배치 × 누적 | 대화 학습 배치 × 누적 |
|---|---:|---:|
| 기본 | 1 × 64 | 1 × 32 |
| H100 | 8 × 8 | 8 × 4 |
| RTX PRO 6000 | 8 × 8 | 8 × 4 |
| H200 | 8 × 8 | 8 × 4 |

기본 문맥은 사전학습 2,048, 대화 학습 1,024토큰입니다.
확대 배치 설정은 검증할 시작 구성이며 실제 최고 처리 속도나 메모리 사용량의 실측값은 아닙니다.
자동 감지 대신 `--gpu-profile balanced`, `--gpu-profile h100`, `--gpu-profile h200`,
`--gpu-profile rtx-pro-6000`을 지정할 수 있습니다.
기존 학습도 다른 학습 옵션과 유효 배치가 같으면 배치·누적 묶음을 변경해 재개합니다.
예를 들어 배치 1 × 누적 64에서 배치 8 × 누적 8로 바꿀 수 있습니다.
원래 실행 식별자와 검증 창 수를 보존하고 실제 배치를 `execution_options`에 기록합니다.
가중치·옵티마이저·난수·다음 샘플 위치를 복원하지만, 부동소수점 계산 순서가 달라져
비트 단위 동일성은 보장하지 않습니다. 데이터·정밀도·예산·학습률 등 다른 변경은 거부합니다.
Hopper·Blackwell 호환성을 위해 Modal 이미지에서 CUDA 12.8 빌드의 PyTorch를 지정합니다.
전체 메모리가 75GiB 이상인 H100에서는 `auto`·`h100` 프로필에서 활성값 재계산을 끕니다.
RTX PRO 6000의 전체 메모리가 80GiB 이상이면 `auto`·`rtx-pro-6000` 프로필에서
활성값 재계산을 끄고 VRAM에 보관합니다. `balanced`는 활성값 재계산을 유지합니다.
이 설정은 메모리와 연산량의 교환이며 체크포인트 파일의 저장·재개와는 별개입니다.
한 GPU만 지원하므로 여러 GPU가 연결되면 `CUDA_VISIBLE_DEVICES=0`으로 하나만 노출하십시오.
기본 프로필은 24GB 이상 GPU에서 짧은 실행으로 확인하고, 가능하면 40GB 이상을 사용하십시오.

## 실행 및 저장 공간

Python 3.10~3.12의 학습 환경에서 저장소를 열고 실행합니다.
Lightning AI Studio와 다른 GPU 서버, 로컬 CUDA 환경을 사용할 수 있습니다.
특정 플랫폼이나 디렉터리를 검사하지 않으며 모델 학습에는 단일 CUDA GPU가 필요합니다.
데이터 준비와 토크나이저 학습은 CPU 환경에서 실행할 수 있습니다.

```bash
python -m pip install -r requirements-native-600m.txt
python run_native_600m.py --plan
python run_native_600m.py
```

CPU 환경에서 전처리한 뒤 GPU로 전환하면 준비 중 GPU 사용 시간을 줄일 수 있습니다.
CPU 작업자 수는 `--cpu-workers`로 지정하며 기존 `--prepare-workers`도 같은 옵션으로 지원합니다.
명령행 기본값과 Modal 실행기는 24를 사용합니다. Hugging Face 원본 읽기·디코딩도
원본 샤드 수 범위에서 최대 24개 프로세스로 수행하며 문서 필터링·검증·실제 토큰 변환을
작업자로 분산합니다. 읽기 순서·작업자 수는 데이터 명세에 기록합니다. 작업자 수가 달라지면
새 데이터의 공급 순서와 할당량에 먼저 도달해 선택되는 문서 집합도 바뀔 수 있습니다.
이미 완료된 데이터는 검증 후 그대로 재사용합니다.
BPE 어휘 학습에는 24개 Rust 스레드를 사용합니다. 각 CPU 작업자의 내부 라이브러리
스레드는 1개로 제한하여 중첩 실행을 방지합니다. 네트워크·저장 대기와 순차 계산에서는
24코어가 모두 사용되지 않을 수 있습니다. 메모리는 스트리밍과 제한된 작업
버퍼로 관리하므로 할당된 RAM 전체를 사용할 필요는 없습니다.
`--reuse-prepared`와 토큰 변환 전의 데이터 검증에도 같은 설정을 적용하며,
`[verify:...]` 로그로 파일 해시와 기록 검증의 진행 상황을 확인할 수 있습니다.

```bash
python run_native_600m.py --prepare-only --cpu-workers 24
# GPU를 연결한 뒤 실행합니다.
python run_native_600m.py --reuse-prepared
```

데이터 캐시, JSONL, 토큰 파일, 체크포인트는 영구 저장 공간에 저장하도록 구성하십시오.
컨테이너 기반 플랫폼에서는 `--artifacts-dir`와 데이터 캐시 경로를 영구 저장 공간에
연결하고 해당 플랫폼의 실행 진입점과 실행 시간 제한을 별도로 설정하십시오.
모델 FP32 가중치 한 벌은 약 1.21GB이고 AdamW 재개 체크포인트는 그보다 큽니다.
사전학습·대화 학습의 best/last, 임시 원자적 저장본, 데이터와 공개 소스 캐시까지
고려하여 **여유 공간 100GB 이상을 준비하고 실제 사용량을 확인**하십시오.
기본 데이터 규모와 소스 캐시에 따라 더 많은 공간이 필요할 수 있습니다.

## 학습량과 기대 수준

설정 파일은 `configs/native_300m.json`입니다. 기본 사전학습 예산은 60억
입력 토큰이고 이후 대화 학습은 2에포크입니다. 사전학습의 한 optimizer step은
기본값에서 131,072토큰이며 약 45,777 step입니다. 이는 실험 시작 예산이지
자연스러운 한국어 대화를 보장하는 기준은 아닙니다.
사전학습 warmup은 1,000 step, 검증 간격은 500 step입니다.
기존 600M·120억 토큰과 비교한 파라미터 수는 약 50.8%, 파라미터 수와 처리 토큰 수의
곱은 약 25.4%입니다. 전처리·손실 계산·저장 시간 등이 포함되므로 실행 시간의 비율을
의미하지 않으며, H100에서 실제 `tokens/s`, VRAM과 검증 손실로 평가해야 합니다.

구조와 학습 예산이 바뀌므로 기존 600M 체크포인트를 그대로 재개하지 않습니다.
300M 결과는 `artifacts_native_300m/`에서 새로 생성하며 600M 결과와 설정은 보존합니다.
데이터 명세도 전체 설정 해시를 검사하므로 새 결과 경로에서 전처리를 수행합니다.
기존 600M을 재개하려면 다음을 사용하십시오.

```bash
python run_native_600m.py --config configs/native_600m.json --artifacts-dir artifacts_native_600m --reuse-prepared
```

코퍼스 토큰 수가 예산보다 작으면 코퍼스를 반복합니다. 실행 명세의
`planned_corpus_passes`로 반복 횟수를 확인하고 과도하면 학습 문서 수를 늘리십시오.
적은 문서를 많이 반복하는 것으로 다양한 한국어 데이터를 대체할 수는 없습니다.
학습 처리 속도와 실제 데이터 품질을 측정하기 전에는 완료 시간을 확정하지 않습니다.

처음에는 검증 손실과 새로운 생성 예시를 보며 진행하십시오. 대화 데이터만 짧게
학습해서는 무작위 모델에 언어 능력이 생길 것으로 기대하기 어렵습니다.
큰 모델과 데이터 규모를 함께 고려하는 학습 예산의 배경은
[Chinchilla 연구](https://arxiv.org/abs/2203.15556)를 참고할 수 있습니다.

## 단계별 실행

```bash
python -m native_600m.prepare
python -m native_600m.tokenizer \
  --input artifacts_native_300m/data/pretrain_train.jsonl \
  --output artifacts_native_300m/tokenizer/tokenizer.json
python -m native_600m.tokenize
python -m native_600m.train --stage pretrain \
  --output-dir artifacts_native_300m/pretrain
python -m native_600m.train --stage sft \
  --init-from artifacts_native_300m/pretrain/best.pt \
  --output-dir artifacts_native_300m/sft
python -m native_600m.export
python -m native_600m.evaluate --device cuda
```

모델 전체를 사전학습하고 대화 학습합니다. 대화 단계는 옵티마이저를 새로 시작하며
사용자·시스템·패딩을 정답에서 제외하고 도우미 응답과 발화 종료 토큰만 학습합니다.
정답과 마스크는 데이터셋에서 한 토큰 이동하고 모델 내부에서는 다시 이동하지 않습니다.

문서와 대화는 BOS/EOS 및 역할·발화 종료 토큰으로 구분하여 연속 스트림으로
저장합니다. 학습 창은 문서 또는 대화 경계를 넘을 수 있으며 하나의 긴 대화가
여러 창으로 나뉠 수 있습니다. 각 창에서는 그 안의 이전 토큰만 문맥으로 보입니다.
학습·검증 분리는 문서와 대화 단위로 먼저 수행합니다.

## 재개 및 실행 검사

같은 설정·데이터·GPU 정밀도에서 `--reuse-prepared`로 재개합니다. 체크포인트에는
옵티마이저, FP16 scaler, Python·PyTorch·CUDA 난수 상태, 다음 샘플을 결정하는 step을
저장합니다. 파일을 임시 이름으로 완성한 뒤 교체하여 기존 체크포인트를 보존합니다.
BF16/FP16, 데이터, 학습 예산이나 유효 배치가 달라지면 새로운 결과 디렉터리를 사용하십시오.
유효 배치를 유지한 배치·누적 재구성은 같은 결과 디렉터리에서 재개할 수 있습니다.
첫 번째 완료 step과 이후 약 5분마다 `last.pt`를 저장하며, 설정된 step·검증·완료 시점에도
저장합니다. 시간 간격은 학습기의 `--checkpoint-seconds`로 변경할 수 있습니다.
저장하지 않은 진행 상황은 복원하지 못합니다. 첫 검증과 `best.pt` 저장도 조기에 수행합니다.

```bash
python run_native_600m.py --reuse-prepared
```

데이터 준비 후 **단일 CUDA GPU 환경에서** 2 step 실행을 확인할 수 있습니다.
짧은 실행의 학습 결과는 대화 품질 평가로 사용하지 마십시오.

```bash
python -m native_600m.train --stage pretrain \
  --tokens-dir artifacts_native_300m/tokens \
  --tokenizer artifacts_native_300m/tokenizer/tokenizer.json \
  --output-dir artifacts_native_300m_smoke/pretrain \
  --gpu-profile balanced --seq-len 128 --batch-size 1 --grad-accumulation 1 \
  --max-steps 2
```

## 데이터 출처와 자체 데이터

사전학습은 FineWeb2 `kor_Hang`, revision
`af9c13333eb981300149d5ca60a8e9d659b276b9`를 사용합니다.
출처는 [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)이며
ODC-By-1.0 및 Common Crawl 이용 조건을 기록합니다. 대화 학습은
[smol-koreantalk](https://huggingface.co/datasets/lemon-mint/smol-koreantalk)의
`04eb12c1f999578d74dd87720bc853cedb7fa156`, 배포자 표기 Apache-2.0을 사용합니다.
배포 시 출처와 적용 라이선스 고지를 함께 유지하십시오.

한국어 비율·문자 길이·역할 순서를 검사하고 특수 제어 토큰을 거릅니다.
문서는 정규화한 본문, 대화는 첫 사용자 요청 단위로 중복을 제거하고 결정적으로
학습·검증에 나눕니다. 다른 표현으로 된 유사 문장까지 완전히 제거하지는 않습니다.
어휘는 사전학습 **학습 분할만** 사용하여 새로 학습합니다.
토크나이저와 자료의 SHA-256, 출처, 기록 수를 명세에 보관합니다.

자체 문서는 UTF-8 JSONL의 `{"text":"한국어 문서 내용"}` 형식이며 대화는 다음과 같습니다.

```json
{"messages":[{"role":"user","content":"내 이름은 지연이야."},{"role":"assistant","content":"반가워요, 지연님."},{"role":"user","content":"내 이름 기억해?"},{"role":"assistant","content":"지연님이라고 말씀하셨어요."}]}
```

기본 설정을 복사하여 자체 데이터 크기에 맞게 기록 수와 학습 예산을 조정한 뒤 실행합니다.

```bash
python run_native_600m.py --config configs/my_native_600m.json \
  --local-pretrain-jsonl /data/text.jsonl \
  --pretrain-license '자체 작성, 학습 및 배포 허용' \
  --local-chat-jsonl /data/chat.jsonl \
  --chat-license '자체 작성, 학습 및 배포 허용' \
  --artifacts-dir artifacts_native_300m_custom
```

## 평가와 독립 실행

최종 모델은 인사, 이름 정정, 선호 기억, 확인 질문, 형식 지시, 간단한 계획,
불확실성, 산수, 요약 등 13개 대화 사례로 검사합니다. 실제 모델 응답을 다음 발화에
넣어 다중 턴을 평가하며 한국어 비율, 반복, 사례별 조건을 기록합니다.
자동 기준 미달 시 모델 파일과 보고서는 보존하고 종료 코드 1을 반환합니다.
키워드 검사의 한계가 있으므로 새로운 대화 30개 이상으로 사람 평가도 진행하십시오.

`model/`에는 Safetensors, 구조 설정, 독립 토크나이저, 출처 명세가 포함됩니다.
실행에는 저장소의 `native_600m` 코드와 범용 라이브러리만 필요합니다.

```bash
python -m native_600m.chat --checkpoint artifacts_native_300m/model \
  --tokenizer artifacts_native_300m/model/tokenizer.json --device cuda
```

참고: [H200 공식 사양](https://www.nvidia.com/en-us/data-center/h200/),
[Tokenizers의 처음부터 학습하는 BPE API](https://huggingface.co/docs/tokenizers/python/latest/tutorials/python/training_from_memory.html).
