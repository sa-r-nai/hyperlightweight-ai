# NativeByteLM-500M English

`hyperlightweight-ai`의 영어 전용 독자 학습 경로입니다. 이 경로는 외부 사전학습 가중치, 외부 토크나이저, 모델별 채팅 템플릿, Hugging Face 모델 클래스를 사용하지 않습니다. 모델은 빈 가중치에서 시작하며, 저장소 안의 순수 PyTorch 구현으로 학습과 추론을 수행합니다. 기본 채팅 생성은 깨진 UTF-8 출력을 방지하기 위해 printable ASCII로 제한됩니다.

## 구조

```mermaid
flowchart LR
    A[사용자 입력] --> B[자체 UTF-8 토크나이저]
    B --> C[인과 Transformer 24층]
    C --> D[다음 토큰 분포]
    D --> E[샘플링·디코딩]
    E --> C
```

## 기본 구성

| 항목 | 값 |
|---|---:|
| 어휘 | 특수 토큰 8개 + UTF-8 바이트 256개 = 264개 |
| 문맥 길이 | 2,048 토큰 |
| 은닉 차원 | 1,280 |
| 디코더 블록 | 24층 |
| 어텐션 헤드 | 20개, 전체 멀티헤드 어텐션 |
| FFN 차원 | 5,504 |
| 위치 정보 | 학습형 절대 위치 임베딩 |
| 정규화·활성화 | Pre-LayerNorm · GELU(tanh 근사) |
| 임베딩/출력 헤드 | 가중치 공유 |
| 파라미터 수 | 약 498.5M |

바이트 단위 토크나이저는 임의의 Unicode 입력을 표현할 수 있지만, 현재 학습 데이터와 기본 시스템 프롬프트는 영어 전용입니다. 채팅 출력은 기본적으로 ASCII 문자만 허용하며 `--allow-unicode`를 지정할 때만 제한을 해제합니다.

## 설치

CUDA를 사용하는 경우 먼저 CUDA에 맞는 PyTorch를 설치한 다음 나머지 의존성을 설치하세요.

```powershell
python -m pip install -r .\requirements-native.txt
```

이 경로에는 `transformers`, 외부 모델 저장소, 외부 tokenizer 패키지가 필요하지 않습니다.

## 데이터 생성과 검증

저장소에 포함할 수 있는 최소한의 자체 작성 시드를 생성합니다.

```powershell
python .\generate_native_data.py
python .\prepare_native_sft.py
python -m unittest -v .\test_native_500m.py
```

`data/native_sft_seed.jsonl`과 `data/native_pretraining_seed.txt`는 기능 점검용 시드입니다. 이것만으로 500M 모델을 유용하게 사전 학습할 수 있다고 주장하지 않습니다. 더 큰 데이터를 추가할 때는 `prepare_native_sft.py`에 입력하고, 출처·라이선스·해시를 별도 매니페스트에 남겨야 합니다.

## 학습

기본값은 CUDA이며 CUDA가 없을 때 CPU로 조용히 전환하지 않습니다.

```powershell
python .\train_native_500m.py `
  --device cuda `
  --data .\data `
  --seq-len 2048 `
  --batch-size 1 `
  --grad-accumulation 8 `
  --grad-checkpointing `
  --max-steps 1000 `
  --output-dir .\checkpoints_native_500m
```

CPU를 명시적으로 선택할 수는 있지만, 500M 모델을 CPU로 처음부터 학습하는 것은 현실적으로 매우 느립니다.

```powershell
python .\train_native_500m.py --device cpu --preset smoke --seq-len 128 --max-steps 10
```

실제 학습을 시작하기 전에는 `--preset smoke`로 데이터 로딩, 손실 계산, 체크포인트 저장이 모두 되는지 확인하세요. CUDA bfloat16이 지원되면 `--dtype auto`가 bfloat16을 선택하고, 그렇지 않으면 float16을 사용합니다. CPU에서는 float32를 사용합니다.

## 추론

학습이 끝난 뒤 다음처럼 단일 질문을 실행할 수 있습니다.

```powershell
python .\chat_native_500m.py `
  --checkpoint .\checkpoints_native_500m\best.pt `
  --device cuda `
  --message "캐시와 백업의 차이를 설명해 줘."
```

대화형 모드는 `--message`를 생략하면 됩니다. 종료 명령은 `/exit`입니다.

```powershell
python .\run_native_chat.ps1 -Checkpoint .\checkpoints_native_500m\best.pt -Device cuda
```

## 재현성과 범위

- `native_500m.py`: 모델, 파라미터 수 계산, 체크포인트 입출력
- `native_tokenizer.py`: 자체 특수 토큰 규약과 UTF-8 바이트 인코더
- `train_native_500m.py`: 사전 학습 루프, AdamW, warmup/cosine schedule, AMP, gradient checkpointing
- `chat_native_500m.py`: 대화 구성, 자동 회귀 생성, temperature/top-k/top-p 샘플링
- `generate_native_data.py`: 자체 작성 학습 시드 생성
- `prepare_native_sft.py`: 검증·중복 제거·분할·출처 매니페스트 생성
- `test_native_500m.py`: 토크나이저, 인과성, 손실, 생성, 파라미터 수 검사
- `docs/native_500m_research.md`: 설계 근거와 학습 계획

이 커밋에는 무작위 초기화 상태의 500M 가중치 파일을 넣지 않습니다. 가중치 파일은 용량이 크고, 학습되지 않은 랜덤 모델은 사용할 수 있는 모델이 아니기 때문입니다. 위 학습 명령으로 사용자가 확보한 데이터와 GPU 환경에서 직접 체크포인트를 생성하도록 구성했습니다.
