# NativeKoreanLM-600M

**외부 사전학습 모델 없이 처음부터 학습하는 한국어 대화 모델**입니다.
메인 모델은 직접 정의한 PyTorch Transformer이며 **596,788,480개 파라미터**를 사용합니다.
무작위 가중치에서 시작하고, 토크나이저와 대화 형식도 별도로 구축합니다.
외부 모델의 가중치·설정·어휘·대화 템플릿을 가져오지 않습니다.

학습 데이터는 공개 한국어 코퍼스와 대화 데이터 또는 자체 JSONL을 사용할 수 있습니다.
PyTorch, Tokenizers, Datasets는 구현 도구로 사용합니다. 모델 학습은
**단일 CUDA GPU 환경**에서 실행하며 특정 플랫폼이나 디렉터리에 제한되지 않습니다.
현재 제공되는 것은 코드와 설정이며 아직 학습된 가중치나 품질 실측값은 없습니다.

## 학습 환경에서 실행

Python 3.10~3.12의 단일 CUDA GPU 환경에서 실행합니다.
Lightning AI Studio 또는 필요한 패키지를 설치한 다른 GPU 환경을 사용할 수 있습니다.

```bash
python -m pip install -r requirements-native-600m.txt
python run_native_600m.py
```

기존 `python run_lightning_600m.py` 명령도 같은 자체 모델 경로를 실행합니다.
RTX PRO 6000 또는 H200이 감지되고 전체 메모리가 80GiB 이상이면 배치 8 구성을
자동 적용합니다. Modal 기본 GPU는 RTX PRO 6000 96GB 한 장입니다.
기존 학습도 유효 배치가 같으면 배치 8로 재개하며 가중치·옵티마이저·학습 위치를 유지합니다.

실행 순서는 한국어 데이터 준비 → 자체 BPE 학습 → 토큰 파일 생성 → 무작위
가중치 사전학습 → 도우미 답변 중심 대화 학습 → 모델 저장 → 대화 평가입니다.

처음부터 학습하므로 사전학습이 필요합니다. 기본 사전학습 예산은 **120억 토큰**이며,
대화 품질을 보장하는 수치는 아닙니다. 데이터 준비 이후 실제 토큰 수, 코퍼스 반복
횟수, 처리 속도를 기록하므로 이를 확인하여 학습 예산을 조정하십시오.

데이터 준비와 토크나이저 학습은 CPU 환경에서 먼저 수행할 수 있습니다.
CPU 작업자는 기본 24개이며 `--cpu-workers`로 조정합니다. 문서 필터링·검증·토큰 변환은
프로세스로 분산하고 토크나이저 어휘 학습은 Rust 스레드를 사용합니다.

```bash
python run_native_600m.py --prepare-only
```

이후 GPU를 연결하고 준비된 자료를 재사용합니다. 중단된 학습도 같은 명령으로 재개합니다.

```bash
python run_native_600m.py --reuse-prepared
```

다운로드나 학습 없이 실행할 명령만 확인하려면 다음을 사용합니다.

```bash
python run_native_600m.py --plan
```

## Modal에서 실행

`modaltest.py`는 CPU에서 데이터를 준비한 뒤 RTX PRO 6000 96GB 한 장에서 모델을 학습합니다.
데이터와 결과는 영구 Volume에 저장하며 자동 재시도는 사용하지 않습니다.

```bash
python -m pip install -r requirements-modal.txt
modal setup
modal run modaltest.py
```

[전처리 전용 실행, 2 step 시험 실행 및 저장 위치](docs/modal.md)를 참고하십시오.

## 학습 결과와 대화

```bash
python -m native_600m.chat --device cuda
```

`/reset`으로 기록을 초기화하고 `/exit`으로 종료합니다.
독립 저장본도 불러올 수 있습니다.

```bash
python -m native_600m.chat \
  --checkpoint artifacts_native_600m/model \
  --tokenizer artifacts_native_600m/model/tokenizer.json \
  --device cuda
```

| 구성 | 기본값 |
|---|---:|
| 파라미터 | 596,788,480 |
| 블록 / hidden size | 32 / 1,280 |
| attention / KV heads | 20 / 5 |
| FFN / 활성화 | 3,456 / SwiGLU |
| 위치 / 정규화 | RoPE / RMSNorm |
| 어휘 / 문맥 | 자체 BPE 32,000 / 2,048토큰 |
| 사전학습 데이터 | 한국어 문서 2,000,000개 + 검증 10,000개 |
| 대화 데이터 | 100,000개 + 검증 2,000개 |

주요 결과는 `artifacts_native_600m/`에 저장됩니다. `pretrain/last.pt`,
`sft/last.pt`는 재개용이고 `sft/best.pt`는 검증 손실 기준 최적 모델입니다.
`model/`에는 자체 모델 Safetensors·설정·토크나이저·출처 명세가 저장됩니다.
`evaluation.json`은 한국어 대화 13개 사례의 자동 검사와 사람 평가 기준을 포함합니다.

## 안내 및 검사

- [GPU 설정, 단계별 실행, 자체 데이터와 학습 예산](docs/native_600m.md)
- 사전학습: [FineWeb2 한국어](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), ODC-By-1.0 및 Common Crawl 이용 조건
- 대화 학습: [smol-koreantalk](https://huggingface.co/datasets/lemon-mint/smol-koreantalk), 배포자 표기 Apache-2.0

코드 검사는 모델·토크나이저 학습을 실행하지 않습니다. 작은 모델의 추론으로
인과성·손실 정렬·저장 및 복원 등을 검사하고 전체 크기는 meta 장치로 확인합니다.

```bash
python -m unittest discover -s tests -p 'test_native_600m_*.py' -v
```

이 저장소는 자체 한국어 600M 모델의 학습·대화·평가 경로만 포함합니다.
