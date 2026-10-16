# KoreanChat-600M

약 600M 규모의 기존 모델을 한국어 대화용으로 추가 학습하는 프로젝트입니다.
기본 모델은 **Qwen3-0.6B**, 고유 파라미터는 **596,049,920개**입니다.
요청 이해, 자연스러운 응답, 대화 맥락 유지에 초점을 맞춥니다.

학습은 **Lightning AI Studio의 단일 CUDA GPU**에서 실행하도록 구성했습니다.
새 학습 명령은 Studio 환경을 확인하며 로컬 PC에서의 학습을 차단합니다.
현재 저장소에는 학습 코드와 설정이 제공됩니다. 추가 학습된 가중치와 대화 평가
결과는 Studio에서 실행한 뒤 생성됩니다.

## Lightning AI Studio에서 실행

Python 3.10~3.12와 단일 CUDA GPU를 사용하는 Studio에서 저장소를 열고 실행합니다.
GPU 메모리는 16GB 이상을 시작 구성으로 권장하며, 실제 사용량은 데이터 길이와
GPU에서 확인해야 합니다. 별도의 PyTorch Lightning 설치는 필요하지 않습니다.

```bash
python -m pip install -r requirements-korean-600m.txt
python run_lightning_600m.py
```

한국어 데이터 준비 → 원본 모델 평가 → LoRA 추가 학습 → 독립 모델 병합 저장 →
추가 학습 모델 평가를 수행합니다. 기본값은 학습 대화 50,000개, 검증 대화
1,000개, 문맥 1,024토큰, 2에포크, LoRA rank 32입니다.
각 도우미 발화가 개별 학습 예시가 되며 실제 예시 수와 토큰 수를 기록합니다.

중단된 작업은 같은 GPU 정밀도와 설정으로 재개합니다.

```bash
python run_lightning_600m.py --reuse-prepared
```

학습·검증 데이터의 해시와 설정을 검사한 뒤 마지막 완성 체크포인트의
모델·옵티마이저·스케줄러·난수 상태를 복원합니다.
실행할 명령만 확인하려면 다음을 사용합니다. 다운로드나 학습을 하지 않습니다.

```bash
python run_lightning_600m.py --plan
```

## 학습 결과로 대화

Studio에서 학습한 뒤 실행합니다. `/reset`으로 기록을 초기화하고 `/exit`으로 종료합니다.

```bash
python -m korean_600m.chat --device cuda
```

병합된 독립 모델을 사용할 수도 있습니다.

```bash
python -m korean_600m.chat --merged artifacts_korean_600m/merged --device cuda
```

| 결과 경로 | 내용 |
|---|---|
| `artifacts_korean_600m/data/` | 한국어 대화, 출처·라이선스·해시 명세 |
| `artifacts_korean_600m/training/checkpoint-*` | 학습 재개용 상태 |
| `artifacts_korean_600m/training/metrics.json` | 학습·검증 손실 |
| `artifacts_korean_600m/adapter/` | 최적 검증 체크포인트의 LoRA 및 토크나이저 |
| `artifacts_korean_600m/merged/` | 약 596M 독립 Safetensors 모델 |
| `artifacts_korean_600m/evaluation_base.json` | 추가 학습 전 대화 보고서 |
| `artifacts_korean_600m/evaluation_tuned.json` | 추가 학습 후 대화 보고서 |

평가는 인사, 맥락 기억과 정정, 확인 질문, 지시 준수, 반복, 한국어 유지 등을
검사합니다. 기준에 미달하면 보고서를 저장하고 오류 상태로 종료합니다.
자동 검사 통과만으로 대화 품질을 확정하지 않으며 보고서의 사람 평가 기준으로
새 대화도 확인해야 합니다.

## 데이터와 상세 안내

- 기본 모델: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), Apache-2.0
- 추가 학습 데이터: [smol-koreantalk](https://huggingface.co/datasets/lemon-mint/smol-koreantalk), 배포자 표기 Apache-2.0
- [Studio 실행, 설정 변경, 짧은 검증 실행, 자체 데이터 사용](docs/korean_600m.md)
- [기존 영어 전용 200M 실험 문서](docs/native_200m_legacy.md)

모델과 데이터는 커밋 SHA로 고정합니다. 한국어 비중 검사, 중복 제거,
첫 사용자 요청 단위의 학습·검증 분리를 적용합니다. 도우미 답변에만 손실을
계산하며 학습·추론에 동일한 비사고 모드 대화 형식을 사용합니다.

## 로컬 코드 검사

다음 검사는 학습, 모델 가중치 다운로드, 역전파를 실행하지 않습니다.
패딩 검사에는 기존 PyTorch 설치가 필요합니다.

```bash
python -m unittest discover -s tests -p 'test_korean_*.py' -v
```
