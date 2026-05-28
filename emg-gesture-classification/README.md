# EMG Gesture Classification: 3-Way Agent Comparison

UCI **EMG Data for Gestures** 데이터셋 (8채널 sEMG, 36명 피험자, 4.2M 샘플)을 가지고 동일한 분류 문제를 세 가지 다른 환경에서 풀어보고 비교한 실험 프로젝트입니다.

## 비교 대상

| # | 환경 | 작업 방식 |
|---|---|---|
| 1 | **Gemini LLM** | Docker 위 Jupyter Notebook에서 사람이 직접 LLM과 대화하며 작성 |
| 2 | **Google Antigravity** | 에이전트가 자동으로 Python 스크립트 작성 + 실행 |
| 3 | **Claude Cowork** | 에이전트가 자동으로 전처리/학습/평가 + 보고서 생성 |

## 결과 한눈에 보기

| 접근법 | 모델 | 라이브러리 | 클래스 | 윈도우 | 평가 프로토콜 | **테스트/검증 Acc** |
|---|---|---|---|---|---|---|
| Gemini v1 | 1D-CNN | PyTorch | 7 (1-7) | 64 / step 32 | **Subject-independent** | **91.51%** |
| Gemini v2 | 1D-CNN + BiLSTM (Label Smoothing) | PyTorch | 7 (1-7) | 128 / step 64 | **Subject-independent** | **93.18%** |
| Antigravity | 1D-CNN (128→256→512) | PyTorch + sklearn | 7 (1-7) | 200 / step 50 | Mixed-subject random | **96.08%** |
| Claude Cowork | MLP + handcrafted features | **numpy only** | 6 (1-6) | 50 / step 10 | Mixed-subject stratified | **96.99%** |

### 평가 프로토콜 차이 — 꼭 읽어주세요

세 접근법의 정확도를 그대로 비교하면 안 됩니다. 평가 방식이 다릅니다.

- **Subject-independent (Gemini)** — 학습에 안 쓴 피험자 6명으로만 검증. 실제 새 사용자에게 모델을 배포했을 때의 성능에 가깝습니다. **더 어렵습니다.**
- **Mixed-subject (Antigravity, Cowork)** — 같은 피험자의 데이터가 train/test에 섞임. 동일 사용자 캘리브레이션 시나리오에 해당합니다. 윈도우 overlap이 있으면 더더욱 쉬워집니다.

같은 프로토콜에서 비교하면 순위가 바뀔 가능성이 높습니다. Cowork의 97%는 *protocol advantage*가 일부 포함된 숫자입니다.

또한 Cowork은 class 0(unmarked)과 class 7(희소)을 제거한 6-class 프로토콜을 사용했고, 나머지 두 접근법은 class 0만 제거하여 7-class입니다. 클래스 수가 적으면 일반적으로 정확도가 더 높게 나옵니다.

## 폴더 구조

```
emg-gesture-classification/
├── README.md
├── .gitignore
│
├── 01_gemini/                       # 사람이 직접 + Gemini 도움받아 작성
│   ├── emg_gesture_gm_v1.ipynb     # 1D-CNN baseline (91.51%)
│   └── emg_gesture_gm_v2.ipynb     # CNN + BiLSTM SOTA (93.18%)
│
├── 02_antigravity/                  # Google Antigravity 에이전트 산출물
│   ├── README.md
│   ├── preprocess.py               # 윈도우 200, step 50, sklearn 사용
│   ├── train.py                    # 1D-CNN, PyTorch, 96.08%
│   ├── report.txt
│   └── confusion_matrix.png
│
└── 03_claude_cowork/                # Claude Cowork 에이전트 산출물
    ├── README.md
    ├── requirements.txt
    ├── LICENSE
    ├── src/
    │   ├── preprocess.py           # numpy only, 220-d features
    │   ├── train.py                # 4-layer MLP (numpy + OpenBLAS)
    │   └── evaluate.py
    ├── results/
    │   ├── confusion_matrix.png    # test 96.99%
    │   ├── training_curve.png
    │   ├── per_class_metrics.png
    │   ├── class_distribution.png
    │   └── final_metrics.json
    ├── data/
    │   └── README.md
    └── docs/
        └── velog_post.md           # 시행착오 회고 블로그
```

## 각 접근법의 특징 요약

### Gemini (직접, v1 → v2)

가장 학구적인 접근. 같은 데이터/protocol을 유지하면서 모델만 업그레이드.

- 전처리: 피험자별 Z-score 정규화 → 학습 1-30번, 검증 31-36번
- v1 모델: 3-layer 1D-CNN (8→32→64→128) + dropout
- v2 모델: CNN의 spatial feature + BiLSTM의 temporal feature를 결합한 **하이브리드** 구조, Label Smoothing 추가
- 결과: v1 91.51% → v2 93.18%로 **+1.67%p 개선**, 가장 어려운 평가 환경에서 가장 의미있는 결과

`emg_gesture_gm_v1.ipynb`, `emg_gesture_gm_v2.ipynb` 참고.

### Antigravity (에이전트)

가장 무난한 PyTorch 표준 1D-CNN.

- 윈도우 200/step 50, sklearn `train_test_split` 사용 (random shuffle → mixed-subject)
- 모델: Conv 128 → 256 → 512 + Dense 1024, Dropout/BatchNorm 풍부
- AdamW + ReduceLROnPlateau, 40 epoch
- 결과: 96.08%

`02_antigravity/` 폴더 참고.

### Claude Cowork (에이전트)

라이브러리 설치가 막힌 환경 (PyTorch/sklearn 사용 불가)에서 **순수 numpy로** 전부 구현. 1D-CNN 대신 핸드크래프트 시간/주파수 특성 + MLP 채택.

- 윈도우 50/step 10, stratified 70/15/15
- 220차원 특성: 시간 도메인 16개 (MAV, RMS, ZC, SSC, Hjorth ...) + FFT 5밴드 파워 + 채널 페어 상관관계 28개
- 피험자별 표준화 + subject one-hot embedding
- 4-layer MLP (768→384→192→6), forward/backward/Adam을 numpy로 손코딩
- **class 0과 class 7 모두 제거** (class 0과 class 1이 모두 "휴식"이라 모델이 구분 불가했던 시행착오)
- 결과: 96.99% (Macro F1 0.97)

자세한 시행착오 회고는 [03_claude_cowork/docs/velog_post.md](03_claude_cowork/docs/velog_post.md).

## 실험에서 얻은 인사이트

1. **에이전트 vs 사람-주도 LLM**: 사람이 흐름을 통제한 Gemini 작업은 더 엄격한 평가 프로토콜을 자연스럽게 채택했고, 에이전트들은 더 쉬운 protocol을 (특별한 지시 없이) 자동으로 선택했습니다. 정확도 숫자만 비교하면 에이전트가 더 잘 한 것처럼 보이지만, **평가 설계의 엄밀함은 사람이 직접 한 쪽이 더 좋았습니다.**
2. **환경 제약이 다른 해법을 낳음**: Cowork는 라이브러리 설치가 막혀서 numpy만으로 풀어야 했고, 그 결과 1D-CNN 대신 핸드크래프트 특성 + MLP라는 전혀 다른 접근으로 갔습니다.
3. **도메인 지식이 모델보다 중요할 때가 있음**: Cowork가 정확도를 74% → 97%로 올린 결정적 변화는 모델 변경이 아니라 **class 0의 의미를 정확히 이해**한 것 (class 0과 class 1이 모두 "rest"라 의미적으로 중복)이었습니다.
4. **모델 복잡도와 결과**: Antigravity의 거대한 CNN과 Cowork의 가벼운 MLP가 비슷한 정확도를 보였습니다. 데이터 규모와 feature 품질이 충분하면 단순 모델로도 SOTA에 근접 가능.

## 데이터셋

[UCI Machine Learning Repository - EMG Data for Gestures](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures)

> Lobov, S., Krilova, N., Kastalskiy, I., Kazantsev, V., & Makarov, V. A. (2018).
> Latent factors limiting the performance of sEMG-interfaces.
> Sensors, 18(4), 1122.

각 접근법의 코드를 실행하려면 위 링크에서 다운로드 후, 각 폴더의 README에 명시된 위치에 CSV를 두면 됩니다.

## License

MIT
