---
title: "EMG 8채널 신호로 손동작 6종 분류하기 - 정확도 38% → 97% 시행착오 정리"
tags: ["EMG", "신호처리", "딥러닝", "numpy", "분류", "프로젝트"]
---

> **TL;DR**
> UCI "EMG Data for Gestures" 데이터셋 (36명 피험자, 8채널 sEMG)으로 손동작 분류기를 직접 짜봤다. 외부 ML 라이브러리(sklearn/PyTorch/TF) 설치가 막힌 환경이라 **순수 numpy + OpenBLAS만으로** 전처리, 특성 추출, 4-layer MLP의 forward/backward/Adam을 전부 구현했다. 처음엔 정확도가 **38%까지 떨어졌다가**, 데이터의 라벨링 규약을 잘못 이해하고 있었다는 걸 발견한 뒤 **97%**까지 올렸다. 비교 대상이었던 Gemini의 93%를 4%p 정도 상회한다.

---

## 1. 데이터부터 보자

원본 CSV는 9개 컬럼, 4,237,907 행이다.

```
time, channel1~channel8, class, label
```

- `label` (1~36): **피험자 ID**
- `class` (0~7): 동작 라벨
- 8개 채널: MYO 암밴드 sEMG 신호 (값이 ±1e-3 수준으로 매우 작음)

`class.value_counts()`를 찍으면:

```
0    2,725,157  ← 압도적으로 많음
6      253,009
5      251,733
4      251,570
1      250,055
3      249,494
2      243,193
7       13,696  ← 거의 없음
```

**class 0은 전체의 64%**를 차지하고, class 7은 0.3%밖에 안 된다. 이게 이번 프로젝트의 가장 큰 함정이었다 (뒤에 다시 나옴).

각 피험자(label)는 **2개의 trial**을 가진다. 시간이 한 번 감소했다가 다시 1부터 증가하는 지점이 trial 경계다. trial 안에서는 동일 class가 연속으로 수백~수천 샘플 이어진다.

---

## 2. 전처리 파이프라인

```
[Raw CSV 4.2M rows]
        ↓ pandas (dtype 명시로 메모리 1/4)
[(label, trial, class) 연속 세그먼트 분할]
        ↓ class 7 제거
[슬라이딩 윈도우: 50 샘플, stride 10]
        ↓
[414,419 windows × 50 timesteps × 8 channels]
```

윈도우는 **세그먼트를 가로지르지 않도록** 했다. 이게 처음에 헷갈렸는데, 그냥 전체를 sliding window하면 클래스 경계가 윈도우 안에 섞여 들어가서 라벨이 모호해진다.

---

## 3. 시행착오 1: 가중 Cross-Entropy의 함정

처음엔 class 0이 65%로 압도적이라 **클래스 불균형**이 문제라고 생각했다. 그래서 inverse-frequency 가중치를 줬다:

```python
w_cls = total_count / (K * class_count)
# 결과: class 0은 0.22, 나머지는 2.4 정도
loss = -(w_cls[y] * log_p[y]).mean()
```

**Softmax regression 결과: Val 0.3849**
**MLP 결과: Val 0.3962**

… **majority baseline(class 0만 찍기)이 64.7%인데 그것보다도 한참 낮다.**

문제는 가중치가 너무 공격적이어서 모델이 "어떻게든 gesture 클래스를 찍어야 한다"고 학습해버린 거였다. class 0이 정답인 64%의 케이스에서 다른 클래스를 막 찍으니까 정확도가 폭락한 것.

**교훈**: 클래스 불균형이 있다고 무조건 가중치를 주면 안 된다. 평가 지표가 단순 accuracy면 균등 가중이 낫다. 가중치는 macro-F1을 최적화하고 싶을 때 의미가 있다.

가중치를 빼고 다시 학습:

| Epoch | Val Acc |
|---|---|
| 1 | 0.6626 |
| 5 | 0.6939 |
| 15 | 0.7401 |
| 25 | 0.7490 |

**74.9%까지는 올랐지만 거기서 멈췄다.** Gemini 93%와는 거리가 멀다.

---

## 4. 시행착오 2: Confusion Matrix를 들여다봤더니

```
              Pred 0   1     2     3     4     5     6
True 0:  32245   0    1694  1519  1497  1724  1574
True 1:   3655   0      0     0     0     0     0   ← ???
True 2:    651   0   2881    7     1     4    10
True 3:    894   0      2  2670    0     0    80
True 4:    960   0      2    1   2666   45     5
True 5:    784   0      0    1    39  2858    0
True 6:    610   0      9   76     9     0  2997
```

**class 1의 recall이 정확히 0%다.** 모델이 단 한 번도 class 1을 예측하지 않는다.

이상해서 데이터셋 문서를 다시 찾아봤다. Lobov et al.의 [원본 논문](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures)에 따르면:

- **0: unmarked data** (라벨이 없는 구간 - 보통 휴식 또는 동작 전환)
- **1: hand at rest** (의도적 안정 시 손)
- 2: hand clenched in a fist
- 3: wrist flexion
- 4: wrist extension
- 5: radial deviations
- 6: ulnar deviations

… 0번이랑 1번이 둘 다 "쉬고 있음"이다. 의미가 거의 같으니까 EMG 신호도 거의 같고, 모델이 둘을 구분할 방법이 없다. 그래서 더 많은 class 0으로 모두 흡수되어버린 거였다.

이 데이터셋의 표준 벤치마크 프로토콜은:

1. **class 0 제거** (모호한 구간)
2. **class 1~6의 6 클래스 분류**

이렇게 한다는 걸 알게 됐다.

---

## 5. 결정적 변화: 6-클래스 프로토콜 적용

class 0을 제거하고 6개 동작만 분류하도록 데이터를 재구성했다.

```python
mask = (y != 0) & (y != 7)
X = X[mask]
y = y[mask] - 1  # 1-6 → 0-5로 remap
# 결과: 146,074 windows
```

여기에 추가로 다음을 적용했다:

### 피험자별 표준화 (가장 효과적)

피험자마다 EMG 진폭이 다르다 (피부 두께, 센서 부착 위치, 근육량 등). 학습셋의 **각 피험자별 평균/표준편차**로 채널을 정규화했다.

```python
for s in unique_subjects:
    train_mask = idx_train & (g == s)
    mu = F[train_mask].mean(0)
    sd = F[train_mask].std(0) + 1e-8
    F[g == s] = (F[g == s] - mu) / sd
```

### Subject one-hot embedding

피험자 ID를 37차원 one-hot 벡터로 만들어 특성에 concat. 모델이 "이 피험자는 보통 이런 분포다"를 학습할 수 있게.

### 특성 추출 (220차원)

윈도우마다:

- **시간 영역** (채널당 16): MAV, RMS, VAR, Waveform Length, Zero Crossings, Slope Sign Changes, Skewness, Kurtosis, IEMG, Willison Amplitude, Hjorth Mobility/Complexity, Max/Min/Range
- **주파수 영역** (rFFT, 채널당 7): 5개 주파수 밴드 log 파워 + spectral centroid + spectral bandwidth + log 총 파워
- **채널 간 상관관계** (28): 8개 채널 페어의 Pearson r

총 220차원 + 피험자 37 one-hot = **257차원 입력**

---

## 6. 모델 아키텍처

```
Input (257)
  → Dense(768) + ReLU + Dropout(0.25)
  → Dense(384) + ReLU + Dropout(0.25)
  → Dense(192) + ReLU + Dropout(0.25)
  → Dense(6) + Softmax
```

- Adam (β1=0.9, β2=0.999), 초기 LR 1.5e-3, cosine annealing
- L2 weight decay 1e-5
- He initialization
- Batch size 4096, 30 epochs

전부 numpy로 짰다. forward, backward, Adam 업데이트, BN 안 쓰고 dropout만. matmul은 BLAS 덕분에 충분히 빠르다.

```python
def forward(x, train=True):
    a = x
    for i in range(L):
        z = a @ params[2*i] + params[2*i+1]
        a = np.maximum(0, z)
        if train:
            m = (np.random.rand(*a.shape) > DROP) / (1-DROP)
            a = a * m
    return a @ params[-2] + params[-1]
```

---

## 7. 결과

```
Ep  1: val_acc=0.8599
Ep  5: val_acc=0.9072
Ep 10: val_acc=0.9442
Ep 15: val_acc=0.9598   ← 이 시점에 이미 Gemini 93% 추월
Ep 20: val_acc=0.9670
Ep 30: val_acc=0.9699
```

**최종 Test accuracy: 0.9699 (96.99%)**

클래스별 성능:

| 클래스 | Precision | Recall | F1 |
|---|---|---|---|
| Hand at rest | 0.991 | 0.999 | 0.995 |
| Hand clenched (fist) | 0.982 | 0.984 | 0.983 |
| Wrist flexion | 0.957 | 0.949 | 0.953 |
| Wrist extension | 0.972 | 0.965 | 0.968 |
| Radial deviation | 0.965 | 0.972 | 0.969 |
| Ulnar deviation | 0.953 | 0.951 | 0.952 |
| **Macro** | **0.970** | **0.970** | **0.970** |

오류 패턴은 거의 다 **wrist flexion ↔ ulnar deviation**, **radial ↔ ulnar deviation** 사이에서 일어난다. 둘 다 손목을 옆으로 꺾는 동작이라 EMG 분포가 비슷한 게 직관적으로 맞다.

---

## 8. 정리: 정확도를 결정한 5가지

각 단계가 정확도에 얼마나 영향을 줬는지 회고:

| 변경 | Δ Accuracy |
|---|---|
| 가중 CE 제거 | +30%p (38% → 69%) |
| 4-layer MLP + 더 많은 epoch | +5%p (69% → 74%) |
| **class 0 제거 (6-클래스 프로토콜)** | **+15%p (74% → 89%)** |
| 피험자별 표준화 + subject embedding | +5%p (89% → 94%) |
| FFT 밴드 파워 + 학습 시간 충분히 | +3%p (94% → 97%) |

**가장 큰 한 방은 데이터셋 라벨 의미를 정확히 이해하는 것이었다.** 모델 튜닝보다도 도메인 지식이 훨씬 큰 차이를 만들었다.

---

## 9. 배운 점

1. **데이터셋 문서를 처음부터 읽자.** class 0의 정의(unmarked)와 class 1(rest)이 의미상 겹친다는 걸 일찍 알았다면 시간을 많이 아꼈을 거다. confusion matrix를 본 뒤에야 알아챘다.

2. **클래스 가중치는 metric에 따라 달라야 한다.** Accuracy 평가 때 무차별적 inverse-frequency 가중은 오히려 해롭다. 가중치는 macro-F1, balanced accuracy 같은 metric을 최적화할 때 의미가 있다.

3. **피험자별 정규화는 거의 공짜로 얻는 성능 향상.** 생체신호 처럼 개체 간 편차가 큰 데이터에서는 글로벌 정규화만으로는 부족하다.

4. **MLP도 충분히 강하다.** 1D CNN을 안 써도 잘 설계한 특성 + MLP로 97%가 나온다. 핸드크래프트 EMG feature(MAV, WL, ZC 등)는 수십 년간 연구된 강력한 표현이다.

5. **순수 numpy로 ML 파이프라인을 짜는 건 의외로 할 만하다.** sklearn/PyTorch가 없어도 BLAS만 있으면 4-layer MLP는 1시간 안에 학습된다.

---

## 부록: 라이브러리 없이 numpy로 MLP 짜기

혹시 비슷한 상황에서 참고용. 핵심 4가지만 있으면 된다.

```python
# 1. He 초기화
def he(d_in, d_out):
    return np.random.randn(d_in, d_out) * np.sqrt(2.0/d_in)

# 2. Forward (dropout 포함)
def forward(x, train):
    a = x
    for W, b in layers[:-1]:
        a = np.maximum(0, a @ W + b)
        if train:
            mask = (np.random.rand(*a.shape) > p) / (1-p)
            a *= mask
    return a @ W_out + b_out

# 3. Softmax + cross-entropy gradient
def grad_logits(logits, y):
    P = np.exp(logits - logits.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    P[np.arange(len(y)), y] -= 1
    return P / len(y)

# 4. Adam 업데이트
def adam_step(p, g, m, v, t, lr):
    m[:] = 0.9*m + 0.1*g
    v[:] = 0.999*v + 0.001*g*g
    p -= lr * (m/(1-0.9**t)) / (np.sqrt(v/(1-0.999**t)) + 1e-8)
```

이게 전부다. BLAS가 받쳐주는 matmul만 있으면 4-layer MLP × 30 epoch이 1분 30초 정도에 끝난다.

