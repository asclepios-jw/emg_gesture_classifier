"""EMG preprocessing + feature extraction pipeline.

입력:  data/EMG-data.csv   (UCI EMG Data for Gestures)
출력:  artifacts/dataset.npz   (features, labels, splits)

파이프라인
----------
1. CSV 로딩 (dtype 명시로 메모리 1/4)
2. 피험자(label)별로 시간 감소를 trial 경계로 사용하여 분할
3. (trial, class) 연속 세그먼트로 잘게 분할
4. class 0(unmarked, class 1과 의미 중복) 및 class 7(희소)을 제거
5. 슬라이딩 윈도우 (50 samples × 8 channels, stride 10)
6. 윈도우별 220차원 특성 추출 (시간 도메인 + FFT + 채널 상관)
7. 학습 데이터만 사용한 피험자별 표준화 → 전역 표준화
8. 70/15/15 stratified split
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

CHANNELS = [f"channel{i}" for i in range(1, 9)]
WINDOW = 50
STRIDE = 10
DROP_CLASSES = {0, 7}  # 0 = unmarked (class 1과 중복), 7 = 희소 (라벨링 노이즈)
SPLIT_SEED = 42
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15


# ---------------------------------------------------------------------------
# Step 1-2: load CSV and detect trial boundaries
# ---------------------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    print(f"Loading {path} ...", flush=True)
    t0 = time.time()
    df = pd.read_csv(
        path,
        usecols=["time", *CHANNELS, "class", "label"],
        dtype={
            "time": np.int32,
            "class": np.int8,
            "label": np.int8,
            **{c: np.float32 for c in CHANNELS},
        },
    )
    print(f"  {len(df):,} rows in {time.time() - t0:.1f}s", flush=True)
    return df


# ---------------------------------------------------------------------------
# Step 3-5: segment + windowing
# ---------------------------------------------------------------------------
def window_dataframe(df: pd.DataFrame):
    """피험자별로 trial 경계를 식별, (trial, class) 세그먼트마다 슬라이딩 윈도우."""
    all_X, all_y, all_g, all_t = [], [], [], []
    for lbl, sub in df.groupby("label", sort=True):
        times = sub["time"].values
        classes = sub["class"].values.astype(np.int8)
        sigs = sub[CHANNELS].values  # float32

        # time이 감소하는 지점이 새로운 trial 시작
        trial_id = np.zeros(len(sub), dtype=np.int8)
        for d in np.where(np.diff(times) < 0)[0] + 1:
            trial_id[d:] += 1

        # (trial, class) 변화점을 세그먼트 경계로
        keys = trial_id.astype(np.int32) * 100 + classes.astype(np.int32)
        change = np.where(np.diff(keys) != 0)[0] + 1
        bounds = np.concatenate(([0], change, [len(sub)]))

        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            klass = int(classes[s])
            if klass in DROP_CLASSES:
                continue
            if e - s < WINDOW:
                continue
            seg = sigs[s:e]
            wv = sliding_window_view(seg, WINDOW, axis=0)[::STRIDE]  # (nw, 8, W)
            wv = np.transpose(wv, (0, 2, 1)).astype(np.float32)      # (nw, W, 8)
            nw = wv.shape[0]
            all_X.append(wv)
            all_y.append(np.full(nw, klass, dtype=np.int8))
            all_g.append(np.full(nw, int(lbl), dtype=np.int8))
            all_t.append(np.full(nw, int(trial_id[s]), dtype=np.int8))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    g = np.concatenate(all_g, axis=0)
    t = np.concatenate(all_t, axis=0)

    # Class 라벨을 0부터 시작하도록 remap (예: 1-6 → 0-5)
    unique = np.array(sorted(set(int(c) for c in np.unique(y))), dtype=np.int8)
    remap = {old: new for new, old in enumerate(unique)}
    y = np.array([remap[c] for c in y], dtype=np.int8)
    print(f"Windowed: X={X.shape}, y unique={np.unique(y)}, subjects={np.unique(g)}", flush=True)
    return X, y, g, t


# ---------------------------------------------------------------------------
# Step 6: feature extraction
# ---------------------------------------------------------------------------
def extract_features(X: np.ndarray) -> np.ndarray:
    """X: (N, W, C) → features (N, 220).

    채널별 19개 시간/주파수 특성 + 채널 페어 28개 상관계수.
    """
    eps = 1e-12
    N, W, C = X.shape

    # 시간 도메인 16개
    mav = np.mean(np.abs(X), axis=1)
    rms = np.sqrt(np.mean(X ** 2, axis=1))
    var = np.var(X, axis=1)
    dX = np.diff(X, axis=1)
    wl = np.sum(np.abs(dX), axis=1)
    iemg = np.sum(np.abs(X), axis=1)

    sign = np.sign(X)
    zc = np.sum(
        (sign[:, 1:, :] * sign[:, :-1, :] < 0) & (np.abs(dX) > 1e-6), axis=1
    ).astype(np.float32)
    ddX = np.diff(dX, axis=1)
    ssc = np.sum(
        (dX[:, 1:, :] * dX[:, :-1, :] < 0) & (np.abs(ddX) > 1e-6), axis=1
    ).astype(np.float32)

    log_rms = np.log(rms + eps)
    mean_ = np.mean(X, axis=1, keepdims=True)
    std_ = np.std(X, axis=1, keepdims=True) + eps
    z = (X - mean_) / std_
    skew = np.mean(z ** 3, axis=1)
    kurt = np.mean(z ** 4, axis=1) - 3.0
    wamp = np.sum(np.abs(dX) > 1e-5, axis=1).astype(np.float32)
    mobility = np.sqrt(np.var(dX, axis=1) / (var + eps))
    complexity = (
        np.sqrt(np.var(ddX, axis=1) / (np.var(dX, axis=1) + eps)) / (mobility + eps)
    )
    mx = np.max(X, axis=1)
    mn = np.min(X, axis=1)
    rng = mx - mn

    # 주파수 도메인 (rFFT): 5 밴드 파워 + centroid + bandwidth + total power
    Xf = np.fft.rfft(X, axis=1)
    pow_ = (Xf.real ** 2 + Xf.imag ** 2).astype(np.float32)
    F_bins = pow_.shape[1]
    nbands = 5
    edges = np.linspace(0, F_bins, nbands + 1, dtype=int)
    bands = []
    for bi in range(nbands):
        s, e = edges[bi], edges[bi + 1]
        bp = pow_[:, s:e, :].sum(axis=1) if e > s else np.zeros((N, C), np.float32)
        bands.append(np.log(bp + eps))
    band_feats = np.concatenate(bands, axis=1)

    freqs = np.arange(F_bins).astype(np.float32)
    total_pow = pow_.sum(axis=1) + eps
    centroid = (pow_ * freqs[None, :, None]).sum(axis=1) / total_pow
    bw = np.sqrt(
        ((freqs[None, :, None] - centroid[:, None, :]) ** 2 * pow_).sum(axis=1)
        / total_pow
    )
    mean_pow = np.log(total_pow + eps)

    F_per_ch = np.concatenate(
        [
            mav, rms, var, wl, zc, ssc, log_rms, skew, kurt, iemg, wamp,
            mobility, complexity, mx, mn, rng, centroid, bw, mean_pow,
        ],
        axis=1,
    ).astype(np.float32)  # 19 * 8 = 152

    # 채널 페어 Pearson correlation (8C2 = 28)
    Xc = X - mean_
    norms = np.sqrt(np.sum(Xc ** 2, axis=1)) + eps
    corr_feats = []
    for i in range(C):
        for j in range(i + 1, C):
            num = np.sum(Xc[:, :, i] * Xc[:, :, j], axis=1)
            corr_feats.append(num / (norms[:, i] * norms[:, j]))
    corr_arr = np.stack(corr_feats, axis=1).astype(np.float32)

    F = np.concatenate([F_per_ch, band_feats, corr_arr], axis=1)
    F[~np.isfinite(F)] = 0.0
    return F  # shape (N, 152 + 5*8 + 28) = (N, 220)


# ---------------------------------------------------------------------------
# Step 7-8: split and standardize
# ---------------------------------------------------------------------------
def stratified_split(y: np.ndarray, seed: int = SPLIT_SEED):
    rng = np.random.default_rng(seed)
    K = int(y.max()) + 1
    idx_train = np.zeros(len(y), bool)
    idx_val = np.zeros(len(y), bool)
    idx_test = np.zeros(len(y), bool)
    for c in range(K):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        n = len(ci)
        n_tr = int(n * TRAIN_RATIO)
        n_va = int(n * VAL_RATIO)
        idx_train[ci[:n_tr]] = True
        idx_val[ci[n_tr:n_tr + n_va]] = True
        idx_test[ci[n_tr + n_va:]] = True
    return idx_train, idx_val, idx_test


def normalize(F: np.ndarray, g: np.ndarray, idx_train: np.ndarray) -> np.ndarray:
    """피험자별 표준화 (학습셋 통계만 사용) → 전역 표준화."""
    F_norm = F.copy()
    for s in np.unique(g):
        tr_mask = idx_train & (g == s)
        if tr_mask.sum() < 10:
            continue
        subj_mask = g == s
        mu = F[tr_mask].mean(0, keepdims=True)
        sd = F[tr_mask].std(0, keepdims=True) + 1e-8
        F_norm[subj_mask] = (F[subj_mask] - mu) / sd

    mu_g = F_norm[idx_train].mean(0, keepdims=True)
    sd_g = F_norm[idx_train].std(0, keepdims=True) + 1e-8
    F_norm = (F_norm - mu_g) / sd_g
    return F_norm.astype(np.float32)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/EMG-data.csv")
    ap.add_argument("--out", default="artifacts/dataset.npz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    df = load_csv(args.csv)
    X, y, g, t = window_dataframe(df)
    print(f"Class counts: {np.bincount(y)}")

    print("Extracting features ...", flush=True)
    t0 = time.time()
    parts = []
    CHUNK = 30000
    for s in range(0, len(X), CHUNK):
        parts.append(extract_features(X[s:s + CHUNK]))
    F = np.concatenate(parts, axis=0)
    print(f"  features {F.shape} in {time.time() - t0:.1f}s", flush=True)

    idx_train, idx_val, idx_test = stratified_split(y)
    F = normalize(F, g, idx_train)

    np.savez(
        args.out,
        F=F,
        y=y,
        g=g,
        t=t,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )
    print(
        f"Saved {args.out}  | train={idx_train.sum()} val={idx_val.sum()} test={idx_test.sum()}"
    )


if __name__ == "__main__":
    main()
