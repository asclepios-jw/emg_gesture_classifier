"""4-layer MLP trainer for the EMG gesture dataset.

순수 numpy로 forward / backward / Adam을 구현했습니다.
체크포인트가 저장되므로 중간에 멈춰도 같은 명령어로 재개됩니다.
"""
import argparse
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np

# ---------------------------------------------------------------------------
# Model & training hyperparameters
# ---------------------------------------------------------------------------
HIDDEN = [768, 384, 192]
DROPOUT = 0.25
WEIGHT_DECAY = 1e-5
LR0 = 1.5e-3
BATCH = 4096
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
TIME_BUDGET_SEC = 30  # per-call wall budget (set high if your env has no limit)


# ---------------------------------------------------------------------------
# Building blocks (pure numpy)
# ---------------------------------------------------------------------------
def he(d_in, d_out):
    return (np.random.randn(d_in, d_out) * np.sqrt(2.0 / d_in)).astype(np.float32)


def init_params(D, K):
    sizes = [D, *HIDDEN, K]
    params = []
    for i in range(len(sizes) - 1):
        params.append(he(sizes[i], sizes[i + 1]))
        params.append(np.zeros(sizes[i + 1], np.float32))
    return params


def forward(x, params, train: bool, dropout: float = DROPOUT):
    L = len(HIDDEN)
    acts = [x]
    masks = []
    a = x
    for i in range(L):
        z = a @ params[2 * i] + params[2 * i + 1]
        a = np.maximum(0, z)
        if train:
            m = (np.random.rand(*a.shape) > dropout).astype(np.float32) / (1 - dropout)
            a = a * m
            masks.append(m)
        else:
            masks.append(None)
        acts.append(a)
    logits = a @ params[2 * L] + params[2 * L + 1]
    return logits, (acts, masks)


def backward(cache, dlogits, params):
    L = len(HIDDEN)
    acts, masks = cache
    grads = [None] * len(params)
    grads[2 * L] = acts[L].T @ dlogits + WEIGHT_DECAY * params[2 * L]
    grads[2 * L + 1] = dlogits.sum(0)
    d = dlogits @ params[2 * L].T
    for i in range(L - 1, -1, -1):
        if masks[i] is not None:
            d *= masks[i]
        d = d * (acts[i + 1] > 0)
        grads[2 * i] = acts[i].T @ d + WEIGHT_DECAY * params[2 * i]
        grads[2 * i + 1] = d.sum(0)
        if i > 0:
            d = d @ params[2 * i].T
    return grads


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def adam_step(params, grads, m_state, v_state, step, lr):
    bc1, bc2 = 1 - BETA1 ** step, 1 - BETA2 ** step
    for i in range(len(params)):
        m_state[i] = BETA1 * m_state[i] + (1 - BETA1) * grads[i]
        v_state[i] = BETA2 * v_state[i] + (1 - BETA2) * (grads[i] * grads[i])
        params[i] -= lr * (m_state[i] / bc1) / (np.sqrt(v_state[i] / bc2) + EPS)


def evaluate(X, y, params, bs=8192):
    correct = 0
    total = 0
    preds = []
    for s in range(0, len(X), bs):
        lo, _ = forward(X[s:s + bs], params, train=False)
        p = lo.argmax(1)
        preds.append(p)
        correct += (p == y[s:s + bs]).sum()
        total += len(p)
    return correct / total, np.concatenate(preds)


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------
def save_state(path, params, m_state, v_state, best, step, epoch, best_val):
    out = {}
    for i, p in enumerate(params): out[f"p{i}"] = p
    for i, m in enumerate(m_state): out[f"m{i}"] = m
    for i, v in enumerate(v_state): out[f"v{i}"] = v
    for i, b in enumerate(best):    out[f"best{i}"] = b
    out["step"] = np.array([step])
    out["epoch"] = np.array([epoch])
    out["best_val"] = np.array([best_val])
    np.savez(path, **out)


def load_state(path, n_params):
    s = np.load(path)
    params = [s[f"p{i}"].copy() for i in range(n_params)]
    m_state = [s[f"m{i}"].copy() for i in range(n_params)]
    v_state = [s[f"v{i}"].copy() for i in range(n_params)]
    best = [s[f"best{i}"].copy() for i in range(n_params)]
    return (params, m_state, v_state, best,
            int(s["step"].item()), int(s["epoch"].item()), float(s["best_val"].item()))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="artifacts/dataset.npz")
    ap.add_argument("--state",   default="artifacts/model_state.npz")
    ap.add_argument("--preds",   default="artifacts/preds.npz")
    ap.add_argument("--epochs",  type=int, default=30)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)

    z = np.load(args.dataset)
    F = z["F"].astype(np.float32)
    y = z["y"].astype(np.int64)
    g = z["g"].astype(np.int64)
    idx_train, idx_val, idx_test = z["idx_train"], z["idx_val"], z["idx_test"]

    K = int(y.max()) + 1
    N_SUBJ = int(g.max()) + 1
    onehot = np.zeros((len(g), N_SUBJ), np.float32)
    onehot[np.arange(len(g)), g] = 1.0
    F = np.concatenate([F, onehot], axis=1)

    Xtr, ytr = F[idx_train], y[idx_train]
    Xva, yva = F[idx_val],   y[idx_val]
    Xte, yte = F[idx_test],  y[idx_test]
    del F
    D = Xtr.shape[1]
    print(f"D={D} K={K} train={len(Xtr):,} val={len(Xva):,} test={len(Xte):,}", flush=True)

    n_params = 2 * (len(HIDDEN) + 1)
    if os.path.exists(args.state):
        print("Resuming from checkpoint...", flush=True)
        params, m_state, v_state, best, step, epoch, best_val = load_state(args.state, n_params)
    else:
        np.random.seed(0)
        params = init_params(D, K)
        m_state = [np.zeros_like(p) for p in params]
        v_state = [np.zeros_like(p) for p in params]
        best = [p.copy() for p in params]
        step, epoch, best_val = 0, 0, 0.0

    BS = BATCH
    order = np.arange(len(Xtr))
    rng = np.random.default_rng(epoch + 1)
    t0 = time.time()
    end_ep = args.epochs

    for ep in range(epoch, args.epochs):
        rng.shuffle(order)
        losses = []
        lr = 0.5 * LR0 * (1 + np.cos(np.pi * ep / args.epochs))
        for s in range(0, len(order), BS):
            bi = order[s:s + BS]
            xb, yb = Xtr[bi], ytr[bi]
            lo, cache = forward(xb, params, train=True)
            P = softmax(lo)
            loss = -np.log(np.clip(P[np.arange(len(yb)), yb], 1e-12, 1)).mean()
            losses.append(loss)
            P[np.arange(len(yb)), yb] -= 1.0
            dlogits = P / len(yb)
            grads = backward(cache, dlogits, params)
            step += 1
            adam_step(params, grads, m_state, v_state, step, lr)
        va, _ = evaluate(Xva, yva, params)
        elapsed = time.time() - t0
        print(
            f"Ep {ep + 1:2d}/{args.epochs}: loss={np.mean(losses):.4f}  "
            f"val_acc={va:.4f}  lr={lr:.4f}  {elapsed:.1f}s",
            flush=True,
        )
        if va > best_val:
            best_val = va
            best = [p.copy() for p in params]
        if elapsed > TIME_BUDGET_SEC and ep + 1 < args.epochs:
            end_ep = ep + 1
            break
    else:
        end_ep = args.epochs

    save_state(args.state, params, m_state, v_state, best, step, end_ep, best_val)
    print(f"Saved checkpoint  | epoch={end_ep}/{args.epochs}  best_val={best_val:.4f}", flush=True)

    if end_ep >= args.epochs:
        for i in range(len(params)):
            params[i][...] = best[i]
        va, vap = evaluate(Xva, yva, params)
        te, tep = evaluate(Xte, yte, params)
        print(f"FINAL: val_acc={va:.4f}  TEST_acc={te:.4f}", flush=True)
        np.savez(args.preds, te_pred=tep, yte=yte, va_pred=vap, yva=yva)
    else:
        print("Time budget hit; re-run the same command to continue training.")


if __name__ == "__main__":
    main()
