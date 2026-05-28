"""Evaluation: metrics + confusion matrix + per-class bars."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = [
    "1. Hand at rest",
    "2. Hand clenched (fist)",
    "3. Wrist flexion",
    "4. Wrist extension",
    "5. Radial deviation",
    "6. Ulnar deviation",
]
SHORT = ["Rest", "Fist", "WFlex", "WExt", "Radial", "Ulnar"]


def confusion(y, p, K):
    M = np.zeros((K, K), int)
    for t, q in zip(y, p):
        M[t, q] += 1
    return M


def per_class_metrics(M):
    rows = []
    for i in range(M.shape[0]):
        tp = M[i, i]
        fn = M[i].sum() - tp
        fp = M[:, i].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append((prec, rec, f1, int(M[i].sum())))
    acc = np.trace(M) / M.sum()
    macro = tuple(np.mean(np.array(rows)[:, k]) for k in range(3))
    return acc, rows, macro


def plot_confusion(M_va, M_te, acc_va, acc_te, out):
    K = M_te.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, M, title in [
        (axes[0], M_va, f"Validation (acc={acc_va:.4f})"),
        (axes[1], M_te, f"Test (acc={acc_te:.4f})"),
    ]:
        Mn = M / M.sum(1, keepdims=True)
        im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(K))
        ax.set_xticklabels(SHORT, rotation=30, ha="right")
        ax.set_yticks(range(K))
        ax.set_yticklabels(SHORT)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(K):
            for j in range(K):
                txt = f"{M[i, j]}\n({Mn[i, j] * 100:.1f}%)"
                color = "white" if Mn[i, j] > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def plot_per_class(rows, out):
    K = len(rows)
    precs = [r[0] for r in rows]
    recs = [r[1] for r in rows]
    f1s = [r[2] for r in rows]
    x = np.arange(K)
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, precs, w, label="Precision", color="#4C72B0")
    ax.bar(x, recs, w, label="Recall", color="#55A868")
    ax.bar(x + w, f1s, w, label="F1", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=20)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1 (Test)")
    for i, v in enumerate(f1s):
        ax.text(i + w, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds",   default="artifacts/preds.npz")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(args.preds)
    te_pred, yte = d["te_pred"], d["yte"]
    va_pred, yva = d["va_pred"], d["yva"]
    K = int(max(yte.max(), te_pred.max())) + 1

    M_te = confusion(yte, te_pred, K)
    M_va = confusion(yva, va_pred, K)
    acc_te, rows_te, mac_te = per_class_metrics(M_te)
    acc_va, _, _ = per_class_metrics(M_va)

    print(f"VAL  acc: {acc_va:.4f}")
    print(f"TEST acc: {acc_te:.4f}")
    print(f"Macro F1: {mac_te[2]:.4f}")
    print()
    print(f"{'class':<25} {'P':<8} {'R':<8} {'F1':<8} {'support':<8}")
    for i, (p, r, f, s) in enumerate(rows_te):
        print(f"{LABELS[i]:<25} {p:<8.4f} {r:<8.4f} {f:<8.4f} {s:<8}")

    report = {
        "test_accuracy": float(acc_te),
        "val_accuracy": float(acc_va),
        "macro_precision": float(mac_te[0]),
        "macro_recall": float(mac_te[1]),
        "macro_f1": float(mac_te[2]),
        "per_class": [
            {
                "name": LABELS[i],
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for i, (p, r, f, s) in enumerate(rows_te)
        ],
        "confusion_matrix_test": M_te.tolist(),
        "confusion_matrix_val": M_va.tolist(),
    }
    with open(os.path.join(args.out_dir, "final_metrics.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plot_confusion(M_va, M_te, acc_va, acc_te,
                   os.path.join(args.out_dir, "confusion_matrix.png"))
    plot_per_class(rows_te, os.path.join(args.out_dir, "per_class_metrics.png"))
    print(f"\nSaved metrics and figures to {args.out_dir}/")


if __name__ == "__main__":
    main()
