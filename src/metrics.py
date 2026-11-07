"""Metrics computed FROM SAVED PREDICTIONS (results/preds/*.npz), identical for PyTorch / ONNX fp32 / int8.

CLI:
    python -m src.metrics results/preds/a_lite0__torch__test.npz
-> prints the report and appends one row to results/metrics.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.stats import bootstrap_ci
from src.utils import BLANK, ROOT, load_config, load_json

# ---------------------------------------------------------------- multi-label (ingredients)


def average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    """AP of one class, same definition as sklearn (ties handled as one threshold). nan if no positive."""
    n_pos = y.sum()
    if n_pos == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    s, t = scores[order], y[order]
    tp, fp = np.cumsum(t), np.cumsum(1 - t)
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]   # last index of each block of tied scores
    tp, fp = tp[last], fp[last]
    precision, recall = tp / (tp + fp), tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def ap_per_class(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.array([average_precision(probs[:, k], targets[:, k]) for k in range(probs.shape[1])])


def map_macro(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean AP over classes that have at least one positive (classes without positives are skipped)."""
    return float(np.nanmean(ap_per_class(probs, targets)))


def map_micro(probs: np.ndarray, targets: np.ndarray) -> float:
    return average_precision(probs.ravel(), targets.ravel())


def f1_at_thresholds(probs: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> dict:
    pred = probs >= thresholds[None, :]
    t = targets.astype(bool)
    tp, fp, fn = (pred & t).sum(0), (pred & ~t).sum(0), (~pred & t).sum(0)
    p_micro = tp.sum() / max(tp.sum() + fp.sum(), 1)
    r_micro = tp.sum() / max(tp.sum() + fn.sum(), 1)
    f1_k = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    has_pos = t.sum(0) > 0
    return {
        "f1_micro": float(2 * p_micro * r_micro / max(p_micro + r_micro, 1e-12)),
        "f1_macro": float(f1_k[has_pos].mean()),
        "precision_micro": float(p_micro),
        "recall_micro": float(r_micro),
        "unknown_rate": float((pred.sum(1) == 0).mean()),   # images where no ingredient is shown
    }


def best_thresholds(probs: np.ndarray, targets: np.ndarray, default: float = 0.5) -> np.ndarray:
    """Per-class threshold that maximises F1 ON THE VALIDATION SET. Classes without positives keep `default`."""
    from sklearn.metrics import precision_recall_curve

    thr = np.full(probs.shape[1], default, dtype=np.float32)
    for k in range(probs.shape[1]):
        if targets[:, k].sum() == 0:
            continue
        p, r, t = precision_recall_curve(targets[:, k], probs[:, k])
        f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
        thr[k] = t[int(np.argmax(f1))]
    return thr


def ece_binary(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error over all (image, class) pairs."""
    p, y = probs.ravel(), targets.ravel()
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


# ---------------------------------------------------------------- single-label (dish)


def topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(1).mean())


def ece_multiclass(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf, correct = probs.max(1), (probs.argmax(1) == labels).astype(float)
    return ece_binary(conf, correct, n_bins)


def macro_f1_dish(probs: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(labels, probs.argmax(1), average="macro"))


# ---------------------------------------------------------------- temperature scaling


def fit_temperature(logits: np.ndarray, targets: np.ndarray, kind: str) -> float:
    """One scalar T > 0 minimising the NLL of logits / T on the VALIDATION set (Guo et al., 2017).
    kind = "multilabel" (BCE) or "multiclass" (CE)."""
    z = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(targets)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        zt = z / log_t.exp()
        loss = (F.binary_cross_entropy_with_logits(zt, y.double()) if kind == "multilabel"
                else F.cross_entropy(zt, y.long()))
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp())
    if not 0.05 <= t <= 20:
        # A huge T means "the model is no better than uniform on val": a bug or an untrained model.
        print(f"[warning] fitted temperature {t:.3g} is out of [0.05, 20]: clipped. Check the model!")
    return min(max(t, 0.05), 20.0)


# ---------------------------------------------------------------- reports


def ingredient_report(probs, targets, thresholds, train_counts=None, frequent_min=None, n_boot=1000) -> dict:
    point, lo, hi = bootstrap_ci(map_macro, probs, targets, n_boot)
    ap = ap_per_class(probs, targets)
    rep = {
        "n_images": len(targets),
        "map_macro": point, "map_macro_lo": lo, "map_macro_hi": hi,
        "map_micro": map_micro(probs, targets),
        "classes_without_positives": int(np.isnan(ap).sum()),
        "ece_ingr": ece_binary(probs, targets),
        **f1_at_thresholds(probs, targets, thresholds),
    }
    if train_counts is not None and frequent_min is not None:
        freq = np.asarray(train_counts) >= frequent_min
        rep["map_frequent"] = float(np.nanmean(ap[freq]))
        rep["map_rare"] = float(np.nanmean(ap[~freq]))
    return rep


def dish_report(probs, labels, unknown_min_prob=None, n_boot=1000) -> dict:
    valid = labels >= 0   # own set: -1 = "not a Food-101 dish"
    pr, y = probs[valid], labels[valid]
    top1 = lambda a, b: float((a.argmax(1) == b).mean())
    point, lo, hi = bootstrap_ci(top1, pr, y, n_boot)
    rep = {
        "n_dish": int(valid.sum()),
        "top1": point, "top1_lo": lo, "top1_hi": hi,
        "top3": topk_accuracy(pr, y, 3), "top5": topk_accuracy(pr, y, 5),
        "dish_macro_f1": macro_f1_dish(pr, y), "ece_dish": ece_multiclass(pr, y),
    }
    if unknown_min_prob is not None:
        rep["dish_unknown_rate"] = float((probs.max(1) < unknown_min_prob).mean())
    return rep


# ---------------------------------------------------------------- helpers shared with other modules


def run_of(pred_path: Path) -> str:
    """results/preds/<run>__<variant>__<split>.npz -> <run>"""
    return Path(pred_path).stem.split("__")[0]


def load_postprocess(run: str) -> dict:
    path = ROOT / "models" / run / "postprocess.json"
    if not path.exists():
        print(f"[warning] {path} not found (run src.calibrate first): using T=1 and thresholds=0.5")
        return {"t_ingr": 1.0, "t_dish": 1.0, "thresholds": None, "train_counts": None}
    return load_json(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("preds", type=Path)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    run, variant, split = args.preds.stem.split("__")
    ckpt_cfg = load_config(ROOT / "models" / run / "config.yaml")
    post = load_postprocess(run)
    data = np.load(args.preds)
    row = {"run": run, "variant": variant, "split": split}

    if "ingr_probs" in data:
        probs, y = data["ingr_probs"], data["ingr_targets"]
        thr = np.asarray(post["thresholds"]) if post.get("thresholds") else np.full(probs.shape[1], 0.5)
        frequent_min = ckpt_cfg["eval"].get("frequent_min_train")
        row.update(ingredient_report(probs, y, thr, post.get("train_counts"),
                                     None if frequent_min == BLANK else frequent_min, args.n_boot))
    if "dish_probs" in data:
        row.update(dish_report(data["dish_probs"], data["dish_targets"], post.get("unknown_dish_min_prob"),
                               args.n_boot))

    for k, v in row.items():
        print(f"{k:>26}: {v:.4f}" if isinstance(v, float) else f"{k:>26}: {v}")
    out = ROOT / "results" / "metrics.csv"
    df = pd.DataFrame([row])
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
        df = df.drop_duplicates(subset=["run", "variant", "split"], keep="last")
    df.to_csv(out, index=False)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
