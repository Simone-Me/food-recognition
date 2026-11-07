"""Statistical uncertainty: bootstrap CIs, paired bootstrap of a difference, McNemar.

CLI — compare two prediction files of the SAME split (e.g. fp32 vs int8):
    python -m src.stats results/preds/a_lite0__torch__test.npz results/preds/a_lite0__model_int8_minmax__test.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def bootstrap_ci(metric_fn: MetricFn, preds: np.ndarray, targets: np.ndarray,
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(point estimate, ci_low, ci_high). Resamples IMAGES (first axis) and recomputes the whole metric."""
    rng = np.random.default_rng(seed)
    n = len(targets)
    point = metric_fn(preds, targets)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(metric_fn(preds[idx], targets[idx]))
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def paired_bootstrap_delta(metric_fn: MetricFn, preds_a: np.ndarray, preds_b: np.ndarray, targets: np.ndarray,
                           n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """CI of metric(a) - metric(b), using the SAME resampled indices for a and b."""
    rng = np.random.default_rng(seed)
    n = len(targets)
    point = metric_fn(preds_a, targets) - metric_fn(preds_b, targets)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(metric_fn(preds_a[idx], targets[idx]) - metric_fn(preds_b[idx], targets[idx]))
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """p-value of McNemar on paired correctness. Only the discordant cells (a ok & b wrong, a wrong & b ok) matter."""
    a, b = correct_a.astype(bool), correct_b.astype(bool)
    table = [[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]]
    discordant = table[0][1] + table[1][0]
    return float(mcnemar(table, exact=discordant < 25).pvalue)


def _variant(path: Path) -> str:
    return path.stem


def main() -> None:
    from src.metrics import f1_at_thresholds, load_postprocess, map_macro, run_of

    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path, help="reference predictions (e.g. fp32)")
    p.add_argument("b", type=Path, help="candidate predictions (e.g. int8)")
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    A, B = np.load(args.a), np.load(args.b)
    row = {"a": _variant(args.a), "b": _variant(args.b)}
    if "ingr_probs" in A and "ingr_probs" in B:
        assert (A["ingr_keys"] == B["ingr_keys"]).all(), "the two files must cover the same images in the same order"
        y = A["ingr_targets"]
        d, lo, hi = paired_bootstrap_delta(map_macro, A["ingr_probs"], B["ingr_probs"], y, args.n_boot)
        row.update(delta_map_macro=d, delta_map_lo=lo, delta_map_hi=hi, map_significant=not (lo <= 0 <= hi))
        thr = np.asarray(load_postprocess(run_of(args.a))["thresholds"])
        f1 = lambda pr, t: f1_at_thresholds(pr, t, thr)["f1_micro"]
        d, lo, hi = paired_bootstrap_delta(f1, A["ingr_probs"], B["ingr_probs"], y, args.n_boot)
        row.update(delta_f1_micro=d, delta_f1_lo=lo, delta_f1_hi=hi)
    if "dish_probs" in A and "dish_probs" in B:
        y = A["dish_targets"]
        ca, cb = A["dish_probs"].argmax(1) == y, B["dish_probs"].argmax(1) == y
        top1 = lambda pr, t: float((pr.argmax(1) == t).mean())
        d, lo, hi = paired_bootstrap_delta(top1, A["dish_probs"], B["dish_probs"], y, args.n_boot)
        row.update(delta_top1=d, delta_top1_lo=lo, delta_top1_hi=hi, mcnemar_p=mcnemar_test(ca, cb))

    for k, v in row.items():
        print(f"{k:>20}: {v:.4f}" if isinstance(v, float) else f"{k:>20}: {v}")
    out = Path("results/comparisons.csv")
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
    print(f"appended to {out}")


if __name__ == "__main__":
    main()
