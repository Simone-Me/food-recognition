"""Post-processing fitted ON THE VALIDATION SET ONLY: temperature(s) and per-class thresholds.

    python -m src.predict --run a_lite0 --split val      # first: PyTorch logits on val
    python -m src.calibrate --run a_lite0

Writes models/<run>/postprocess.json, read by predict (T), metrics (thresholds) and export (T + thresholds).
"""
from __future__ import annotations

import argparse

import numpy as np

from src.data import foodseg_train_counts
from src.metrics import best_thresholds, ece_binary, ece_multiclass, fit_temperature
from src.predict import sigmoid, softmax
from src.utils import ROOT, load_config, save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--unknown-dish-min-prob", type=float, default=None,
                   help="[D3] chosen on the personal set with scripts.unknown_threshold")
    p.add_argument("--unknown-ingr-min-prob", type=float, default=None,
                   help="[A9] 'not recognised' when the highest ingredient probability is below this (personal set)")
    args = p.parse_args()

    rdir = ROOT / "models" / args.run
    cfg = load_config(rdir / "config.yaml")
    val = np.load(ROOT / "results" / "preds" / f"{args.run}__torch__val.npz")
    out: dict = {"fitted_on": "val", "unknown_dish_min_prob": args.unknown_dish_min_prob,
                 "unknown_ingr_min_prob": args.unknown_ingr_min_prob}

    z, y = val["ingr_logits"], val["ingr_targets"]
    t = fit_temperature(z, y, "multilabel")
    probs = sigmoid(z, t)
    out.update(t_ingr=t, ece_ingr_before=ece_binary(sigmoid(z, 1.0), y), ece_ingr_after=ece_binary(probs, y),
               thresholds=best_thresholds(probs, y).round(4).tolist(),
               train_counts=foodseg_train_counts(cfg).tolist())

    if "dish_logits" in val:
        z, y = val["dish_logits"], val["dish_targets"]
        t = fit_temperature(z, y, "multiclass")
        out.update(t_dish=t, ece_dish_before=ece_multiclass(softmax(z, 1.0), y),
                   ece_dish_after=ece_multiclass(softmax(z, t), y))
    else:
        out["t_dish"] = 1.0

    save_json(out, rdir / "postprocess.json")
    for k, v in out.items():
        if not isinstance(v, list):
            print(f"{k:>18}: {v}")
    print(f"thresholds: min={min(out['thresholds']):.3f} median={np.median(out['thresholds']):.3f} "
          f"max={max(out['thresholds']):.3f}")
    print(f"saved {rdir / 'postprocess.json'}")


if __name__ == "__main__":
    main()
