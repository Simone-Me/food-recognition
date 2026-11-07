"""Error analysis from a prediction file (no inference).

    python -m scripts.error_analysis results/preds/a_lite0__torch__test.npz

-> results/errors/<variant>_classes.csv (AP, F1, train count per class, worst first)
   results/errors/<variant>_confusions.csv (false positive of X while Y is present)
   + false-positive rate of allergen-like classes (a "peanut" that is not there is a SERIOUS error)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import INGREDIENT_IDS, load_foodseg_labels
from src.metrics import ap_per_class, load_postprocess, run_of
from src.utils import ROOT

ALLERGEN_LIKE = ["peanut", "almond", "cashew", "walnut", "shrimp", "crab", "shellfish", "egg", "milk",
                 "cheese butter", "soy", "fish"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("preds", type=Path)
    a = p.parse_args()
    d = np.load(a.preds)
    names = load_foodseg_labels(ROOT / "data" / "labels" / "foodseg103.json")
    cls = [names[i] for i in INGREDIENT_IDS]
    post = load_postprocess(run_of(a.preds))
    probs, y = d["ingr_probs"], d["ingr_targets"].astype(bool)
    thr = np.asarray(post["thresholds"]) if post.get("thresholds") else np.full(probs.shape[1], 0.5)
    pred = probs >= thr

    tp, fp, fn = (pred & y).sum(0), (pred & ~y).sum(0), (~pred & y).sum(0)
    df = pd.DataFrame({"class": cls, "ap": ap_per_class(probs, y.astype(float)), "test_pos": y.sum(0),
                       "train_count": post.get("train_counts") or np.nan,
                       "f1": 2 * tp / np.maximum(2 * tp + fp + fn, 1), "fp": fp, "fn": fn}).sort_values("ap")
    out = ROOT / "results" / "errors"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{a.preds.stem}_classes.csv", index=False)
    print("10 worst classes by AP:\n", df.head(10).to_string(index=False))

    # Confusions: how often X is predicted (wrongly) on images that contain Y.
    fp_mask = pred & ~y
    conf = fp_mask.T.astype(int) @ y.astype(int)          # (K, K): [x, y] = #images with FP x while y present
    np.fill_diagonal(conf, 0)
    pairs = [(cls[i], cls[j], conf[i, j]) for i, j in zip(*np.unravel_index(np.argsort(-conf, axis=None)[:30], conf.shape))]
    pd.DataFrame(pairs, columns=["false_positive", "while_present", "images"]).to_csv(
        out / f"{a.preds.stem}_confusions.csv", index=False)
    print("\ntop confusions (FP of X while Y present):")
    for x_, y_, n in pairs[:10]:
        print(f"  {x_:>20} <- {y_:<20} {n}")

    print("\nallergen-like classes — false positives / negatives in the test images:")
    for n in ALLERGEN_LIKE:
        k = cls.index(n)
        print(f"  {n:>14}: FP={fp[k]:4d}  FN={fn[k]:4d}  (positives={y[:, k].sum()})")


if __name__ == "__main__":
    main()
