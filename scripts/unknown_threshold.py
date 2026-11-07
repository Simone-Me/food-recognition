"""Open world: choose the "not recognised" operating point on the PERSONAL set (food + non-food photos).

    python -m src.predict --run d_lite0 --split own
    python -m scripts.unknown_threshold results/preds/d_lite0__torch__own.npz

Ingredients (A): an image is "not recognised" when no class passes its threshold -> prints how often that
happens on food photos (bad) and on non-food photos (good).
Dish (D): for each candidate threshold on the max probability: coverage (share of food photos answered),
top-1 among the answered ones, and share of non-food photos correctly rejected.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.metrics import load_postprocess, run_of


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("preds", type=Path)
    a = p.parse_args()
    d = np.load(a.preds)
    post = load_postprocess(run_of(a.preds))

    y = d["ingr_targets"]
    food = y.sum(1) > 0
    thr = np.asarray(post["thresholds"]) if post.get("thresholds") else np.full(y.shape[1], 0.5)
    empty = (d["ingr_probs"] >= thr).sum(1) == 0
    print(f"[ingredients] food photos with nothing recognised  : {empty[food].mean():.2%} ({food.sum()} photos)")
    print(f"[ingredients] non-food photos correctly 'nothing'  : {empty[~food].mean():.2%} ({(~food).sum()} photos)")

    if "dish_probs" not in d:
        return
    probs, lab = d["dish_probs"], d["dish_targets"]
    conf, pred = probs.max(1), probs.argmax(1)
    is_dish = lab >= 0
    print("\n[dish] thr  coverage(dish photos)  top1(answered)  non-dish rejected")
    for t in np.arange(0.0, 0.95, 0.05):
        ans = conf >= t
        cov = ans[is_dish].mean() if is_dish.any() else np.nan
        acc = (pred[ans & is_dish] == lab[ans & is_dish]).mean() if (ans & is_dish).any() else np.nan
        rej = (~ans[~is_dish]).mean() if (~is_dish).any() else np.nan
        print(f"       {t:.2f}  {cov:20.2%}  {acc:14.2%}  {rej:17.2%}")


if __name__ == "__main__":
    main()
