"""Exploratory data analysis of FoodSeg103 (train split) -> figures and tables in results/eda/.

    python -m scripts.eda --config configs/base.yaml

Under every figure, write ONE sentence in the journal: what it implies for the model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data import INGREDIENT_IDS, NUM_FOODSEG_IDS, foodseg_root, load_foodseg_labels, load_split, parse_ids, read_index
from src.utils import ROOT, load_config, need


def main(cfg: dict) -> None:
    out = ROOT / "results" / "eda"
    out.mkdir(parents=True, exist_ok=True)
    names = load_foodseg_labels(ROOT / need(cfg, "data.labels"))
    index = read_index(foodseg_root(cfg) / "index.csv")
    keys = load_split(need(cfg, "data.foodseg_split"))["train"]

    img_count = np.zeros(NUM_FOODSEG_IDS, int)
    pix_count = np.zeros(NUM_FOODSEG_IDS, np.int64)
    areas = {c: [] for c in INGREDIENT_IDS}       # area fraction when present
    n_ingr, bg_frac, pairs = [], [], Counter()
    for k in tqdm(keys, desc="masks"):
        m = np.array(Image.open(foodseg_root(cfg) / "masks" / f"{k}.png"))
        bc = np.bincount(m.ravel(), minlength=NUM_FOODSEG_IDS)
        present = [c for c in np.flatnonzero(bc) if c != 0]
        img_count[present] += 1
        pix_count += bc
        for c in present:
            areas[c].append(bc[c] / m.size)
        n_ingr.append(len(present))
        bg_frac.append(bc[0] / m.size)
        pairs.update(combinations(sorted(present), 2))

    df = pd.DataFrame({
        "class_id": INGREDIENT_IDS, "name": [names[c] for c in INGREDIENT_IDS],
        "train_images": img_count[1:], "train_pixels": pix_count[1:],
        "area_median": [np.median(areas[c]) if areas[c] else np.nan for c in INGREDIENT_IDS],
        "area_p5": [np.percentile(areas[c], 5) if areas[c] else np.nan for c in INGREDIENT_IDS],
        "relevant_for_covertcook": "",   # fill by hand: yes / no
    }).sort_values("train_images", ascending=False)
    df.to_csv(out / "classes.csv", index=False)

    cum = np.cumsum(df.train_images.values) / df.train_images.sum()
    print(f"classes covering 80% of occurrences: {int(np.searchsorted(cum, 0.8)) + 1} / {len(df)}")
    print(f"classes with < 20 train images: {(df.train_images < 20).sum()}")
    print(f"classes whose median area < 1% of the image: {(df.area_median < 0.01).sum()}")

    fig, ax = plt.subplots(2, 1, figsize=(18, 9))
    ax[0].bar(df.name, df.train_images); ax[0].set_yscale("log"); ax[0].set_title("images per class (train)")
    s = df.sort_values("train_pixels", ascending=False)
    ax[1].bar(s.name, s.train_pixels); ax[1].set_yscale("log"); ax[1].set_title("pixels per class (train)")
    for a in ax:
        a.tick_params(axis="x", rotation=90, labelsize=6)
    fig.tight_layout(); fig.savefig(out / "1_class_frequency.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].hist(n_ingr, bins=range(0, max(n_ingr) + 2)); ax[0].set_title("ingredients per image")
    ax[1].hist(bg_frac, bins=30); ax[1].set_title("background fraction per image")
    w = [int(index[k]["orig_w"]) for k in keys]; h = [int(index[k]["orig_h"]) for k in keys]
    ax[2].scatter(w, h, s=2, alpha=0.3); ax[2].set_title("original image sizes (w, h)")
    fig.tight_layout(); fig.savefig(out / "2_per_image.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(18, 4))
    a = df.sort_values("area_median")
    ax.bar(a.name, a.area_median, label="median"); ax.bar(a.name, a.area_p5, label="p5")
    ax.set_yscale("log"); ax.axhline(0.01, color="r", ls="--", label="1% of the image")
    ax.tick_params(axis="x", rotation=90, labelsize=6); ax.legend(); ax.set_title("area of an ingredient when present")
    fig.tight_layout(); fig.savefig(out / "3_area.png", dpi=120); plt.close(fig)

    top = pd.DataFrame([(names[a], names[b], n) for (a, b), n in pairs.most_common(20)],
                       columns=["ingredient_a", "ingredient_b", "images"])
    top.to_csv(out / "4_cooccurrences_top20.csv", index=False)
    print(top.to_string(index=False))
    print(f"figures and tables in {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    main(load_config(p.parse_args().config))
