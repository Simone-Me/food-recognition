"""Reproducible train / val / test splits, saved as lists of KEYS (not row indices).

    python -m scripts.make_splits --config configs/base.yaml --dataset foodseg
    python -m scripts.make_splits --config configs/base.yaml --dataset food101     # part D

test = the HF "validation" split (= official test set, comparable with the papers). Never used to choose anything.
val  = a fraction of the HF "train" split. FoodSeg103: multi-label iterative stratification (Sechidis 2011).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data import INGREDIENT_IDS, food101_root, foodseg_root, parse_ids, read_index
from src.utils import ROOT, load_config, need, save_json


def foodseg_splits(cfg: dict) -> dict[str, list[str]]:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    index = read_index(foodseg_root(cfg) / "index.csv")
    train_keys = sorted(k for k, r in index.items() if r["hf_split"] == "train")
    test_keys = sorted(k for k, r in index.items() if r["hf_split"] == "validation")
    # Binary matrix (n_images, 103), background excluded (present almost everywhere: no information).
    Y = np.zeros((len(train_keys), len(INGREDIENT_IDS)), dtype=np.int8)
    for i, k in enumerate(train_keys):
        for c in parse_ids(index[k]["classes"]):
            if c:
                Y[i, c - 1] = 1
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=need(cfg, "prep.val_fraction"),
                                            random_state=need(cfg, "prep.split_seed"))
    tr, va = next(msss.split(np.zeros(len(train_keys)), Y))
    splits = {"train": sorted(train_keys[i] for i in tr), "val": sorted(train_keys[i] for i in va),
              "test": test_keys}

    # Images per class in each split -> results/split_counts.csv
    def counts(keys):
        c = np.zeros(len(INGREDIENT_IDS), dtype=int)
        for k in keys:
            for cid in parse_ids(index[k]["classes"]):
                if cid:
                    c[cid - 1] += 1
        return c

    df = pd.DataFrame({"class_id": INGREDIENT_IDS, **{s: counts(v) for s, v in splits.items()}})
    df.to_csv(ROOT / "results" / "split_counts.csv", index=False)
    print(f"classes with 0 val images: {(df.val == 0).sum()} | with < 5: {(df.val < 5).sum()}")
    print(f"classes with < 20 train images: {(df.train < 20).sum()}")
    return splits


def food101_splits(cfg: dict) -> dict[str, list[str]]:
    from sklearn.model_selection import train_test_split

    index = read_index(food101_root(cfg) / "index.csv")
    train_keys = sorted(k for k, r in index.items() if r["hf_split"] == "train")
    test_keys = sorted(k for k, r in index.items() if r["hf_split"] == "validation")
    labels = [index[k]["label"] for k in train_keys]
    tr, va = train_test_split(train_keys, test_size=need(cfg, "prep.food101_val_fraction"), stratify=labels,
                              random_state=need(cfg, "prep.split_seed"))
    return {"train": sorted(tr), "val": sorted(va), "test": test_keys}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--dataset", choices=["foodseg", "food101"], required=True)
    a = p.parse_args()
    c = load_config(a.config)
    s = foodseg_splits(c) if a.dataset == "foodseg" else food101_splits(c)
    out = ROOT / need(c, f"data.{a.dataset}_split")
    save_json(s, out)
    all_keys = [k for v in s.values() for k in v]
    assert len(all_keys) == len(set(all_keys)), "a key is in two splits!"
    print({k: len(v) for k, v in s.items()}, "->", out)
