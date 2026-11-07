"""Part D, use the two heads to check each other (ANALYSES, not official metrics).

    python -m scripts.cross_dataset --run d_lite0

1. Ingredient profile of each Food-101 dish: mean ingredient probability on the Food-101 TEST images,
   compared with the "expected_ingredients" column of data/labels/mapping_foodseg_food101.csv.
2. Agreement of the heads on the concepts present in both datasets (pizza, hamburg/hamburger, steak, ...):
   on FoodSeg103 test images containing the ingredient, how often does the dish head say that dish?
-> results/cross_dataset_<run>_profiles.csv, results/cross_dataset_<run>_agreement.csv
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import pandas as pd

from src.models import load_checkpoint
from src.predict import torch_outputs
from src.data import food101_dataset, foodseg_dataset, load_foodseg_labels, make_loader
from src.utils import ROOT, get_device, load_config, load_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    a = p.parse_args()
    rdir = ROOT / "models" / a.run
    cfg = load_config(rdir / "config.yaml")
    info, post = load_json(rdir / "run_info.json"), load_json(rdir / "postprocess.json")
    names = load_foodseg_labels(ROOT / "data" / "labels" / "foodseg103.json")
    ingr_names = [names[i] for i in info["ingredient_ids"]]
    dishes = info["dish_names"]
    with open(ROOT / "data" / "labels" / "mapping_foodseg_food101.csv", newline="", encoding="utf-8") as f:
        mapping = list(csv.DictReader(f))

    device = get_device()
    model, _ = load_checkpoint(rdir / "best.pt", device)
    thr = np.asarray(post["thresholds"])

    # 1. ingredient head on Food-101 test images
    ds = food101_dataset(cfg, "test", info["mean"], info["std"])
    ingr_logits, _, t = torch_outputs(model, make_loader(ds, cfg, shuffle=False), device)
    probs = 1 / (1 + np.exp(-ingr_logits / post["t_ingr"]))
    rows = []
    for m in mapping:
        if not m.get("expected_ingredients"):
            continue
        d = dishes.index(m["food101"])
        sel = t[0] == d
        prof = probs[sel].mean(0)
        expected = [e.strip() for e in m["expected_ingredients"].split(";") if e.strip()]
        rows.append({"dish": m["food101"], "n": int(sel.sum()),
                     "top5_predicted": "; ".join(ingr_names[k] for k in np.argsort(-prof)[:5]),
                     **{f"freq_{e}": float((probs[sel][:, ingr_names.index(e)] >= thr[ingr_names.index(e)]).mean())
                        for e in expected}})
    pd.DataFrame(rows).to_csv(ROOT / "results" / f"cross_dataset_{a.run}_profiles.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    # 2. dish head on FoodSeg103 test images that contain a "dish-like" ingredient
    ds = foodseg_dataset(cfg, "test", info["mean"], info["std"])
    _, dish_logits, t = torch_outputs(model, make_loader(ds, cfg, shuffle=False), device)
    pred_dish = dish_logits.argmax(1)
    agree = []
    for m in mapping:
        if not m.get("foodseg"):
            continue
        k = ingr_names.index(m["foodseg"])
        sel = t[0][:, k] == 1
        d = dishes.index(m["food101"])
        agree.append({"foodseg": m["foodseg"], "food101": m["food101"], "n_images": int(sel.sum()),
                      "dish_head_agrees": float((pred_dish[sel] == d).mean()) if sel.any() else np.nan})
    pd.DataFrame(agree).to_csv(ROOT / "results" / f"cross_dataset_{a.run}_agreement.csv", index=False)
    print(pd.DataFrame(agree).to_string(index=False))


if __name__ == "__main__":
    main()
