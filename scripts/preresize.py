"""Copy a dataset from the Hugging Face cache to data/<name>/, short side resized once and for all.

    python -m scripts.preresize --config configs/base.yaml --dataset foodseg
    python -m scripts.preresize --config configs/base.yaml --dataset food101      # part D (~100k images)

FoodSeg103 -> data/foodseg/images/<key>.jpg, masks/<key>.png (PNG = lossless; resized with NEAREST), index.csv
Food-101   -> data/food101/images/<key>.jpg, index.csv
key = "<hf_split>_<id>" so that keys stay unique even if the two HF splits reuse the same ids.
Also runs the dataset integrity checks of the TP (sizes, mask values, classes_on_image vs mask).
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data import FOOD101_HF_ID, FOODSEG_HF_ID, NUM_FOODSEG_IDS
from src.utils import load_config, need, path_from


def resize_short(img: Image.Image, short: int, resample) -> Image.Image:
    w, h = img.size
    s = short / min(w, h)
    return img if s >= 1 else img.resize((round(w * s), round(h * s)), resample)


def write_index(path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def prep_foodseg(cfg: dict) -> None:
    from datasets import load_dataset

    short = need(cfg, "prep.foodseg_short_side")
    out = path_from(cfg, "data") / "foodseg"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    ds = load_dataset(FOODSEG_HF_ID)
    print(ds)
    rows, checks = [], {"size_mismatch": 0, "invalid_values": 0, "classes_on_image_differs": 0, "rgb_mask": 0}
    for hf_split in ds:
        for ex in tqdm(ds[hf_split], desc=hf_split):
            key = f"{hf_split}_{ex['id']}"
            img = ex["image"].convert("RGB")
            m = np.array(ex["label"])
            if m.ndim == 3:            # mask stored as RGB: the class id is in every channel
                checks["rgb_mask"] += 1
                m = m[..., 0]
            if m.max() >= NUM_FOODSEG_IDS:
                checks["invalid_values"] += 1
            mask = Image.fromarray(m.astype(np.uint8))
            if mask.size != img.size:
                checks["size_mismatch"] += 1
                mask = mask.resize(img.size, Image.NEAREST)
            if set(np.unique(m).tolist()) - {0} != set(ex["classes_on_image"]) - {0}:
                checks["classes_on_image_differs"] += 1

            orig_w, orig_h = img.size
            img = resize_short(img, short, Image.BICUBIC)
            mask = resize_short(mask, short, Image.NEAREST)
            img.save(out / "images" / f"{key}.jpg", quality=95)
            mask.save(out / "masks" / f"{key}.png")
            classes = np.unique(np.array(mask)).tolist()   # recomputed AFTER the resize
            rows.append({"key": key, "hf_split": hf_split, "hf_id": ex["id"], "orig_w": orig_w, "orig_h": orig_h,
                         "w": img.size[0], "h": img.size[1], "classes": ";".join(map(str, classes)),
                         "classes_hf": ";".join(map(str, sorted(ex["classes_on_image"])))})
    assert len({r["key"] for r in rows}) == len(rows), "keys are not unique: the `id` field is not an identifier"
    write_index(out / "index.csv", rows)
    print(f"{len(rows)} images written to {out}")
    print("integrity checks:", checks)


def prep_food101(cfg: dict) -> None:
    from datasets import load_dataset

    short = need(cfg, "prep.food101_short_side")
    out = path_from(cfg, "data") / "food101"
    (out / "images").mkdir(parents=True, exist_ok=True)
    ds = load_dataset(FOOD101_HF_ID)
    print(ds)
    names = ds["train"].features["label"].names
    rows = []
    for hf_split in ds:
        for i, ex in enumerate(tqdm(ds[hf_split], desc=hf_split)):
            key = f"{hf_split}_{i:06d}"
            path = out / "images" / f"{key}.jpg"
            img = ex["image"]
            if not path.exists():   # resumable: 100k images take a while
                resize_short(img.convert("RGB"), short, Image.BICUBIC).save(path, quality=92)
            rows.append({"key": key, "hf_split": hf_split, "label": ex["label"], "label_name": names[ex["label"]],
                         "orig_w": img.size[0], "orig_h": img.size[1]})
    write_index(out / "index.csv", rows)
    print(f"{len(rows)} images written to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--dataset", choices=["foodseg", "food101"], required=True)
    a = p.parse_args()
    c = load_config(a.config)
    prep_foodseg(c) if a.dataset == "foodseg" else prep_food101(c)
