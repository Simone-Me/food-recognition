"""Visual checks: images with their mask overlaid and the names of the classes present.

    python -m scripts.show_samples --config configs/a_lite0.yaml --n 16                  # raw samples (label table check)
    python -m scripts.show_samples --config configs/a_lite0.yaml --n 16 --augmented      # AFTER augmentation (alignment)
    python -m scripts.show_samples --config configs/base.yaml --class-name pizza --n 6   # gallery of one class

The --augmented view shows the images exactly as the model sees them (de-normalised for display),
with the multi-hot target computed from the transformed mask.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image

from src.data import (INGREDIENT_IDS, foodseg_dataset, foodseg_root, load_foodseg_labels, load_split, parse_ids,
                      read_index)
from src.utils import ROOT, load_config, need

CMAP = plt.get_cmap("tab20", 104)


def overlay(ax, img: np.ndarray, mask: np.ndarray, title: str) -> None:
    ax.imshow(img)
    colored = CMAP(mask % 104)
    colored[..., 3] = np.where(mask > 0, 0.5, 0.0)
    ax.imshow(colored)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--augmented", action="store_true")
    p.add_argument("--class-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    cfg = load_config(a.config)
    names = load_foodseg_labels(ROOT / need(cfg, "data.labels"))
    keys = load_split(need(cfg, "data.foodseg_split"))[a.split]
    if a.class_name:
        cid = names.index(a.class_name)
        index = read_index(foodseg_root(cfg) / "index.csv")
        keys = [k for k in keys if cid in parse_ids(index[k]["classes"])]
    rng = np.random.default_rng(a.seed)
    keys = list(rng.choice(keys, size=min(a.n, len(keys)), replace=False))

    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(keys) / cols)), cols, figsize=(4 * cols, 4 * np.ceil(len(keys) / cols)))
    axes = np.atleast_1d(axes).ravel()
    if a.augmented:
        dc = timm.data.resolve_data_config({}, model=timm.create_model(need(cfg, "model.name"), pretrained=False))
        ds = foodseg_dataset(cfg, a.split, dc["mean"], dc["std"], train=True, return_mask=True)
        ds.keys = keys
        mean, std = torch.tensor(dc["mean"])[:, None, None], torch.tensor(dc["std"])[:, None, None]
        for ax, i in zip(axes, range(len(keys))):
            x, y, m = ds[i]
            img = (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            present = [names[INGREDIENT_IDS[k]] for k in torch.nonzero(y).flatten().tolist()]
            overlay(ax, img, m.numpy(), "target: " + ", ".join(present))
    else:
        root = foodseg_root(cfg)
        for ax, k in zip(axes, keys):
            img = np.array(Image.open(root / "images" / f"{k}.jpg"))
            m = np.array(Image.open(root / "masks" / f"{k}.png"))
            overlay(ax, img, m, f"{k}: " + ", ".join(names[c] for c in np.unique(m) if c))
    for ax in axes[len(keys):]:
        ax.axis("off")
    out = ROOT / "results" / "eda" / f"samples_{a.split}{'_aug' if a.augmented else ''}{'_' + a.class_name if a.class_name else ''}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
