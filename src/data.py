"""FoodSeg103 / Food-101 / personal set: datasets, targets and transforms.

Images are read from the pre-resized copy on disk written by scripts/preresize.py,
not from the Hugging Face cache: decoding large JPEGs at every epoch starves the GPU.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from src.utils import load_json, need, path_from, seed_worker

FOODSEG_HF_ID = "EduardoPacheco/FoodSeg103"
FOOD101_HF_ID = "ethz/food101"
NUM_FOODSEG_IDS = 104                              # 103 ingredients + background
BACKGROUND_ID = 0
INGREDIENT_IDS = list(range(1, NUM_FOODSEG_IDS))   # model output k <-> original id INGREDIENT_IDS[k]
AUGMENT_LEVELS = ("none", "basic", "photo", "strong")


# ---------------------------------------------------------------- labels, splits, index

def load_foodseg_labels(path: str | Path) -> list[str]:
    """data/labels/foodseg103.json -> list of names indexed by ORIGINAL id (0 = background)."""
    d = load_json(path)
    return [d[str(i)] for i in range(len(d))]


def load_split(path: str | Path) -> dict[str, list[str]]:
    return load_json(path)


def read_index(path: str | Path) -> dict[str, dict]:
    """index.csv written by preresize.py -> {key: row}."""
    with open(path, newline="", encoding="utf-8") as f:
        return {row["key"]: row for row in csv.DictReader(f)}


def parse_ids(s: str) -> list[int]:
    return [int(x) for x in s.split(";") if x != ""]


def foodseg_root(cfg: dict) -> Path:
    return path_from(cfg, "data") / "foodseg"


def food101_root(cfg: dict) -> Path:
    return path_from(cfg, "data") / "food101"


def foodseg_train_counts(cfg: dict) -> np.ndarray:
    """Number of TRAIN images containing each ingredient (K,), from the index (area > 0)."""
    index = read_index(foodseg_root(cfg) / "index.csv")
    keys = load_split(need(cfg, "data.foodseg_split"))["train"]
    counts = np.zeros(len(INGREDIENT_IDS), dtype=np.int64)
    for k in keys:
        for cid in parse_ids(index[k]["classes"]):
            if cid != BACKGROUND_ID:
                counts[cid - 1] += 1
    return counts


def food101_names(cfg: dict) -> list[str]:
    index = read_index(food101_root(cfg) / "index.csv")
    names = {int(r["label"]): r["label_name"] for r in index.values()}
    return [names[i] for i in range(len(names))]


# ---------------------------------------------------------------- targets

def mask_to_multihot(mask: torch.Tensor, class_ids: list[int], min_area: float) -> torch.Tensor:
    """mask (H, W) of original ids -> float (K,) with 1 where area(class)/(H*W) >= min_area.

    min_area = 0 means "at least one pixel", i.e. the same as classes_on_image.
    """
    area = torch.bincount(mask.flatten().long(), minlength=NUM_FOODSEG_IDS).float() / mask.numel()
    area = area[torch.as_tensor(class_ids)]
    return ((area > 0) & (area >= min_area)).float()


def ids_to_multihot(ids: list[int], class_ids: list[int]) -> torch.Tensor:
    pos = {c: k for k, c in enumerate(class_ids)}
    y = torch.zeros(len(class_ids))
    for i in ids:
        if i in pos:
            y[pos[i]] = 1.0
    return y


# ---------------------------------------------------------------- transforms

def build_transforms(level: str, resolution: int, mean, std, crop_scale_min: float = 1.0) -> v2.Compose:
    """torchvision v2 pipeline. Called as t(image) or t(image, mask): geometric ops are applied
    identically to both (masks always with NEAREST), colour ops only to the image.

    level: none (eval) | basic (crop + flip) | photo (+ colour) | strong (+ perspective, blur)
    """
    if level not in AUGMENT_LEVELS:
        raise ValueError(f"augment level must be one of {AUGMENT_LEVELS}, got {level!r}")
    if level == "none":
        # Eval / browser: resize short side then center crop. The JS code reproduces exactly this.
        ops = [v2.Resize(resolution, antialias=True), v2.CenterCrop(resolution)]
    else:
        ops = [v2.RandomResizedCrop(resolution, scale=(crop_scale_min, 1.0), antialias=True),
               v2.RandomHorizontalFlip()]
        if level in ("photo", "strong"):
            # Simulates indoor phone photos: dim light, white balance, low contrast.
            ops.append(v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.02))
        if level == "strong":
            ops += [v2.RandomPerspective(distortion_scale=0.2, p=0.3),
                    v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2)]
    return v2.Compose([
        v2.ToImage(),
        *ops,
        v2.ToDtype({tv_tensors.Image: torch.float32, "others": None}, scale=True),
        v2.Normalize(mean=list(mean), std=list(std)),
    ])


def transforms_from_cfg(cfg: dict, mean, std, train: bool) -> v2.Compose:
    res = need(cfg, "data.resolution")
    if not train:
        return build_transforms("none", res, mean, std)
    level = need(cfg, "train.augment")
    crop_min = need(cfg, "train.crop_scale_min") if level != "none" else 1.0
    return build_transforms(level, res, mean, std, crop_min)


# ---------------------------------------------------------------- datasets

class FoodSegDataset(Dataset):
    """FoodSeg103 from disk. Returns (image (3,R,R) float, multi-hot target (K,)).

    The target is computed AFTER the augmentation, from the transformed mask: an ingredient
    cut out by the random crop is not labelled as present.
    """

    def __init__(self, root: Path, keys: list[str], transform, min_area: float,
                 class_ids: list[int] = INGREDIENT_IDS, return_mask: bool = False):
        self.root = Path(root)
        self.keys = list(keys)
        self.transform = transform
        self.min_area = float(min_area)
        self.class_ids = list(class_ids)
        self.return_mask = return_mask

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int):
        key = self.keys[i]
        image = Image.open(self.root / "images" / f"{key}.jpg").convert("RGB")
        mask = tv_tensors.Mask(torch.from_numpy(np.array(Image.open(self.root / "masks" / f"{key}.png"))))
        image, mask = self.transform(image, mask)
        target = mask_to_multihot(mask, self.class_ids, self.min_area)
        if self.return_mask:
            return image.as_subclass(torch.Tensor), target, mask.as_subclass(torch.Tensor)
        return image.as_subclass(torch.Tensor), target


class Food101Dataset(Dataset):
    """Food-101 from disk. Returns (image, dish label as int64 scalar)."""

    def __init__(self, root: Path, keys: list[str], transform):
        self.root = Path(root)
        self.keys = list(keys)
        self.transform = transform
        index = read_index(self.root / "index.csv")
        self.labels = [int(index[k]["label"]) for k in self.keys]

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int):
        image = Image.open(self.root / "images" / f"{self.keys[i]}.jpg").convert("RGB")
        return self.transform(image).as_subclass(torch.Tensor), torch.tensor(self.labels[i])


class OwnDataset(Dataset):
    """Your personal photos (data/own/images + data/own/labels.csv). NEVER used for training.

    labels.csv columns: file, ingredient_ids ("58;73;9", empty = nothing), dish (Food-101 name, empty = none).
    Returns (image, ingredient multi-hot (K,), dish index or -1).
    """

    def __init__(self, root: Path, transform, dish_names: list[str] | None = None,
                 class_ids: list[int] = INGREDIENT_IDS):
        self.root = Path(root)
        self.transform = transform
        with open(self.root / "labels.csv", newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.keys = [r["file"] for r in self.rows]
        self.class_ids = class_ids
        self.dish_pos = {n: i for i, n in enumerate(dish_names or [])}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        # Phone photos carry an EXIF rotation: apply it, as the browser does.
        image = ImageOps.exif_transpose(Image.open(self.root / "images" / r["file"])).convert("RGB")
        ingr = ids_to_multihot(parse_ids(r.get("ingredient_ids", "")), self.class_ids)
        dish = self.dish_pos.get((r.get("dish") or "").strip(), -1)
        return self.transform(image).as_subclass(torch.Tensor), ingr, torch.tensor(dish)


# ---------------------------------------------------------------- builders

def foodseg_dataset(cfg: dict, split: str, mean, std, train: bool = False, return_mask: bool = False):
    keys = load_split(need(cfg, "data.foodseg_split"))[split]
    # Train targets use YOUR tau (an ablation knob); eval targets use a FIXED tau so that
    # every model is scored against the same ground truth.
    min_area = need(cfg, "data.train_min_area") if train else need(cfg, "eval.min_area")
    return FoodSegDataset(foodseg_root(cfg), keys, transforms_from_cfg(cfg, mean, std, train),
                          min_area, return_mask=return_mask)


def food101_dataset(cfg: dict, split: str, mean, std, train: bool = False):
    keys = load_split(need(cfg, "data.food101_split"))[split]
    return Food101Dataset(food101_root(cfg), keys, transforms_from_cfg(cfg, mean, std, train))


def own_dataset(cfg: dict, mean, std, dish_names: list[str] | None = None):
    return OwnDataset(path_from(cfg, "data") / "own", transforms_from_cfg(cfg, mean, std, False), dish_names)


def make_loader(ds: Dataset, cfg: dict, shuffle: bool, batch_size: int | None = None) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    workers = need(cfg, "data.num_workers")
    return DataLoader(
        ds,
        batch_size=batch_size or need(cfg, "train.batch_size"),
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )
