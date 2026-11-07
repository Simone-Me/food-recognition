"""Inference with PyTorch OR ONNX -> one .npz of raw predictions. Metrics are computed elsewhere (src.metrics).

    python -m src.predict --run a_lite0 --split val                        # PyTorch, best.pt
    python -m src.predict --run a_lite0 --split test --onnx model_int8_minmax_200_pc.onnx

Output: results/preds/<run>__<variant>__<split>.npz with, depending on the model:
    ingr_keys, ingr_probs (N,K), ingr_targets (N,K), ingr_logits (PyTorch only)
    dish_keys, dish_probs (N,D), dish_targets (N,),  dish_logits (PyTorch only)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import food101_dataset, foodseg_dataset, make_loader, own_dataset
from src.utils import ROOT, get_device, load_config, load_json


@torch.no_grad()
def torch_outputs(model, loader, device) -> tuple[np.ndarray, np.ndarray | None, list[np.ndarray]]:
    """-> (ingredient logits, dish logits or None, [targets of each batch position])."""
    model.eval()
    ingr, dish, targets = [], [], None
    for batch in loader:
        x = batch[0].to(device, memory_format=torch.channels_last)
        out = model(x)
        if isinstance(out, tuple):
            ingr.append(out[0].float().cpu())
            dish.append(out[1].float().cpu())
        else:
            ingr.append(out.float().cpu())
        targets = [[] for _ in batch[1:]] if targets is None else targets
        for t, b in zip(targets, batch[1:]):
            t.append(b)
    targets = [torch.cat(t).numpy() for t in targets]
    return torch.cat(ingr).numpy(), (torch.cat(dish).numpy() if dish else None), targets


def onnx_outputs(onnx_path: str | Path, loader) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """Same loader, same preprocessing as PyTorch; the graph has batch = 1 so we feed images one by one."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]
    outs = {n: [] for n in names}
    targets = None
    for batch in loader:
        for img in batch[0].numpy():
            for n, v in zip(names, sess.run(None, {in_name: img[None].astype(np.float32)})):
                outs[n].append(v[0])
        targets = [[] for _ in batch[1:]] if targets is None else targets
        for t, b in zip(targets, batch[1:]):
            t.append(b)
    return {n: np.stack(v) for n, v in outs.items()}, [torch.cat(t).numpy() for t in targets]


def sigmoid(z, t):
    return 1 / (1 + np.exp(-z / t))


def softmax(z, t):
    z = z / t
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def run_split(run: str, split: str, onnx_file: str | None) -> Path:
    rdir = ROOT / "models" / run
    cfg = load_config(rdir / "config.yaml")
    info = load_json(rdir / "run_info.json")
    mean, std, multitask = info["mean"], info["std"], info["num_dishes"] > 0
    post_path = rdir / "postprocess.json"
    post = load_json(post_path) if post_path.exists() else {"t_ingr": 1.0, "t_dish": 1.0}

    # Loaders: FoodSeg for the ingredient head, Food-101 for the dish head, or the personal set for both.
    jobs = []   # (head, loader, keys)
    if split == "own":
        ds = own_dataset(cfg, mean, std, info.get("dish_names"))
        jobs.append(("own", make_loader(ds, cfg, shuffle=False), ds.keys))
    else:
        ds = foodseg_dataset(cfg, split, mean, std, train=False)
        jobs.append(("ingr", make_loader(ds, cfg, shuffle=False), ds.keys))
        if multitask:
            ds = food101_dataset(cfg, split, mean, std, train=False)
            jobs.append(("dish", make_loader(ds, cfg, shuffle=False), ds.keys))

    res: dict[str, np.ndarray] = {}
    if onnx_file is None:
        from src.models import load_checkpoint

        device = get_device()
        model, _ = load_checkpoint(rdir / "best.pt", device)
        model = model.to(memory_format=torch.channels_last)
        variant = "torch"
        for head, loader, keys in jobs:
            ingr_logits, dish_logits, targets = torch_outputs(model, loader, device)
            if head in ("ingr", "own"):
                res.update(ingr_keys=np.array(keys), ingr_logits=ingr_logits,
                           ingr_probs=sigmoid(ingr_logits, post["t_ingr"]), ingr_targets=targets[0])
            if head in ("dish", "own") and dish_logits is not None:
                res.update(dish_keys=np.array(keys), dish_logits=dish_logits,
                           dish_probs=softmax(dish_logits, post["t_dish"]), dish_targets=targets[-1])
    else:
        onnx_path = Path(onnx_file) if Path(onnx_file).exists() else rdir / onnx_file
        variant = onnx_path.stem
        for head, loader, keys in jobs:
            outs, targets = onnx_outputs(onnx_path, loader)
            if head in ("ingr", "own"):
                res.update(ingr_keys=np.array(keys), ingr_probs=outs["ingredients"], ingr_targets=targets[0])
            if head in ("dish", "own") and "dish" in outs:
                res.update(dish_keys=np.array(keys), dish_probs=outs["dish"], dish_targets=targets[-1])

    out = ROOT / "results" / "preds" / f"{run}__{variant}__{split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    print(f"saved {out}  ({', '.join(f'{k}{tuple(v.shape)}' for k, v in res.items())})")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="folder name in models/")
    p.add_argument("--split", required=True, choices=["val", "test", "own"])
    p.add_argument("--onnx", default=None, help="ONNX file (in models/<run>/ or a path). Omit for PyTorch.")
    args = p.parse_args()
    run_split(args.run, args.split, args.onnx)


if __name__ == "__main__":
    main()
