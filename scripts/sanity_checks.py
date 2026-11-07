"""Step A7, checks to run BEFORE a real training (Karpathy, "A Recipe for Training Neural Networks").

    python -m scripts.sanity_checks configs/a_lite0.yaml --check init      # loss at step 0 vs expected value
    python -m scripts.sanity_checks configs/a_lite0.yaml --check overfit   # 1 batch, no augmentation, 200 steps
    python -m scripts.sanity_checks configs/a_lite0.yaml --check lr        # LR range test -> results/lr_range_<model>.png
"""
from __future__ import annotations

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.metrics import map_macro
from src.models import LN2, entropy_bce, param_groups
from src.train import batch_stream, setup, step_loss
from src.utils import ROOT, load_config, need, set_key


def check_init(S) -> None:
    S.model.train()
    batch = next(batch_stream(S))
    with torch.no_grad():
        loss, _ = step_loss(S.model, batch, S)
    p = S.train_counts / len(S.ds_train)
    print(f"loss at init                : {loss.item():.4f}")
    print(f"expected, BCE with bias = 0 : {LN2:.4f}  (ln 2)")
    print(f"expected, BCE with bias = prior (train.bias_init: true): {entropy_bce(p):.4f}")
    print("ASL / pos_weight change the scale: compare the TREND between configs, not the absolute value.")
    if S.multitask:
        print(f"dish CE expected at init    : {math.log(len(S.dish_names)):.4f} (ln D), weighted by dish_loss_weight")


def check_overfit(cfg: dict, steps: int = 200) -> None:
    set_key(cfg, "train.augment", "none")
    S = setup(cfg)
    batch = next(batch_stream(S))
    opt = torch.optim.AdamW(S.model.parameters(), lr=need(cfg, "train.lr"))
    for i in range(steps):
        S.model.train()
        loss, _ = step_loss(S.model, batch, S)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 20 == 0 or i == steps - 1:
            print(f"step {i:4d}  loss {loss.item():.5f}")
    S.model.eval()
    with torch.no_grad():
        x, y = batch["ingr"]
        out = S.model(x.to(S.device, memory_format=torch.channels_last))
        logits = (out[0] if isinstance(out, tuple) else out).float().cpu().numpy()
    print(f"mAP on the memorised batch: {map_macro(1 / (1 + np.exp(-logits)), y.numpy()):.3f} (expected ~1.0)")


def check_lr(cfg: dict, steps: int = 200, lr_min: float = 1e-6, lr_max: float = 1.0) -> None:
    S = setup(cfg)
    opt = torch.optim.AdamW(param_groups(S.model, lr_min, 1.0, 0.0))
    gamma = (lr_max / lr_min) ** (1 / steps)
    stream, lrs, losses, smooth = batch_stream(S), [], [], None
    for i in range(steps):
        lr = lr_min * gamma ** i
        for g in opt.param_groups:
            g["lr"] = lr
        S.model.train()
        with torch.autocast(S.device.type, dtype=torch.float16, enabled=need(cfg, "train.amp")):
            loss, _ = step_loss(S.model, next(stream), S)
        opt.zero_grad()
        loss.backward()
        opt.step()
        smooth = loss.item() if smooth is None else 0.9 * smooth + 0.1 * loss.item()
        lrs.append(lr)
        losses.append(smooth)
        if not math.isfinite(smooth) or smooth > 4 * min(losses):
            break
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lrs, losses)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("smoothed loss")
    ax.set_title(f"LR range test — {need(cfg, 'model.name')}")
    out = ROOT / "results" / f"lr_range_{need(cfg, 'model.name')}.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}. Pick an LR about 10x below the point where the loss is lowest / starts to rise.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--check", choices=["init", "overfit", "lr"], required=True)
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args()
    c = load_config(a.config, a.set)
    set_key(c, "mlflow.enabled", False)
    {"init": lambda: check_init(setup(c)), "overfit": lambda: check_overfit(c), "lr": lambda: check_lr(c)}[a.check]()
