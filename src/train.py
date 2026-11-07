"""Training for part A (task: multilabel) and part D (task: multitask).

    python -m src.train configs/a_lite0.yaml
    python -m src.train configs/a_lite0.yaml --set train.loss=asl --set run_name=a_lite0_asl   # ablation

Writes models/<run_name>/{best.pt, config.yaml, run_info.json} and results/runs/<run_name>_history.csv,
and logs everything to MLflow (mlflow ui --backend-store-uri sqlite:///mlflow.db).
"""
from __future__ import annotations

import argparse
import itertools
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.data import (INGREDIENT_IDS, food101_dataset, food101_names, foodseg_dataset, foodseg_train_counts,
                      make_loader)
from src.losses import build_dish_loss, build_ingredient_loss
from src.metrics import map_macro
from src.models import (create_model, data_config, freeze_backbone, ingredient_head, init_bias_from_frequencies,
                        param_groups)
from src.predict import sigmoid, torch_outputs
from src.utils import (ROOT, flatten, get_device, git_commit, load_config, need, run_dir, save_json,
                       seed_everything, sha256)


def cycle(loader):
    """Infinite iterator over a DataLoader (a new shuffle at every pass)."""
    while True:
        yield from loader


def setup(cfg: dict) -> SimpleNamespace:
    """Everything a run needs: model, loaders, losses. Also used by scripts/sanity_checks.py."""
    seed_everything(cfg["seed"])
    device = get_device()
    multitask = need(cfg, "task") == "multitask"
    dish_names = food101_names(cfg) if multitask else []

    model = create_model(cfg, len(INGREDIENT_IDS), len(dish_names))
    dc = data_config(model)
    ds_train = foodseg_dataset(cfg, "train", dc["mean"], dc["std"], train=True)
    ds_val = foodseg_dataset(cfg, "val", dc["mean"], dc["std"], train=False)
    train_counts = foodseg_train_counts(cfg)
    if need(cfg, "train.bias_init"):
        init_bias_from_frequencies(ingredient_head(model), train_counts / len(ds_train))

    S = SimpleNamespace(
        cfg=cfg, device=device, multitask=multitask, dish_names=dish_names, data_cfg=dc,
        train_counts=train_counts, ds_train=ds_train,
        model=model.to(device, memory_format=torch.channels_last),
        crit_ingr=build_ingredient_loss(cfg, train_counts, len(ds_train)).to(device),
        loaders={"ingr_train": make_loader(ds_train, cfg, shuffle=True),
                 "ingr_val": make_loader(ds_val, cfg, shuffle=False)},
    )
    if multitask:
        bs = need(cfg, "multitask.dish_batch_size")
        S.crit_dish = build_dish_loss(cfg).to(device)
        S.loaders["dish_train"] = make_loader(food101_dataset(cfg, "train", dc["mean"], dc["std"], True), cfg, True, bs)
        S.loaders["dish_val"] = make_loader(food101_dataset(cfg, "val", dc["mean"], dc["std"], False), cfg, False, bs)
        S.w_ingr = need(cfg, "multitask.ingr_loss_weight")
        S.w_dish = need(cfg, "multitask.dish_loss_weight")
    return S


def batch_stream(S):
    """A: one FoodSeg batch per step. D: one FoodSeg batch + one Food-101 batch per step."""
    ingr = cycle(S.loaders["ingr_train"])
    if not S.multitask:
        return ({"ingr": b} for b in ingr)
    dish = cycle(S.loaders["dish_train"])
    return ({"ingr": a, "dish": b} for a, b in zip(ingr, dish))


def steps_per_epoch(S) -> int:
    return need(S.cfg, "multitask.steps_per_epoch") if S.multitask else len(S.loaders["ingr_train"])


def step_loss(model, batch, S) -> tuple[torch.Tensor, dict]:
    dev = S.device
    xi, yi = batch["ingr"]
    if "dish" not in batch:
        x = xi.to(dev, non_blocking=True, memory_format=torch.channels_last)
        return S.crit_ingr(model(x), yi.to(dev, non_blocking=True)), {}
    # D: both batches go through the shared backbone in ONE forward pass; each head is supervised
    # only on the images of its own dataset.
    xd, yd = batch["dish"]
    x = torch.cat([xi, xd]).to(dev, non_blocking=True, memory_format=torch.channels_last)
    ingr_logits, dish_logits = model(x)
    n = xi.shape[0]
    li = S.crit_ingr(ingr_logits[:n], yi.to(dev, non_blocking=True))
    ld = S.crit_dish(dish_logits[n:].float(), yd.to(dev, non_blocking=True))
    return S.w_ingr * li + S.w_dish * ld, {"loss_ingr": li.item(), "loss_dish": ld.item()}


def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Linear warmup then cosine decay, stepped PER BATCH (smooth even with few epochs)."""
    warmup_steps = max(1, warmup_steps)
    warm = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cos = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    return SequentialLR(optimizer, [warm, cos], milestones=[warmup_steps])


def train_one_epoch(S, stream, n_steps, optimizer, scheduler, scaler, frozen: bool) -> dict:
    """Returns the MEAN loss over the epoch (the last batch alone is noise)."""
    model, cfg = S.model, S.cfg
    model.train()
    freeze_backbone(model, frozen)
    amp, clip = need(cfg, "train.amp"), cfg["train"].get("grad_clip")
    sums = defaultdict(float)
    for _ in range(n_steps):
        batch = next(stream)
        with torch.autocast(S.device.type, dtype=torch.float16, enabled=amp):
            loss, parts = step_loss(model, batch, S)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if clip:
            scaler.unscale_(optimizer)   # clip the TRUE gradients, not the scaled ones
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        sums["train_loss"] += loss.item()
        for k, v in parts.items():
            sums[k] += v
    return {k: v / n_steps for k, v in sums.items()}


@torch.no_grad()
def evaluate(S) -> dict:
    logits, _, t = torch_outputs(S.model, S.loaders["ingr_val"], S.device)
    y = t[0]
    m = {
        "val_loss": S.crit_ingr(torch.from_numpy(logits).to(S.device), torch.from_numpy(y).to(S.device)).item(),
        "val_map": map_macro(sigmoid(logits, 1.0), y),
    }
    if S.multitask:
        _, dish_logits, td = torch_outputs(S.model, S.loaders["dish_val"], S.device)
        m["val_top1"] = float((dish_logits.argmax(1) == td[0]).mean())
        m["select"] = (m["val_map"] + m["val_top1"]) / 2   # model selection: both heads count equally
    else:
        m["select"] = m["val_map"]
    return m


def main(cfg: dict) -> dict:
    S = setup(cfg)
    run = need(cfg, "run_name")
    rdir = run_dir(cfg)
    epochs = need(cfg, "train.epochs")
    freeze_epochs = need(cfg, "train.freeze_backbone_epochs")
    patience = need(cfg, "train.early_stop_patience")
    lr = need(cfg, "train.lr")

    optimizer = torch.optim.AdamW(param_groups(S.model, lr, need(cfg, "train.backbone_lr_mult"),
                                               need(cfg, "train.weight_decay")))
    n_steps = steps_per_epoch(S)
    scheduler = build_scheduler(optimizer, epochs * n_steps, round(need(cfg, "train.warmup_epochs") * n_steps))
    scaler = torch.amp.GradScaler(S.device.type, enabled=need(cfg, "train.amp"))

    # Everything needed to reload / evaluate / export this run later, saved BEFORE training.
    (rdir / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")},
                                                     sort_keys=False), encoding="utf-8")
    split_file = ROOT / need(cfg, "data.foodseg_split")
    info = {"run": run, **S.data_cfg, "resolution": need(cfg, "data.resolution"),
            "num_ingredients": len(INGREDIENT_IDS), "ingredient_ids": INGREDIENT_IDS,
            "num_dishes": len(S.dish_names), "dish_names": S.dish_names,
            "git_commit": git_commit(), "split_sha": sha256(split_file),
            "torch": torch.__version__}
    save_json(info, rdir / "run_info.json")

    tracker = Tracker(cfg, info)
    history, best, bad_epochs = [], -np.inf, 0
    stream = batch_stream(S)
    for epoch in range(epochs):
        t0 = time.time()
        frozen = epoch < freeze_epochs
        tr = train_one_epoch(S, stream, n_steps, optimizer, scheduler, scaler, frozen)
        ev = evaluate(S)
        row = {"epoch": epoch, "frozen": frozen, **tr, **ev,
               "lr_head": optimizer.param_groups[1]["lr"], "sec": time.time() - t0}
        history.append(row)
        tracker.log(row, epoch)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

        if ev["select"] > best:
            best, bad_epochs = ev["select"], 0
            torch.save({"state_dict": S.model.state_dict(), "config": cfg, "epoch": epoch,
                        "num_ingredients": len(INGREDIENT_IDS), "num_dishes": len(S.dish_names)},
                       rdir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    hist_path = ROOT / "results" / "runs" / f"{run}_history.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hist_path, index=False)
    best_row = max(history, key=lambda r: r["select"])
    summary = {"run": run, "best_epoch": best_row["epoch"], "best_select": best_row["select"],
               "best_val_map": best_row["val_map"], "best_val_top1": best_row.get("val_top1"),
               "sec_per_epoch": float(np.mean([r["sec"] for r in history]))}
    save_json({**info, **summary}, rdir / "run_info.json")
    tracker.finish(summary, [hist_path, rdir / "config.yaml", rdir / "run_info.json"])
    return summary


class Tracker:
    """Thin MLflow wrapper (mlflow.enabled: false in the config turns it off)."""

    def __init__(self, cfg: dict, info: dict):
        self.on = cfg.get("mlflow", {}).get("enabled", True)
        if not self.on:
            return
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        mlflow.start_run(run_name=cfg["run_name"])
        mlflow.log_params({k: str(v) for k, v in itertools.islice(flatten(cfg).items(), 0, 400)})
        mlflow.log_params({"git_commit": info["git_commit"], "split_sha": info["split_sha"]})

    def log(self, row: dict, step: int) -> None:
        if self.on:
            self.mlflow.log_metrics({k: float(v) for k, v in row.items()
                                     if isinstance(v, (int, float)) and k != "epoch"}, step=step)

    def finish(self, summary: dict, artifacts) -> None:
        if self.on:
            self.mlflow.log_metrics({k: float(v) for k, v in summary.items() if isinstance(v, (int, float))})
            for a in artifacts:
                self.mlflow.log_artifact(str(a))
            self.mlflow.end_run()


if __name__ == "__main__":   # mandatory on Windows with num_workers > 0
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--set", action="append", default=[], help="override, e.g. --set train.lr=3e-4")
    a = p.parse_args()
    main(load_config(a.config, a.set))
