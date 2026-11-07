"""Images/second of the training DataLoader alone (no model), for several num_workers.

    python -m scripts.bench_dataloader --config configs/a_lite0.yaml --workers 0 2 4 6

Compare with the training throughput printed by src.train (images per epoch / sec per epoch):
if they are close, training is I/O-bound (the GPU waits for the CPU).
"""
from __future__ import annotations

import argparse
import time

import timm

from src.data import foodseg_dataset, make_loader
from src.utils import load_config, need, set_key


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 6])
    p.add_argument("--batches", type=int, default=40)
    a = p.parse_args()
    cfg = load_config(a.config)
    dc = timm.data.resolve_data_config({}, model=timm.create_model(need(cfg, "model.name"), pretrained=False))
    ds = foodseg_dataset(cfg, "train", dc["mean"], dc["std"], train=True)
    for w in a.workers:
        set_key(cfg, "data.num_workers", w)
        loader = make_loader(ds, cfg, shuffle=True)
        it = iter(loader)
        next(it)   # worker start-up is not part of the steady state
        t0, n = time.time(), 0
        for _ in range(a.batches):
            n += next(it)[0].shape[0]
        print(f"num_workers={w}: {n / (time.time() - t0):7.1f} images/s")


if __name__ == "__main__":
    main()
