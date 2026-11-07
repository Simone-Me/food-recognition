"""Step A6, short-budget screening of the candidates with the SAME recipe, and seed variance.

    python -m scripts.screening --config configs/screening.yaml                     # every candidate, 1 seed
    python -m scripts.screening --config configs/screening.yaml --seeds 0 1 2 --only tf_efficientnet_lite0

-> results/screening.csv (one row per model x seed: best val mAP, epoch, sec/epoch)
"""
from __future__ import annotations

import argparse
import copy

import pandas as pd

from src.models import CANDIDATES
from src.train import main as train_main
from src.utils import ROOT, load_config, need, set_key


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/screening.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--only", nargs="+", default=None)
    a = p.parse_args()
    base = load_config(a.config)
    res = need(base, "data.resolution")
    models = a.only or base.get("screening", {}).get("candidates", CANDIDATES)
    out = ROOT / "results" / "screening.csv"
    for name in models:
        for seed in a.seeds or [base["seed"]]:
            cfg = copy.deepcopy(base)
            set_key(cfg, "model.name", name)
            set_key(cfg, "seed", seed)
            set_key(cfg, "run_name", f"screen_{name}_s{seed}")
            s = train_main(cfg)
            row = {"model": name, "seed": seed, "variant": f"{name}_{res}", "val_map": s["best_val_map"],
                   "best_epoch": s["best_epoch"], "sec_per_epoch": s["sec_per_epoch"]}
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(row)
    df = pd.read_csv(out)
    print(df.groupby("model").val_map.agg(["mean", "std", "count"]).sort_values("mean", ascending=False))


if __name__ == "__main__":
    main()
