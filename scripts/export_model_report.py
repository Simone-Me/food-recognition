"""One JSON with everything the "How the model works" page of CovertCook shows (route /scan/model).

    python -m scripts.export_model_report --dest C:\\Users\\hector\\covertcook\\public\\models

Reads only committed results (nothing is recomputed): training curves (results/runs/), test metrics
(results/metrics.csv), int8 variants (results/quant_runs.csv + results/comparisons.csv) and phone
latency (results/device_bench.csv, after scripts.collect_bench). Writes <dest>/model_report.json.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ROOT, load_config, load_json

FINALISTS = {"a_convnext": "ConvNeXt-Atto", "a_mnv3": "MobileNetV3-Large", "a_lite0": "EfficientNet-Lite0"}
DEPLOYED = ("a_convnext", "model_int8_minmax_200_pc_only-conv_rr")
INT8_CHOSEN = {"a_convnext": "model_int8_minmax_200_pc_only-conv_rr", "a_lite0": "model_int8_minmax_200_pc_rr"}


def _num(x, nd=4):
    """JSON-safe number: NaN -> None, rounded."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd)


def training(run: str) -> dict:
    rdir = ROOT / "models" / run
    cfg, info = load_config(rdir / "config.yaml"), load_json(rdir / "run_info.json")
    h = pd.read_csv(ROOT / "results" / "runs" / f"{run}_history.csv")
    return {
        "run": run, "name": FINALISTS[run], "backbone": cfg["model"]["name"],
        "resolution": info["resolution"], "lr": cfg["train"]["lr"], "augment": cfg["train"]["augment"],
        "best_epoch": info["best_epoch"], "best_val_map": _num(info["best_val_map"]),
        "epochs_run": int(len(h)), "sec_per_epoch": _num(h["sec"].median(), 1),
        "history": [{"epoch": int(r.epoch), "train_loss": _num(r.train_loss), "val_loss": _num(r.val_loss),
                     "val_map": _num(r.val_map), "select": _num(r.select), "sec": _num(r.sec, 1)}
                    for r in h.itertuples()],
    }


def test_metrics() -> dict:
    m = pd.read_csv(ROOT / "results" / "metrics.csv")
    m = m[m["split"] == "test"].drop_duplicates(["run", "variant"], keep="last")
    return {f"{r.run}__{r.variant}": {"map": _num(r.map_macro), "f1_micro": _num(r.f1_micro),
                                      "map_frequent": _num(r.map_frequent), "map_rare": _num(r.map_rare)}
            for r in m.itertuples()}


def quantization() -> list[dict]:
    q = pd.read_csv(ROOT / "results" / "quant_runs.csv").drop_duplicates(["run", "variant"], keep="last")
    c = pd.read_csv(ROOT / "results" / "comparisons.csv")
    c = c[c["b"].str.contains("int8") & c["b"].str.endswith("__val")].copy()
    c["run"] = c["b"].str.split("__").str[0]
    c["variant"] = c["b"].str.split("__").str[1]
    c = c.drop_duplicates(["run", "variant"], keep="last")
    j = q.merge(c[["run", "variant", "delta_map_macro", "delta_map_lo", "delta_map_hi"]], on=["run", "variant"])
    out = []
    for r in j.sort_values(["run", "delta_map_macro"]).itertuples():
        out.append({
            "run": r.run, "variant": r.variant, "method": r.method, "calib_size": int(r.calib_size),
            "per_channel": bool(r.per_channel), "sampling": r.sampling,
            "op_types": r.op_types if isinstance(r.op_types, str) else "all",
            "reduce_range": bool(r.reduce_range) if not pd.isna(r.reduce_range) else False,
            "size_mb": _num(r.size_int8_MB, 2), "size_fp32_mb": _num(r.size_fp32_MB, 2),
            "loss_val": _num(r.delta_map_macro), "loss_val_lo": _num(r.delta_map_lo), "loss_val_hi": _num(r.delta_map_hi),
            "chosen": INT8_CHOSEN.get(r.run) == r.variant,
        })
    return out


def comparison(tests: dict) -> list[dict]:
    c = pd.read_csv(ROOT / "results" / "comparisons.csv")
    rows = []
    for run, variant in INT8_CHOSEN.items():
        fp, q = tests.get(f"{run}__model_fp32"), tests.get(f"{run}__{variant}")
        d = c[(c["a"] == f"{run}__model_fp32__test") & (c["b"] == f"{run}__{variant}__test")].tail(1)
        qr = pd.read_csv(ROOT / "results" / "quant_runs.csv")
        qr = qr[(qr["run"] == run) & (qr["variant"] == variant)].tail(1)
        rows.append({
            "run": run, "name": FINALISTS[run], "variant": variant, "deployed": (run, variant) == DEPLOYED,
            "fp32": {"map": fp["map"], "f1_micro": fp["f1_micro"], "size_mb": _num(qr["size_fp32_MB"].iloc[0], 2)},
            "int8": {"map": q["map"], "f1_micro": q["f1_micro"], "size_mb": _num(qr["size_int8_MB"].iloc[0], 2)},
            "loss": _num(d["delta_map_macro"].iloc[0]), "loss_lo": _num(d["delta_map_lo"].iloc[0]),
            "loss_hi": _num(d["delta_map_hi"].iloc[0]),
        })
    return rows


def phone() -> dict | None:
    path = ROOT / "results" / "device_bench.csv"
    if not path.exists():
        return None
    b = pd.read_csv(path)
    b["device"] = b["device"].str.strip()
    g = (b.groupby(["device", "variant", "backend", "numThreads"])
          .agg(run_ms=("runMedian", "median"), pre_ms=("preMs", "median"), n=("runMedian", "size")).reset_index())
    g["total_ms"] = g["run_ms"] + g["pre_ms"]
    return {"device": str(g["device"].iloc[0]),
            "rows": [{"variant": r.variant, "backend": r.backend, "threads": int(r.numThreads),
                      "run_ms": _num(r.run_ms, 1), "pre_ms": _num(r.pre_ms, 1), "total_ms": _num(r.total_ms, 1),
                      "measures": int(r.n)} for r in g.itertuples()]}


def screening() -> list[dict]:
    s = pd.read_csv(ROOT / "results" / "screening.csv").groupby("model").val_map.agg(["mean", "count"]).reset_index()
    c = pd.read_csv(ROOT / "results" / "candidates.csv")
    j = s.merge(c[["model", "params_M", "mult_adds_G"]], on="model").sort_values("mean", ascending=False)
    return [{"model": r.model, "val_map": _num(r.mean), "seeds": int(r.count), "params_m": _num(r.params_M, 2),
             "mult_adds_g": _num(r.mult_adds_G, 3)} for r in j.itertuples()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, required=True)
    a = p.parse_args()
    tests = test_metrics()
    targets = load_config(ROOT / "configs" / "base.yaml")["targets"]
    report = {
        "generated": date.today().isoformat(),
        "dataset": {"name": "FoodSeg103", "classes": 103, "train": 4225, "val": 758, "test": 2135},
        "targets": targets,
        "deployed": {"run": DEPLOYED[0], "variant": DEPLOYED[1]},
        "screening": screening(),
        "models": [training(r) | {"test": tests.get(f"{r}__torch")} for r in FINALISTS],
        "quantization": quantization(),
        "comparison": comparison(tests),
        "phone": phone(),
    }
    a.dest.mkdir(parents=True, exist_ok=True)
    out = a.dest / "model_report.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
