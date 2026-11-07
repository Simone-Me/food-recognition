"""Static post-training quantization (PTQ) int8, QDQ format, with onnxruntime (Niveau 2).

    python -m src.optimize --run a_lite0                                   # parameters from quant.* in the config
    python -m src.optimize --run a_lite0 --method Entropy --calib-size 200 --per-channel false   # one experiment

Writes models/<run>/model_int8_<method>_<n>_<pc|pt>[_<tag>].onnx and appends to results/quant_runs.csv.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from src.data import foodseg_dataset, parse_ids, read_index, foodseg_root
from src.utils import ROOT, file_mb, load_config, load_json, need

METHODS = {"MinMax": CalibrationMethod.MinMax, "Entropy": CalibrationMethod.Entropy,
           "Percentile": CalibrationMethod.Percentile}


def pick_calibration_keys(cfg: dict, n: int, sampling: str, seed: int = 0) -> list[str]:
    """Images of the VALIDATION split (never test).
    random   : n images at random.
    coverage : greedy — repeatedly take the image containing the classes seen the fewest times so far,
               so that rare ingredients also produce activations during calibration."""
    from src.data import load_split

    keys = load_split(need(cfg, "data.foodseg_split"))["val"]
    rng = np.random.default_rng(seed)
    if sampling == "random":
        return list(rng.choice(keys, size=min(n, len(keys)), replace=False))
    if sampling != "coverage":
        raise ValueError("quant.calib_sampling must be 'random' or 'coverage'")
    index = read_index(foodseg_root(cfg) / "index.csv")
    classes = {k: set(parse_ids(index[k]["classes"])) - {0} for k in keys}
    seen: Counter = Counter()
    remaining, chosen = set(keys), []
    while remaining and len(chosen) < n:
        k = min(sorted(remaining), key=lambda k: sum(seen[c] for c in classes[k]) / max(len(classes[k]), 1))
        chosen.append(k)
        remaining.remove(k)
        seen.update(classes[k])
    return chosen


class FoodCalibReader(CalibrationDataReader):
    """Feeds the calibration images one at a time, preprocessed EXACTLY like at inference
    (same eval transform as src.predict / the browser)."""

    def __init__(self, cfg: dict, info: dict, keys: list[str]):
        self.ds = foodseg_dataset(cfg, "val", info["mean"], info["std"], train=False)
        self.ds.keys = keys
        self.i = 0

    def get_next(self) -> dict | None:
        if self.i >= len(self.ds):
            return None
        x = self.ds[self.i][0]
        self.i += 1
        return {"input": x[None].numpy().astype(np.float32)}

    def rewind(self) -> None:
        self.i = 0


def op_histogram(path: Path) -> Counter:
    return Counter(n.op_type for n in onnx.load(str(path)).graph.node)


def quantize(run: str, method: str, calib_size: int, per_channel: bool, sampling: str, tag: str = "",
             exclude: list[str] | None = None) -> Path:
    rdir = ROOT / "models" / run
    cfg = load_config(rdir / "config.yaml")
    info = load_json(rdir / "run_info.json")
    src = rdir / "model_fp32.onnx"
    pre = rdir / "model_fp32_pre.onnx"
    name = f"model_int8_{method.lower()}_{calib_size}_{'pc' if per_channel else 'pt'}{'_' + tag if tag else ''}"
    dst = rdir / f"{name}.onnx"

    # Graph optimisation (BN folding, constant folding) + shape inference before quantization.
    quant_pre_process(str(src), str(pre))
    keys = pick_calibration_keys(cfg, calib_size, sampling)
    quantize_static(
        str(pre), str(dst),
        calibration_data_reader=FoodCalibReader(cfg, info, keys),
        quant_format=QuantFormat.QDQ,          # QuantizeLinear/DequantizeLinear pairs: readable in Netron, portable
        per_channel=per_channel,
        activation_type=QuantType.QUInt8,      # activations uint8 (asymmetric: ReLU outputs are >= 0)
        weight_type=QuantType.QInt8,           # weights int8 (symmetric around 0)
        calibrate_method=METHODS[method],
        nodes_to_exclude=exclude or [],
    )
    ops = op_histogram(dst)
    row = {"run": run, "variant": name, "method": method, "calib_size": calib_size, "per_channel": per_channel,
           "sampling": sampling, "excluded_nodes": len(exclude or []),
           "size_fp32_MB": round(file_mb(src), 3), "size_int8_MB": round(file_mb(dst), 3),
           "n_QuantizeLinear": ops["QuantizeLinear"], "n_DequantizeLinear": ops["DequantizeLinear"]}
    print(f"saved {dst}")
    for k, v in row.items():
        print(f"{k:>20}: {v}")
    print("op types in the int8 graph:", dict(ops.most_common()))
    out = ROOT / "results" / "quant_runs.csv"
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
    return dst


def _bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "y")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--method", choices=list(METHODS))
    p.add_argument("--calib-size", type=int)
    p.add_argument("--per-channel", type=_bool)
    p.add_argument("--sampling", choices=["random", "coverage"])
    p.add_argument("--exclude", nargs="*", default=None, help="node names to keep in float (mixed precision)")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    c = load_config(ROOT / "models" / a.run / "config.yaml")
    torch.manual_seed(0)
    quantize(
        a.run,
        a.method or need(c, "quant.calib_method"),
        a.calib_size or need(c, "quant.calib_size"),
        need(c, "quant.per_channel") if a.per_channel is None else a.per_channel,
        a.sampling or need(c, "quant.calib_sampling"),
        a.tag,
        a.exclude if a.exclude is not None else c["quant"].get("nodes_to_exclude", []),
    )
