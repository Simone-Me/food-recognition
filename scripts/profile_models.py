"""Step A6, "dry" measurements of every candidate (no training) + raw ONNX for a first phone benchmark.

    python -m scripts.profile_models --config configs/screening.yaml

-> results/candidates.csv and models/candidates/<model>_<res>.onnx (+ .meta.json) (random head, ImageNet
   backbone: the latency is the same as for the trained model, the predictions are meaningless).
"""
from __future__ import annotations

import argparse

import onnx
import pandas as pd
import timm
import torch

from src.data import INGREDIENT_IDS
from src.models import CANDIDATES, profile
from src.utils import ROOT, file_mb, load_config, need, save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/screening.yaml")
    a = p.parse_args()
    cfg = load_config(a.config)
    res, opset, dynamo = need(cfg, "data.resolution"), need(cfg, "export.opset"), need(cfg, "export.dynamo")
    out_dir = ROOT / "models" / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in cfg.get("screening", {}).get("candidates", CANDIDATES):
        row = profile(name, res)
        m = timm.create_model(name, pretrained=True, num_classes=103).eval()
        path = out_dir / f"{name}_{res}.onnx"
        torch.onnx.export(m, (torch.randn(1, 3, res, res),), str(path), input_names=["input"],
                          output_names=["ingredients"], opset_version=opset, dynamo=dynamo)
        # Re-save as ONE self-contained file: the exporter can write the weights to <name>.onnx.data, which the
        # browser would not load (and the file size below would not count them).
        onnx.save_model(onnx.load(str(path)), str(path), save_as_external_data=False)
        path.with_name(path.name + ".data").unlink(missing_ok=True)
        onnx.checker.check_model(str(path))
        # Minimal metadata so that the benchmark page can preprocess a real photo for this model.
        save_json({"version": path.stem, "resolution": res, "mean": row["mean"], "std": row["std"],
                   "input_name": "input", "outputs": {"ingredients": "ingredients"},
                   "ingredient_ids": INGREDIENT_IDS, "ingredient_names": [str(i) for i in INGREDIENT_IDS],
                   "thresholds": [0.5] * len(INGREDIENT_IDS), "dish_names": [], "unknown_dish_min_prob": None},
                  path.with_suffix(".meta.json"))
        row["variant"] = path.stem
        row["onnx_fp32_MB"] = round(file_mb(path), 2)
        row["fits_size_target_fp32"] = row["onnx_fp32_MB"] <= need(cfg, "targets.size_mb")
        row["fits_size_target_int8_est"] = row["size_int8_MB_est"] <= need(cfg, "targets.size_mb")
        rows.append(row)
        print(row)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "candidates.csv", index=False)
    print(df[["model", "params_M", "mult_adds_G", "onnx_fp32_MB", "size_int8_MB_est"]].to_string(index=False))


if __name__ == "__main__":
    main()
