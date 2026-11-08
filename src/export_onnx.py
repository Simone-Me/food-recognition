"""Export a trained run to ONNX (Niveau 1 — packaging) + metadata for the browser + parity test.

    python -m src.export_onnx --run a_lite0

Writes in models/<run>/:
    model_fp32.onnx    input "input" (1,3,R,R) normalised float32
                       output "ingredients" (1,K) probabilities [+ "dish" (1,D) for part D]
    model_meta.json    everything the JS code needs: resolution, mean/std, labels, thresholds, output names
and results/parity_<run>.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch

from src.data import foodseg_dataset, load_foodseg_labels, make_loader
from src.models import ExportWrapper, load_checkpoint
from src.utils import ROOT, file_mb, load_config, load_json, need, save_json


def export(run: str) -> Path:
    rdir = ROOT / "models" / run
    cfg = load_config(rdir / "config.yaml")
    info = load_json(rdir / "run_info.json")
    post = load_json(rdir / "postprocess.json")
    res, opset, dynamo = info["resolution"], need(cfg, "export.opset"), need(cfg, "export.dynamo")

    model, _ = load_checkpoint(rdir / "best.pt", "cpu")
    wrapper = ExportWrapper(model, post["t_ingr"], post.get("t_dish", 1.0)).eval()   # eval(): BN/dropout frozen
    multitask = info["num_dishes"] > 0
    outputs = ["ingredients", "dish"] if multitask else ["ingredients"]
    out_path = rdir / "model_fp32.onnx"

    dummy = torch.randn(1, 3, res, res)   # batch fixed to 1: the browser classifies one photo at a time
    torch.onnx.export(wrapper, (dummy,), str(out_path), input_names=["input"], output_names=outputs,
                      opset_version=opset, dynamo=dynamo)
    # Re-save as ONE self-contained file (some exporters write weights as external data).
    m = onnx.load(str(out_path))
    onnx.save_model(m, str(out_path), save_as_external_data=False)
    out_path.with_name(out_path.name + ".data").unlink(missing_ok=True)   # leftover copy of the weights, now unused
    onnx.checker.check_model(str(out_path))
    print(f"exported {out_path} ({file_mb(out_path):.2f} MB, opset {opset}, dynamo={dynamo})")

    names = load_foodseg_labels(ROOT / need(cfg, "data.labels"))
    meta = {
        "version": f"{run}-v1",
        "resolution": res, "mean": info["mean"], "std": info["std"],
        "preprocess": "resize short side to resolution (bilinear, antialias), center crop, /255, (x-mean)/std, CHW",
        "input_name": "input",
        "outputs": {"ingredients": "ingredients", **({"dish": "dish"} if multitask else {})},
        "ingredient_ids": info["ingredient_ids"],
        "ingredient_names": [names[i] for i in info["ingredient_ids"]],
        "thresholds": post["thresholds"],
        "dish_names": info["dish_names"],
        "unknown_dish_min_prob": post.get("unknown_dish_min_prob"),
        "unknown_ingr_min_prob": post.get("unknown_ingr_min_prob"),   # max ingredient prob below -> "not recognised"
    }
    save_json(meta, rdir / "model_meta.json")
    return out_path


def parity_test(run: str, onnx_file: str = "model_fp32.onnx", n_images: int = 64) -> dict:
    """PyTorch vs ONNX on REAL validation images (already preprocessed by the eval loader)."""
    import onnxruntime as ort

    rdir = ROOT / "models" / run
    cfg = load_config(rdir / "config.yaml")
    info = load_json(rdir / "run_info.json")
    post = load_json(rdir / "postprocess.json")
    thr = np.asarray(post["thresholds"])

    model, _ = load_checkpoint(rdir / "best.pt", "cpu")
    wrapper = ExportWrapper(model, post["t_ingr"], post.get("t_dish", 1.0)).eval()
    ds = foodseg_dataset(cfg, "val", info["mean"], info["std"], train=False)
    ds.keys = ds.keys[:n_images]
    x = torch.cat([b[0] for b in make_loader(ds, cfg, shuffle=False, batch_size=16)])

    with torch.no_grad():
        out = wrapper(x)
    p_torch = (out[0] if isinstance(out, tuple) else out).numpy()
    sess = ort.InferenceSession(str(rdir / onnx_file), providers=["CPUExecutionProvider"])
    p_onnx = np.concatenate([sess.run(["ingredients"], {"input": xi[None].numpy()})[0] for xi in x])

    r = {
        "onnx": onnx_file, "n_images": len(x),
        "max_abs_diff": float(np.abs(p_torch - p_onnx).max()),
        "decision_agreement": float(((p_torch >= thr) == (p_onnx >= thr)).mean()),
    }
    save_json(r, ROOT / "results" / f"parity_{run}.json")
    for k, v in r.items():
        print(f"{k:>20}: {v}")
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    a = p.parse_args()
    export(a.run)
    parity_test(a.run)
