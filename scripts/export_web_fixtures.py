"""Fixtures for the JS parity test (Vitest): Python's view of a few validation images.

    python -m scripts.export_web_fixtures --run a_lite0 --n 3

-> web/fixtures/parity_<run>.json with, per image:
   rgb    : the uint8 pixels AFTER resize + center crop (H*W*3, row-major)  -> input of the JS toTensor()
   tensor : the normalised CHW float32 tensor Python feeds to the model     -> expected output of toTensor()
   probs  : the ONNX fp32 output                                            -> input of the JS postprocess()
   ingredients : names above the thresholds                                 -> expected output of postprocess()
"""
from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.transforms import v2

from src.data import foodseg_dataset, foodseg_root
from src.utils import ROOT, load_config, load_json, save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=3)
    a = p.parse_args()
    rdir = ROOT / "models" / a.run
    cfg = load_config(rdir / "config.yaml")
    meta = load_json(rdir / "model_meta.json")
    res = meta["resolution"]
    ds = foodseg_dataset(cfg, "val", meta["mean"], meta["std"], train=False)
    geo = v2.Compose([v2.ToImage(), v2.Resize(res, antialias=True), v2.CenterCrop(res)])
    sess = ort.InferenceSession(str(rdir / "model_fp32.onnx"), providers=["CPUExecutionProvider"])
    thr = np.asarray(meta["thresholds"])

    items = []
    for i in range(a.n):
        key = ds.keys[i]
        rgb = geo(Image.open(foodseg_root(cfg) / "images" / f"{key}.jpg").convert("RGB"))   # (3,R,R) uint8
        x = ds[i][0]
        probs = sess.run(["ingredients"], {"input": x[None].numpy()})[0][0]
        items.append({
            "key": key,
            "rgb": rgb.permute(1, 2, 0).reshape(-1).tolist(),
            "tensor": np.round(x.numpy().reshape(-1), 5).tolist(),
            "probs": np.round(probs, 6).tolist(),
            "ingredients": [meta["ingredient_names"][k] for k in np.flatnonzero(probs >= thr)],
        })
    out = ROOT / "web" / "fixtures" / f"parity_{a.run}.json"
    save_json({"resolution": res, "mean": meta["mean"], "std": meta["std"], "thresholds": meta["thresholds"],
               "ingredient_names": meta["ingredient_names"], "items": items}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
