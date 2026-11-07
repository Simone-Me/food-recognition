"""Copy ONNX variants + metadata into the CovertCook repo, with names that identify the run.

    python -m scripts.copy_to_web --run a_lite0 --variants model_fp32 model_int8_minmax_200_pc --dest C:\\dev\\CovertCook\\public\\models

-> <dest>/a_lite0__model_fp32.onnx, <dest>/a_lite0__model_int8_minmax_200_pc.onnx, <dest>/a_lite0__meta.json
The file stem ("a_lite0__model_fp32") is the `variant` reported by the benchmark page.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.utils import ROOT, file_mb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--dest", type=Path, required=True)
    a = p.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    rdir = ROOT / "models" / a.run
    for v in a.variants:
        dst = a.dest / f"{a.run}__{v}.onnx"
        shutil.copy2(rdir / f"{v}.onnx", dst)
        print(f"{dst}  ({file_mb(dst):.2f} MB)")
    shutil.copy2(rdir / "model_meta.json", a.dest / f"{a.run}__meta.json")
    print(f"{a.dest / f'{a.run}__meta.json'}")


if __name__ == "__main__":
    main()
