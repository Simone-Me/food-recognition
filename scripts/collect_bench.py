"""Merge the JSON files copied from the phone benchmark page into results/device_bench.csv.

    python -m scripts.collect_bench results/bench_raw/*.json
"""
from __future__ import annotations

import argparse
import glob
import json

import pandas as pd

from src.utils import ROOT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    a = p.parse_args()
    rows = []
    for pattern in a.files:              # PowerShell does not expand globs: do it here
        for f in glob.glob(pattern):
            data = json.load(open(f, encoding="utf-8"))
            rows += data if isinstance(data, list) else [data]
    if not rows:     # stop BEFORE writing: never overwrite existing measurements with an empty file
        raise SystemExit(f"\n>>> No benchmark results found in: {' '.join(a.files)}\n"
                         f"    Run the phone benchmark first (TP step A12) and paste its \"copy JSON\" output\n"
                         f"    into a .json file in results/bench_raw/. results/device_bench.csv was not changed.\n")
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "device_bench.csv"
    df.to_csv(out, index=False)
    cols = [c for c in ["variant", "backend", "numThreads", "loadMs", "preMs", "runMedian", "runP95", "postMs"] if c in df]
    print(df[cols].groupby(["variant", "backend", "numThreads"]).median().round(1).to_string())
    print(f"{len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
