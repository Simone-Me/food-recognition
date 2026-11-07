"""Quality vs on-phone latency, with the D-002 thresholds drawn. Generated from the CSVs, never by hand.

    python -m scripts.pareto --stage screening --backend wasm   # results/screening.csv + device_bench.csv
    python -m scripts.pareto --stage final --backend wasm       # results/metrics.csv (test) + device_bench.csv

final also writes results/final_table.md (the table of TP step A13).
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ROOT, file_mb, load_config, need


def latency_table(backend: str, threads: int) -> pd.DataFrame:
    path = ROOT / "results" / "device_bench.csv"
    if not path.exists():
        raise SystemExit("\n>>> results/device_bench.csv not found: the Pareto front needs the phone latencies.\n"
                         "    Run the phone benchmark (TP step A12), save its JSON in results/bench_raw/, then:\n"
                         "    python -m scripts.collect_bench results/bench_raw/*.json\n")
    b = pd.read_csv(path)
    # One thread count at a time: a median over 1 and 4 threads would describe neither. WebGPU is always logged as 1.
    b = b[(b.backend == backend) & (b.numThreads == (1 if backend == "webgpu" else threads))]
    # median over the campaigns (each campaign already reports the median of its runs)
    agg = b.groupby("variant")[["runMedian", "runP95", "loadMs", "preMs", "postMs"]].median()
    agg.columns = ["run_median_ms", "run_p95_ms", "load_ms", "pre_ms", "post_ms"]
    agg["total_median_ms"] = agg.run_median_ms + agg.pre_ms.fillna(0) + agg.post_ms.fillna(0)
    return agg.reset_index()


def plot(df: pd.DataFrame, x: str, y: str, label: str, cfg: dict, title: str, out) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df[x], df[y])
    for _, r in df.iterrows():
        ax.annotate(r[label], (r[x], r[y]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axvline(need(cfg, "targets.latency_ms"), color="r", ls="--", label="latency target (D-002)")
    ax.axhline(need(cfg, "targets.min_map"), color="g", ls="--", label="quality target (D-002)")
    ax.set_xlabel(f"{x} on the phone (ms)")
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["screening", "final"], required=True)
    p.add_argument("--backend", default="wasm")
    p.add_argument("--threads", type=int, default=1, help="WASM thread count to read from device_bench.csv")
    p.add_argument("--config", default="configs/base.yaml")
    a = p.parse_args()
    cfg = load_config(a.config)
    lat = latency_table(a.backend, a.threads)

    if a.stage == "screening":
        q = pd.read_csv(ROOT / "results" / "screening.csv").groupby("variant").val_map.agg(["mean", "std"]).reset_index()
        df = q.merge(lat, on="variant", how="left").rename(columns={"mean": "val_map"})
        print(df.to_string(index=False))
        plot(df.dropna(subset=["run_median_ms"]), "run_median_ms", "val_map", "variant", cfg,
             f"Screening — {a.backend}", ROOT / "results" / "pareto_screening.png")
        return

    m = pd.read_csv(ROOT / "results" / "metrics.csv")
    test = m[(m.split == "test") & (m.variant != "torch")].copy()
    test["variant_id"] = test.run + "__" + test.variant
    own = m[m.split == "own"].assign(variant_id=lambda d: d.run + "__" + d.variant)[["variant_id", "map_macro"]]
    df = test.merge(own.rename(columns={"map_macro": "own_map_macro"}), on="variant_id", how="left")
    df["size_MB"] = [round(file_mb(ROOT / "models" / r / f"{v}.onnx"), 2) for r, v in zip(df.run, df.variant)]
    df = df.merge(lat.rename(columns={"variant": "variant_id"}), on="variant_id", how="left")
    cols = [c for c in ["variant_id", "size_MB", "map_macro", "map_macro_lo", "map_macro_hi", "f1_micro", "ece_ingr",
                        "top1", "top3", "total_median_ms", "run_p95_ms", "load_ms", "own_map_macro"] if c in df]
    table = df[cols].round(4)
    (ROOT / "results" / "final_table.md").write_text(table.to_markdown(index=False), encoding="utf-8")
    print(table.to_string(index=False))
    plot(df.dropna(subset=["total_median_ms"]), "total_median_ms", "map_macro", "variant_id", cfg,
         f"Final variants — {a.backend}", ROOT / "results" / "pareto_final.png")


if __name__ == "__main__":
    main()
