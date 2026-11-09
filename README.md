# food-recognition

Ingredient recognition (FoodSeg103, multi-label, 103 classes), quantized to int8 and run **on the phone** in the
CovertCook PWA with onnxruntime-web: [covertcook.fr/scan](https://covertcook.fr/scan) (Profile → "Analyse your
plate"). The photo never leaves the phone. The page [covertcook.fr/scan/model](https://covertcook.fr/scan/model)
("How does the model work?") shows the curves, the int8 variants and the phone latency, from a `model_report.json`
written by `scripts.export_model_report`. A signed-in user can press "Evaluate" to remove the wrong ingredients and add
the missing ones: only the ingredient ids are sent (no photo) to the Supabase table `scan_feedback` (CovertCook
migration 0111), with the model version, precision and recall, to measure the model in real use. Part D (dish +
ingredients, Food-101) is not done yet.

## Results

| | fp32 | int8 (deployed) |
|---|---|---|
| Model | ConvNeXt-Atto, 256 px | MinMax, 200 val images, per channel, convolutions only, `reduce_range` |
| mAP macro, test (2 135 images) | 0.388 | **0.367** (-2.1 points, 95 % CI [1.0 ; 3.1]) |
| File size | 14.1 MB | **3.8 MB** |
| Phone latency, model only (Realme GT Master, wasm, 4 threads / 1 thread) | 55 / 178 ms | 73 / 227 ms |
| Phone latency, photo + model (4 threads) | 474 ms | 491 ms |

- The int8 model is 3.7 times smaller but not faster in the browser; the real bottleneck on the phone is decoding the
  photo (350-560 ms on the phone, against 20-40 ms on a PC). Phone measures: `results/device_bench.csv`.
- Targets fixed before measuring (`configs/base.yaml`): size ≤ 5 MB **met**; mAP ≥ 0.40 **not met**; int8 loss
  ≤ 1.5 points met by Lite0 (0.0), not by ConvNeXt (2.1).

## Documents

| Document | Content |
|---|---|
| [docs/presentation_modele.md](docs/presentation_modele.md) | The model: data, screening, ablations, evaluation, problems and solutions (French, for the oral) |
| [docs/presentation_deploiement.md](docs/presentation_deploiement.md) | Quantization and phone deployment: process, problems, solutions, results (French, for the oral) |
| [docs/decisions.md](docs/decisions.md) | Decision journal with every measure (French) |
| [TP1_riconoscimento_piatti.md](TP1_riconoscimento_piatti.md) | Every command, in order (Italian) |
| [TP1_riconoscimento_piatti.v3.md](TP1_riconoscimento_piatti.v3.md) | Full explanations and theory |

## Reproduce (Windows, PowerShell, from the repo root, venv active)

```powershell
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
pytest

python -m scripts.make_labels
python -m scripts.preresize --dataset foodseg
python -m scripts.make_splits --dataset foodseg
python -m src.train configs/a_convnext.yaml
python -m src.predict --run a_convnext --split val
python -m src.calibrate --run a_convnext --unknown-ingr-min-prob 0.4
python -m src.predict --run a_convnext --split test
python -m src.metrics results/preds/a_convnext__torch__test.npz
python -m src.export_onnx --run a_convnext
python -m src.optimize --run a_convnext --op-types Conv --reduce-range true
python -m src.predict --run a_convnext --split test --onnx model_int8_minmax_200_pc_only-conv_rr.onnx
python -m src.stats results/preds/a_convnext__model_fp32__test.npz results/preds/a_convnext__model_int8_minmax_200_pc_only-conv_rr__test.npz
python -m scripts.copy_to_web --run a_convnext --variants model_int8_minmax_200_pc_only-conv_rr --dest <CovertCook>\public\models
```

Every value to choose is in [configs/base.yaml](configs/base.yaml), with its meaning and alternatives. Layout: `src/`
library, `scripts/` one-shot commands, `configs/` experiments, `tests/` pytest, `web/` TypeScript copied into
CovertCook (source of truth for `foodModel.ts`, `bench.ts`, `FoodScan.tsx`, `LabBench.tsx`), `results/` committed
measurements (`metrics.csv`, `comparisons.csv`, `quant_runs.csv`, `runs/`, `bench_raw/`).

`src.optimize` options: `--method MinMax|Entropy|Percentile`, `--calib-size`, `--per-channel`, `--sampling
random|coverage`, `--op-types` (mixed precision), `--reduce-range` (7-bit weights, for x86 CPUs without VNNI). The
classifier head (linear, temperature, sigmoid) always stays in float.
