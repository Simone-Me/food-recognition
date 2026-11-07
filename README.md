# food-recognition

Ingredient recognition (FoodSeg103, multi-label) then dish + ingredients (Food-101 + FoodSeg103, shared backbone
with two heads), quantized to int8 and run **on the phone** in the CovertCook PWA with onnxruntime-web.

Every command, in order, is in [TP1_riconoscimento_piatti.md](TP1_riconoscimento_piatti.md). The choices and how the
model works are in [docs/decisions.md](docs/decisions.md); the full explanations are in
[TP1_riconoscimento_piatti.v3.md](TP1_riconoscimento_piatti.v3.md).

## Reproduce (Windows, PowerShell, from the repo root, venv active)

```powershell
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
pytest

python -m scripts.make_labels
python -m scripts.preresize --dataset foodseg
python -m scripts.make_splits --dataset foodseg
python -m src.train configs/a_lite0.yaml
python -m src.predict --run a_lite0 --split val
python -m src.calibrate --run a_lite0
python -m src.predict --run a_lite0 --split test
python -m src.metrics results/preds/a_lite0__torch__test.npz
python -m src.export_onnx --run a_lite0
python -m src.optimize --run a_lite0
python -m src.predict --run a_lite0 --split test --onnx model_int8_<...>.onnx
python -m src.stats results/preds/a_lite0__torch__test.npz results/preds/a_lite0__model_int8_<...>__test.npz
```

Every value to choose is in [configs/base.yaml](configs/base.yaml), with its meaning and alternatives; if one is
reset to `___`, the code stops and names it. Layout: `src/` library, `scripts/` one-shot commands, `configs/` experiments, `tests/` pytest,
`web/` TypeScript to copy into CovertCook, `results/` committed measurements.
