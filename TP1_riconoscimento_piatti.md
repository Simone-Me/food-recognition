# TP1 — Comandi da lanciare

PowerShell 7, dalla radice del repo, con il venv attivo. Scelte e funzionamento del modello: [docs/decisions.md](docs/decisions.md). Versione con tutte le spiegazioni e la teoria per l'orale: [TP1_riconoscimento_piatti.v3.md](TP1_riconoscimento_piatti.v3.md).

```powershell
cd C:\Users\hector\Desktop\Projects\EFREI\M2\MLops\food-recognition
.\.venv\Scripts\Activate.ps1
```

---

# PARTE A — Ingredienti

## A1 — Ambiente ✅

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
python -m pip freeze | Out-File -Encoding ascii requirements.lock.txt
python -m scripts.check_env
pytest
```

## A2 — Etichette e dati su disco ✅

```powershell
python -m scripts.make_labels
python -m scripts.preresize --dataset foodseg
```

## A3 — Split ✅

```powershell
python -m scripts.make_splits --dataset foodseg
Get-FileHash data\splits\foodseg_v1.json
python -m scripts.make_splits --dataset foodseg
Get-FileHash data\splits\foodseg_v1.json        # stesso hash = riproducibile
python -m scripts.show_samples --n 16
```

## A4 — EDA ✅

```powershell
python -m scripts.eda
python -m scripts.show_samples --class-name "french fries" --n 6
python -m scripts.show_samples --class-name pepper --n 6
```

## A5 — DataLoader

```powershell
python -m scripts.bench_dataloader --config configs/a_lite0.yaml --workers 0 2 4 6
# scrivi in base.yaml, data.num_workers, il valore oltre il quale le immagini/s non aumentano più
python -m scripts.show_samples --config configs/a_lite0.yaml --n 16 --augmented
```

## A6 — Screening

```powershell
python -m scripts.screening --config configs/screening.yaml                                                       # ✅
python -m scripts.screening --config configs/screening.yaml --only tf_efficientnet_lite0 --seeds 1 2              # in corso

# Quando i seed sono finiti: rigenera i file ONNX (pesi inclusi in un solo file) e la tabella
python -m scripts.profile_models --config configs/screening.yaml
python -c "import pandas as pd; s=pd.read_csv('results/screening.csv').groupby('variant').val_map.agg(['mean','std','count']).reset_index(); c=pd.read_csv('results/candidates.csv'); print(s.merge(c[['variant','params_M','mult_adds_G','size_int8_MB_est','fits_size_target_int8_est']]).sort_values('mean',ascending=False).to_string(index=False))"
```

## A7 — Controlli prima del training

```powershell
python -m scripts.sanity_checks configs/a_convnext.yaml --check init
python -m scripts.sanity_checks configs/a_convnext.yaml --check overfit
python -m scripts.sanity_checks configs/a_convnext.yaml --check lr
python -m scripts.sanity_checks configs/a_mnv3.yaml --check lr
python -m scripts.sanity_checks configs/a_lite0.yaml --check lr
# dai grafici results/lr_range_*.png: LR ~10 volte sotto il punto in cui la loss è minima
```

## A8 — Training e ablation

Secondo terminale, per seguire le run:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db       # http://127.0.0.1:5000
```

Terminale principale:

```powershell
# Ricetta base, sul finalista più rapido da addestrare
python -m src.train configs/a_convnext.yaml

# Batch size: guarda la memoria con nvidia-smi
python -m src.train configs/a_convnext.yaml --set train.batch_size=48 --set run_name=a_convnext_b48
python -m src.train configs/a_convnext.yaml --set train.batch_size=64 --set run_name=a_convnext_b64

# Ablation: un cambiamento alla volta, stesso seed
python -m src.train configs/a_convnext.yaml --set train.freeze_backbone_epochs=3 --set run_name=a_convnext_probe
python -m src.train configs/a_convnext.yaml --set train.loss=bce_posw --set run_name=a_convnext_posw
python -m src.train configs/a_convnext.yaml --set train.loss=asl --set run_name=a_convnext_asl
python -m src.train configs/a_convnext.yaml --set data.train_min_area=0.005 --set run_name=a_convnext_tau005
python -m src.train configs/a_convnext.yaml --set data.train_min_area=0.02 --set run_name=a_convnext_tau02
python -m src.train configs/a_convnext.yaml --set train.augment=photo --set run_name=a_convnext_photo
python -m src.train configs/a_convnext.yaml --set train.augment=strong --set run_name=a_convnext_strong
python -m src.train configs/a_convnext.yaml --set train.crop_scale_min=0.35 --set run_name=a_convnext_crop35
python -m src.train configs/a_convnext.yaml --set data.resolution=256 --set run_name=a_convnext_r256
python -m src.train configs/a_convnext.yaml --set data.resolution=320 --set run_name=a_convnext_r320

# Ricetta finale in base.yaml: 256 px + augment photo (D-013). Un ultimo test: freeze 3 sulla ricetta finale
python -m src.train configs/a_convnext.yaml --set train.freeze_backbone_epochs=3 --set run_name=a_convnext_final_probe

# Finalisti con la ricetta finale e il LR di A7 (se final_probe >= 0,485, aggiungi --set train.freeze_backbone_epochs=3)
python -m src.train configs/a_convnext.yaml
python -m src.train configs/a_mnv3.yaml --set train.lr=0.0005
python -m src.train configs/a_lite0.yaml --set train.lr=0.0006
```

## A9 — Valutazione

```powershell
foreach ($r in "a_convnext", "a_mnv3", "a_lite0") {
  python -m src.predict --run $r --split val
  python -m src.calibrate --run $r
  python -m src.predict --run $r --split test
  python -m src.metrics "results/preds/${r}__torch__test.npz"
  python -m scripts.error_analysis "results/preds/${r}__torch__test.npz"
}

# Solo dopo aver preparato data/own/images/*.jpg e data/own/labels.csv (colonne: file,ingredient_ids,dish)
foreach ($r in "a_convnext", "a_mnv3", "a_lite0") {
  python -m src.predict --run $r --split own
  python -m src.metrics "results/preds/${r}__torch__own.npz"
  python -m scripts.unknown_threshold "results/preds/${r}__torch__own.npz"
}
```

## A10 — Export ONNX

```powershell
foreach ($r in "a_convnext", "a_mnv3", "a_lite0") {
  python -m src.export_onnx --run $r
  python -m src.predict --run $r --split test --onnx model_fp32.onnx
  python -m src.metrics "results/preds/${r}__model_fp32__test.npz"
  python -m src.stats "results/preds/${r}__torch__test.npz" "results/preds/${r}__model_fp32__test.npz"
}
```

## A11 — Quantizzazione int8

```powershell
# Variante di riferimento per ogni finalista
foreach ($r in "a_convnext", "a_mnv3", "a_lite0") {
  python -m src.optimize --run $r
}

# Esperimenti sul finalista migliore: un parametro alla volta
$r = "a_convnext"
python -m src.optimize --run $r --method Entropy
python -m src.optimize --run $r --method Percentile
python -m src.optimize --run $r --per-channel false
python -m src.optimize --run $r --calib-size 50
python -m src.optimize --run $r --calib-size 500
python -m src.optimize --run $r --sampling coverage

# Misura tutte le varianti int8 create, con il confronto appaiato contro fp32
foreach ($r in "a_convnext", "a_mnv3", "a_lite0") {
  foreach ($v in Get-ChildItem "models\$r\model_int8_*.onnx") {
    python -m src.predict --run $r --split test --onnx $v.Name
    python -m src.metrics "results/preds/${r}__$($v.BaseName)__test.npz"
    python -m src.stats "results/preds/${r}__model_fp32__test.npz" "results/preds/${r}__$($v.BaseName)__test.npz"
  }
}
Get-ChildItem models\a_*\model_int8_*.onnx | Select-Object FullName, @{n='MB';e={[math]::Round($_.Length/1MB,2)}}
```

## A12 — Benchmark sul telefono (solo a modello finito)

⚠️ CovertCook è l'app in produzione: Netlify pubblica `main`. Lavora solo su questo branch, non unirlo a `main` e non usare `--prod`.

```powershell
cd C:\Users\hector\covertcook
git fetch origin
git switch -c lab/phone-bench origin/main
Add-Content .git\info\exclude "public/models/","public/ort/",".netlify/"
npm install
npm install onnxruntime-web

$ML = "C:\Users\hector\Desktop\Projects\EFREI\M2\MLops\food-recognition"
New-Item -ItemType Directory -Force public\ort, public\models, src\lib\__fixtures__, src\pages
Copy-Item node_modules\onnxruntime-web\dist\*.wasm public\ort\
Copy-Item node_modules\onnxruntime-web\dist\*.mjs public\ort\
Copy-Item "$ML\web\src\lib\foodModel.ts", "$ML\web\src\lib\bench.ts" src\lib\
Copy-Item "$ML\web\src\pages\LabBench.tsx" src\pages\

# Route /lab/bench
$f = "src\App.tsx"
(Get-Content $f -Raw).Replace("const HelpPage = page(() => import('./features/legal/HelpPage'), 'HelpPage')", "const HelpPage = page(() => import('./features/legal/HelpPage'), 'HelpPage')`nconst LabBench = lazy(() => import('./pages/LabBench'))").Replace('<Route path="/help" element={<HelpPage />} />', '<Route path="/lab/bench" element={<LabBench />} /> <Route path="/help" element={<HelpPage />} />') | Set-Content $f -NoNewline

# Il service worker non deve togliere COOP/COEP a /lab
$f = "src\sw.ts"
(Get-Content $f -Raw).Replace("/^\/rest\//,", "/^\/rest\//,`n      /^\/lab\//,") | Set-Content $f -NoNewline

# CSP (WebAssembly) e isolamento cross-origin solo su /lab
$f = "public\_headers"
(Get-Content $f -Raw).Replace("script-src 'self';", "script-src 'self' 'wasm-unsafe-eval';") | Set-Content $f -NoNewline
Add-Content $f "`n/lab/*`n  Cross-Origin-Opener-Policy: same-origin`n  Cross-Origin-Embedder-Policy: require-corp"

# Modelli da misurare
$f = "src\pages\LabBench.tsx"
(Get-Content $f -Raw) -replace '(?s)const VARIANTS = \[.*?\];', 'const VARIANTS = ["/models/a_convnext__model_fp32.onnx", "/models/a_convnext__model_int8_minmax_200_pc.onnx", "/models/a_mnv3__model_fp32.onnx", "/models/a_mnv3__model_int8_minmax_200_pc.onnx", "/models/a_lite0__model_fp32.onnx", "/models/a_lite0__model_int8_minmax_200_pc.onnx"];' | Set-Content $f -NoNewline

git diff --stat

# Copia i modelli (dal repo ML)
cd $ML
foreach ($r in "a_convnext", "a_mnv3", "a_lite0") {
  python -m scripts.copy_to_web --run $r --variants model_fp32 model_int8_minmax_200_pc --dest C:\Users\hector\covertcook\public\models
}

# Deploy di bozza in HTTPS
cd C:\Users\hector\covertcook
npm run build
npx netlify-cli login
npx netlify-cli link
npx netlify-cli deploy --dir=dist
```

Sul telefono apri `<URL della bozza>/lab/bench`, in carica e con le altre app chiuse. Controlla `crossOriginIsolated: true`, scegli una foto vera e `campaigns` = 3. Lancia `run` con `wasm threads` = 1, premi `copy JSON` e salva. Poi ripeti con `wasm threads` = 4: ogni `run` cancella i risultati precedenti.

```powershell
cd C:\Users\hector\Desktop\Projects\EFREI\M2\MLops\food-recognition
New-Item -ItemType Directory -Force results\bench_raw
# salva i JSON come results\bench_raw\2026-11-xx_<telefono>_t1.json e ..._t4.json
python -m scripts.collect_bench results/bench_raw/*.json
```

## A13 — Scelta della variante

```powershell
python -m scripts.pareto --stage final --backend wasm --threads 1
python -m scripts.pareto --stage final --backend wasm --threads 4
python -m scripts.pareto --stage final --backend webgpu
```

## A14 — Integrazione in CovertCook

```powershell
# La variante scelta in A13
$RUN = "a_convnext"
$VAR = "model_int8_minmax_200_pc"

python -m scripts.export_web_fixtures --run $RUN --n 3
Copy-Item "web\fixtures\parity_$RUN.json" C:\Users\hector\covertcook\src\lib\__fixtures__\parity.json
Copy-Item web\src\lib\foodModel.test.ts C:\Users\hector\covertcook\src\lib\
Copy-Item web\src\pages\FoodScan.tsx C:\Users\hector\covertcook\src\pages\

cd C:\Users\hector\covertcook

$f = "src\pages\FoodScan.tsx"
(Get-Content $f -Raw).Replace("/models/a_lite0__model_int8_minmax_200_pc.onnx", "/models/${RUN}__$VAR.onnx").Replace("/models/a_lite0__meta.json", "/models/${RUN}__meta.json") | Set-Content $f -NoNewline

$f = "src\App.tsx"
(Get-Content $f -Raw).Replace("const LabBench = lazy(() => import('./pages/LabBench'))", "const LabBench = lazy(() => import('./pages/LabBench'))`nconst FoodScan = lazy(() => import('./pages/FoodScan'))").Replace('<Route path="/lab/bench" element={<LabBench />} />', '<Route path="/lab/bench" element={<LabBench />} /> <Route path="/scan" element={<FoodScan />} />') | Set-Content $f -NoNewline

# Modelli e runtime disponibili offline
$f = "src\sw.ts"
(Get-Content $f -Raw).Replace("// The allergen and diet tiles, kept out of the precache", "registerRoute(({ url }) => url.pathname.startsWith('/models/') || url.pathname.startsWith('/ort/'), new CacheFirst({ cacheName: 'ml-models', plugins: [new ExpirationPlugin({ maxEntries: 20 })] }))`n`n// The allergen and diet tiles, kept out of the precache") | Set-Content $f -NoNewline

git diff --stat
npx vitest run src/lib/foodModel.test.ts
npm run build
npx netlify-cli deploy --dir=dist
```

Sul telefono: apri `/scan`, fai una foto, poi attiva la modalità aereo e riprova. Registra il video del checkpoint di fine Parte A.

---

# PARTE D — Piatto + ingredienti

Da iniziare solo dopo il checkpoint A14.

## D1 — Food-101

```powershell
cd C:\Users\hector\Desktop\Projects\EFREI\M2\MLops\food-recognition
python -m scripts.preresize --dataset food101
python -m scripts.make_splits --dataset food101
# completa a mano data/labels/mapping_foodseg_food101.csv
```

## D2 — Training multi-task

```powershell
# in configs/d_multitask.yaml: model.name e run_name del backbone vincitore di A13
python -m scripts.sanity_checks configs/d_multitask.yaml --check init
python -m src.train configs/d_multitask.yaml
```

## D3 — Valutazione

```powershell
python -m src.predict --run d_lite0 --split val
python -m src.calibrate --run d_lite0
python -m src.predict --run d_lite0 --split test
python -m src.metrics results/preds/d_lite0__torch__test.npz
python -m src.predict --run d_lite0 --split own
python -m scripts.unknown_threshold results/preds/d_lite0__torch__own.npz
# 0.5 è provvisorio: sostituiscilo con la soglia scelta dalla tabella qui sopra
python -m src.calibrate --run d_lite0 --unknown-dish-min-prob 0.5
python -m scripts.cross_dataset --run d_lite0
```

## D4 — Export e quantizzazione

```powershell
python -m src.export_onnx --run d_lite0
python -m src.optimize --run d_lite0
python -m src.predict --run d_lite0 --split test --onnx model_fp32.onnx
python -m src.predict --run d_lite0 --split test --onnx model_int8_minmax_200_pc.onnx
python -m src.metrics results/preds/d_lite0__model_int8_minmax_200_pc__test.npz
python -m src.stats results/preds/d_lite0__model_fp32__test.npz results/preds/d_lite0__model_int8_minmax_200_pc__test.npz
```

## D5 — (Opzionale) Baseline solo piatto

```powershell
python -m src.train configs/d_multitask.yaml --set multitask.ingr_loss_weight=0 --set run_name=c_dish_only
```

## D6 — Benchmark e integrazione

```powershell
python -m scripts.copy_to_web --run d_lite0 --variants model_fp32 model_int8_minmax_200_pc --dest C:\Users\hector\covertcook\public\models

cd C:\Users\hector\covertcook
$f = "src\pages\LabBench.tsx"
(Get-Content $f -Raw) -replace '(?s)const VARIANTS = \[.*?\];', 'const VARIANTS = ["/models/d_lite0__model_fp32.onnx", "/models/d_lite0__model_int8_minmax_200_pc.onnx"];' | Set-Content $f -NoNewline
npm run build
npx netlify-cli deploy --dir=dist
# ripeti il benchmark sul telefono come in A12, e salva i JSON in results\bench_raw\ con _d_ nel nome

cd C:\Users\hector\Desktop\Projects\EFREI\M2\MLops\food-recognition
python -m scripts.collect_bench results/bench_raw/*.json
python -m scripts.pareto --stage final --backend wasm --threads 1
```
