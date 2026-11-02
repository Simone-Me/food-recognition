# TP guidato 1 — Riconoscere un piatto da una foto, sul telefono

> Progetto EFREI M2 MLOps — soggetto **« Embarquer un modèle »**
> Dispositivo: telefono Android, modello eseguito nel browser dentro la PWA **CovertCook** (onnxruntime-web).
> Livello atteso: M2. Ogni scelta deve essere **tua**, **misurata** e **giustificabile all'orale**.

---

## Come usare questo TP

Il TP non ti dà "la" soluzione. Ad ogni tappa trovi:

- **🎯 Obiettivo**: cosa deve esistere alla fine della tappa.
- **📚 Da sapere**: i concetti che devi padroneggiare *da solo* prima di procedere, con domande di autoverifica. Se non sai rispondere, studia prima di scrivere codice: all'orale ti chiederanno proprio queste cose.
- **🔀 Scelte**: le opzioni realistiche, con i criteri per decidere. Non c'è un default: decidi tu, sulla base di misure.
- **🛠️ Da fare**: i passi concreti, con scheletri di codice da completare (i `TODO` sono tuoi).
- **📝 Journal**: cosa annotare nel tuo *registro delle decisioni* (vedi §0.4). È quello che ti salva all'orale.
- **✅ Checkpoint**: come sapere che la tappa è finita.

Hai già fatto in corso il lab `quantization/pytorch-fashionMNIST-tutorial` (PyTorch → ONNX → `*_int8.onnx`). Questo TP riprende la stessa catena, ma su un problema reale: immagini a colori, 101 classi, rumore nelle etichette, un dispositivo vero e un utente vero.

---

## 0. Inquadramento

### 0.1 Cosa chiede il soggetto e dove lo copri

| Richiesta del soggetto | Dove nel TP |
|---|---|
| Entraîner ou récupérer un modèle sur un dataset de choix | §2, §3, §4 |
| Le faire tourner sur un dispositif embarqué | §8, §10 |
| 1 à 3 optimisations (quantization, distillation, etc.) | §7 |
| **Niveau 1** — packager un modèle | §6 (ONNX), §10 (integrazione) |
| **Niveau 2** — une ou plusieurs quantizations (post-training, training-aware…) | §7 |
| **Niveau 2** — comparer quantifié vs non quantifié (métrique, taille, vitesse) | §5, §8, §9 |
| **Niveau 2** — déployer le meilleur sur la cible | §9, §10 |

### 0.2 Il problema, formalmente

Classificazione di immagini multi-classe: data un'immagine RGB $x$, stimare $p(y \mid x)$ per $y \in \{1,\dots,K\}$ (con Food-101, $K=101$), e restituire le top-k classi.

Vincoli del dispositivo che **cambiano il problema** rispetto a un notebook:

- **Latenza**: l'utente aspetta. Fissa tu una soglia (es. < 300 ms per immagine sul tuo telefono) e giustificala.
- **Dimensione**: il modello si scarica via rete mobile e finisce nella cache del service worker. Fissa una soglia (es. < 10 MB).
- **Runtime**: non è PyTorch. È onnxruntime-web (WASM o WebGPU), che non supporta tutti gli operatori né tutti i formati quantizzati allo stesso modo.
- **Dominio**: le foto degli utenti (fatte in casa, luce scarsa, piatto di traverso) non sono quelle del dataset.
- **Mondo aperto**: un utente può fotografare qualcosa che non è nelle 101 classi. Il modello risponderà comunque qualcosa.

> 📝 **Journal**: scrivi le tue tre soglie (latenza, dimensione, perdita di accuracy massima accettabile dopo l'ottimizzazione) **prima** di misurare. Fissarle dopo aver visto i numeri è un bias, e un docente lo nota.

### 0.3 Hardware disponibile

- PC: GTX 1650 Ti (4 GB VRAM, architettura Turing, compute capability 7.5), 16 GB RAM, Python 3.13.
- Opzionale: Kaggle Notebooks (GPU gratuita, quota settimanale) o Google Colab, per i modelli grandi (insegnante per la distillazione).
- Telefono Android con Chrome: è il **dispositivo di destinazione**, le misure finali si fanno lì.

### 0.4 Il registro delle decisioni (obbligatorio per te)

Crea `docs/decisions.md` nel repo. Una voce per ogni scelta:

```markdown
## D-003 — Scelta della risoluzione di input
- Data: 2026-11-xx
- Contesto: la latenza su telefono con 224px è 410 ms, sopra la soglia di 300 ms.
- Opzioni considerate: 224 / 192 / 160 ; modello più piccolo ; int8.
- Misure: (tabella o link a results/xxx.csv)
- Decisione: 192px
- Perché: -0.9 pt top-1, -35% latenza ; il costo è accettabile rispetto alla soglia D-001.
- Cosa mi farebbe cambiare idea: se la WebGPU fosse disponibile su più telefoni.
```

All'orale, ogni domanda "perché hai scelto X?" ha una risposta già scritta.

### 0.5 Struttura del repo consigliata

Il codice di addestramento è Python e non appartiene a CovertCook (che è un'app React). Due possibilità, sceglila tu:

| Opzione | Pro | Contro |
|---|---|---|
| **Repo separato** `food-recognition` | Storia pulita, valutabile da solo dal docente, dipendenze Python isolate | Devi copiare il `.onnx` finale in CovertCook |
| Cartella `ml/` dentro CovertCook | Un solo repo, il modello e l'app evolvono insieme | Mescola due ecosistemi; il docente vede anche tutta l'app |

```
food-recognition/
├── README.md               # come riprodurre tutto, dall'inizio
├── requirements.txt        # versioni FISSATE (pip freeze)
├── configs/                # un file yaml per esperimento
├── data/                   # gitignore — dataset scaricati e preprocessati
├── src/
│   ├── data.py             # dataset, split, trasformazioni
│   ├── models.py           # creazione dei modelli candidati
│   ├── train.py
│   ├── evaluate.py         # metriche su PyTorch E su ONNX
│   ├── export_onnx.py
│   ├── optimize.py         # fp16, quantizzazioni
│   └── stats.py            # bootstrap, McNemar
├── notebooks/              # EDA, analisi errori (non codice di produzione)
├── models/                 # gitignore (o Git LFS) — checkpoint e .onnx
├── results/                # CSV delle misure, figure: questi SÌ in git
└── docs/
    ├── decisions.md
    └── report.md
```

> 📚 **Da sapere — riproducibilità**
> - Perché fissare `torch.manual_seed`, `numpy.random.seed`, `random.seed` non basta per avere risultati identici su GPU? (cerca `torch.backends.cudnn.deterministic`, operazioni non deterministiche).
> - Perché si salvano le misure in CSV versionati invece di copiarle a mano nel report?
> - Che differenza c'è tra fissare le versioni con `pip freeze` e con un lockfile (`uv lock`, `pip-tools`)?

---

## 1. Ambiente

### 🎯 Obiettivo
Un ambiente Python isolato in cui `torch.cuda.is_available()` restituisce `True` e un modello timm fa una forward pass sulla GPU.

### 🔀 Scelte

| Gestore | Quando sceglierlo |
|---|---|
| `venv` + `pip` | Il più semplice, già con Python 3.13 |
| `uv` | Molto più veloce, lockfile integrato, gestisce anche la versione di Python |
| `conda` / `mamba` | Se ti servono librerie native non disponibili via pip (qui non dovrebbe essere il caso) |

**Versione di Python**: 3.13 è recente. Prima di iniziare verifica che *tutte* le librerie di cui avrai bisogno abbiano wheel per 3.13 (torch, onnxruntime, onnxconverter-common, eventualmente nncf). Se una manca, un ambiente 3.11/3.12 ti evita ore di compilazione. → 📝 Journal.

### 🛠️ Da fare

```bash
python -m venv .venv
.venv\Scripts\activate
```

Per PyTorch con CUDA **non** fare `pip install torch` a caso: vai su pytorch.org → "Get Started", seleziona Windows / Pip / CUDA, e usa il comando che ti dà (contiene un `--index-url` con la versione CUDA). La 1650 Ti è supportata dalle build CUDA 12.x.

Poi:

```bash
pip install timm onnx onnxruntime onnxconverter-common torchinfo pandas scikit-learn statsmodels matplotlib pyyaml tqdm
```

Verifica:

```python
import torch, timm
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
m = timm.create_model("efficientnet_b0", pretrained=True).cuda().eval()
print(m(torch.randn(1, 3, 224, 224, device="cuda")).shape)  # torch.Size([1, 1000])
```

> 📚 **Da sapere**
> - Differenza tra driver NVIDIA, CUDA toolkit e la CUDA "runtime" contenuta nella wheel di PyTorch. Perché non devi installare il CUDA toolkit per usare PyTorch?
> - Cos'è un "tensor core" e perché la 1650 Ti (TU117) non ne ha. Cosa implica per la mixed precision (AMP): guadagni memoria, ma di velocità?

### ✅ Checkpoint
La forward pass sulla GPU funziona; `requirements.txt` committato.

---

## 2. Dati

### 🎯 Obiettivo
Un dataset scaricato, esplorato, diviso in train/val/test in modo riproducibile, più un piccolo **set personale** di foto tue.

### 📚 Da sapere

- **Food-101** (Bossard et al., ECCV 2014): 101 classi, 1000 immagini per classe, 750 train / 250 test per classe. Le immagini di **train contengono volutamente rumore** (etichette sbagliate, colori strani); quelle di test sono state pulite a mano. Leggi la sezione "Dataset" del paper: devi saperlo citare.
- Perché **non** si usa il test set per scegliere gli iperparametri? Cos'è il *data leakage*?
- Cosa sono la *distribution shift* e in particolare la *covariate shift*? Perché le tue foto di casa sono un caso da manuale?
- Cosa succede a un classificatore softmax davanti a un'immagine di una classe che non conosce (*open-set recognition*)?

### 🔀 Scelta 1 — Quale dataset

| Dataset | Classi | Immagini | Note |
|---|---|---|---|
| **Food-101** | 101 | 101k | Lo standard; classi molto "internazionali" (sushi, takoyaki, poutine…); disponibile in `torchvision` |
| UEC Food-256 | 256 | ~31k | Cucina soprattutto giapponese; include bounding box |
| ISIA Food-500 | 500 | ~400k | Molto più grande; addestramento più lungo |
| Food2K | 2000 | ~1M | Ricerca; troppo grande per una 1650 Ti |
| Food-101 + classi tue | 101+n | — | Aggiungi piatti che cucini davvero (es. *risotto*, *tiramisù* ci sono già, ma *vitello tonnato* no) — implica raccogliere e pulire immagini |

Criteri: pertinenza per gli utenti di CovertCook (cene tra amici, cucina europea), costo di addestramento sulla tua GPU, disponibilità e licenza. → 📝 Journal.

### 🔀 Scelta 2 — Tutte le classi o un sottoinsieme?

- **Tutte le 101**: confrontabile con la letteratura (puoi citare risultati pubblicati), ma include classi poco utili.
- **Sottoinsieme** (es. 30 classi "europee"): modello più facile, più accurato, ma non confrontabile e la scelta del sottoinsieme va giustificata.

### 🔀 Scelta 3 — Come costruire la validazione

Food-101 ha solo train/test. Ti serve un val set preso dal train.
- **Split stratificato** 90/10 dal train (es. `sklearn.model_selection.train_test_split(..., stratify=labels, random_state=42)`): salva gli indici su disco, così lo split è identico in tutti gli esperimenti.
- Problema: il val viene dal train, che è **rumoroso**. Le tue misure su val saranno quindi pessimiste. Va bene, ma devi saperlo spiegare.

### 🛠️ Da fare

1. **Scaricare**

   ```python
   from torchvision.datasets import Food101
   train = Food101(root="data", split="train", download=True)   # ~5 GB
   test  = Food101(root="data", split="test",  download=True)
   ```

2. **EDA** (`notebooks/01_eda.ipynb`): distribuzione delle classi, distribuzione delle dimensioni (larghezza/altezza), 5 immagini a caso per 10 classi, prova a trovare tu qualche etichetta sbagliata nel train.

3. **Pre-ridimensionamento (opzionale, ma misuralo)**: le immagini sono fino a 512 px. Decodificare JPEG grandi a ogni epoca è spesso il collo di bottiglia su Windows (la GPU aspetta la CPU). Ridimensionare una volta il lato corto a ~256 px su disco può moltiplicare il throughput. Misura immagini/secondo prima e dopo. → 📝 Journal.

4. **Il tuo set personale** (`data/own/<classe>/*.jpg`): da 50 a 150 foto di piatti veri (tuoi, di amici, di un ristorante), solo per classi presenti nel dataset, più 20-30 foto di **cose che non sono nel dataset** (un piatto fuori classe, un tavolo, un gatto). Non userai mai questo set per addestrare: serve a misurare lo scarto tra il dataset e la realtà, e la gestione del mondo aperto (§5.4).

> ⚠️ Su Windows, un `DataLoader` con `num_workers > 0` richiede che lo script principale sia protetto da `if __name__ == "__main__":`. Senza, i worker rilanciano lo script all'infinito.

### ✅ Checkpoint
`data/splits/train_idx.npy` e `val_idx.npy` salvati; notebook EDA con almeno 3 osservazioni scritte; set personale raccolto.

---

## 3. Scelta del modello

### 🎯 Obiettivo
Una lista ragionata di candidati, una prima selezione a budget ridotto, e **due finalisti** da addestrare a fondo.

### 📚 Da sapere

Prima di confrontare architetture devi saper spiegare:

- **Parametri vs FLOPs/MACs vs latenza reale**: perché due modelli con gli stessi FLOPs possono avere latenze molto diverse su telefono? (accesso alla memoria, operatori depthwise, supporto del runtime).
- **Depthwise separable convolution** (MobileNetV1): perché riduce il costo? Quanto, in formula?
- **Inverted residual + linear bottleneck** (MobileNetV2).
- **Squeeze-and-Excitation** e **hard-swish** (MobileNetV3): perché aiutano l'accuracy ma possono **danneggiare la quantizzazione int8**?
- **Compound scaling** (EfficientNet) e perché esiste la variante **EfficientNet-Lite** (niente SE, ReLU6 invece di swish → pensata per int8).
- **Vision Transformer**: cos'è la self-attention, perché è costosa in $O(n^2)$ nel numero di patch, e perché LayerNorm/GELU/softmax sono difficili da quantizzare.
- **Transfer learning**: perché partire da pesi ImageNet funziona su cibo?

Letture base: MobileNetV3 (Howard et al., 2019), EfficientNet (Tan & Le, 2019), MobileViT (Mehta & Rastegari, 2022).

### 🔀 Candidati

I numeri sono **ordini di grandezza** per ImageNet: verificali tu con `torchinfo` e con il file `results-imagenet.csv` del repo GitHub di timm, e completa la tabella con le tue misure.

| Famiglia | Nome timm (da verificare con `timm.list_models(pretrained=True)`) | Parametri | Perché considerarlo | Rischio |
|---|---|---|---|---|
| MobileNetV3 | `mobilenetv3_large_100` | ~5.5 M | Pensato per mobile, molto veloce | SE + hard-swish: perdita int8 potenzialmente alta |
| MobileNetV3 small | `mobilenetv3_small_100` | ~2.5 M | Il più veloce | Accuracy più bassa |
| EfficientNet-Lite | `tf_efficientnet_lite0` | ~4.7 M | Progettato per int8 | Leggermente meno accurato di B0 in fp32 |
| EfficientNet | `efficientnet_b0` | ~5.3 M | Ottimo compromesso accuracy/dimensione | Swish + SE |
| MobileViT | `mobilevit_xs`, `mobilevit_s` | ~2.3 / 5.6 M | Ibrido CNN-Transformer, interessante da discutere | Supporto runtime e quantizzazione più delicati |
| ConvNeXt piccoli | `convnext_atto`, `convnext_femto` | ~3.7 / 5.2 M | CNN "moderna" | LayerNorm e GELU |
| RegNet | `regnety_004` | ~4.3 M | Progettato per essere regolare e veloce | — |
| **Riferimento** | `resnet50` | ~25 M | Punto di confronto classico, non per il telefono | Troppo grande |
| **Insegnante** (opz.) | `vit_base_patch16_224`, `efficientnetv2_rw_s`, `convnext_tiny` | 20–90 M | Per la distillazione (§7.5) | Va addestrato su Kaggle/Colab |

Scegli **4-6 candidati** coprendo famiglie diverse. Una scelta di soli MobileNet non ti insegna nulla e non si difende all'orale.

### 🛠️ Da fare — protocollo di selezione

**Passo A — misure "a vuoto" (senza addestrare)**, per ogni candidato:

```python
import timm, torch
from torchinfo import summary

def profile(name, res=224):
    m = timm.create_model(name, pretrained=True, num_classes=101).eval()
    s = summary(m, input_size=(1, 3, res, res), verbose=0)
    cfg = timm.data.resolve_data_config({}, model=m)   # mean/std/res attesi dal modello!
    return {"model": name, "params_M": s.total_params / 1e6,
            "mult_adds_G": s.total_mult_adds / 1e9, "cfg": cfg}
```

Poi esporta ognuno in ONNX (§6) e misura la latenza **già sul telefono** (§8). Sembra presto, ma è l'informazione che più spesso fa scartare un candidato: un modello che è "piccolo" su carta ma lento in onnxruntime-web non ti serve.

**Passo B — screening a budget ridotto**: stesso protocollo per tutti (stesso sottoinsieme, es. 20% del train; stesse 3-5 epoche; stessa ricetta di §4), misura l'accuracy su val.

> 📚 **Da sapere — equità del confronto**
> - Uno screening breve favorisce i modelli che convergono in fretta, non necessariamente quelli migliori a convergenza. Come lo mitighi? (curve di apprendimento, due budget diversi, …)
> - Usare la *stessa* ricetta per tutti è equo? Un ViT e una CNN non hanno gli stessi iperparametri ottimali. Cosa scegli e come lo giustifichi?

**Passo C — fronte di Pareto**: grafico latenza-sul-telefono (asse x) vs accuracy-val (asse y), un punto per modello. I modelli *dominati* (più lenti E meno accurati di un altro) si eliminano. Scegli 2 finalisti sul fronte, idealmente di natura diversa (es. uno "int8-friendly" e uno più accurato).

### 📝 Journal
Tabella completa dei candidati, il grafico di Pareto, i 2 finalisti e perché.

### ✅ Checkpoint
`results/screening.csv` + `results/pareto.png` + voce di journal.

---

## 4. Addestramento

### 🎯 Obiettivo
I 2 finalisti addestrati a convergenza, con curve, checkpoint e iperparametri registrati.

### 📚 Da sapere

- **Strategie di transfer learning**: *linear probing* (solo l'ultimo strato), *fine-tuning completo*, *sblocco graduale*. Quando si usa ciascuna? Cos'è il *catastrophic forgetting*?
- **Ottimizzatori**: SGD+momentum vs AdamW. Cos'è il *weight decay* "decoupled" di AdamW (Loshchilov & Hutter)?
- **Scheduling del learning rate**: warmup + cosine decay. Perché il warmup è utile soprattutto quando si parte da pesi pre-addestrati?
- **Label smoothing**: perché è particolarmente sensato su Food-101 (etichette di train rumorose)?
- **Data augmentation**: `RandomResizedCrop`, flip, `TrivialAugment`/`RandAugment`, `MixUp`/`CutMix`. Quali hanno senso per il cibo? (Un piatto capovolto verticalmente è ancora lo stesso piatto? E una rotazione di 90°?)
- **Mixed precision (AMP)** e `GradScaler`: cosa risolve il *loss scaling*?
- **Normalizzazione dell'input**: ogni modello pre-addestrato si aspetta una sua media/deviazione standard e risoluzione (`timm.data.resolve_data_config`). Sbagliarla non dà errori, dà solo un modello peggiore, ed è il bug n.1 quando porterai il modello nel browser.
- **Batch Normalization**: cosa cambia tra `train()` ed `eval()`? Perché un batch piccolo (4 GB di VRAM!) è un problema per la BN?

### 🔀 Scelte (tutte da giustificare)

| Decisione | Opzioni | Come scegliere |
|---|---|---|
| Strategia | linear probe → fine-tune completo / fine-tune diretto | Prova entrambe su un finalista, confronta val acc e tempo |
| Ottimizzatore | AdamW / SGD | Un piccolo *LR range test* (Smith, 2017) per trovare l'ordine di grandezza |
| LR | Da scoprire | LR range test; LR più basso per il backbone che per la testa (*discriminative LR*) |
| Epoche | 10–30 | Early stopping sulla val loss/accuracy |
| Batch size | 32 / 64 + gradient accumulation | Il massimo che entra in 4 GB con AMP; misura |
| Augmentation | base / + TrivialAugment / + MixUp-CutMix | Ablation: aggiungi una alla volta |
| Label smoothing | 0 / 0.1 | Ablation |
| Risoluzione | 224 / 192 / 160 | La risoluzione è **anche un'ottimizzazione** (§7.6): pensaci già qui |

### 🛠️ Da fare — scheletro di `train.py`

```python
import torch, timm, time, csv
from torch import nn
from torch.utils.data import DataLoader

def main(cfg):
    torch.manual_seed(cfg["seed"])
    device = "cuda"

    model = timm.create_model(cfg["model"], pretrained=True, num_classes=101)
    data_cfg = timm.data.resolve_data_config({}, model=model)
    train_tf = timm.data.create_transform(**data_cfg, is_training=True)   # TODO: decidi l'augmentation
    eval_tf  = timm.data.create_transform(**data_cfg, is_training=False)

    train_ds, val_ds = ...  # TODO: Food101 + i tuoi indici salvati (torch.utils.data.Subset)
    train_dl = DataLoader(train_ds, batch_size=cfg["bs"], shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["bs"] * 2, num_workers=4, pin_memory=True)

    model = model.to(device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])  # TODO: o SGD
    scheduler = ...  # TODO: warmup lineare + cosine (torch.optim.lr_scheduler.SequentialLR)
    scaler = torch.amp.GradScaler("cuda")

    best = 0.0
    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        for x, y in train_dl:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        val_acc = evaluate(model, val_dl, device)       # TODO: in evaluate.py
        log_row(cfg, epoch, loss.item(), val_acc, time.time() - t0)  # TODO: CSV o MLflow
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), f"models/{cfg['run_name']}.pt")

if __name__ == "__main__":   # obbligatorio su Windows con num_workers > 0
    main(load_yaml(...))
```

**Tracking degli esperimenti** — scegli:

| Strumento | Pro | Contro |
|---|---|---|
| CSV in `results/` | Zero dipendenze, versionato in git | Niente UI, confronti fatti a mano |
| **MLflow** (locale, `mlflow ui`) | Registra parametri, metriche, artefatti; model registry; l'hai visto in corso | Un servizio in più |
| Weights & Biases | UI eccellente | Servizio esterno, account |

MLflow non è obbligatorio per questo soggetto (lo è per il soggetto MLOps), ma ti fa confrontare decine di run senza perderti.

### 📝 Journal
Per ogni ablation: cosa hai cambiato, l'effetto misurato, la decisione.

### ✅ Checkpoint
2 checkpoint `.pt`, curve train/val (loss e accuracy), nessun segno evidente di overfitting non gestito, tempo per epoca annotato.

---

## 5. Protocollo di valutazione

### 🎯 Obiettivo
Uno script `evaluate.py` che, dato **un modello PyTorch o un file ONNX**, produce le stesse metriche sullo stesso test set. Lo riuserai per ogni variante ottimizzata, quindi deve essere solido.

### 📚 Da sapere

- **Top-1 / top-5 accuracy**: perché la top-3 o top-5 è pertinente per la tua UI (mostri 3 proposte all'utente)?
- **Macro-F1** vs accuracy: su Food-101 test (bilanciato) coincidono quasi; sul tuo set personale (sbilanciato) no.
- **Matrice di confusione**: quali piatti si confondono e perché (es. *steak* / *filet mignon* / *prime rib*)? Distingui errori "accettabili" da errori gravi.
- **Calibrazione** (Guo et al., 2017, *On Calibration of Modern Neural Networks*): un modello che dice "90%" ha ragione il 90% delle volte? *Expected Calibration Error* (ECE), *reliability diagram*, *temperature scaling*. È necessario per §5.4.
- **Incertezza statistica**: 25 250 immagini di test danno un'accuracy con un intervallo di confidenza. Quanto è largo? (bootstrap). Una differenza di 0,3 punti tra fp32 e int8 è significativa?
- **Test di McNemar**: confronta due classificatori *sulle stesse immagini*. Perché è più adatto di confrontare due accuracy con due intervalli indipendenti?

### 🛠️ Da fare

```python
import numpy as np

def bootstrap_ci(correct: np.ndarray, n_boot=2000, alpha=0.05, seed=0):
    """correct: vettore 0/1, un elemento per immagine di test."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(axis=1)
    return np.quantile(accs, [alpha / 2, 1 - alpha / 2])

from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray):
    # TODO: costruisci la tabella 2x2 [[a&b, a&!b], [!a&b, !a&!b]]
    table = ...
    return mcnemar(table, exact=False, correction=True).pvalue
```

Salva per ogni modello valutato un file `results/preds_<variante>.npy` con le predizioni per immagine: così ogni confronto statistico futuro è immediato, senza rieseguire l'inferenza.

### 5.4 Mondo aperto: "non lo so"

Il modello deve poter rispondere *"piatto non riconosciuto"*.

**🔀 Opzioni**:
1. Soglia sulla probabilità massima della softmax (*maximum softmax probability*, Hendrycks & Gimpel, 2017), su un modello calibrato con *temperature scaling*.
2. Soglia sull'entropia della distribuzione.
3. Una classe "altro" addestrata (richiede dati "negativi": quali?).

Per scegliere la soglia, usa il tuo set personale (dentro + fuori classe): traccia la curva *copertura vs accuracy sulle risposte date* e scegli il punto operativo. → 📝 Journal.

### ✅ Checkpoint
`evaluate.py` produce: top-1, top-5, macro-F1, IC al 95%, ECE, matrice di confusione, i 10 errori più frequenti; sia sul test di Food-101 sia sul set personale.

---

## 6. Export ONNX (Niveau 1 — packaging)

### 🎯 Obiettivo
Un `.onnx` per ogni finalista, **numericamente equivalente** al modello PyTorch.

### 📚 Da sapere

- Cos'è ONNX: un formato di **grafo** di operatori standardizzati, versionato da un *opset*. Perché il runtime del browser ha bisogno di un formato del genere invece dei pesi PyTorch?
- Export via *tracing* vs export via `torch.export` (dynamo). Nelle versioni recenti di PyTorch, `torch.onnx.export` può usare l'uno o l'altro (parametro `dynamo`): verifica il comportamento predefinito della tua versione nella documentazione e annota quale hai usato.
- Assi dinamici (`dynamic_axes` / `dynamic_shapes`): servono un batch dinamico? Nel browser fai batch=1. Un asse fisso permette più ottimizzazioni.
- Opset: perché un opset troppo recente può non essere supportato da onnxruntime-web?

### 🛠️ Da fare

```python
import torch, timm, onnx, onnxruntime as ort, numpy as np

model = timm.create_model(NAME, num_classes=101)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()
dummy = torch.randn(1, 3, RES, RES)

torch.onnx.export(model, dummy, OUT, input_names=["input"], output_names=["logits"],
                  opset_version=17)   # TODO: decidi e giustifica opset e assi dinamici

onnx.checker.check_model(onnx.load(OUT))

# Test di parità: OBBLIGATORIO
sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
x = np.random.randn(8, 3, RES, RES).astype(np.float32)  # o immagini vere, meglio
with torch.no_grad():
    ref = model(torch.from_numpy(x)).numpy()
out = sess.run(None, {"input": x})[0]   # adatta se hai fissato batch=1
print("max |diff|:", np.abs(ref - out).max())
print("accordo top-1:", (ref.argmax(1) == out.argmax(1)).mean())
```

Ispeziona il grafo con **Netron** (netron.app): devi saper mostrare all'orale dove sono le convoluzioni, la testa, e (dopo la §7) i nodi `QuantizeLinear`/`DequantizeLinear`.

### 📝 Journal
Opset, exporter, assi dinamici, e il risultato del test di parità (atteso: differenze ~1e-5, accordo top-1 ≈ 100%).

### ✅ Checkpoint
`evaluate.py` sul `.onnx` dà la stessa accuracy del `.pt` (entro l'IC).

---

## 7. Ottimizzazioni (Niveau 2)

### 🎯 Obiettivo
Da **1 a 3** ottimizzazioni applicate ai finalisti, ognuna misurata con lo stesso protocollo (§5 + §8). Non devi fare tutto: scegli e giustifica.

### 📚 Da sapere — la quantizzazione, seriamente

Letture: Jacob et al., 2018 (*Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*); Krishnamoorthi, 2018 (*Quantizing deep convolutional networks for efficient inference: A whitepaper*); Nagel et al., 2021 (*A White Paper on Neural Network Quantization*). Devi saper spiegare a voce:

- **Quantizzazione affine**: $q = \text{round}(x / s) + z$, $\hat{x} = s\,(q - z)$. Cosa sono la *scale* $s$ e lo *zero-point* $z$? Come si calcolano da un intervallo $[\alpha, \beta]$?
- **Simmetrica vs asimmetrica**; **per-tensor vs per-channel** (perché il per-channel sui pesi delle convoluzioni recupera molta accuracy?).
- **Errore di arrotondamento vs errore di clipping**: il compromesso dietro la scelta dell'intervallo. Perché min/max è sensibile agli outlier?
- **Pesi vs attivazioni**: i pesi sono noti in anticipo, le attivazioni dipendono dall'input. Da qui la differenza tra quantizzazione **dinamica** (scala delle attivazioni calcolata a runtime) e **statica** (scala calcolata una volta con un set di **calibrazione**).
- **Metodi di calibrazione**: MinMax, Entropy (KL), Percentile.
- **Formato QDQ vs QOperator** in ONNX: nodi `QuantizeLinear`/`DequantizeLinear` intorno agli operatori, oppure operatori interi dedicati (`QLinearConv`). Il runtime decide poi se "fondere" i QDQ in calcoli interi veri.
- **Batch-norm folding**: perché si fonde la BN nella convoluzione prima di quantizzare?
- **Perché la quantizzazione può non accelerare**: se il runtime/backend non ha kernel interi per un operatore, dequantizza e calcola in float, quindi il modello è più piccolo ma non più veloce (a volte più lento). Questo dipende dal backend (WASM vs WebGPU) e va **misurato sul telefono**.

### 🔀 Il menu delle ottimizzazioni

| # | Ottimizzazione | Costo per te | Cosa ti aspetti | Rischi / punti da discutere |
|---|---|---|---|---|
| 7.1 | **FP16** (pesi in mezza precisione) | Basso | Dimensione ÷2; velocità ↑ solo se il backend calcola in fp16 (WebGPU sì, WASM in generale no) | Overflow raro; spesso `keep_io_types=True` |
| 7.2 | **PTQ dinamica** int8 | Basso | Dimensione ÷~4 | Pensata soprattutto per MatMul/Gemm (transformer, RNN); su una CNN guadagno di velocità spesso modesto: verifica quali nodi sono stati quantizzati |
| 7.3 | **PTQ statica** int8 (QDQ, per-channel) | Medio | Dimensione ÷~4, velocità ↑ se il backend ha kernel int8 | Scelta del set di calibrazione (quante immagini? quali?) e del metodo; modelli con SE/hard-swish possono perdere punti |
| 7.4 | **QAT** (quantization-aware training) | Alto | Recupera la perdita della PTQ | La catena PyTorch → ONNX con QAT è la più fragile (API `torch.ao` in transizione verso `torchao`/PT2E; alternativa: NNCF di OpenVINO). Fallo solo se la PTQ statica perde più della tua soglia |
| 7.5 | **Distillazione** (insegnante grande → studente piccolo) | Medio-alto | Accuracy dello studente ↑ a parità di dimensione | Serve addestrare l'insegnante (Kaggle); scegliere $T$ e $\alpha$ |
| 7.6 | **Risoluzione di input ridotta** (224 → 192 → 160) | Basso | Latenza ∝ circa pixel² | Meno dettagli: perdita di accuracy da misurare; cambia anche il preprocessing nel browser |
| 7.7 | **Pruning** | Medio | Dipende | Il pruning *non strutturato* rende i pesi sparsi ma un runtime denso non va più veloce; quello *strutturato* (canali interi) sì, ma richiede re-training |

**Una combinazione sensata** (da non prendere come default: giustifica la tua) è: una variante "economica" (7.1 o 7.6), una quantizzazione int8 (7.3), e un tentativo di recupero (7.4 o 7.5) se la perdita supera la soglia.

### 🛠️ Da fare

**7.1 — FP16**

```python
import onnx
from onnxconverter_common import float16
m = onnx.load("models/b0.onnx")
m16 = float16.convert_float_to_float16(m, keep_io_types=True)
onnx.save(m16, "models/b0_fp16.onnx")
```

**7.2 / 7.3 — Quantizzazione con onnxruntime**

```python
from onnxruntime.quantization import (quantize_dynamic, quantize_static, QuantType,
                                      QuantFormat, CalibrationDataReader, CalibrationMethod)
from onnxruntime.quantization.shape_inference import quant_pre_process

# Preprocessing raccomandato da ORT (ottimizzazione del grafo + shape inference)
quant_pre_process("models/b0.onnx", "models/b0_pre.onnx")

# Dinamica
quantize_dynamic("models/b0_pre.onnx", "models/b0_dyn.onnx", weight_type=QuantType.QInt8)

# Statica
class FoodCalibReader(CalibrationDataReader):
    def __init__(self, images, input_name="input"):
        # TODO: images = array (N, 3, H, W) già preprocessate ESATTAMENTE come in inferenza
        self.it = iter([{input_name: img[None]} for img in images])
    def get_next(self):
        return next(self.it, None)

quantize_static("models/b0_pre.onnx", "models/b0_int8.onnx",
                calibration_data_reader=FoodCalibReader(calib_images),
                quant_format=QuantFormat.QDQ,
                per_channel=True,                       # TODO: confronta True/False
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax)  # TODO: confronta MinMax/Entropy/Percentile
```

Esperimenti da fare sulla PTQ statica (ognuno è una riga nella tua tabella):
- numero di immagini di calibrazione: 50 / 200 / 1000 (da **val**, mai dal test);
- metodo: MinMax / Entropy / Percentile;
- per-channel: sì / no;
- se un modello perde molto: identifica i livelli sensibili (quantizza tutto tranne un blocco alla volta, oppure usa il debugger di quantizzazione di ORT, `onnxruntime.quantization.qdq_loss_debug`) e prova a lasciarli in float (`nodes_to_exclude`). È una *quantizzazione mista*, discutibile e interessante all'orale.

**7.5 — Distillazione** (perdita di Hinton et al., 2015)

```python
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, y, T=4.0, alpha=0.7):
    soft = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean") * (T * T)
    hard = F.cross_entropy(student_logits, y)
    return alpha * soft + (1 - alpha) * hard
```

> 📚 Perché si moltiplica per $T^2$? Cosa contengono le "dark knowledge" dei logit dell'insegnante che le etichette one-hot non hanno? Per non far girare l'insegnante a ogni epoca sulla 1650 Ti, puoi pre-calcolare i suoi logit una volta per tutte (su Kaggle) e salvarli: ma con l'augmentation casuale i logit pre-calcolati non corrispondono più all'immagine vista dallo studente. Come gestisci il problema?

### 📝 Journal
Una voce per ogni ottimizzazione: ipotesi → misura → conclusione. Anche (soprattutto) quando non funziona: "la PTQ dinamica non accelera la CNN perché solo 3 nodi su 80 sono stati quantizzati" è un ottimo risultato da presentare.

### ✅ Checkpoint
Ogni variante ha: un `.onnx`, la sua dimensione su disco, le metriche di §5 (con IC e McNemar rispetto a fp32).

---

## 8. Benchmark sul dispositivo

### 🎯 Obiettivo
Misure di latenza **affidabili** di ogni variante, su **telefono vero**, nel **runtime vero**.

### 📚 Da sapere

- **onnxruntime-web** ha due backend principali:
  - **WASM** (CPU): sempre disponibile; può usare SIMD e più thread. I thread richiedono `SharedArrayBuffer`, quindi una pagina **cross-origin isolated** (header `Cross-Origin-Opener-Policy: same-origin` e `Cross-Origin-Embedder-Policy: require-corp` o `credentialless`).
  - **WebGPU** (GPU del telefono): più veloce per fp32/fp16 sui modelli supportati, ma la disponibilità dipende da telefono e versione di Chrome, e il supporto int8 è più limitato.
  - Verifica tu nella documentazione attuale di onnxruntime-web cosa supporta ciascun backend: le cose cambiano velocemente.
- **Metodologia di benchmark**: riscaldamento (le prime esecuzioni compilano/allocano), numero di ripetizioni, mediana e p95 invece della media, *thermal throttling* (il telefono rallenta scaldandosi), altre app aperte, modalità risparmio batteria.
- **Cosa si misura**: solo `session.run`? O anche il preprocessing (resize su canvas, normalizzazione)? E il tempo di caricamento del modello (*cold start*)? Sono tre numeri diversi: misurali tutti e tre.

### ⚠️ Vincoli specifici di CovertCook

- `public/_headers` (Netlify) definisce già degli header di sicurezza. La **Content-Security-Policy** attuale ha `script-src 'self'`: **blocca la compilazione di WebAssembly**, quindi onnxruntime-web fallirà finché non aggiungi `'wasm-unsafe-eval'` a `script-src`. Capisci cosa autorizza questa direttiva (e perché è molto meno pericolosa di `'unsafe-eval'`) prima di aggiungerla: è una modifica di sicurezza da giustificare. Anche i file `.wasm`/`.onnx` devono essere caricabili (`connect-src 'self'` va bene se li servi tu).
- Attivare COOP/COEP su **tutto** il sito può rompere risorse caricate da altri domini (immagini, Supabase). Valuta di attivarli **solo sul percorso della pagina di benchmark/riconoscimento** (Netlify permette header per percorso), oppure di restare in single-thread e documentarlo. → 📝 Journal.
- onnxruntime-web carica i suoi file `.wasm` a runtime: con Vite devi decidere se servirli dal tuo sito (copiandoli in `public/`) o da un CDN (`ort.env.wasm.wasmPaths`). Il CDN contraddice la CSP e la logica offline della PWA: perché?

### 🛠️ Da fare — una pagina di benchmark

Una pagina nascosta in CovertCook (es. `/lab/bench`, non nel menu), che:

```ts
import * as ort from "onnxruntime-web";

async function bench(modelUrl: string, ep: "wasm" | "webgpu", res: number, runs = 100, warmup = 10) {
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(modelUrl, { executionProviders: [ep] });
  const loadMs = performance.now() - t0;

  const input = new ort.Tensor("float32", new Float32Array(3 * res * res), [1, 3, res, res]);
  const feeds = { [session.inputNames[0]]: input };
  for (let i = 0; i < warmup; i++) await session.run(feeds);

  const times: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t = performance.now();
    await session.run(feeds);
    times.push(performance.now() - t);
  }
  times.sort((a, b) => a - b);
  return { loadMs, median: times[Math.floor(runs / 2)], p95: times[Math.floor(runs * 0.95)] };
  // TODO: invia o mostra il risultato in JSON, con modello, backend, numThreads, user agent
}
```

Matrice minima delle misure: **variante × backend (wasm 1 thread / wasm N thread / webgpu) × 3 ripetizioni della campagna intera**, con il telefono a temperatura ambiente all'inizio di ogni campagna.

> 📚 Perché `performance.now()` e non `Date.now()`? Perché alcuni browser riducono la precisione di `performance.now()` se la pagina non è cross-origin isolated, e cosa cambia per misure di qualche millisecondo?

### ✅ Checkpoint
`results/device_bench.csv`: una riga per (variante, backend, campagna), con load, mediana, p95, dimensione del file, modello del telefono.

---

## 9. Scegliere la variante da deployare

### 🎯 Obiettivo
Una decisione motivata, e il grafico che la sostiene.

### 🛠️ Da fare

1. Tabella finale (una riga per variante):

| Variante | Backend | Dim. (MB) | Top-1 (IC 95%) | Top-5 | Δ vs fp32 (p McNemar) | ECE | Latenza mediana / p95 (ms) | Load (ms) | Set personale top-1 |
|---|---|---|---|---|---|---|---|---|---|

2. Grafico accuracy vs latenza-sul-telefono, con le tue soglie di §0.2 disegnate.
3. Applica le soglie: elimina ciò che non le rispetta, poi scegli sul fronte di Pareto.

> 📚 **Domande che ti faranno**
> - "La versione int8 è più piccola ma non più veloce. Perché?"
> - "La perdita di 0,4 punti è significativa?" (→ IC, McNemar)
> - "Su un altro telefono senza WebGPU, la tua scelta resta valida?" (→ fallback)
> - "Perché non hai semplicemente preso il modello più piccolo?"

### 📝 Journal
La decisione finale, e cosa la farebbe cambiare.

---

## 10. Integrazione in CovertCook (deploy sul dispositivo)

### 🎯 Obiettivo
Nella PWA, l'utente scatta o sceglie una foto e vede le 3 ipotesi più probabili (o "non riconosciuto"), con l'inferenza **interamente sul telefono**.

### 📚 Da sapere

- Perché **embarqué** qui è un argomento di prodotto e non solo tecnico: la foto non lascia mai il telefono (privacy, RGPD), funziona offline, non costa server.
- **Preprocessing in JS identico a Python**: resize (quale interpolazione?), center crop, conversione HWC→CHW, divisione per 255, sottrazione di media e divisione per deviazione standard **del modello scelto**. Un errore qui non lancia eccezioni, fa solo crollare l'accuracy. Per questo serve un **test di parità end-to-end** (stessa immagine in Python e nel browser → stessi top-3).
- **Service worker** (vite-plugin-pwa / Workbox, già in CovertCook): come mettere in cache un file da diversi MB senza appesantire l'installazione iniziale (*runtime caching* invece di *precache*)? Strategie `CacheFirst` vs `StaleWhileRevalidate`, e come forzare l'aggiornamento quando pubblichi una nuova versione del modello (nome di file versionato, es. `food-v3-int8.onnx`).
- **Web Worker**: l'inferenza nel thread principale blocca l'interfaccia. Spostarla in un worker è una scelta da valutare (misura il blocco dell'UI).
- **Caricamento differito** (*lazy loading*): onnxruntime-web e il modello si caricano solo quando l'utente apre la funzione, non all'avvio dell'app.

### 🔀 Scelte
- Dove mettere la funzione nell'app: una pagina dedicata (più semplice, coerente con il principio "simple" del README di CovertCook) o dentro il flusso di una cena (più coerente col prodotto, ma l'app non ha utenti oggi). Per il progetto universitario, una pagina dedicata basta.
- Mostrare le probabilità all'utente o solo l'ordine?
- Comportamento sotto soglia (§5.4): "non riconosciuto" o "forse…"?

### 🛠️ Da fare
1. `src/lib/foodModel.ts`: caricamento differito della sessione, preprocessing, inferenza, post-processing (softmax con la temperatura di calibrazione, top-3, soglia).
2. Test unitario del preprocessing (Vitest, già in CovertCook): confronta il tensore prodotto in JS per un'immagine fissa con quello esportato da Python (`.npy` → JSON).
3. La pagina, accessibile dall'app, con input fotocamera (`<input type="file" accept="image/*" capture="environment">`).
4. Le etichette delle classi tradotte (CovertCook è multilingue).
5. Prova sul telefono, in modalità aereo dopo il primo caricamento: deve funzionare offline.

### ✅ Checkpoint
Video di 30-60 secondi: foto di un piatto vero → risultato, telefono in modalità aereo.

---

## 11. Consegna e orale

### Da consegnare
- Il repo, riproducibile dal README (da `git clone` alla tabella finale).
- `docs/report.md` (o PDF):
  1. Problema e vincoli (soglie di §0.2)
  2. Dati (scelte, EDA, set personale)
  3. Selezione del modello (screening, Pareto)
  4. Addestramento (ricetta, ablation)
  5. Ottimizzazioni (teoria in breve, esperimenti, risultati — anche quelli negativi)
  6. Benchmark sul dispositivo (metodologia!)
  7. Scelta finale e deploy
  8. Limiti e lavori futuri (dominio, mondo aperto, altri dispositivi)
- Il video della demo.
- `docs/decisions.md`.

### Domande d'orale da preparare
1. Spiega la quantizzazione affine con un esempio numerico a mano.
2. Statica vs dinamica: perché la tua scelta?
3. Perché per-channel sui pesi e per-tensor sulle attivazioni?
4. Come hai scelto il set di calibrazione? Cosa succede se non è rappresentativo?
5. Perché MobileNetV3 perde più punti di EfficientNet-Lite in int8 (se è il caso nei tuoi risultati)?
6. Come hai garantito che il preprocessing nel browser è identico a quello di training?
7. La differenza fp32 / int8 è statisticamente significativa?
8. Il tuo modello su foto vere: quanto perde rispetto al test di Food-101, e perché?
9. Cosa fa la tua app con una foto che non è un piatto?
10. Se dovessi rifare il progetto con un mese in più, cosa cambieresti?

---

## 12. Piano di lavoro indicativo

| Settimana | Tappe |
|---|---|
| 1 | §0-§2: ambiente, dati, EDA, set personale, soglie, struttura del repo |
| 2 | §3: profili dei candidati, export grezzo, prima latenza su telefono, screening, Pareto |
| 3 | §4-§5: addestramento dei finalisti, ablation, `evaluate.py` completo |
| 4 | §6-§7: export, parità, ottimizzazioni |
| 5 | §8-§9: benchmark sul telefono, scelta |
| 6 | §10-§11: integrazione, video, report, prova d'orale |

---

## Riferimenti

- Bossard, Guillaumin, Van Gool — *Food-101 – Mining Discriminative Components with Random Forests*, ECCV 2014.
- Howard et al. — *Searching for MobileNetV3*, ICCV 2019.
- Tan, Le — *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019.
- Mehta, Rastegari — *MobileViT*, ICLR 2022.
- Jacob et al. — *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR 2018.
- Krishnamoorthi — *Quantizing deep convolutional networks for efficient inference: A whitepaper*, 2018.
- Nagel et al. — *A White Paper on Neural Network Quantization*, 2021.
- Hinton, Vinyals, Dean — *Distilling the Knowledge in a Neural Network*, 2015.
- Guo et al. — *On Calibration of Modern Neural Networks*, ICML 2017.
- Hendrycks, Gimpel — *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*, ICLR 2017.
- Loshchilov, Hutter — *Decoupled Weight Decay Regularization*, ICLR 2019.
- Documentazione: PyTorch (`torch.onnx`, AMP), timm, ONNX Runtime (*Quantize ONNX models*, *onnxruntime-web*), Netron, vite-plugin-pwa / Workbox.
