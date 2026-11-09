# Optimisation et déploiement sur téléphone

Support de présentation, partie 2 sur 2. La partie 1, sur le modèle, est dans [presentation_modele.md](presentation_modele.md).

---

## 1. La chaîne complète

```
PyTorch (best.pt)
  -> export ONNX fp32 (opset 18, exporteur dynamo, un seul fichier)       [Niveau 1 : packaging]
  -> test de parité PyTorch / ONNX sur 64 images réelles
  -> quantification statique int8 (onnxruntime, format QDQ)               [Niveau 2 : optimisation]
  -> mesure de la perte de mAP (val pour choisir, test une fois)
  -> CovertCook (PWA React + Vite) : onnxruntime-web, WebAssembly ou WebGPU
  -> déploiement Netlify : brouillon pour tester, puis production (covertcook.fr/scan)
```

**Ce qui part sur le téléphone :** un fichier `.onnx` (3,8 Mo) et un `meta.json` (5 Ko) qui contient tout ce qui est propre au modèle : résolution, moyenne et écart-type, noms des ingrédients, seuils par classe, seuil « non reconnu ». Rien n'est recopié à la main entre Python et JavaScript.

---

## 2. Export ONNX (Niveau 1)

| Modèle | Taille fp32 | Écart max PyTorch / ONNX | Décisions identiques | mAP test ONNX |
|---|---|---|---|---|
| ConvNeXt-Atto | 14,1 Mo | 4,2e-6 | 100 % | 0,3876 (identique) |
| MobileNetV3-Large | 17,7 Mo | 2,0e-6 | 100 % | 0,3634 (identique) |
| EfficientNet-Lite0 | 14,2 Mo | 2,1e-6 | 100 % | 0,3453 (identique) |

- **La température et la sigmoïde sont dans le graphe :** le JavaScript reçoit directement des probabilités calibrées.
- **Un seul fichier :** l'exporteur écrivait les poids à part (`.onnx.data`), que le navigateur ne sait pas charger. Le fichier est ré-enregistré avec les poids intégrés.
- **Conclusion :** l'export ne change rien. Toute perte mesurée ensuite vient de la quantification.

---

## 3. Quantification int8 (Niveau 2)

**Méthode :** quantification statique post-entraînement (PTQ). Les poids passent en int8 ; les plages des activations sont calibrées sur 200 images de validation. Format QDQ (paires `QuantizeLinear` / `DequantizeLinear`), lisible dans Netron et portable.

### 3.1 Premier essai : invalide, et pourquoi

Perte de mAP val de **6 à 19 points** selon le modèle et les réglages, pour un maximum visé de 1,5. Trois causes trouvées, une par une :

| # | Problème | Diagnostic | Solution |
|---|---|---|---|
| 1 | La sortie sigmoïde était elle aussi quantifiée en uint8 | 256 probabilités possibles (pas de 0,004) ; la probabilité médiane est 0,003 : 40 % des probabilités tombaient à 0, plus que 71 à 91 valeurs distinctes sur 220 000. La mAP mesure un classement : il était détruit | **La tête reste en float** (couche linéaire, température, sigmoïde) : quelques centaines de Ko |
| 2 | Les noyaux int8 du PC saturent | Processeur sans instructions VNNI (Ryzen 7 4800HS). Lite0 sur 300 images : 0,391 avec les noyaux optimisés, 0,447 en exécution simulée | **`reduce_range`** : poids sur 7 bits, plus de saturation (les deux exécutions donnent alors le même résultat) |
| 3 | ConvNeXt supporte mal l'int8 | LayerNorm et GELU produisent des activations avec quelques valeurs très grandes, qui écrasent l'échelle des autres | **Précision mixte** : seules les convolutions en int8 (`--op-types Conv`) |

Défauts de méthode corrigés au passage : la variante `coverage` écrasait la variante de référence (même nom de fichier) ; et la variante était d'abord choisie sur le **test**, désormais sur la **validation**.

### 3.2 Les variantes comparées

Perte de mAP sur la validation (758 images), fp32 moins int8, avec l'intervalle de confiance à 95 % d'un bootstrap apparié. Source : `results/comparisons.csv`, `results/quant_runs.csv`.

| Modèle | Variante | Perte val (points) | Taille |
|---|---|---|---|
| ConvNeXt | MinMax, 200 images, par canal | 16,5 [13,6 ; 18,9] | 3,84 Mo |
| ConvNeXt | Entropy | 16,5 (fichier identique à MinMax : anomalie) | 3,84 Mo |
| ConvNeXt | Percentile | 13,0 [10,2 ; 15,1] | 3,84 Mo |
| ConvNeXt | par tenseur | 9,5 [7,3 ; 11,4] | 3,68 Mo |
| ConvNeXt | 50 / 500 images de calibration | 19,0 / 18,8 | 3,84 Mo |
| ConvNeXt | images choisies pour couvrir les classes rares | 15,0 [12,3 ; 17,5] | 3,84 Mo |
| **ConvNeXt** | **convolutions seules + `reduce_range`** | **4,5 [2,7 ; 5,8]** | **3,80 Mo** |
| MobileNetV3 | MinMax, 200 images, par canal | 12,2 [9,1 ; 14,2] | 5,16 Mo (> 5) |
| Lite0 | MinMax, 200 images, par canal | 6,1 [4,6 ; 8,3] | 4,46 Mo |
| **Lite0** | **`reduce_range`** | **1,2 [-0,3 ; 2,4]** | **4,46 Mo** |

> Ces mesures sont faites avec les noyaux int8 du PC. Le téléphone utilise d'autres noyaux (WebAssembly) : les variantes sans `reduce_range` y perdraient probablement moins.

### 3.3 Les trois optimisations tentées

1. **Calibration de la PTQ :** méthode (MinMax, Entropy, Percentile), nombre d'images (50, 200, 500), choix des images (aléatoire ou couvrant les classes rares), par canal ou par tenseur.
2. **Précision mixte :** les opérations sensibles (LayerNorm, GELU, tête) restent en float, les convolutions passent en int8.
3. **Réduction de plage (`reduce_range`) :** poids sur 7 bits pour éviter la saturation des noyaux x86 sans VNNI.

---

## 4. Quantifié contre non quantifié : métrique, taille, vitesse

### 4.1 Métrique et taille (test, 2 135 images)

| Modèle | mAP fp32 | mAP int8 | Perte (IC 95 %) | Taille fp32 | Taille int8 |
|---|---|---|---|---|---|
| **ConvNeXt-Atto** | 0,388 | **0,367** | 2,1 [1,0 ; 3,1] | 14,1 Mo | **3,80 Mo** (÷ 3,7) |
| EfficientNet-Lite0 | 0,345 | 0,345 | 0,0 [-0,7 ; 0,7] | 14,2 Mo | 4,46 Mo (÷ 3,2) |

- **Lite0** respecte la perte maximale (1,5 point) ; **ConvNeXt** la dépasse mais reste **le plus précis en valeur absolue**.
- **Modèle déployé :** ConvNeXt int8, le quantifié le plus performant, comme le demande le sujet.

### 4.2 Vitesse

Latence médiane d'une inférence (ms), 100 exécutions après 10 de chauffe, 3 campagnes ; mesurée par la page `/lab/bench`. Sources : `results/bench_raw/` (téléphone) et `results/bench_raw/pc/` (PC).

**PC (Ryzen 7 4800HS, Chrome) :**

| Variante | CPU 1 thread | CPU 4 threads | GPU (WebGPU) |
|---|---|---|---|
| ConvNeXt fp32 | 98 | 41 | **9** |
| ConvNeXt int8 | 107 | 45 | 345-394 |
| Lite0 fp32 | 94 | 30 | **7-8** |
| Lite0 int8 | 112 | 42 | 269-305 |

**Téléphone (Realme GT Master, Android, Chrome 156) :** 3 campagnes, 24 mesures par fichier. Source : `results/device_bench.csv` (après `scripts.collect_bench`), `scripts.pareto --stage final`.

| Variante | Modèle seul, CPU 1 thread | Modèle seul, CPU 4 threads | Modèle seul, GPU | **Total** (photo + modèle), CPU 4 threads |
|---|---|---|---|---|
| ConvNeXt fp32 | 178 | 55 | 57 | 474 |
| **ConvNeXt int8 (déployé)** | 227 | 73 | 794 | **491** |
| Lite0 fp32 | 115 | **34** | 64 | 590 |
| Lite0 int8 | 167 | 48 | 498 | 536 |

Préparation de la photo (décodage, redressement, recadrage, 256 px) : **350 à 560 ms** selon la variante et la campagne. Elle est la même pour tous les modèles : ses écarts sont du bruit, qui suffit à inverser l'ordre des totaux (Lite0 a le modèle le plus rapide mais le total le plus lent).

**Lecture :**
- **Le gain de l'int8 est la taille, pas la vitesse.** En format QDQ, onnxruntime-web ajoute des conversions : sur le téléphone, l'int8 est **30 à 45 % plus lent** que le fp32 sur CPU.
- **4 threads divisent le temps du modèle par 3 à 3,5** (ConvNeXt fp32 : 178 → 55 ms).
- **L'int8 est très lent sur WebGPU** (500 à 800 ms) : ces opérations ne sont pas accélérées par le GPU. En fp32, le GPU du téléphone n'est pas plus rapide que 4 threads CPU.
- **Le modèle seul tient l'objectif de 300 ms** sur le téléphone (34-73 ms, 4 threads ; 115-227 ms, 1 thread).
- **La chaîne complète ne le tient pas :** 474 à 590 ms, dont 350 à 560 ms de préparation de la photo, contre 20-40 ms sur le PC. Piste : décoder directement en basse résolution (`createImageBitmap` avec `resizeWidth`/`resizeHeight`) au lieu de décoder 12 Mpx puis réduire.
- **Avec 1 thread** (le cas de `/scan`, qui n'est pas isolée), ConvNeXt int8 prend 227 ms pour le modèle seul.

---

## 5. Intégration dans CovertCook

| Élément | Rôle |
|---|---|
| `src/lib/foodModel.ts` | Charge le modèle (avec avancement du téléchargement), prépare la photo exactement comme Python, applique seuils et « non reconnu » |
| `src/pages/FoodScan.tsx` (`/scan`) | Prendre une photo ou choisir dans la galerie, barre d'avancement, ingrédients affichés en lecture seule ; le bouton « Évaluer » permet de retirer les ingrédients faux et d'ajouter ceux qui manquent |
| Profil → « Analyser votre assiette » | Ouvre `/scan` (marqué « en développement ») |
| `src/pages/ModelReport.tsx` (`/scan/model`) | Page « Comment fonctionne le modèle ? » : courbes d'entraînement, variantes int8, fp32 contre int8, latence sur téléphone, en sections repliables |
| `public/models/model_report.json` | Les données de `/scan/model`, écrites par `scripts/export_model_report.py` à partir des résultats versionnés (rien n'est recalculé) |
| Supabase : table `scan_feedback`, fonction `submit_scan_feedback()` (migration 0111) | Reçoit une évaluation : version du modèle, ingrédients proposés avec leurs probabilités, retirés, ajoutés. **Aucune photo.** Précision, rappel et Jaccard calculés côté serveur ; utilisateurs connectés seulement, 30 envois par heure au maximum ; aucun accès direct à la table |
| `src/pages/LabBench.tsx` (`/lab/bench`) | Page cachée de mesure de latence, pour le TP |
| `public/models/` | Le modèle de `/scan` et son `meta.json`, versionnés |
| `scripts/copy-ort.mjs` | Copie le runtime onnxruntime-web (WebAssembly) dans `public/ort/` à chaque build |
| `public/_headers` | CSP `'wasm-unsafe-eval'` ; isolation COOP/COEP sur `/lab/*` et `/ort/*` |
| `src/sw.ts` | Modèle et runtime en cache : `/scan` fonctionne hors ligne |

**Vérifications :**
- **Test de parité Python / JavaScript (Vitest, 6 tests) :** même tenseur d'entrée et mêmes ingrédients que Python sur 3 images.
- **Chrome headless piloté par script :** `/scan` testé sur 2 photos de plats et 2 sans nourriture, résultats proches de Python (une photo proche du seuil 0,4 change de côté : le redimensionnement d'une photo de 4 000 px dans le navigateur n'est pas identique à celui de Python) ; `/lab/bench` testé avec 1 et 4 threads.
- **WebKit (moteur de Safari, profil iPhone 15) piloté par Playwright :** `/scan` testé sur 4 photos, dont des JPEG progressifs de WhatsApp ; résultats cohérents avec Python après le repli de décodage.
- **Processus :** une branche par changement, un **déploiement de brouillon Netlify** (URL HTTPS propre, en-têtes réels) testé sur le téléphone, puis la migration Supabase appliquée (`npx supabase db push --linked`) **avant** le merge sur `main`, qui publie en production.

---

## 6. Problèmes rencontrés et solutions (côté déploiement)

| Problème | Cause | Solution |
|---|---|---|
| Poids séparés du `.onnx` | L'exporteur écrit un `.onnx.data` que le navigateur ne charge pas | Ré-enregistrement en un seul fichier |
| Perte int8 énorme | Sortie quantifiée, saturation x86, ConvNeXt (§ 3.1) | Tête en float, `reduce_range`, précision mixte |
| `covertcook.fr/scan` tournait sans fin | Modèle et runtime exclus de git : la build Netlify ne les avait pas, le serveur renvoyait la page HTML | Runtime copié à chaque build, modèle versionné ; erreur affichée si un fichier manque |
| Benchmark bloqué à 0 avec 4 threads | Les workers WebAssembly d'une page isolée (COEP) doivent eux aussi être servis avec COEP | En-tête COEP sur `/ort/*` |
| Téléphone : 1 mesure sur 24 | Chaque mesure décodait la photo (≈ 48 Mo) sans libérer la mémoire | `bitmap.close()` après le redimensionnement (aussi dans `/scan`) |
| Écran qui s'éteint pendant la mesure | Chrome suspend la page | Verrou d'écran (Screen Wake Lock) pendant la mesure |
| « copy JSON » refusé sur le téléphone | Presse-papiers bloqué hors geste reconnu | Bouton « download JSON » |
| Photos sans nourriture | Le modèle trouve toujours des ingrédients | « Non reconnu » si probabilité max < 0,4 |
| `/scan` en WebKit (moteur de Safari) : « erreur de lecture » sur les photos WhatsApp | `createImageBitmap` refuse les JPEG progressifs dans ce WebKit | Repli : décodage par un `<img>` (applique aussi la rotation EXIF) ; vérifié en WebKit, profil iPhone 15 |
| `/lab/bench` ne marche pas sur iPhone | **Non résolu** (pas d'iPhone pour le diagnostic ; le repli ci-dessus pourrait suffire) | La page affiche maintenant l'erreur |

---

## 7. Ce qu'il faut retenir

- **Taille :** ÷ 3,7 avec l'int8 (14,1 → 3,8 Mo), sous l'objectif de 5 Mo.
- **Précision :** -2,1 points de mAP pour ConvNeXt après trois corrections ; 0,0 pour Lite0.
- **Vitesse :** l'int8 n'accélère pas dans le navigateur ; le modèle seul tient les 300 ms (73 ms, 4 threads), la chaîne complète non (491 ms), à cause de la préparation de la photo.
- **Une quantification se vérifie à chaque étape :** la sortie, le matériel qui exécute et l'architecture peuvent chacun casser le résultat.
- **Sur téléphone, ce qui compte est toute la chaîne :** téléchargement, décodage de la photo, mémoire, en-têtes HTTP, mise en veille.

---

## 8. Où trouver les tableaux

| Contenu | Fichier |
|---|---|
| Variantes int8 : taille, nœuds quantifiés | `results/quant_runs.csv` |
| Pertes fp32 → int8 (val et test, IC appariés) | `results/comparisons.csv` |
| mAP, F1, ECE de chaque variante | `results/metrics.csv` |
| Mesures brutes de latence | `results/bench_raw/*.json` (téléphone), `results/bench_raw/pc/*.json` (PC) |
| Latences agrégées et front de Pareto | `results/device_bench.csv` (après `scripts.collect_bench`), `scripts.pareto` |
| Fichiers déployés | `models/a_convnext/model_int8_minmax_200_pc_only-conv_rr.onnx`, `models/a_convnext/model_meta.json` |
| Code web (source) | `web/src/lib/foodModel.ts`, `web/src/lib/bench.ts`, `web/src/pages/*.tsx` |
| Commandes de bout en bout | `TP1_riconoscimento_piatti.md`, étapes A10 à A14 |

---

## 9. Thèmes principaux

- **ONNX :** format d'échange, opset, exporteur dynamo, test de parité.
- **Quantification post-entraînement (PTQ) statique :** calibration (MinMax, Entropy, Percentile), par canal ou par tenseur, format QDQ.
- **Précision mixte :** garder en float les opérations sensibles.
- **Matériel et noyaux int8 :** saturation u8×s8 sans VNNI, `reduce_range`.
- **Comparaison appariée :** bootstrap apparié de la différence de mAP, sur les mêmes images.
- **Inférence dans le navigateur :** onnxruntime-web, WebAssembly (threads, SIMD) et WebGPU.
- **Sécurité web :** CSP, isolation cross-origin (COOP / COEP) nécessaire aux threads.
- **PWA hors ligne :** service worker, cache du modèle et du runtime.
- **Benchmark embarqué :** chauffe, médiane et p95, plusieurs campagnes, latence de bout en bout.
- **Déploiement continu :** branches, brouillons Netlify, production protégée.
