# Journal de décisions — Reconnaissance alimentaire embarquée

Dernière mise à jour : 2026-11-08. Les paramètres sont dans [configs/base.yaml](../configs/base.yaml), les mesures dans `results/`. Tout ce qui n'a pas encore été mesuré est marqué **à mesurer**.

## Retour sur le projet de recognition image alimentaires

Avant propos :
- Pour rèaliser le projet de deployer un modele sur un téléphone, j'ai choisi de travailler sur un application déjà développé avant qui consiste à organiser des diners avec des echanges de recettes parmi les utilisateurs. En complement ce modele permettera de reconnaitre le type de plat cousiné par l'utilisateur et dans un second temps, qui ne est pas lié à le devoir demandé de informer la personne sur les calories et les nutriments du plat.
- Le modele de réconnaissance d'image devra avoir une latence maximal de 300 ms, avec des dimentions petite inferieur à 5Mo, avec une accuracy de 80% apres la quantification. Le modele devra etre capable de reconnaitre au moins 10 types de plats differents.
- Les photos prises par les utilisateurs seront de qualité moyenne, avec une mauvaise luminosité et un angle de prise de vue non optimal. Le modele devra donc répondre à ces conditions.

---

## 1. Comment fonctionne le modèle

**Sur le téléphone**, dans la PWA CovertCook, avec onnxruntime-web :

1. L'utilisateur prend une photo. Elle ne quitte jamais le téléphone.
2. La photo est redressée (EXIF), recadrée au centre et redimensionnée en 256 × 256 px, puis normalisée avec la moyenne et l'écart-type du backbone, lus dans `model_meta.json`.
3. Le modèle ONNX int8 calcule une probabilité par ingrédient (103 sorties). La sigmoïde et la température de calibration sont dans le graphe.
4. Le JavaScript applique un seuil par classe et affiche les ingrédients retenus, ou « non reconnu » si aucun ne passe.
5. **Partie D :** le même backbone a une seconde tête, qui propose les 3 plats les plus probables parmi 101.

**À l'entraînement :**

1. Un backbone pré-entraîné sur ImageNet reçoit une nouvelle tête de 103 classes.
2. Chaque image et son masque subissent le même recadrage et le même retournement. La cible multi-étiquettes est recalculée à partir du masque transformé.
3. L'optimisation utilise AdamW, un LR plus faible pour le backbone, un warmup linéaire puis un cosinus, et l'AMP fp16.
4. Le meilleur modèle est celui qui a la meilleure mAP macro sur la validation.

**Après l'entraînement :** calibration sur la validation (température et seuils), évaluation sur le test, export ONNX, quantification statique int8, benchmark sur le téléphone.

---

## 2. Paramètres actuels

Statut : **Fixé**, **Provisoire** (remplacé par une mesure) ou **Ablation** (valeur de départ à comparer).

| Paramètre | Valeur | Signification | Statut / autres valeurs à tester |
|---|---|---|---|
| `prep.foodseg_short_side` | 512 | Petit côté sur disque | Fixé (~2× la plus grande résolution testée) |
| `prep.val_fraction` | 0.15 | Part du train HF pour la validation | Fixé |
| `data.resolution` | 256 | Côté de l'entrée en px | Fixé par l'ablation (voir D-013) ; repli 224 si latence trop haute |
| `data.num_workers` | 4 | Processus de chargement | Provisoire : `bench_dataloader` |
| `data.train_min_area` | 0 | Aire minimale pour qu'un ingrédient compte à l'entraînement | Ablation : 0.005, 0.02 |
| `model.drop_rate` | 0.2 | Dropout avant la tête | À tester : 0.1, 0.3 |
| `train.epochs` | 40 (6 au screening) | Maximum, avec early stopping | À tester : 30, 50 |
| `train.batch_size` | 32 | Taille du batch | Provisoire : 48, 64 selon la VRAM |
| `train.lr` | 0.0003 | LR de la tête | Fixé pour `convnext_atto` (voir D-012), 0.001 au screening |
| `train.backbone_lr_mult` | 0.1 | LR backbone = `lr` × cette valeur | À tester : 0.2, 0.3 |
| `train.weight_decay` | 0.05 | Régularisation AdamW | À tester : 0.01, 0.03 |
| `train.warmup_epochs` | 1 | Montée linéaire du LR | À tester : 2 |
| `train.freeze_backbone_epochs` | 0 | Époques avec seule la tête entraînée | Ablation : 3 |
| `train.early_stop_patience` | 6 | Époques sans progrès avant l'arrêt | À tester : 5, 8 |
| `train.bias_init` | true | Biais de la tête initialisé à la fréquence des classes | Fixé, à comparer une fois avec false |
| `train.loss` | bce | Fonction de perte | Ablation : bce_posw, asl |
| `train.augment` | photo | Niveau d'augmentation | Fixé par l'ablation (voir D-013) |
| `train.crop_scale_min` | 0.5 | Aire minimale du recadrage | Ablation : 0.35, 0.6 |
| `eval.min_area` | 0 | Aire minimale pour les cibles d'évaluation | Fixé : identique pour tous les modèles |
| `eval.frequent_min_train` | 50 | Seuil d'une classe « fréquente » | Fixé |
| `export.opset` | 18 | Version des opérateurs ONNX | Fixé (voir D-008) |
| `export.dynamo` | true | Exporteur `torch.export` | Fixé, false en cas d'échec |
| `quant.calib_size` | 200 | Images de calibration (validation) | À tester : 50, 500 |
| `quant.calib_method` | MinMax | Choix de l'intervalle des activations | À tester : Entropy, Percentile |
| `quant.per_channel` | true | Une échelle par filtre de convolution | À tester : false |
| `quant.calib_sampling` | random | Choix des images | À tester : coverage |
| `multitask.dish_batch_size` | 16 | Images Food-101 par pas | Provisoire : 32 si la VRAM suffit |
| `multitask.steps_per_epoch` | 130 | Longueur d'une « époque » | À ajuster si `batch_size` change |
| `multitask.ingr_loss_weight` / `dish_loss_weight` | 1.0 / 1.0 | Poids des deux pertes | À rééquilibrer si l'une domine |
| `multitask.label_smoothing` | 0.1 | Lissage de la perte du plat | À tester : 0, 0.05 |

---

## 3. Décisions

### D-001 — Formulation : ingrédients multi-étiquettes (A), puis plat + ingrédients (D)
- **Contexte :** FoodSeg103 annote des ingrédients, jamais des plats. Le nom du plat vient de Food-101.
- **Décision :** d'abord 103 ingrédients en multi-étiquettes, puis un backbone partagé avec deux têtes.
- **Alternative écartée :** la segmentation. Elle aiderait pour les portions et les calories, mais demande un modèle plus lourd et une quantification plus délicate.

### D-002 — Objectifs fixés
| Objectif | Valeur | Porte sur | Statut |
|---|---|---|---|
| Latence totale médiane | ≤ 300 ms | Téléphone, pré + inférence + post | Fixé, téléphone et backend à préciser |
| Taille | ≤ 5 Mo | Fichier **int8** déployé (le fp32 peut dépasser) | Fixé |
| mAP macro ingrédients | ≥ 0,40 | Test, modèle int8 | Estimation sans référence publiée pour cette métrique |
| Perte int8 vs fp32 | ≤ 1,5 point | mAP, test | Fixé |
| Accuracy plat | ≥ 80 % top-1 | Test, modèle int8 (partie D) | **À confirmer** |

**Pourquoi l'objectif de 80 % de l'avant-propos porte sur le plat :** une accuracy par étiquette serait trompeuse sur les ingrédients. Chaque photo contient environ 5 ingrédients sur 103, donc répondre « aucun » partout donnerait déjà environ 95 %. La mAP macro est bien plus sévère, car les classes rares comptent autant que les autres.

### D-003 — Backbones candidats et finalistes
- **Finalistes actuels :** `convnext_atto`, `mobilenetv3_large_100`, `tf_efficientnet_lite0`.
- **Pourquoi :** `convnext_atto` a la meilleure mAP au screening et le temps d'entraînement le plus court. MobileNetV3 Large a le moins d'opérations parmi les modèles les plus précis. Lite0 a été conçu pour l'int8.
- **Risque :** LayerNorm et GELU peuvent mal se quantifier sur `convnext_atto`, et c'est le candidat avec le plus d'opérations. Il sera gardé seulement si A11 et A12 le confirment.
- **À mesurer :** variance entre seeds, perte int8, latence.

### D-004 — Pertes comparées
- `bce` sert de référence, avec le biais de la tête initialisé. `bce_posw` renforce les classes rares, avec un plafond de `pos_weight` à 20. `asl` réduit le poids des négatifs faciles (`gamma_neg` 4, `gamma_pos` 0, `clip` 0.05).
- **Pourquoi :** environ 95 % des cibles valent 0.
- **Décision finale :** à mesurer (ablation A8).

### D-005 — Données et split
- **Données :** copie redimensionnée sur disque, images en JPG et masques en PNG (sans perte, les valeurs de pixels restent des identifiants de classe).
- **Split :** le test est le split `validation` officiel de Hugging Face. La validation (15 %) est extraite du train par stratification itérative, pour garder les classes rares.
- **Règle :** seuils, température et choix de modèle ne viennent jamais du test.

### D-006 — Cible recalculée après l'augmentation
- **Décision :** le même recadrage s'applique à l'image et au masque, et la cible vient du masque transformé.
- **Observation :** avec `train_min_area` = 0 et `crop_scale_min` = 0.5, la cible a peu changé. Sur 16 images, seule `train_4263` perd un ingrédient (`cheese butter`). Un ingrédient presque invisible au bord compte encore, d'où l'ablation sur ce seuil.

### D-007 — Environnement
- **Dépôt hors de OneDrive :** la synchronisation bloque des fichiers et consomme le quota avec un venv de plusieurs Go. Un venv ne se déplace pas, il a donc été recréé.
- **Versions :** Python 3.13.1, PyTorch 2.15.0+cu132, GTX 1650 Ti (4 Go).
- **Pilote NVIDIA :** le pilote 555.97 s'arrêtait à CUDA 12.5, alors que la wheel utilise CUDA 13.2, et PyTorch ne voyait pas la GPU. La GPU est utilisée à 99 % après la mise à jour vers 621.14 (CUDA 13.5).
- **Suivi :** MLflow en local (SQLite), et CSV dans `results/` pour le rapport.

### D-008 — Export ONNX : opset 18
- **Contexte :** avec l'opset 17, la conversion échouait sur au moins un modèle (`Failed to convert the model to the target version 17`) et le modèle restait à 18.
- **Décision :** opset 18, natif pour l'exporteur `dynamo`, avec les poids réunis dans un seul fichier, car le navigateur ne chargerait pas un `.onnx.data` séparé.
- **À vérifier :** le chargement dans onnxruntime-web, au benchmark.
- **Dans le graphe :** sigmoïde et température. **Hors du graphe :** normalisation et seuils, dans `model_meta.json`.

### D-009 — Optimisation : quantification statique int8 (QDQ, per-channel)
- **Pourquoi :** elle divise la taille par environ 4, ce qui est nécessaire pour tenir dans 5 Mo.
- **Alternatives écartées :** fp16 (utile sur WebGPU seulement), quantification dynamique (presque rien sur une CNN), QAT (seulement si la perte dépasse l'objectif).
- **Résultats :** à mesurer (A11).

### D-010 — Le téléphone passe après le modèle
- **Décision :** le benchmark est fait sur les modèles entraînés et quantifiés (A12), plutôt que sur les candidats en A6.
- **Conséquence :** les finalistes sont choisis sur la mAP, les mult-adds et la taille estimée, et les mult-adds n'approximent que la latence.
- **Si un finaliste dépasse 300 ms :** réduire la résolution ou passer à un autre finaliste.

### D-011 — Déploiement sans toucher à la production
- **Décision :** branche `lab/phone-bench` et déploiement de brouillon Netlify, car Netlify publie `main` sur covertcook.fr.
- **Changements de sécurité sur la branche :** `'wasm-unsafe-eval'` dans la CSP (compilation WebAssembly seulement, pas `eval()`), et COOP/COEP uniquement sur `/lab/*`.
- **Service worker :** `/lab/` est exclu du fallback de navigation, sinon la page perdrait COOP/COEP dès la deuxième visite.
- **Mesure :** une seule configuration de threads par graphique, car une médiane sur 1 et 4 threads ne décrirait aucune des deux.

### D-012 — Learning rate de `convnext_atto` : 0,0003
- **Contexte :** deux entraînements complets, avec la même recette et le même seed.
- **Mesures :** voir le tableau « Learning rate » de la section 4 (`results/runs/`).
- **Décision :** 0,0003, comme nouvelle référence pour les ablations.
- **Pourquoi :** la meilleure mAP est équivalente (écart de 0,24 point, sous le seuil d'environ 1 point). Mais 0,0003 oscille moins d'une époque à l'autre et atteint une `val_loss` plus basse, ce qui rend les comparaisons d'ablation plus fiables.
- **Coût :** environ 14 minutes par entraînement, contre 7.
- **Ce qui me ferait changer d'avis :** un budget de temps trop serré pour les ablations.

### D-013 — Recette finale : 256 px + augmentation photo
- **Contexte :** 12 ablations sur `convnext_atto`, un changement à la fois, seed 42 (tableau « Ablation » de la section 4).
- **Décision :** `data.resolution` = 256 et `train.augment` = photo. Le reste de la recette de base est inchangé.
- **Pourquoi :** la combinaison donne 0,4749, soit +2,8 points sur la référence. 320 px n'apporte rien de plus que 256 pour 1,6 fois plus de pixels. `bce_posw`, `asl` et `train_min_area` ne gagnent pas.
- **Hypothèse :** la recette est appliquée telle quelle à `mobilenetv3_large_100` et `tf_efficientnet_lite0`, sans ablation propre. LR : 0,0005 pour MNV3 et 0,0006 pour Lite0 (LR range test).
- **Ouvert :** `freeze_backbone_epochs` = 3 gagne 1,1 point seul, à peine au-dessus du seuil ; il est testé une fois sur la recette finale.
- **Ce qui me ferait changer d'avis :** une latence au-dessus de 300 ms à 256 px. Les checkpoints à 224 px (`a_convnext_photo`, `a_convnext_lr3e4`) servent de repli.

---

## 4. Mesures

### Screening (A6) — 6 époques, seed 42, 224 px

| Modèle | mAP val | Paramètres (M) | Mult-adds (G) | Int8 estimé (Mo) | Temps par époque (s) |
|---|---|---|---|---|---|
| `convnext_atto` | 0,426 | 3,41 | 0,551 | 3,41 | 52 |
| `mobilenetv3_large_100` | 0,366 | 4,33 | 0,215 | 4,33 | 157 |
| `efficientnet_b0` | 0,322 | 4,14 | 0,385 | 4,14 | 252 |
| `mobilenetv3_small_100` | 0,310 | 1,62 | 0,056 | 1,62 | 57 |
| `mobilevit_xs` | 0,303 | 1,97 | 0,470 | 1,97 | 372 |
| `tf_efficientnet_lite0` | 0,280 | 3,50 | 0,384 | 3,50 | 255 |

Ces valeurs viennent d'un budget court, sur la validation et en fp32. Elles ne se comparent pas directement aux objectifs de D-002.

**Variance entre seeds (Lite0, seeds 42, 1 et 2) :** mAP val moyenne 0,275, écart-type 0,005. Un écart de moins d'environ 1 point (2 écarts-types) entre deux modèles n'est donc pas interprétable. L'écart entre `convnext_atto` et `mobilenetv3_large_100` (6 points) dépasse largement ce seuil, tandis que `mobilenetv3_small_100` et `mobilevit_xs` (0,7 point) restent indiscernables. Limite : la variance n'a été mesurée que sur Lite0, avec 3 seeds.

### Contrôles avant l'entraînement (A7) — `convnext_atto`

- **`init` :** perte initiale 0,1335, pour 0,1290 attendu avec le biais initialisé à la fréquence des classes (0,6931 sans). L'initialisation fonctionne.
- **`overfit` :** perte de 0,136 à 0,0005 en 200 pas sur un seul batch, mAP 1,000. Le modèle, la perte et les cibles peuvent apprendre.
- **LR range test (lecture visuelle) :** `convnext_atto` minimum vers 2e-3, divergence vers 2e-1, soit un LR candidat de 0,0003. `mobilenetv3_large_100` minimum vers 6e-3, divergence vers 5e-2, soit un LR candidat de 0,0005. La vallée est peu profonde : le LR sera comparé à 0,001, qui a déjà fonctionné au screening.

### Learning rate (A8) — `convnext_atto`, recette de base, seed 42

| LR | Meilleure mAP val | Époque | `val_loss` minimale | `train_loss` finale | Époques jouées |
|---|---|---|---|---|---|
| 0,001 | 0,4490 | 5 | 0,0770 | 0,0301 | 12 |
| 0,0003 | 0,4466 | 16 | 0,0761 | 0,0392 | 23 |

**Observation :** dans les deux cas, la `train_loss` continue de baisser alors que la `val_loss` cesse de baisser puis remonte : c'est du surapprentissage. L'early stopping s'arrête bien avant les 40 époques, donc augmenter `epochs` n'aiderait pas. Les prochaines ablations visent la généralisation.

### Ablation (A8) — `convnext_atto`, LR 0,0003, seed 42

Référence : `a_convnext_lr3e4`. Un seul changement par run. Seuil d'interprétation : environ 1 point, soit une mAP d'au moins 0,457.

| Run | Changement | Meilleure mAP val | Écart (points) | Époque | `val_loss` minimale | Verdict |
|---|---|---|---|---|---|---|
| `a_convnext_r256_photo` | 256 px + `train.augment` = photo | 0,4749 | +2,83 | 22 | 0,0747 | Retenu (D-013) |
| `a_convnext_r320` | `data.resolution` = 320 | 0,4625 | +1,59 | 20 | 0,0748 | Comme 256, pour 1,6 fois plus de pixels |
| `a_convnext_r256` | `data.resolution` = 256 | 0,4612 | +1,46 | 16 | 0,0752 | Au-dessus du seuil |
| `a_convnext_probe` | `train.freeze_backbone_epochs` = 3 | 0,4576 | +1,10 | 29 | 0,0762 | À peine au-dessus du seuil |
| `a_convnext_photo` | `train.augment` = photo | 0,4544 | +0,78 | 22 | 0,0761 | Sous le seuil |
| `a_convnext_strong` | `train.augment` = strong | 0,4503 | +0,37 | 22 | 0,0759 | Sous le seuil |
| `a_convnext_crop35` | `train.crop_scale_min` = 0.35 | 0,4486 | +0,20 | 24 | 0,0760 | Sous le seuil |
| `a_convnext_lr3e4` | Référence | 0,4466 | — | 16 | 0,0761 | — |
| `a_convnext_tau02` | `data.train_min_area` = 0.02 | 0,4491 | +0,25 | 16 | 0,0769 | Sans effet |
| `a_convnext_tau005` | `data.train_min_area` = 0.005 | 0,4459 | -0,07 | 20 | 0,0761 | Sans effet |
| `a_convnext_posw` | `train.loss` = bce_posw | 0,4146 | -3,20 | 26 | non comparable | Rejeté |
| `a_convnext_asl` | `train.loss` = asl | 0,4242 | -2,24 | 34 | non comparable | Rejeté |

**Observations :**
- La résolution est le seul changement au-dessus du seuil. Elle multiplie le nombre de pixels par 1,31 ; le coût en latence reste à mesurer sur le téléphone.
- Les augmentations ne baissent pas la `val_loss` minimale, qui reste vers 0,076 : elles ralentissent le surapprentissage sans donner de gain net.
- La `val_loss` de l'ASL n'est pas comparable, car cette perte a une autre échelle : seule la mAP sert à la comparaison.
- Limite : un seul seed par run, et un seuil issu de la variance de Lite0. L'écart de `r256` reste à confirmer.

- `bce_posw` et `asl` perdent nettement : la BCE simple, avec le biais initialisé, suffit.
- Les tailles de batch 48 et 64 n'ont pas été testées : batch 32 tient en mémoire, et le changer modifierait la dynamique du LR.

### Évaluation (A9) — PyTorch fp32, seuils calibrés sur val

| Modèle | mAP test (2135 images) | F1 micro test | mAP photos perso (108) | F1 micro perso | Rejet des photos sans nourriture |
|---|---|---|---|---|---|
| `a_convnext` | 0,388 | 0,485 | 0,306 | 0,206 | 0 / 13 |
| `a_mnv3` | 0,363 | 0,422 | 0,259 | 0,139 | 1 / 13 |
| `a_lite0` | 0,345 | 0,404 | 0,264 | 0,156 | 0 / 13 |

**Observations :**
- Le classement est le même sur le test et sur les photos personnelles : ConvNeXt est devant.
- **Cible D-002 non atteinte :** `convnext_atto` atteint 0,388 de mAP test en fp32, sous les 0,40 visés, avant même la quantification. La cible n'est pas déplacée.
- **Écart val / test :** 0,475 en val contre 0,388 en test pour ConvNeXt. Une partie vient de la sélection sur val (13 runs comparées sur les mêmes 750 images), le reste n'est pas expliqué.
- **Photos personnelles :** environ 8 points de moins que le test, ce qui mesure l'écart entre FoodSeg et des photos de téléphone. 46 des 103 classes n'y apparaissent pas, et les étiquettes sont une première version faite sur miniatures, à vérifier.
- **Monde ouvert :** avec les seuils par classe, le modèle trouve toujours des ingrédients, même sur des photos sans nourriture. Il n'a jamais vu de non-nourriture à l'entraînement, et la précision est faible (0,17 sur les photos perso).
- **Piste :** une règle « non reconnu » sur la probabilité maximale. Pour ConvNeXt, un seuil de 0,4 garde 85 % des photos de nourriture et rejette 10 photos sans nourriture sur 13. Le seuil est choisi et mesuré sur les mêmes 13 photos : estimation optimiste, à confirmer sur d'autres photos.
- **Intervalles de confiance :** la valeur mesurée tombe en bas de l'intervalle bootstrap (test ConvNeXt : 0,388 dans [0,384 ; 0,424]), voire en dehors (Lite0 perso). Le bootstrap sur les images fait disparaître les classes rares de certains tirages, et ces classes ont une AP plus faible : l'intervalle est décalé vers le haut. À corriger avant de citer ces intervalles.

### Export ONNX (A10) — fp32, opset 18, dynamo

| Modèle | Taille `.onnx` | Écart max PyTorch / ONNX (64 images) | Décisions identiques | mAP test ONNX | Écart de mAP |
|---|---|---|---|---|---|
| `a_convnext` | 14,1 Mo | 4,2e-6 | 100 % | 0,3876 | 0,0000 |
| `a_mnv3` | 17,7 Mo | 2,0e-6 | 100 % | 0,3634 | 0,0000 |
| `a_lite0` | 14,2 Mo | 2,1e-6 | 100 % | 0,3453 | 0,0000 |

**Observation :** l'export ne change rien, ni aux probabilités (écarts de l'ordre de 1e-6, du bruit de calcul en float32) ni à la mAP. Toute perte mesurée en A11 viendra donc de la quantification. En fp32, les trois fichiers dépassent la cible de 5 Mo, comme prévu.

### Quantification int8 (A11) — premier essai, invalide

Premier essai, MinMax, 200 images, par canal : perte de mAP test de 13,2 points pour ConvNeXt, 9,9 pour MNV3 et 5,4 pour Lite0, contre 1,5 au maximum visé. Aucune option de calibration ne ramène la perte sous 8 points.

- **Cause :** la sortie sigmoïde était elle aussi quantifiée en uint8, soit 256 probabilités possibles (pas de 0,004). La probabilité médiane d'un ingrédient est 0,003 : environ 40 % des probabilités tombaient à 0, et il ne restait que 71 à 91 valeurs distinctes sur 220 000. L'AP mesure un classement, qui est détruit par ces égalités. Le problème touche donc tous les modèles, même Lite0, conçu pour l'int8.
- **Correction :** la tête (couche linéaire, division par la température, sigmoïde) reste en float. Le coût est une petite matrice en float, quelques centaines de Ko.
- **Second défaut :** le nom du fichier ne contenait pas l'échantillonnage, donc la variante `coverage` avait écrasé la variante de référence `random`. Le suffixe `_cov` est maintenant ajouté.
- **Méthode :** la variante int8 est désormais choisie sur val, et seule la variante retenue est mesurée sur test.

### Quantification int8 (A11) — après correction

Même avec la tête en float, la perte restait de 6 à 19 points sur val. Deux autres causes ont été trouvées.

- **Noyaux int8 du PC :** sur un processeur x86 sans instructions VNNI (Ryzen 7 4800HS, Zen 2), les noyaux uint8 × int8 d'onnxruntime saturent. Sur 300 images de val, Lite0 int8 donne 0,391 avec les noyaux optimisés et 0,447 en exécution simulée (mêmes calculs sans ces noyaux). Avec `reduce_range` (poids sur 7 bits), les deux exécutions donnent le même résultat. Le téléphone utilise d'autres noyaux : le comportement y reste à vérifier.
- **ConvNeXt :** LayerNorm et GELU supportent mal l'int8. Quantifier seulement les convolutions (`--op-types Conv`) ramène la perte de 9,1 à 3,6 points sur 300 images de val.
- **Anomalie non expliquée :** les fichiers `Entropy` et `MinMax` de ConvNeXt sont identiques octet par octet. La méthode de calibration n'a donc pas été prise en compte, à vérifier.

| Variante retenue | Taille | mAP val fp32 → int8 | Perte val | mAP test fp32 → int8 | Perte test (IC 95 %) |
|---|---|---|---|---|---|
| `a_convnext`, MinMax 200, par canal, convolutions seules, `reduce_range` | 3,80 Mo | 0,471 → 0,426 | 4,5 points | 0,388 → 0,367 | 2,1 points [1,0 ; 3,1] |
| `a_lite0`, MinMax 200, par canal, `reduce_range` | 4,46 Mo | 0,400 → 0,388 | 1,2 points | 0,345 → 0,345 | 0,0 point [-0,7 ; 0,7] |

- **Lite0** respecte la perte maximale de 1,5 point ; **ConvNeXt** la dépasse, mais reste plus précis en valeur absolue.
- **MNV3** est écarté : 5,16 Mo, au-dessus de la cible, et 12 points de perte avant correction.
- **Premier déploiement :** ConvNeXt, sur la page `/scan` d'un déploiement de brouillon. Le choix final (A13) attend la latence mesurée sur le téléphone.

### À compléter

- [ ] Compteurs `integrity checks` de `preresize` (A2)
- [ ] Classes avec 0 ou moins de 5 images en validation (A3)
- [ ] 5 observations de l'EDA, chacune avec sa conséquence (A4)
- [ ] `num_workers` mesuré (A5)
- [x] Variance entre seeds de Lite0 (A6)
- [ ] Résultats `overfit` et LR par finaliste (A7)
- [ ] Tableau d'ablation (A8)
- [ ] mAP + IC, F1, ECE, classes fréquentes et rares, photos personnelles (A9)
- [x] Parité PyTorch / ONNX (A10)
- [x] Taille réelle et perte int8 de chaque variante (A11)
- [ ] Latence par variante, backend et threads, avec le modèle du téléphone (A12)
- [ ] Décision finale de déploiement (A13)
- [ ] Top-1 et top-3 du plat, seuil « non reconnu », coût du multi-tâche (D3-D5)
