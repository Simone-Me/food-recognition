# Le modèle — reconnaissance des ingrédients d'une assiette

Support de présentation, partie 1 sur 2. La partie 2, sur l'optimisation et le déploiement sur téléphone, est dans [presentation_deploiement.md](presentation_deploiement.md). Les choix détaillés sont dans [decisions.md](decisions.md).

---

## 1. Le sujet et ce qui y répond

**Sujet :** « Embarquer un modèle » non trivial sur un dispositif embarqué, entraîné sur un dataset choisi, avec une à trois optimisations.

| Exigence du sujet | Ce qui a été fait | Où |
|---|---|---|
| Modèle non trivial, entraîné sur un dataset choisi | Classification **multi-étiquettes** de 103 ingrédients (FoodSeg103), backbone pré-entraîné ImageNet affiné | Ce document |
| **Niveau 1** : packager le modèle | Export ONNX autonome (un seul fichier) + `model_meta.json` (prétraitement, noms, seuils), servi par la PWA | Partie 2, § 2 |
| **Niveau 2** : une ou plusieurs quantifications | Quantification statique post-entraînement int8 (PTQ), 11 variantes comparées | Partie 2, § 3 |
| Comparer quantifié et non quantifié (métrique, taille, vitesse) | mAP test, taille du fichier, latence sur PC et sur téléphone | Partie 2, § 4 |
| Déployer le quantifié le plus performant sur la cible | ConvNeXt-Atto int8 dans CovertCook, page `/scan`, sur le téléphone | Partie 2, § 5 |

**Cible :** un smartphone, dans le navigateur. Le modèle tourne dans la PWA CovertCook avec onnxruntime-web : la photo ne quitte jamais le téléphone.

**Pourquoi ce sujet :** CovertCook est une application existante qui organise des dîners avec échange de recettes. Reconnaître les ingrédients d'un plat photographié est la première brique d'« Analyser votre assiette » (sur `/scan` ; calories et nutriments à venir).

---

## 2. Objectifs fixés avant de mesurer

Fixés au départ ([configs/base.yaml](../configs/base.yaml), `targets`) et jamais déplacés après avoir vu les résultats :

| Objectif | Valeur | Atteint ? |
|---|---|---|
| Latence totale sur le téléphone | ≤ 300 ms | Modèle seul oui (55-73 ms pour ConvNeXt, 4 threads) ; **chaîne complète non** (491 ms, dont 350-560 ms de préparation de la photo) |
| Taille du modèle téléchargé, après quantification | ≤ 5 Mo | **Oui** : 3,80 Mo |
| mAP macro test du modèle déployé | ≥ 0,40 | **Non** : 0,367 |
| Perte de mAP due à l'int8 | ≤ 1,5 point | ConvNeXt **non** (2,1) ; Lite0 **oui** (0,0) |

> L'objectif « 80 % de précision » de l'avant-propos concerne la reconnaissance **du plat** (partie D, Food-101, top-1), pas encore réalisée. Pour les ingrédients, la métrique est la mAP macro, qui n'a pas d'équivalent « 80 % ».

---

## 3. Les données

**FoodSeg103** : photos de plats avec un masque de segmentation par ingrédient. Chaque image contient plusieurs ingrédients : c'est un problème **multi-étiquettes**. La cible d'une image est un vecteur de 103 zéros et uns.

| Split | Images | Origine |
|---|---|---|
| train | 4 225 | 85 % du train officiel |
| val | 758 | 15 % du train officiel, stratifié par classe |
| test | 2 135 | test officiel, utilisé **une seule fois** par modèle final |

- **Déséquilibre fort :** 3 classes n'ont aucune image en validation, 29 en ont moins de 5. Voir `results/split_counts.csv` et `results/eda/1_class_frequency.png`.
- **Cible recalculée après l'augmentation :** un ingrédient coupé par le recadrage aléatoire disparaît de la cible. La cible vient du masque transformé, pas de la liste d'origine.
- **Photos personnelles :** 108 photos de téléphone, dont 95 de plats et 13 sans nourriture. Elles servent de test « monde réel » ; les étiquettes sont une première version faite sur miniatures, à vérifier.

**Figures :** `results/eda/` (fréquence des classes, ingrédients par image, aire, cooccurrences, exemples avec et sans augmentation).

---

## 4. Le modèle

```
photo -> redressement EXIF -> recadrage centré -> 256 x 256 -> normalisation
      -> backbone CNN pré-entraîné (ImageNet) -> tête linéaire 103 sorties
      -> division par la température T -> sigmoïde -> seuil par classe -> ingrédients
```

- **Transfer learning :** le backbone vient de `timm`, pré-entraîné sur ImageNet ; seule une nouvelle tête de 103 classes est créée.
- **Entraînement :** BCE, AdamW, LR du backbone 10 fois plus faible que celui de la tête, warmup puis cosinus, AMP fp16 sur une GTX 1650 Ti (4 Go), early stopping sur la mAP de validation.
- **Après l'entraînement :** calibration sur la validation (température T, seuil par classe qui maximise le F1), puis une seule évaluation sur le test.

---

## 5. La démarche, étape par étape

### 5.1 Screening des backbones (A6)

Six backbones mobiles, même budget court (6 époques, 224 px, même recette). Source : `results/screening.csv`, `results/candidates.csv`.

| Backbone | mAP val | Paramètres | Mult-adds | int8 estimé |
|---|---|---|---|---|
| **convnext_atto** | **0,426** | 3,4 M | 0,55 G | 3,4 Mo |
| mobilenetv3_large_100 | 0,366 | 4,3 M | 0,22 G | 4,3 Mo |
| efficientnet_b0 | 0,322 | 4,1 M | 0,39 G | 4,1 Mo |
| mobilenetv3_small_100 | 0,310 | 1,6 M | 0,06 G | 1,6 Mo |
| mobilevit_xs | 0,303 | 2,0 M | 0,47 G | 2,0 Mo |
| tf_efficientnet_lite0 | 0,275 (moyenne 3 seeds) | 3,5 M | 0,38 G | 3,5 Mo |

- **Variance entre seeds :** Lite0 avec 3 seeds donne un écart-type de 0,005. Un écart de moins d'environ **1 point** (2 écarts-types) n'est pas interprétable : c'est le seuil utilisé pour toutes les comparaisons.
- **Finalistes :** ConvNeXt-Atto (le plus précis), MobileNetV3-Large (le plus léger en calcul), EfficientNet-Lite0 (conçu pour l'int8 : pas de SE, ReLU6).

### 5.2 Contrôles avant l'entraînement (A7)

- **Perte initiale :** 0,1335 mesurée pour 0,1290 attendue, grâce au biais initialisé à la fréquence des classes (0,6931 sans).
- **Sur-apprentissage d'un batch :** perte de 0,136 à 0,0005 en 200 pas, mAP 1,000. Modèle, perte et cibles peuvent apprendre.
- **LR range test :** `results/lr_range_*.png`. LR retenus : 0,0003 (ConvNeXt), 0,0005 (MNV3), 0,0006 (Lite0).

### 5.3 Ablations (A8)

Sur ConvNeXt, un seul changement par run, même seed. Référence : 0,4466 de mAP val. Source : `results/runs/a_convnext_*_history.csv`, tableau complet dans [decisions.md](decisions.md).

| Changement | mAP val | Écart (points) | Verdict |
|---|---|---|---|
| **256 px + augmentation photo** | **0,4749** | **+2,8** | **Retenu** |
| 320 px | 0,4625 | +1,6 | Comme 256 px, pour 1,6 fois plus de pixels |
| 256 px | 0,4612 | +1,5 | Au-dessus du seuil |
| Backbone gelé 3 époques | 0,4576 | +1,1 | À peine au-dessus ; sans effet sur la recette finale |
| Augmentation photo | 0,4544 | +0,8 | Sous le seuil seul |
| Augmentation forte | 0,4503 | +0,4 | Sous le seuil |
| Recadrage plus fort (0,35) | 0,4486 | +0,2 | Sous le seuil |
| Seuil d'aire 0,02 / 0,005 | 0,4491 / 0,4459 | +0,3 / -0,1 | Sans effet |
| Perte ASL | 0,4242 | -2,2 | Rejeté |
| BCE pondérée (pos_weight) | 0,4146 | -3,2 | Rejeté |

- **Diagnostic :** dans toutes les runs, la perte d'entraînement continue de baisser alors que la perte de validation remonte : **surapprentissage**. Ajouter des époques n'aide pas ; ce qui aide est de voir plus de détails (résolution) et des photos plus variées (augmentation).

### 5.4 Les trois finalistes, recette finale (256 px, augmentation photo)

| Modèle | mAP val | mAP **test** | F1 micro test | mAP classes fréquentes / rares (test) |
|---|---|---|---|---|
| **ConvNeXt-Atto** | 0,471 | **0,388** | 0,485 | 0,511 / 0,228 |
| MobileNetV3-Large | 0,431 | 0,363 | 0,422 | 0,482 / 0,211 |
| EfficientNet-Lite0 | 0,400 | 0,345 | 0,404 | 0,470 / 0,184 |

Source : `results/metrics.csv` (lignes `torch`, `test`), `results/errors/` (erreurs par classe et confusions).

### 5.5 Photos personnelles et « monde ouvert » (A9)

| Modèle | mAP photos perso (108) | Photos sans nourriture rejetées |
|---|---|---|
| ConvNeXt-Atto | 0,306 | 0 / 13 avec les seuils par classe |
| MobileNetV3-Large | 0,259 | 1 / 13 |
| EfficientNet-Lite0 | 0,264 | 0 / 13 |

- **Écart de domaine :** environ 8 points de moins que le test FoodSeg ; le classement reste le même.
- **Monde ouvert :** le modèle n'a jamais vu de photo sans nourriture et trouve toujours des ingrédients. Règle ajoutée : « non reconnu » si la probabilité maximale est sous **0,4**. Sur les photos perso, ConvNeXt garde alors 85 % des plats et rejette 10 photos sans nourriture sur 13 (seuil choisi sur ces mêmes photos : estimation optimiste).

---

## 6. Problèmes rencontrés et solutions (côté modèle)

| Problème | Cause | Solution |
|---|---|---|
| CUDA indisponible | Pilote NVIDIA trop ancien pour les wheels cu132 | Mise à jour du pilote (555 → 621) |
| Cible fausse après recadrage | La liste d'ingrédients ne suit pas l'augmentation | Cible recalculée à partir du masque transformé |
| Comparaisons trompeuses | Bruit d'entraînement | Variance mesurée sur 3 seeds : seuil d'interprétation de 1 point |
| Surapprentissage | 4 225 images pour 103 classes | Résolution 256 + augmentation photo, early stopping |
| Choix du LR | Deux LR à égalité | 0,0003 : même mAP, courbes plus stables |
| Écart val / test (0,471 contre 0,388) | Sélection sur la même val (13 runs), classes rares | Signalé ; test utilisé une seule fois |
| Intervalles de confiance décalés | Le bootstrap sur images fait disparaître les classes rares de certains tirages | Signalé, à corriger avant de citer les IC |
| Photos sans nourriture | Jamais vues à l'entraînement | Seuil « non reconnu » sur la probabilité max (0,4) |

---

## 7. Ce qu'il faut retenir

- **Meilleur modèle :** ConvNeXt-Atto, 256 px, mAP test 0,388 en fp32 et **0,367 en int8 déployé**.
- **Objectif de mAP non atteint** (0,40), ce qui est dit tel quel : la cible avait été fixée avant de mesurer.
- **Les classes rares sont le point faible** (mAP 0,23 contre 0,51 pour les classes fréquentes).
- **Suite prévue (partie D) :** reconnaissance du plat (Food-101) avec le même backbone et une seconde tête.

---

## 8. Où trouver les tableaux et figures

| Contenu | Fichier |
|---|---|
| Paramètres et leur sens | `configs/base.yaml` |
| Journal de toutes les décisions et mesures | `docs/decisions.md` |
| Répartition des classes par split | `results/split_counts.csv` |
| EDA (figures) | `results/eda/*.png` |
| Screening | `results/screening.csv`, `results/candidates.csv` |
| LR range test | `results/lr_range_*.png` |
| Courbes de chaque run (loss, mAP par époque) | `results/runs/*_history.csv`, et l'interface MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) |
| Métriques test, photos perso, int8 | `results/metrics.csv` |
| Erreurs par classe, confusions | `results/errors/` |
| Modèles entraînés | `models/<run>/` (`best.pt`, `config.yaml`, `postprocess.json`) |

---

## 9. Thèmes principaux

- **Classification multi-étiquettes :** une sigmoïde par classe, perte BCE, seuil par classe.
- **Transfer learning / fine-tuning :** backbone ImageNet, LR discriminatif (backbone 10 fois plus lent que la tête).
- **mAP macro :** moyenne de l'Average Precision par classe ; chaque ingrédient compte autant, rare ou fréquent.
- **Déséquilibre de classes :** biais initialisé à la fréquence, pertes pondérées (testées et rejetées), analyse fréquentes / rares.
- **Rigueur expérimentale :** objectifs fixés avant de mesurer, un changement à la fois, même seed, variance entre seeds, test utilisé une seule fois.
- **Contrôles de santé :** perte initiale, sur-apprentissage d'un batch, LR range test.
- **Surapprentissage :** diagnostic par les courbes train / val, réponse par résolution et augmentation.
- **Calibration :** température (temperature scaling), ECE, seuils choisis sur la validation.
- **Monde ouvert :** rejet par seuil sur la probabilité maximale.
- **Suivi d'expériences :** MLflow, configuration YAML versionnée, splits reproductibles (hash identique).

---

## 10. Guide de la page « Comment fonctionne le modèle » (`/scan/model`)

Dans l'application, la page est volontairement courte : une ligne sous chaque graphique, et chaque partie s'ouvre d'un clic (▶). Les explications ci-dessous sont à dire à l'oral, section par section. Tous les chiffres de la page viennent de `model_report.json`, produit par `scripts/export_model_report.py` à partir de `results/`.

### 10.1 Le modèle en bref
- La photo ne quitte jamais le téléphone. Elle est redressée, recadrée au carré et réduite à 256 × 256 pixels.
- Un ConvNeXt-Atto quantifié en int8 (3,8 MB) tourne dans le navigateur avec onnxruntime-web (WebAssembly).
- Il donne une probabilité pour chacun des 103 ingrédients. Chaque ingrédient a son propre seuil. Si aucun n'atteint 0,4, la réponse est « non reconnu ».
- Les trois tuiles : mAP test 0,367, taille 3,8 MB, 73 ms pour le modèle seul sur le téléphone (4 threads).

### 10.2 Entraînement
**Pourquoi seulement 3 modèles :** ce sont les finalistes. Avant, un criblage de 6 backbones avec une recette courte (tableau « Criblage », ✓ = finalistes). ConvNeXt-Atto et MobileNetV3-Large étaient les meilleurs. EfficientNet-Lite0 est arrivé dernier (0,275) mais a été gardé, parce qu'il est conçu pour l'int8 sur téléphone (pas de squeeze-and-excitation, activation ReLU6).

**Les noms :**
- **ConvNeXt-Atto :** la plus petite taille (3,4 M paramètres) de ConvNeXt (Meta, 2022), un réseau convolutif moderne qui reprend des idées des Transformers. « Atto » est le préfixe 10⁻¹⁸.
- **MobileNetV3-Large :** réseau de Google (2019) conçu pour les téléphones, variante « Large » (l'autre est « Small »).
- **EfficientNet-Lite0 :** EfficientNet (Google, 2019) simplifié pour l'embarqué ; « 0 » est la plus petite taille.

**Graphique mAP de validation :** `val_map` après chaque époque. Le rond marque l'époque gardée. `select` est le nombre qui sert à choisir cette époque : dans ces runs il vaut exactement `val_map`. Il ne serait différent qu'avec la tête « plat » de la partie D (moyenne de val_map et du top-1).

**Graphique de la loss, comment lire le surapprentissage :**
1. Au début, les deux courbes descendent ensemble : le modèle apprend des règles qui valent aussi pour des photos nouvelles.
2. Vers l'époque 16 (ConvNeXt), la loss d'entraînement (pointillé, photos déjà vues) continue de baisser jusqu'à 0,039, tandis que la loss de validation (photos jamais vues) s'arrête vers 0,075 et remonte légèrement.
3. L'écart qui grandit entre les deux, c'est le surapprentissage : le modèle apprend ses photos par cœur au lieu de règles générales.

**Subtilité :** la loss de validation est au plus bas à l'époque 16, mais l'époque gardée est la 23. Le choix se fait sur la mAP (« les bons ingrédients sont-ils en tête ? ») et non sur la loss (« les probabilités sont-elles bien calibrées ? »), et la mAP a continué de progresser jusqu'à la 23. La calibration est corrigée ensuite par la température. L'early stopping arrête l'entraînement après 6 époques sans amélioration.

**Tableau :**
- **Époque meilleure / jouées :** l'époque gardée et le nombre total d'époques avant l'arrêt.
- **mAP test :** sur les 2 135 photos de test de FoodSeg103, mesurée une seule fois à la fin, sur le PC. Elle ne vient pas des JSON du téléphone, qui ne mesurent que le temps.
- **s / époque :** coût de l'entraînement sur le PC.
- **LR (learning rate) :** de combien les poids bougent à chaque pas. Trop haut, l'entraînement diverge ; trop bas, il apprend lentement. Chaque backbone a le sien (ConvNeXt 0,0003, MobileNetV3 0,0005, Lite0 0,0006).

### 10.3 Quantification
Quantification statique post-entraînement (PTQ) en int8 avec onnxruntime : poids sur 8 bits, plages des activations calibrées sur des photos de validation. 11 variantes ; chaque barre est la mAP perdue par rapport au modèle float, mesurée sur la validation. La ligne verticale est l'objectif fixé avant de mesurer (1,5 point).

**Pourquoi la variante retenue perd 4,5 points ici mais 2,1 sur le test :** la validation n'a que 758 photos ; les ingrédients rares y ont très peu d'exemples, donc l'estimation est bruitée. La validation sert à **choisir** la variante, le test (2 135 photos) à **mesurer** le résultat final, pour ne pas choisir en regardant le test. Le chiffre de référence est 2,1 points.

**Le vocabulaire des variantes :**
- **MinMax / Entropy / Percentile :** trois façons de choisir la plage de valeurs à convertir en entiers. MinMax : du minimum au maximum observés. Entropy : la plage qui conserve le mieux la forme de la distribution. Percentile : on coupe les valeurs extrêmes les plus rares.
- **50 / 200 / 500 :** nombre de photos de calibration.
- **per-channel / per-tensor :** une échelle par filtre de convolution (plus précis, le défaut) ou une seule par couche.
- **coverage :** photos de calibration choisies pour couvrir les 103 ingrédients, au lieu du hasard.
- **reduce_range :** poids sur 7 bits au lieu de 8, pour éviter la saturation des calculs int8 sur les processeurs x86 sans instructions VNNI (le Ryzen du PC de mesure).
- **only Conv :** précision mixte, seules les convolutions passent en int8 ; LayerNorm, GELU et la tête restent en float.

**Les 3 corrections :**
1. **Tête en float :** la sortie sigmoïde était elle aussi quantifiée, ce qui ne laissait que 256 probabilités possibles, dont 40 % tombaient à 0.
2. **reduce_range :** les noyaux int8 d'un PC sans VNNI saturent.
3. **Seules les convolutions en int8 :** ConvNeXt (LayerNorm, GELU) supporte mal l'int8 sur ces couches.

### 10.4 Float contre int8
Les deux modèles répondent sur les **mêmes** 2 135 photos de test. Un bootstrap apparié de la différence (on rééchantillonne les photos des milliers de fois) donne l'intervalle de confiance à 95 %.
- **ConvNeXt :** 0,388 → 0,367, perte de 2,1 points, IC [1,0 ; 3,1]. L'intervalle ne contient pas 0 : la perte est petite mais réelle.
- **Lite0 :** 0,345 → 0,345, perte nulle.
- En échange, le fichier est 3,7 fois plus petit (14,1 → 3,8 MB).
- 1 point = 0,01 de mAP.

### 10.5 Sur le téléphone
Temps médian d'une analyse sur un Realme GT Master (Chrome), 3 campagnes de 100 exécutions.
- **CPU 1 / CPU 4 :** nombre de cœurs du processeur utilisés en parallèle (WebAssembly). **GPU :** la puce graphique via WebGPU.
- **Total :** préparation de la photo + modèle, en CPU 4.
- L'int8 n'est **pas** plus rapide dans le navigateur : son avantage est la taille. Le modèle seul respecte les 300 ms ; la chaîne complète non, parce que préparer une photo de 12 mégapixels prend de 350 à 560 ms.

### 10.6 Ce qui est en production, et pourquoi
ConvNeXt-Atto int8 : la plus précise des variantes quantifiées (0,367 de mAP test, 3,8 MB), ce que demande le sujet (« déployez le modèle quantifié ayant la meilleure performance »).
**Limites :** l'objectif de mAP 0,40 n'est pas atteint, les ingrédients rares sont bien moins reconnus que les fréquents, et le modèle n'a jamais vu de photos sans nourriture à l'entraînement.
