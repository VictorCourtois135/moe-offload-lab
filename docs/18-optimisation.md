# 18 — Optimiser moelab : le plan complet

Objectif : rendre le moteur aussi rapide que le matériel le permet, en couvrant toutes
les pistes d'optimisation, **une à la fois, mesurée avant et après**. Point de départ
(benchmark de référence du 30/09, OLMoE) : une couche MoE coûte 9,5 ms en `lo` même
quand tout est en VRAM, 12 ms en `hi`. Le calcul GPU pur de 8 experts pour un token
devrait prendre de l'ordre de 0,1 à 0,3 ms : il y a donc un facteur 30 à 100 entre
le moteur et le plancher physique.

## La méthode

1. **Mesurer d'abord** (phase 0) : savoir où part chaque milliseconde, et quel est le
   plancher physique de la machine. Sans ça, on optimise au hasard.
2. **Une optimisation à la fois**, chacune avec :
   - un test d'exactitude (en fp16, mêmes logits que transformers, comme aujourd'hui) ;
   - une mesure avant/après avec `profile_moe.py` et `benchmark.py` à budget fixé ;
   - une ligne dans le tableau de suivi ci-dessous, gain mesuré compris, même nul.
3. **Garder une option pour désactiver** chaque optimisation, pour pouvoir mesurer son
   effet seul et revenir en arrière.

## Le budget d'un token (à remplir avec la phase 0)

Un token = 16 couches. Chaque couche = une partie dense (attention, normes, routeur)
+ une partie MoE. Hypothèses à vérifier :

| Poste | Estimation actuelle | Plancher physique | Source |
|---|---|---|---|
| Couche MoE, tout en VRAM (`lo`) | 9,5 ms | ~0,1-0,3 ms | benchmark du 30/09 |
| Partie dense d'une couche | ~1 ms | ? | (169 ms/token − 16 × 9,5 ms) / 16 |
| Lancement d'un noyau GPU (Windows) | ? | — | `profile_moe.py` |
| Synchro GPU → CPU | ? | — | `profile_moe.py` |

## Le catalogue : toutes les pistes

### A. Surcoût du moteur (le plus gros gisement)

| # | Optimisation | Pourquoi | Gain attendu |
|---|---|---|---|
| A1 | Profilage (phase 0) | savoir quoi optimiser | — |
| A2 | Décisions vectorisées : HOBBIT, regroupement des tokens et recherche en cache sans boucles Python par expert | ~0,3 ms/couche de pure comptabilité mesurée sur le modèle miniature | 0,3-0,5 ms/couche |
| A3 | Cache VRAM en « plaques » préallouées (slots fixes par précision) au lieu d'un `torch.empty` par expert | plus d'allocations, adresses fixes (prérequis de A5 et C1) | faible seul, indispensable ensuite |
| A4 | Calcul groupé des experts d'une couche (bmm sur les slots) au lieu d'une boucle de ~15 noyaux par expert | 8 experts × 15 noyaux = ~120 lancements par couche | 2-5 ms/couche |
| A5 | Chemin de décodage sans synchro : décisions prises en tenseurs sur GPU quand tout est résident | la synchro empêche le CPU de préparer la couche suivante pendant que le GPU calcule | à mesurer |
| A6 | CUDA graphs pour le décodage (couche MoE résidente, puis partie dense) | rejoue une séquence de noyaux en un seul lancement | ×2-5 sur ce qui reste |
| A7 | Cache KV statique + `torch.compile` de la partie dense | la partie dense de transformers est aussi en mode « eager » | à mesurer |

### B. Calcul des experts

| # | Optimisation | Pourquoi | Gain attendu |
|---|---|---|---|
| B1 | Noyau int4 fusionné (Triton) : produit matrice-vecteur qui lit l'int4 directement | aujourd'hui l'int4 est décompressé en fp16 (7 noyaux, 4× plus d'octets) à chaque usage | ×2-4 sur les experts int4 |
| B2 | Garder décompressés en VRAM les experts int4 les plus utilisés | alternative sans Triton | selon la place |
| B3 | Branche CPU efficace : int4 lu directement (numpy/numba), plusieurs threads | B_Host int4 mesuré à 0,23 Go/s, 15× plus lent que le fp16 | rend q* utile |

### C. Transferts RAM → VRAM

| # | Optimisation | Pourquoi | Gain attendu |
|---|---|---|---|
| C1 | Copies groupées vers des slots VRAM fixes, plusieurs en vol | une copie et un événement par expert aujourd'hui | à mesurer |
| C2 | Prefetch à 2 couches d'avance | plus de temps pour cacher la copie | selon le rappel |
| C3 | Prefetch par précision : ne précharger en fp16 que les experts prédits importants | évite de remplir la VRAM de fp16 inutiles | hits en hausse |

### D. Disque (décisif pour Qwen3-Next-80B)

| # | Optimisation | Pourquoi | Gain attendu |
|---|---|---|---|
| D1 | Lectures sans cache de fichiers Windows (non tamponnées, alignées 4096) | mesures fiables, pas de double copie mémoire | stabilité des mesures |
| D2 | Profondeur de file d'attente adaptée au disque (NVMe : 8-32, HDD : 1-2 triées) | le NVMe n'atteint son débit qu'avec plusieurs lectures en vol | ×2-4 en lecture NVMe |
| D3 | Lectures triées par position sur le HDD | le positionnement de la tête domine (10 ms) | ×1,5-2 sur HDD |
| D4 | Store réparti NVMe / HDD selon l'historique d'usage | les experts chauds sur le disque rapide | décisif sur 80B |

### E. Politique de précision

| # | Optimisation | Pourquoi | Gain attendu |
|---|---|---|---|
| E1 | Seuil t1 par couche, calibré | le proxy du routeur est plus ou moins fiable selon la couche | qualité à octets égaux |
| E2 | Échelle int8 / int4 / saut | un palier intermédiaire | à mesurer |
| E3 | Précision selon le coût : fp16 si déjà en VRAM, int4 si l'expert manque | l'idée d'origine de HOBBIT (basse précision sur les cache miss) | moins de transferts |

## Les mesures du 30/09 (RTX 2060, Windows)

Fichier : `results/profile_olmoe-hf_2026-09-30_1616.md`.

**Le constat principal : sous Windows, un lancement de noyau GPU coûte ~52 µs**, et le
moteur en lançait ~3 500 par token. Un aller-retour GPU → CPU coûte à peu près pareil
(53 µs). Le GPU, lui, attend : 8 experts fp16 groupés se calculent en 0,35 ms
(286 Go/s, la bande passante de la carte), contre 1,8 ms un par un. Chaque noyau
supprimé fait gagner du temps ; c'est le fil directeur de la suite.

| Politique | base | + A2 | + A2 + prefetch adaptatif | + B1 (Triton) | tout (A2, adaptatif, B1, D1) |
|---|---|---|---|---|---|
| lo (tout en VRAM) | 6,73 tok/s | 7,12 | 8,23 | 11,82 | **11,79** |
| hobbit t1=0,8 | 4,12 | 5,25 | 5,58 | 5,41 | **5,87** |
| hi | (faussé) | (faussé) | 3,94 | 3,68 | **5,18** |

(Mode `async`. Les deux premières mesures de `hi` sont faussées : la RAM disponible
était juste au-dessus du budget de 5 Gio, Windows s'est mis à paginer, avec des
attentes de 100 à 380 ms par couche.)

- **B1, le noyau Triton int4, est le plus gros gain** : en `lo`, le calcul des experts
  passe de 5,9 à 2,5 ms par couche, et le débit de 6,7 à 11,8 tok/s.
- **A2 + prefetch adaptatif** : +22 % en `lo`, +35 % en `hobbit`.
- **B2 (int4 déballé) est rejeté** : 2 fois plus de place en VRAM fait tomber le taux de
  hit de 100 % à 93 %, et les copies coûtent plus que la décompression économisée.
- **En `lo`, la partie dense pèse maintenant presque autant que les experts** : 36 ms
  par token (2,3 ms par couche) sur 85. Elle aussi est limitée par les lancements.
- **En `hi` et `hobbit`, prefetch et copies coûtent 3 à 5 ms par couche** : lancer une
  copie RAM → VRAM de 12 Mio prend ~240 µs de temps CPU sous Windows.

Décisions : A2, prefetch adaptatif et B1 activés par défaut (repli automatique sans
Triton). Étape suivante : **A4+B1, les noyaux groupés** (tous les experts d'une couche en
2 lancements par précision au lieu de ~60), puis les **CUDA graphs** pour la partie
dense (A6).

**À tester, sans code** : l'option Windows « Planification du GPU à accélération
matérielle » (Paramètres > Système > Affichage > Graphiques). Elle change la façon dont
Windows soumet le travail au GPU et peut réduire le coût d'un lancement ; `profile_moe.py`
donne `launch_us` avant et après.

## Les mesures du 01/10 : noyaux groupés

Fichier : `results/profile_olmoe-hf_2026-10-01_0951.md`. Lancement d'un noyau : **19 µs**
(52 la veille), synchro : 25 µs.

| Politique | base | défaut du 30/09 | + A4 (noyaux groupés) | + A4 + D1 | + A4 + D1 + C2 |
|---|---|---|---|---|---|
| lo (tout en VRAM) | 6,29 tok/s | 12,42 | **20,16** | 20,01 | 18,09 |
| hobbit t1=0,8 | 4,74 | 5,56 | 6,21 | **6,80** | 5,01 |
| hi | (faussé) | (faussé) | (faussé) | (faussé) | 4,42 |

- **A4+B1, les noyaux groupés : +62 % en `lo`, +12 % en `hobbit`.** En `lo`, une couche
  MoE passe de 2,9 à 0,9 ms. Activés par défaut.
- **D1 (lecture sans cache) : +9 % en `hobbit`**, neutre en `lo` (rien n'est lu). Activé
  par défaut, avec repli automatique si le disque ne le permet pas.
- **C2 (prefetch à 2 couches) est rejeté** : il remplit la VRAM d'experts prédits à tort,
  le taux de hit tombe de 78 à 72 %, et tout ralentit.
- `hi` reste inexploitable : 12 Gio de fp16 pour 5 Gio de RAM, les mesures varient d'un
  facteur 5 d'une passe à l'autre (pagination de Windows).
- **La partie dense pèse maintenant 70 % d'un token en `lo`** : 35 ms sur 50. La trace
  de `torch.profiler` (décodage seul) compte 1 246 lancements par token, dont ~1 000 pour
  la partie dense (~60 par couche : 4 normes RMS de 7 noyaux, RoPE, cache KV, attention,
  routeur). Le GPU, lui, ne lit que 0,9 Gio de poids : ~3 ms.
- **En `hobbit`, c'est le disque** : 11,6 experts fp16 lus par token (139 Mo), et
  ~6 ms par couche entre copies, attente et prefetch.
- Un défaut corrigé au passage : la table d'adresses des noyaux groupés était copiée vers
  le GPU depuis de la mémoire non épinglée, ce qui **attendait la fin du travail GPU** à
  chaque couche (0,6 ms de « phase 6 » en `lo`). Elle passe maintenant par un anneau de
  tampons épinglés, sans synchronisation.

Étape suivante, en cours : **A6, la partie dense en CUDA graphs** (ci-dessous).

## A6 : la partie dense en CUDA graphs

Un CUDA graph enregistre une suite de noyaux et la relance en un seul appel. La couche
MoE ne peut pas en faire partie (ses décisions sont prises sur CPU), d'où le découpage :

```
segment 0 : embeddings -> couche 0 (normes, attention, routeur)        [1 graph]
couche MoE 0 : notre runtime, en Python, comme avant
segment 1 : résiduel de la couche 0 -> couche 1 (normes, attention, routeur)   [1 graph]
...
segment 16 : résiduel de la couche 15 -> norme finale                  [1 graph]
tête de sortie (1 noyau)
```

17 graphs au lieu de ~1 000 lancements. La capture se fait sur une vraie passe du
modèle transformers : chaque appel d'un module d'experts ferme le graph en cours et en
ouvre un nouveau. Rien de propre à OLMoE : le même code découpe Qwen3-Next (testé sur
le modèle miniature, attention linéaire comprise).

Conditions : un cache KV **statique** (transformers `StaticCache` : la position courante
est un tenseur sur le GPU, jamais lue par Python), donc une taille maximale fixée à
l'avance (`--graph-max-len`, automatique depuis le 03/10 : 2 048 tokens pour OLMoE, ~270 Mo
de VRAM ; 8 192 pour Qwen3-Next, ~200 Mo).
L'attention lit tout le cache statique à chaque token (~1 ms par token à 2 048).

Test sans GPU : un émulateur de graphs (`moelab/graphs.py`) enregistre les opérations
de la capture et les rejoue sur les mêmes tenseurs, avec les règles d'un vrai graph :
une lecture de valeur pendant la capture est une erreur, la capture ne modifie rien.
`tests/test_graphs.py` vérifie que la génération rejouée est identique à la génération
normale (OLMoE et Qwen3-Next miniatures), et `check_setup.py` fait la même vérification
avec de vrais CUDA graphs sur la carte.

Activé par défaut depuis la mesure du 01/10 (`--no-cuda-graphs` pour le couper). Si la
capture échoue, la raison est affichée et la génération continue sans graphs.

## Les mesures du 01/10 (2) : CUDA graphs

Fichier : `results/profile_olmoe-hf_2026-10-01_1026.md`. `check_setup.py` : génération
identique avec et sans graphs sur la carte.

| Politique | défaut du 30/09 | défaut du 01/10 matin | + CUDA graphs |
|---|---|---|---|
| lo | 12,62 tok/s | 20,86 | **41,88** |
| hobbit t1=0,8 | (2,65, mesure perturbée) | 6,75 | **7,98** |

- **La partie dense passe de 34 ms à 1,5 ms par token** : 17 graphs rejoués au lieu
  d'environ 1 000 lancements (187 lancements restants par token, tous dans les couches MoE).
- En `lo`, un token coûte 24 ms, dont ~22 dans les couches MoE. Dans chaque couche, le
  CPU attend le GPU (lecture du routage : 0,7 ms, c'est l'attention qui finit), puis le
  GPU attend le CPU (décisions, table, lancement des noyaux groupés). Et la trace montre
  que **les noyaux groupés v1 eux-mêmes prennent ~0,77 ms par couche pour 25 Mo d'int4,
  soit ~35 Go/s** sur les ~240 que lit la carte : 12 ms par token à eux seuls.
- En `hobbit`, la partie dense ne pèse plus rien (3 ms) : il reste les octets à déplacer
  (11 experts fp16 lus sur disque par token, ~28 copies RAM → VRAM).

D'où les **noyaux groupés v2** : adresses alignées déclarées (lectures de 16 octets), pas
de masques quand les dimensions tombent juste, une échelle par groupe au lieu d'une par
octet, le noyau B réparti sur (lignes, experts) avec addition atomique, et les tailles de
blocs choisies par `triton.autotune` sur la carte. `profile_moe.py` mesure v1 et v2 seuls
(planchers : `grouped_int4_v1/v2_gbs`) et dans la génération (variantes `cg_v1` et
`defaut`).

Défaut corrigé dans `profile_moe.py` : la trace de `torch.profiler` était faite sans
remettre les caches à zéro après `hobbit`, d'où des copies RAM → VRAM qui n'existent pas
en régime normal de `lo`.

## Les mesures du 01/10 (3) : noyaux groupés v2, et la suite

Fichier : `results/profile_olmoe-hf_2026-10-01_1054.md`.

| Noyaux groupés, 8 experts, 1 token | ms GPU | Go/s effectifs |
|---|---|---|
| int4 v1 | 0,501 | 53 |
| int4 v2 | 0,387 | 69 |
| fp16 v2 (pour comparer) | 0,382 | **263** |
| plancher int4 (lecture des octets) | 0,094 | 286 |

| tok/s | CUDA graphs + v1 | CUDA graphs + v2 |
|---|---|---|
| lo | 41,8 | **48,4** |
| hobbit t1=0,8 | (perturbé) | 7,6 |

Deux constats :

1. **Le noyau fp16 atteint la bande passante de la carte, pas l'int4.** Les deux font
   le même nombre de multiplications par seconde (~135 milliards) : l'int4 est limité par
   le calcul, pas par la mémoire. Le suspect : la conversion entier → flottant de chaque
   quartet (instruction I2F, 4 fois plus lente qu'une addition sur Turing). **v3** la
   remplace par une astuce de bits : `(q | 0x4B000000)` relu comme un float32 vaut
   exactement 2^23 + q, une soustraction donne q − 8. Résultat identique au bit près.
2. **En `lo`, le CPU pèse plus que les noyaux** : par couche, ~0,7 ms de Python pour
   préparer et lancer les noyaux groupés (vues des records, table, 4 lancements Triton
   coûteux sous Windows), contre ~0,4 ms de calcul. **A6 bis** : le calcul groupé de
   chaque couche (copie de la table, noyaux int4 et fp16, somme) est capturé en UN CUDA
   graph. Par token, le CPU écrit seulement k adresses et k poids dans une table
   épinglée (numpy), puis relance le graph. La table a une taille fixe (k lignes par
   précision) ; une ligne de poids 0 ne fait rien. La branche CPU de q* reste possible
   (son résultat est ajouté dans le graph).

`profile_moe.py` mesure v1 / v2 / v3 seuls (Go/s), le temps mur d'un appel normal contre
un appel en graph, et les variantes `cg_v2`, `cg_v3`, `defaut` (v3 + A6 bis).

## Les mesures du 01/10 (4) : calcul groupé en graph, v3

Sortie console du profil de 11h16 (`check_setup.py` : tout OK, graphs de calcul groupé
actifs en hi, lo et hobbit).

| tok/s | CUDA graphs + v2 | + v3 | + v3 + calcul groupé en graph (A6 bis) |
|---|---|---|---|
| lo | 46,7 | 47,0 | **89,6** |
| hobbit t1=0,8 | 6,4 | 7,9 | **8,6** |

- **A6 bis double la vitesse en `lo`** : un appel du calcul groupé passe de 0,383 ms
  (temps mur, CPU compris) à 0,162 ms en graph. Par couche : 1,23 → 0,61 ms.
- **v3 ne gagne rien** (68 Go/s contre 69) : la conversion des quartets n'était pas le
  frein. Hypothèse suivante : la forme du noyau (blocs 3D, deux réductions par tour,
  changements de disposition des données entre les deux). **v4** reprend la forme du
  noyau fp16 (blocs 2D d'un groupe, accumulateur 2D, une seule réduction à la fin).
  Par défaut, les noyaux sont maintenant **choisis par mesure** au démarrage (v2 ou v4,
  le plus rapide sur la carte).
- En `lo`, par couche : 0,48 ms d'attente du GPU (lecture du routage) sur 0,61. Le GPU
  travaille presque tout le temps : les noyaux int4 (~0,39 ms) sont le premier poste.
- En `hobbit`, par couche : copies 1,1 ms + attente 2,3 ms + prefetch 3,1 ms de temps
  CPU. Une trace de `hobbit` dira où (`--trace-policy hobbit@0.8`).

## Les mesures du 01/10 (5) : v4, et ce que montrent les traces

Fichiers : `results/profile_olmoe-hf_2026-10-01_1133.md` (lo + hobbit) et `_1137.md`
(trace de hobbit). `lo` : **89,1 tok/s** ; `hobbit@0.8` : 8,5 à 10,4 tok/s.

1. **v2, v3 et v4 donnent tous ~70 Go/s, et c'était une erreur de mesure.** Le temps
   « GPU » des noyaux groupés se mesurait en appels Python enchaînés : le temps mur d'un
   appel (0,37 ms) égale le « temps GPU » mesuré, donc le GPU attendait le CPU. Le même
   calcul rejoué en graph prend 0,15 ms tout compris : les noyaux sont au moins 2,5 fois
   plus rapides que mesuré. Le microbenchmark mesure maintenant les noyaux en graph
   (`grouped_int4_vN_kernel_gbs`), et le choix automatique v2/v4 aussi.
2. **`lo` (trace) : le reste est surtout dans la partie dense et l'attention.** Projections
   de l'attention et tête de sortie (cuBLAS) ~40 % du temps GPU, attention ~18 %, noyaux
   groupés ~25 %. L'attention lit les 2 048 positions du cache statique à chaque token,
   même pour une conversation de 100 tokens. D'où les **paliers de longueur** (ci-dessous).
3. **`hobbit` (trace) : le PCIe est saturé.** ~46 copies fp16 de 12 Mio par token
   (~1 ms chacune) + ~7 int4 : **~49 ms de transferts RAM → VRAM par token** sur ~96. Le
   lancement d'une copie ne coûte que ~20 µs de CPU : le coût est le volume. C'est le
   cœur du problème pour Qwen3-Next-80B : il faut transférer moins, pas plus vite. Le
   levier : E3 (`--miss-t1`), qui calcule en int4 (4 fois moins d'octets) un expert
   absent de la VRAM et peu important. Le benchmark mesure maintenant la qualité **token
   par token** (`--quality-mode decode`, défaut) : en une passe sur tout le texte, E3
   n'agissait pas et la qualité de `hobbit_miss` était faussement identique à `hobbit`.
4. Rare : quelques copies depuis de la mémoire non épinglée (pool RAM plein), 0,5 à
   2,5 ms de CPU chacune, ~1,4 par token en `hobbit`.

**Paliers de longueur** : un jeu de graphs par palier (128, 256, 512, 1 024, 2 048) ;
pendant la capture, le cache est remplacé par une vue de ses L premières positions
(mêmes données). Chaque token rejoue le plus petit palier qui le contient : l'attention
ne lit que ce qui existe. Testé avec l'émulateur (OLMoE et Qwen3-Next, passage de
8 → 16 → 32 positions, logits identiques). Variante `sans_paliers` pour comparer.

## Les mesures du 01/10 (6) : paliers, vrais temps des noyaux, E3

Fichiers : `results/profile_olmoe-hf_2026-10-01_1155.md`, `results/benchmark_olmoe-hf_2026-10-01_1203.md`.

- **Paliers de longueur : `lo` 89,4 → 105,9 tok/s** (+18 %), `hobbit` 8,4 → 9,0.
- **Temps réels des noyaux (en graph)**, 8 experts : int4 v2 0,160 ms (167 Go/s), **v3
  0,152 ms (176 Go/s)**, v4 0,220 ms (121 Go/s) ; plancher 0,094 ms. v3 entre dans le choix
  automatique.
- **`hobbit@0.8` en génération : 28 transferts à la demande + 41 prefetch par token,
  ~680 Mo RAM → VRAM par token**, prefetch juste à 84 %. Au benchmark (8 prompts variés),
  le hit VRAM tombe à 59 % et le disque monte à 25 lectures fp16 par token : 5,6 tok/s.
- **E3 ne vaut pas le coup ici** : `hobbit_miss` (miss_t1 = 0,3) gagne 6 % de vitesse
  (5,55 → 5,89 tok/s) et perd beaucoup en qualité (KL 0,0074 → 0,0176, accord top-1
  98,0 → 94,9 %). Rejeté.
- Qualité (token par token, référence fp16) : `lo` KL 0,07, top-1 89,5 %, perplexité
  +10,4 % ; `hobbit@0.8` KL 0,0074, top-1 98,0 %, perplexité +1,1 %.

**Conclusion pour `hobbit`** : avec 12 Mio par expert fp16, le cache VRAM (3 Gio) et le
cache RAM (5 Gio) ne tiennent pas l'ensemble de travail ; tout se joue sur le volume.
D'où **int8 comme haute précision** : un store int8 + int4 (`build_store --hi int8`)
divise par 2 la taille d'un expert « haute précision » (6,4 Mio), donc double ce que
tiennent les caches et divise par 2 les octets copiés. L'int8 par groupes de 64 est
quasi sans perte (bien plus proche du fp16 que l'int4). Noyaux groupés int8 ajoutés
(même forme que v2, testés sous l'interprète et en graph, store int8 + int4 en hobbit).

Attention à la mesure : avec un store int8 + int4, la référence de qualité du benchmark
(`hi`) est l'int8, pas le fp16. La perte int8 → fp16 se mesure à part (ci-dessous).

## Les mesures du 01/10 (7) : int8 comme haute précision

Fichiers : `results/profile_olmoe-hf_2026-10-01_1227.md`, `results/benchmark_olmoe-hf_2026-10-01_1232.md`
(store `C:\moe-store\olmoe-i8`, int8 + int4, 9,4 Gio, construit en 6 min).

| hobbit@0.8, budgets VRAM 3 Gio / RAM 5 Gio | store fp16 + int4 | store int8 + int4 |
|---|---|---|
| profil (1 prompt) | 9,0 tok/s, 680 Mo RAM → VRAM par token, hit 78 % | **36,2 tok/s**, 130 Mo par token, hit 94 % |
| benchmark (8 prompts) | 5,6 tok/s, 25 lectures disque par token | **22,6 tok/s**, 2,6 lectures disque par token |
| KL (référence = haute précision du store) | 0,0074 | 0,0066 |

**×4 en `hobbit`**, pour une qualité du même ordre. Le noyau int8 groupé tourne à
221 Go/s. Avec 6,2 Mio par expert au lieu de 12, les caches en tiennent deux fois plus
et chaque défaut de cache coûte deux fois moins : le nombre de copies chute (8 à la
demande + 14 prefetch par token, contre 28 + 41).

Reste à chiffrer la perte int8 → fp16 elle-même (la référence du benchmark était l'int8) :
`benchmark.py --save-ref` sur le store fp16, puis `--ref` sur le store int8 (commande dans
le GUIDE). Attendu : très faible (int8 par groupes de 64).

## Bilan de l'optimisation sur OLMoE (RTX 2060, Windows)

| Étape | lo (tok/s) | hobbit@0.8 (tok/s, profil) |
|---|---|---|
| Version de base (29-30/09) | 6,7 | 4,1 |
| A2 + prefetch adaptatif + noyau Triton int4 (B1) | 11,8 | 5,9 |
| Noyaux groupés (A4) + lecture sans cache (D1) | 20,2 | 6,8 |
| CUDA graphs pour la partie dense (A6) | 41,9 | 8,0 |
| Noyaux groupés v2 | 48,4 | 7,6 |
| Calcul groupé en graph (A6 bis) | 89,6 | 8,6 |
| Paliers de longueur du cache statique (A8) | **105,9** | 9,0 |
| int8 comme haute précision (E4) | (104,4) | 36,2 |
| A9 (prédiction du prefetch dans le graph) | 104,7 | 39,4 |
| D5 (partie dense en int8, optionnel) | 118,9 | 46,7 |
| Q1 + Q2 (int4 mse puis GPTQ : qualité, pas vitesse) | 117,5 | 51,5 |
| A5 (token entier en un graph) | **132,1** | **51,5** |

×20 en `lo`, ×13 en `hobbit` (×4 en `hobbit` sans D5 ni int8 : la version de base faisait 4,1 tok/s). Ce qui n'a pas marché, et pourquoi : B2 (int4 déballé :
trop de place en VRAM), C2 (prefetch à 2 couches : remplit la VRAM d'erreurs), E3 (int4
sur défaut de cache : +6 % de vitesse, qualité nettement moins bonne), v3/v4 (gains nuls
ou négatifs une fois les noyaux mesurés correctement).

Ce qu'il reste en `lo` (8,4 ms par token avec D5, plancher physique ~5 ms) : la lecture
du routage par le CPU à chaque couche (A5, codé le 02/10, ci-dessous), la tête de
sortie. En `hobbit` : le volume transféré, déjà divisé par 6.

## A9 : la prédiction du prefetch dans le graph

Au profil `hobbit` int8, le prefetch coûtait encore 0,87 ms de CPU par couche (14 ms sur
28 par token) : appliquer le routeur de la couche suivante (4 petits noyaux lancés en
Python), puis une 2e lecture GPU → CPU par couche. Maintenant, au moment de couper le
graph avant chaque couche MoE, la capture ajoute aussi ce routeur ET rassemble routage
courant + prédiction en un seul tenseur. Par couche : une seule lecture, aucun noyau
lancé pour prédire.

## Les mesures du 02/10 : A9, et la perte int8 → fp16

Fichiers : `results/profile_olmoe-hf_2026-10-02_0923.md` (store int8),
`results/benchmark_olmoe-hf_2026-10-02_0928.md` (store int8, **référence fp16**).

- **A9 : `hobbit@0.8` 35,1 → 39,4 tok/s** (+12 %) au profil, 22,6 → 26,2 au benchmark ;
  prefetch 0,89 → 0,58 ms par couche. Neutre en `lo` (102,7 → 104,7).
- Qualité contre le vrai fp16 :

| Réglage (store int8 + int4) | tok/s | KL | accord top-1 | perplexité |
|---|---|---|---|---|
| hi (tout int8) | 23,8 | 0,0027 | 98,0 % | +0,38 % |
| hobbit@0.8 | 26,2 | 0,0079 | 98,0 % | +0,53 % |
| lo (tout int4) | 86,9 | 0,070 | 89,5 % | +10,6 % |
| *rappel : hobbit@0.8, store fp16 + int4* | *5,6* | *0,0074* | *98,0 %* | *+1,1 %* |

**hobbit int8 a la même qualité que hobbit fp16, 4,7 fois plus vite.**

Plancher de la mesure (`results/benchmark_olmoe-hf_2026-10-02_0945.md`) : `hi` sur le store
fp16, token par token, contre la référence fp16 calculée en une passe : **KL 0,0006**,
top-1 98,8 %. C'est le bruit des arrondis fp16 entre les deux chemins de calcul. La perte
propre à l'int8 est donc d'environ **KL 0,002** (0,0027 − 0,0006).

## D5 : la partie dense en int8

Les experts sont maintenant rapides ; la partie dense (attention, tête de sortie) est
relue en entier à chaque token : ~0,75 Go en fp16 sur OLMoE. Sur Qwen3-Next-80B, c'est
~4 Go dont une tête de sortie de 622 Mo, à faire tenir dans 6 Go de VRAM avec les
experts : impossible en fp16. `--dense-int8` passe toutes les couches Linear de la partie
dense (sauf les routeurs) en int8 par groupes de 64 : noyau GEMV Triton en génération
(capturé dans les CUDA graphs), déquantification par blocs en prefill. En placement
`split`, la tête de sortie reste alors sur le GPU (311 Mo pour Qwen3-Next au lieu d'un
calcul sur CPU). Le choix automatique du placement tient compte de la taille int8.

## Les mesures du 02/10 (2) : D5, partie dense en int8

Fichiers : `results/profile_olmoe-hf_2026-10-02_0952.md`, `results/benchmark_olmoe-hf_2026-10-02_0954.md`.
Partie dense : 65 couches Linear, 0,69 → 0,36 Gio ; budget VRAM des experts 2,98 → 3,10 Gio.
Tête 50304 x 2048 : 0,66 ms en fp16, 0,35 ms en int8.

| store int8 + int4 | sans D5 | avec D5 | qualité (KL contre fp16) sans → avec |
|---|---|---|---|
| lo, profil | 104,7 tok/s | **118,9** | 0,070 → 0,070 |
| lo, benchmark | 86,9 | **109,1** | |
| hobbit@0.8, profil | 39,4 | **46,7** | 0,0079 → 0,0089 |
| hobbit@0.8, benchmark (2 passes chacun, sans mesure de qualité) | 26,1 | **26,8** | perplexité +0,53 % → +1,01 % |

- D5 accélère nettement les deux politiques au profil, et `lo` au benchmark.
- Le 17,5 tok/s d'un premier benchmark `hobbit` était une mesure aberrante : revérifié en
  deux passes avec et sans D5 (`results/benchmark_olmoe-hf_2026-10-02_1000.md` et `_1001.md`),
  26,1 → 26,8 tok/s. Au benchmark, `hobbit` est limité par les défauts de cache (8 prompts
  variés), donc D5 y gagne peu ; au profil (+18 %), c'est le calcul qui domine.
- Coût en qualité : nul en `lo` (l'int4 des experts domine l'erreur), léger en `hobbit`.
- Prefill : TTFT 0,74 s avec D5 contre 0,75 s sans (même benchmark) ; le 1,03 s de la
  première mesure faisait partie de la même passe aberrante.
- Conclusion : **optionnel sur OLMoE** (la partie dense fp16 tient en VRAM), **nécessaire
  sur Qwen3-Next** (4 Go de partie dense en fp16 dans 6 Go de VRAM).

## Les pistes du 02/10 : ce que dit la littérature récente

Recherche du 02/10 (papiers 2025-2026 sur l'offloading MoE et la quantification) :
beaucoup d'idées récentes sont déjà dans moelab (prefetch par le routeur de la couche
suivante, précision mixte HOBBIT, int8 en haute précision, CUDA graphs). Cinq pistes
restaient ; toutes sont codées, testées sur CPU, et désactivables. Aucune n'est encore
mesurée sur la RTX 2060.

### Q1 : une meilleure échelle int4, sans calibration (qualité)

Jusqu'ici, l'échelle d'un groupe valait `max|w| / 7` : la valeur -8 ne servait jamais,
et un seul poids extrême fixait le pas de tout le groupe. `quantize(..., method="mse")`
essaie 32 échelles par groupe (max|w|/7 et « max signé / -8 », comme le Q4_0 de
llama.cpp, chacune multipliée par 1,00 ... 0,70) et garde celle de plus petite erreur.
Jamais pire, groupe par groupe (le candidat d'origine en fait partie). Même format,
mêmes noyaux, même vitesse. Sur des poids synthétiques à queue lourde : erreur des poids
-30 %, erreur de sortie -30 %.

- Nouveaux stores : `mse` par défaut pour l'int4 (`build_store.py --int4-scale absmax`
  pour l'ancien comportement). L'int8 garde `absmax` (déjà presque sans perte).
- Store existant : `requant_store.py --method mse` réécrit l'int4 en place (une couche de
  place libre suffit), garde la haute précision et l'historique d'usage.

### Q2 : GPTQ sur les experts (qualité)

GPTQ (Frantar et al., 2023) quantifie les colonnes une à une et reporte l'erreur de chacune
sur les suivantes, d'après les vraies entrées de la couche : ce qui compte, c'est l'erreur
sur la sortie W·x. Pour un MoE, chaque expert a ses propres entrées : `requant_store.py
--method gptq` fait passer un texte de calibration, enregistre les entrées de chaque
couche MoE (fichiers numpy sur le disque) et construit pour chaque expert la Hessienne
H = somme des (w_t x_t)(w_t x_t)ᵀ sur les tokens qui l'ont choisi, pondérée par le
poids du routeur (la sortie de l'expert est multipliée par w_t, son erreur aussi). Même
format de sortie que Q1. Synthétique : erreur de sortie -49 % contre `absmax`.

Texte de calibration : par défaut, écrit par le modèle lui-même à partir de 16 débuts
variés (auto-calibration, Williams et al. 2025), jamais tiré de `bench/` (le texte
d'évaluation de la qualité) ; ou `--calib-file`.

### S3 : éviction least-stale et prefetch selon le score (vitesse, hobbit)

D'après SpecMD (Apple, ICML 2026), mesuré justement sur OLMoE :

- `cache_eviction="stale"` : les experts utilisés ou préchargés pendant la passe en
  cours ne sont évincés qu'en dernier (le modèle parcourt toujours ses couches dans le
  même ordre : ils vont encore servir). SpecMD : jusqu'à 85 fois moins de « collisions »
  (un expert préchargé évincé avant d'avoir servi) qu'en LRU.
- `prefetch_mass=0.9` : précharger les experts prédits jusqu'à 90 % de la probabilité
  du routeur (au plus 2k), au lieu des k meilleurs. SpecMD : meilleur que le top-k fixe,
  même avec une précision de prédiction plus faible.

### A5 : le token entier en un graph, décisions sur le GPU (vitesse, lo)

Dans `lo`, le reste est surtout la lecture du routage par le CPU (0,3 ms par couche,
~4,8 ms sur 8,4 par token). Quand tous les experts demandés sont en VRAM, les décisions
se réduisent à la précision voulue et à l'adresse de l'expert : `fullgraph.py` les prend
sur le GPU (table des adresses des experts résidents, tenue à jour par le cache VRAM),
et le token entier devient UN graph. Une lecture GPU → CPU par token au lieu de 16. Si
un expert manquait, le token est refait par le chemin normal (cache KV réécrit aux mêmes
positions, compteurs de position et états de l'attention linéaire restaurés). Tenté
seulement après un token sans défaut de cache. Testé sur OLMoE et Qwen3-Next miniatures
(émulateur + interprète Triton), y compris le défaut forcé ; `check_setup.py` le vérifie
sur la carte. Option `full_graph` (`--full-graph`), désactivée par défaut jusqu'à la mesure.
SeqMoE (2026) va plus loin (tout l'offloading en graphs, prédiction par un modèle
séquentiel entraîné) : à envisager pour Qwen3-Next si A5 tient ses promesses.

### S5 : décodage spéculatif, mesurer avant de construire

Sans perte de qualité (chaque token proposé est vérifié par le modèle), mais rentable
seulement si le brouillon est souvent juste ET si vérifier K+1 tokens coûte bien moins
que K+1 pas. Sur cette machine, ce n'est pas évident : en `lo`, vérifier plusieurs tokens
relit plus d'experts (l'union des choix de chaque token) et la lecture du routage reste.
`measure_lookup.py` mesure les deux facteurs (brouillon « prompt lookup », sans
entraînement, rejoué sur les vraies réponses ; coût d'une passe de K+1 tokens) et donne
le gain attendu. On ne construit que s'il dépasse 15 %. DraftExpert (2026) obtient ×1,45
avec des experts « brouillons » entraînés : hors de portée sans entraînement.

### Écarté après lecture

Prédiction par un modèle Mamba entraîné (SeqMoE : à entraîner, moins fiable hors des
textes d'entraînement) ; experts en 2 bits ou sautés (DyMoE : coûte de la qualité) ;
rotations de Hadamard (peu utiles avec des groupes de 64, d'après « The Great
Inversion », 2026) ; prefetch par le résiduel (DALI : c'est déjà ce que fait le prefetch).

### À mesurer sur la RTX 2060

```powershell
python -m pytest -q
python scripts\check_setup.py --paths C:\ D:\
python scripts\profile_moe.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --ram-gb 5 --dense-int8 --variants defaut,a5,s3,tout
python scripts\requant_store.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --ram-gb 5 --method mse
python scripts\benchmark.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --ram-gb 5 --dense-int8 --configs lo,hobbit@0.8 --ref results\ref_fp16.pt
python scripts\requant_store.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --ram-gb 5 --method gptq --work-dir D:\moe-calib
python scripts\benchmark.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --ram-gb 5 --dense-int8 --configs lo,hobbit@0.8 --ref results\ref_fp16.pt
python scripts\measure_lookup.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --ram-gb 5 --dense-int8 --policy lo
```

Points de comparaison (02/10, store int8 + int4 « absmax », D5) : `lo` KL 0,070,
top-1 89,1 %, perplexité +10,2 % ; `hobbit@0.8` KL 0,0089 ; profil `lo` 118,9 tok/s,
`hobbit@0.8` 46,7 tok/s.

## Les mesures du 02/10 (3) : les pistes de la littérature

Fichiers : `results/profile_olmoe-hf_2026-10-02_1104.md`, `benchmark_olmoe-hf_2026-10-02_1115.md`
(après Q1), `_1219.md` (après Q2), `lookup_olmoe-hf_2026-10-02_1221.json`.

**Q1 et Q2 : le plus gros gain de qualité du projet, sans rien coûter en vitesse.**
Qualité contre le vrai fp16, store int8 + int4, D5, mêmes 256 tokens :

| int4 du store | `lo` KL | `lo` accord top-1 | `lo` perplexité | `hobbit@0.8` KL | `lo` tok/s (bench.) |
|---|---|---|---|---|---|
| absmax (avant) | 0,070 | 89,1 % | +10,2 % | 0,0089 | 109,1 |
| mse (Q1, 6,9 min) | 0,050 | 93,4 % | +5,0 % | 0,0065 | 112,7 |
| gptq (Q2, 60 min) | **0,049** | **93,4 %** | **+3,4 %** | **0,0061** | 114,6 |

- Q1 seul fait presque tout : KL de `lo` -29 %, perplexité divisée par 2, accord top-1
  +4,3 points. `hobbit` gagne aussi (KL -27 %) : ses experts peu importants sont en int4.
- Q2 ajoute peu par-dessus (perplexité +5,0 → +3,4 %, KL -2 % à -6 %), pour une heure de
  calcul. Gardé sur le store d'OLMoE puisqu'il est fait ; pour Qwen3-Next, Q1 d'abord.
- Le +1,10 % puis -0,87 % de perplexité de `hobbit` sont du bruit (256 tokens) ; le KL,
  lui, baisse régulièrement.

**A5 : un bug, corrigé.** 0 % des tokens en un graph, 48 refaits sur 48 en `lo`. Le
compteur de défauts n'était jamais remis à zéro entre deux replays : un seul vrai
défaut (un expert pas encore chargé) faisait échouer tous les tokens suivants, chacun
calculé deux fois (8,5 → 15,8 ms par token). Remise à zéro maintenant faite dans le
graph ; le test force un défaut puis vérifie que le chemin A5 reprend. À remesurer.

**S3, prefetch selon le score : rejeté sur OLMoE.** Avec `prefetch_mass=0.9`, `hobbit`
tombe de 49,8 à 20,7 tok/s : 49,8 experts préchargés par token au lieu de 9,6, dont la
moitié pour rien (précision 49 %), 286 Mo de transferts par token au lieu de 89. Le
routeur d'OLMoE est trop « plat » : ses 8 meilleurs experts sur 64 ne font pas 90 % de
la probabilité, et la masse en réclame jusqu'à 16. L'éviction least-stale n'a pas été
mesurée seule (variante `stale`, à lancer) ; en `lo`, rien n'est évincé, elle est neutre.

**S5, décodage spéculatif : rejeté.** Le brouillon « prompt lookup » n'est juste que
pour 10 à 12 % des tokens (1,11 à 1,13 token par vérification) : même une vérification
gratuite ne donnerait que +13 %. Et elle coûte ici 35 à 47 pas (passe prefill sans
graphs, union des experts). Rien à construire.

**Point ouvert : `hobbit` au benchmark.** 18,8 tok/s avec D5 et la mesure de qualité,
contre 26,1 sans D5 (01/10, budget VRAM 2,98 Gio). Hypothèse : avec D5, le budget VRAM
automatique monte à 3,38 Gio (la partie dense libère 0,33 Gio), et les graphs capturés
par la mesure de qualité (paliers 128 et 256) s'ajoutent ensuite : la carte de 6 Go
serait à la limite, et Windows déplace de la mémoire. À vérifier avec `--vram-gb 3.0`.

## Les mesures du 02/10 (4) : A5 corrigé, least-stale, budget VRAM

Fichiers : `results/profile_olmoe-hf_2026-10-02_1242.md`, `benchmark_olmoe-hf_2026-10-02_1255.md`
(store int8 + int4 GPTQ, D5).

| Profil (1 prompt) | `lo` tok/s | `hobbit@0.8` tok/s |
|---|---|---|
| défaut avant A5 | 117,5 | 51,5 |
| **A5** (token entier en un graph) | **132,1** (100 % des tokens, 0 refait) | 45,7 (aucun token tenté, 1 refait) |
| éviction least-stale | 118,4 | 43,1 |

- **A5 : +12 % en `lo`, activé par défaut** (`--no-full-graph` pour le couper). Les 16
  lectures du routage par token disparaissent, et le chemin `sync` donne le même temps
  que `async` (7,6 ms) : plus aucune attente du CPU dans le token. En `hobbit`, aucun
  token n'a été sans défaut de cache, donc A5 n'est presque jamais tenté ; l'écart
  `async` (45,7 contre 51,5) vient de la capture du graph et du seul token refait,
  comptés dans les 48 tokens chronométrés (en `sync`, 40,9 contre 40,7 : identique).
- **Least-stale : rejeté.** `hobbit` 51,5 → 43,1 tok/s : 12,4 prefetch par token au lieu
  de 9,5, 99 Mo de transferts au lieu de 88. Protéger les experts de la passe en cours
  évince à la place des experts chauds des passes précédentes, qu'il faut recharger.
  L'idée suppose des experts peu réutilisés d'un token à l'autre ; ceux d'OLMoE le sont
  beaucoup (LRFU les garde).
- **Budget VRAM : l'hypothèse « VRAM trop pleine » est fausse.** Avec `--vram-gb 3.0`
  (au lieu de 3,38 automatique), `hobbit` descend à 16,1 tok/s (hit VRAM 79 %), `lo` à
  102,4 : moins de place = moins de hits, c'est tout. Le budget automatique est le bon.
- `hobbit` au benchmark (~19 tok/s avec D5 et la mesure de qualité) : expliqué et corrigé
  dans la section suivante (ordre des réglages, pool de RAM épinglée).

## Les mesures du 02/10 (5) : le mystère de `hobbit` au benchmark

Fichiers : `results/benchmark_olmoe-hf_2026-10-02_1424.md`, `_1426.md`, `_1438.md`
(`hobbit@0.8` seul, store int8 + int4 GPTQ).

| `hobbit@0.8` seul dans le benchmark | tok/s | KL contre fp16 |
|---|---|---|
| D5 + mesure de qualité | **28,9** | 0,0058 |
| D5, sans mesure de qualité | 29,0 | — |
| sans D5, avec mesure de qualité | 24,8 | 0,0069 |
| *rappel : D5 + qualité, lancé APRÈS `lo` (12h55)* | *16,1 - 18,9* | |

- Ni D5 ni la mesure de qualité n'étaient en cause : **D5 fait gagner 16 % à `hobbit`**
  (24,8 → 28,9), avec une qualité au moins aussi bonne. **D5 est donc activé par défaut**
  dans les scripts (`--no-dense-int8` pour le couper).
- Le coupable : **l'ordre des réglages**. `hobbit` ralentissait seulement quand il passait
  après `lo` dans le même processus. Le pool de RAM épinglée (`hostpool.py`) est fait de
  plaques, chacune découpée pour UNE précision, jamais rendue : `lo` remplissait le
  plafond de plaques int4 ; ensuite, `hobbit` n'avait plus de place pour ses records
  int8, et recevait des tampons ordinaires (non épinglés), lents à copier vers le GPU.
- Correction : une plaque entièrement libre d'une précision est redécoupée pour une autre
  quand le plafond est atteint (test : pool rempli d'int4, puis 20 records int8 servis
  sans nouvelle allocation). Les benchmarks à plusieurs réglages comparent maintenant
  `hobbit` dans les mêmes conditions que seul.
- Conséquence sur les chiffres : les `hobbit` des benchmarks du 02/10 lancés après `lo`
  (17,5 à 18,9 tok/s) sous-estimaient la vitesse ; la bonne valeur est ~29 tok/s.
- **Vérifié** (`results/benchmark_olmoe-hf_2026-10-02_1500.md`, `lo` puis `hobbit` dans le
  même processus, réglages par défaut) : `lo` 117,2 tok/s, KL 0,0495, top-1 93,8 % ;
  `hobbit@0.8` **30,0 tok/s**, KL 0,0062, top-1 98,4 %.

## Bilan final d'OLMoE (02/10)

| Réglage (défauts : store int8 + int4 GPTQ, D5, A5) | profil (1 prompt) | benchmark (8 prompts) | KL contre fp16 | accord top-1 | perplexité |
|---|---|---|---|---|---|
| `lo` | 132,1 tok/s (×20) | 117,2 tok/s | 0,0495 | 93,8 % | +3,4 % |
| `hobbit@0.8` | 51,5 tok/s (×13) | 30,0 tok/s | 0,0062 | 98,4 % | dans le bruit |

Phase d'optimisation sur OLMoE terminée. Prochaine étape : Qwen3-Next-80B.

## Qwen3-Next-80B : premiers pas (02/10)

Store int4 seul (Q1, 38,2 Gio) sur le SSD C:, partie dense en int8 (D5 : 3,73 → 1,92 Gio),
placement `split` (embeddings sur CPU, tête int8 sur GPU), VRAM 1,84 Gio et RAM 4 Gio
pour les experts.

Problèmes rencontrés en chemin, tous corrigés :
- la reprise de `build_store.py` comptait deux fois la place déjà écrite ;
- `plan.py` ignorait D5 (0,75 Gio de VRAM annoncés au lieu de ~1,9) ;
- la bibliothèque safetensors plantait sous Windows (« access violation ») en lisant la
  partie dense : lecteur maison sans projection mémoire (`checkpoint.SafeFile`).

Chat (réponses correctes et nuancées en français) : 0,87 tok/s avec 0,29 Gio de RAM pour
le cache (282 lectures disque par token), **1,5 à 1,9 tok/s avec `--ram-gb 4`**.

Profil (`results/profile_qwen3-next-80b-fp8_2026-10-02_1850.md`, 1 prompt, experts chauds
préchargés depuis `usage.json`) : **3,49 tok/s**, 287 ms par token, dont 262 ms dans les
48 couches MoE (5,5 ms par couche) :

| Phase, par couche | ms |
|---|---|
| attente disque / copies + calcul des manquants | 2,92 |
| lecture du routage (le CPU attend le GPU) | 1,36 |
| prefetch (temps CPU) | 1,01 |
| le reste | 0,18 |

Hit VRAM 84 %, 60,8 lectures disque par token (1,3 par couche), 226 Mo RAM → VRAM par
token. Le SSD ne lit que 0,33 Go/s avec une seule lecture en vol (~5 ms pour un expert) :
chaque couche qui attend un expert lu à la demande perd plusieurs millisecondes. D'où la
première optimisation propre à Qwen3-Next :

- **Lectures disque à l'avance** (`--prefetch-disk-ahead N`) : les experts prédits des
  couches L+2 ... L+1+N sont lus du disque vers la RAM pendant que la couche L calcule
  (jamais copiés en VRAM, pour ne pas reproduire la pollution de C2). La prédiction est
  calculée dans le graph, avec celle de L+1 (même lecture GPU → CPU). Testé : même
  génération (Qwen3-Next miniature, avec et sans graphs).

**Mesuré le 03/10 (`results/profile_qwen3-next-80b-fp8_2026-10-03_1413.md`) : rejeté.**

| Variante | tok/s | lectures disque / token | attente disque / copies (ms/couche) |
|---|---|---|---|
| défaut | 3,35 | 60,9 | 3,06 |
| `ahead1` | 3,01 | 71,4 (dont 48 à l'avance) | 3,05 |
| `ahead2` | 2,83 | 83,2 (dont 95 à l'avance) | 3,04 |

L'attente ne baisse pas du tout. Les 76,6 copies à la demande par token correspondent
exactement aux 16 % d'experts que le prefetch ne prévoit pas (480 × 0,16 = 77) : ce sont
eux qu'on attend, et une prédiction à 2 couches ne les prévoit pas mieux. Les lectures à
l'avance ne font qu'ajouter des lectures inutiles. L'option reste, désactivée.

**Ce que l'attente contient vraiment : le tri du cache.** À chaque insertion qui doit
évincer, `TierCache` triait toutes ses entrées pour trouver la victime : 0,26 ms avec les
~1 200 entrées de la VRAM, 0,54 ms avec les ~2 500 de la RAM (mesuré sur un CPU de
serveur). Avec ~4 insertions en VRAM et ~1,3 en RAM par couche, ça fait ~1,5 à 2 ms par
couche, soit une bonne part des 3 ms d'« attente ». Sur OLMoE, avec 4 fois moins
d'entrées, ça passait inaperçu. Correction : l'ordre des scores LRFU ne dépend pas de
l'instant de la comparaison (log score = log f - t log gamma + constante), donc un tas avec
invalidation paresseuse donne la même victime en O(log n) : 0,01 ms par insertion au
lieu de 0,26 à 0,54. Test : 3 000 opérations aléatoires (insertions, accès, épinglages),
mêmes victimes que le tri complet.

**Mesuré (`results/profile_qwen3-next-80b-fp8_2026-10-03_1430.md`) : ×2.**

| Par couche (ms) | tri complet | tas |
|---|---|---|
| attente disque / copies | 3,06 | 1,78 |
| lecture du routage | 1,38 | 0,56 |
| prefetch | 1,06 | 0,28 |
| **tok/s** | **3,35** | **6,82** |

Le temps « dense » passe aussi de 25 à 10,6 ms par token : il contenait le rangement en
RAM des lectures terminées, qui triait lui aussi. Même nombre de lectures disque (60,7 par
token) et même taux de hit (84 %) : l'éviction choisit les mêmes victimes, seulement plus vite.

**Reste par couche : 2,84 ms**, dont 1,78 d'attente des experts que le prefetch ne
prévoit pas (~1,3 lecture disque par couche). Un expert attendu seul est lu par UNE
lecture, au débit d'une lecture en vol (0,33 Go/s sur ce SSD, contre 1,15 Go/s avec
quatre). D'où la piste suivante, codée : **lire chaque expert en N morceaux simultanés**
(`read_parts`, mesuré par `measure_bandwidth.py`, variantes `split1` / `split4`). Même
résultat octet pour octet (test).

**Mesuré (`results/profile_qwen3-next-80b-fp8_2026-10-03_1445.md`) : +59 %.**

| | 1 morceau | 4 morceaux |
|---|---|---|
| attente disque / copies (ms/couche) | 1,82 | **0,65** |
| ms par couche MoE | 2,83 | 1,64 |
| **tok/s** | 6,82 | **10,85** |

Activé par défaut quand le disque gagne au moins 20 % à avoir plusieurs lectures en vol
(`measure_bandwidth.py`, d'après la courbe des lectures en parallèle, plafonné à 4). La
mesure directe d'un expert lu en k morceaux (32 lectures) s'est montrée trop bruitée pour
décider (1 → 0,70, 2 → 1,05, 4 → 0,41 Go/s) : elle reste affichée, à titre indicatif.

En chat, questions variées (03/10, 15 h) : **6,2 et 6,6 tok/s** (hit VRAM 69-72 %, 47 à 115
lectures disque par token selon le sujet), contre 1,5-1,9 tok/s la veille.

Qwen3-Next en `lo` sur la RTX 2060, profil : **3,35 → 10,85 tok/s** dans la journée
(éviction en tas ×2, lectures découpées +59 %). Reste par couche 1,64 ms : lecture du
routage 0,40, attente des experts manquants 0,65, prefetch 0,32.

### Budgets RAM et VRAM (03/10, 15 h 10 - 15 h 20)

Benchmark `lo`, 8 prompts de 64 tokens, `results/benchmark_qwen3-next-80b-fp8_2026-10-03_15*.md` :

| RAM experts | réserve VRAM | VRAM experts | hit VRAM | disque/token | **tok/s** | TTFT |
|---|---|---|---|---|---|---|
| 4 Gio | 1,0 Gio | 1,84 Gio | 56 % | 140,0 | 6,03 | 6,9 s |
| 5 Gio | 1,0 Gio | 1,84 Gio | 58 % | 122,6 | 5,93 | 7,3 s |
| 6 Gio | 1,0 Gio | 1,84 Gio | 58 % | 109,1 | 6,38 | 6,8 s |
| 5 Gio | **0,6 Gio** | **2,24 Gio** | **61 %** | 117,2 | **6,81** | 6,9 s |

- **Plus de RAM** : −22 % de lectures disque de 4 à 6 Gio, mais la vitesse reste dans le bruit
  (±5 %). Les lectures découpées rendent un défaut RAM peu cher ; ce qui coûte, c'est un défaut
  VRAM. 5 à 6 Gio suffisent ; au-delà de ~35 % de la RAM, Windows pagine.
- **Plus de VRAM** : +0,4 Gio de cache d'experts (250 experts) → hit 58 → 61 %, **+15 %**.
  La réserve fixe de 1 Gio était taillée pour OLMoE (cache KV de 256 Mio à 2 048 tokens) ;
  Qwen3-Next n'a d'attention complète que sur 12 couches sur 48, avec 2 têtes KV : 24 Ko par
  token, 48 Mio à 2 048 tokens.
- **Réserve automatique** (`plan.auto_vram_reserve`, défaut depuis le 03/10) : 0,55 Gio
  d'activations + le cache KV statique à la longueur max. **Longueur max automatique**
  (`plan.auto_graph_max_len`) : la plus grande de 8 192 / 4 096 / 2 048 dont le cache KV
  tient dans 256 Mio. OLMoE : 2 048 tokens, réserve 0,80 Gio (au lieu de 1,0). Qwen3-Next :
  **8 192 tokens** (conversations 4 fois plus longues en chat), réserve 0,74 Gio, soit
  ~0,26 Gio de cache d'experts de plus qu'avant. `--vram-reserve-gb` et `--graph-max-len`
  forcent toujours une valeur.
- **Vérifié avec les réglages automatiques (15 h 42)** : Qwen3-Next `lo`, RAM 5 Gio, réserve
  0,74 Gio, cache de 8 192 tokens : VRAM d'experts 2,11 Gio, hit 59 %, **6,76 tok/s** — autant
  qu'avec 0,6 Gio et 2 048 tokens (6,81), avec des conversations 4 fois plus longues. OLMoE
  (réserve 0,80 Gio, VRAM d'experts 3,58 Gio, pas de manque de mémoire) : `lo` 116,2 tok/s,
  KL 0,051 ; `hobbit` t1 = 0,6 : 29,4 tok/s, KL 0,0146 — dans le bruit des mesures d'avant.
- TTFT ~7 s partout : c'est le prefill, la prochaine cible.

## Le prefill : le temps avant le premier token (03/10)

Qwen3-Next met ~7 s avant son premier token, même pour un prompt de 64 tokens. Le prefill
calcule tous les tokens du prompt d'un coup : chaque couche demande alors 10 experts par
token, soit, pour 64 tokens, plusieurs centaines d'experts distincts sur 512. Avec ~6 % des
experts en RAM et ~5 % en VRAM, l'essentiel vient du disque : à 1,59 Mio l'expert et 48
couches, de l'ordre de 15 à 20 Go à lire. Hypothèse : le prefill est limité par le disque.
À vérifier avant d'optimiser : `scripts/profile_prefill.py` donne, par variante, le temps
du prefill, sa part MoE / dense / attente disque, les experts par couche (VRAM, RAM,
disque), le débit disque obtenu comparé au débit mesuré, et le détail de chaque phase.

En lisant le code, une découverte plus grave : **`chat.py` recalculait toute la
conversation à chaque message** (le cache était remis à zéro à chaque prefill). Au
10e message, le prefill relit donc tout l'historique, et sur Qwen3-Next presque tout le
store.

| Piste | Quoi | Exact ? | Défaut |
|---|---|---|---|
| P0 | `protect` copié en `frozenset` à chaque expert inséré en VRAM : O(n²), quelques centaines d'experts par couche au prefill | oui | corrigé |
| P1 | `kv_reuse` : le cache garde la conversation ; seuls les tokens nouveaux sont calculés. Attention seule (OLMoE) : on coupe au premier token qui diffère. Attention linéaire (Qwen3-Next, dont l'état ne se rembobine pas) : suite exacte si l'historique n'a pas changé, sinon retour à la fin du prompt précédent (états sauvegardés sur CPU, ~70 Mo pour Qwen3-Next) | oui (testé : mêmes logits qu'en recalculant tout) | **activé** |
| P2 | `prefill_order completed` : les experts manquants calculés dans l'ordre où le disque les rend, au lieu d'attendre d'abord le plus demandé | oui | à mesurer |
| P3 | `prefill_prefetch` : pendant le calcul de la couche L, lecture disque -> RAM des experts prédits de L+1 (routeur de L+1 sur l'état de L, tous les tokens) | oui | à mesurer |
| P4 | `prefill_cache ram` : les experts lus pour le prompt ne chassent pas ceux de la génération du cache VRAM | oui | à mesurer |
| P5 | `prefill_skip` : un expert à lire sur le disque dont le plus grand poids du routeur reste sous le seuil est sauté (moins d'octets lus) | non : qualité mesurée (KL, top-1 des tokens qui suivent) | à mesurer, avec ton accord |

### Mesures du 03/10, 16 h (`results/prefill_qwen3-next-80b-fp8_2026-10-03_1613.md`)

Prompts de 17 à 22 tokens (66 à 77 experts distincts par couche) et de 64 tokens (153).

| Variante | court 1 (17 t.) | long (64 t.) | tok/s ensuite (court 1 / long) | KL (long) |
|---|---|---|---|---|
| defaut | 5,83 s | 11,42 s | 8,95 / 5,93 | référence |
| P2 ordre d'arrivée | 5,88 | 11,63 | 8,88 / 5,47 | 0,0016 |
| P3 prefetch disque | 6,25 | 13,64 | 8,26 / 5,39 | 0,0030 |
| P4 hors cache VRAM | 5,82 | 11,80 | 8,38 / 4,88 | 0,0024 |
| P5 seuil 0,05 | 5,52 | 11,46 | 8,43 / 5,25 | 0,0036 |
| P5 seuil 0,10 | **3,42** | **7,29** | 7,90 / 4,89 | **0,0126** |

**P1 en conversation** (3 messages) : 1er token du 3e message **14,0 → 6,0 s** (104 des 129
tokens repris du cache). Le gain grandit avec la conversation.

Ce que disent les chiffres :

- **Le disque ne fait pas attendre** : 0,1 s d'attente sur 5,8 s. Le prefill lit 4,5 Go à
  0,77 Go/s, pour 1,14 Go/s mesurés au mieux : le plancher disque serait ~4,0 s.
- **C'est le fil principal qui traîne** : par couche, 40 ms pour calculer ~8 experts déjà en
  VRAM (5 ms par expert, pour ~30 petites opérations GPU) et 87 ms pour les ~56 lus sur le
  disque. Cette phase 4 tombe à 17-22 ms quand moins de lectures tournent en même temps (P3,
  P5 à 0,10), et la partie dense passe de 0,9 à 2,2 s quand P3 fait tourner des lectures
  pendant qu'elle calcule. Les fils de lecture (16, plus 16 pour les lectures découpées) se
  disputent le GIL de Python avec le fil qui lance les noyaux.
- P2 : rien (le disque n'est pas l'attente). P3 : pire (plus de concurrence). P4 : prefill
  égal, génération qui suit plus lente (les experts du prompt servent à la réponse). Rejetés.
- P5 : 0,10 coupe 40 % du prefill, mais KL 0,0126 sur le prompt long, environ 4 fois le
  bruit des variantes exactes (0,002-0,003 : sommes dans un autre ordre, quasi-égalités du
  top-1 sur 24 tokens). À trancher avec Victor.
- Bruit : les variantes exactes donnent KL 0,0001 sur les prompts courts, 0,002-0,003 sur le
  long. En dessous, une différence ne veut rien dire.

### 2e série : alléger le fil principal (P6 à P8)

| Piste | Quoi | Exact ? |
|---|---|---|
| P6 | `prefill_grouped 64` : prompt court, les noyaux groupés de la génération calculés token par token (2 lancements par token et par couche au lieu d'~30 par expert) | oui (testé avec un équivalent PyTorch des noyaux) |
| P7 | `prefill_split off` : au prefill, un expert = une lecture (16 en vol suffisent ; 4 fois moins de tâches Python) | oui |
| P8 | `gil_switch_ms 0.5` : le GIL change de main toutes les 0,5 ms au lieu de 5 | oui |

**Mesuré le 05/10 (`results/prefill_qwen3-next-80b-fp8_2026-10-05_0925.md`)** :

| Variante | court 1 (17 t.) | long (64 t.) | débit disque | tok/s ensuite | KL (long) |
|---|---|---|---|---|---|
| defaut (sans P6) | 5,95 s | 11,66 s | 0,76 Go/s | 8,13 | référence |
| **P6** | **3,76 s** | **7,90 s** | **1,20 Go/s** | 8,88 | 0,0023 (bruit) |
| P7 | 6,31 | 12,83 | 0,72 | 8,54 | 0,0000 |
| P8 | 12,20 | 22,68 | 0,37 | 4,02 | 0,0020 |
| P6+P7+P8 | 4,79 | 9,25 | 0,94 | 4,45 | 0,0039 |
| P6+P7+P8 + P5 0,05 | 4,37 | 8,54 | 0,88 | 4,69 | 0,0046 |
| P6+P7+P8 + P5 0,10 | 2,48 | 6,45 | 0,30 | 3,94 | 0,0122 |

- **P6 activé par défaut** (prompts jusqu'à 64 tokens) : **−37 %** sur le premier token. La
  phase 4 (experts déjà en VRAM) passe de 38 ms à 0,07 ms par couche, et le disque tourne
  enfin à son débit maximal (1,14 Go/s mesuré) : **le prefill est désormais limité par le
  disque**. Plancher pour le prompt de 17 tokens : 4,5 Go / 1,14 Go/s ≈ 4 s, atteint.
- **Vérifié au benchmark** (`benchmark_qwen3-next-80b-fp8_2026-10-05_0940.md`, 8 prompts) :
  TTFT **6,71 → 4,70 s**, génération inchangée (6,76 → 6,80 tok/s).
- **P7 rejeté** : même au prefill, le disque veut les lectures découpées.
- **P8 rejeté** : un GIL qui bascule 10 fois plus souvent multiplie les bascules entre les
  32 fils au lieu de rendre la main plus vite. Prefill ×2, génération ÷2 (le réglage
  restait actif pendant les tokens qui suivent, d'où les tok/s des lignes P8).
- Il ne reste qu'un moyen d'aller plus vite : **lire moins** (P5). Avec P6, P5 à 0,05
  gagnerait ~10 % (KL dans le bruit), à 0,10 ~35 % (KL ×4). Décision de Victor.
- **Seuil de P6 mesuré le 05/10, 11 h 41** (`prefill_qwen3-next-80b-fp8_2026-10-05_1141.md`) :

  | Prompt | sans P6 | P6 | |
  |---|---|---|---|
  | 17 tokens | 4,01 s | 3,67 | P6 gagne |
  | 93 tokens | 14,33 s | **10,51** | −27 % |
  | 407 tokens | 22,53 s | 27,88 | P6 perd : chaque token relit ses 10 experts |

  Bascule estimée vers 220 tokens : **seuil 192 par défaut**. Pour les longs prompts, le
  vrai remède serait un noyau qui calcule chaque expert pour tous ses tokens d'un coup.
- (avant le 05/10, 11 h 41) Au-delà de 64 tokens, le chemin expert par expert reprend. Avec P1, les messages suivants
  d'une conversation ne calculent que le nouveau message, souvent sous 64 tokens ; la
  variante `p6_256` de `profile_prefill.py` mesurera un seuil plus haut sur un long prompt.

`chat.py` affiche désormais le temps avant le premier token et le nombre de tokens repris
du cache, et des tokens/s **de génération seule** : les 6,2-6,6 tok/s mesurés en chat le
03/10 comptaient aussi le prefill (ils sont donc sous-estimés).

## Suivi

Chaque optimisation est une option, avec son test d'exactitude ; elle n'est activée
par défaut qu'après une mesure sur la RTX 2060.

| # | Option | Statut | Test | Mesure |
|---|---|---|---|---|
| A1 | `profile_moe.py` | fait | chronomètres par phase | à lancer |
| A2 | activé par défaut | fait | génération identique à transformers | GPU : +6 % en lo, +27 % en hobbit |
| — | prefetch adaptatif, activé par défaut | fait | génération identique | lo : prefetch 0,45 → 0,007 ms/couche ; avec A2 : +22 % (lo), +35 % (hobbit) |
| A3 | cache VRAM en plaques | reporté | — | utile seulement pour A6 (CUDA graphs) : l'allocateur de PyTorch réutilise déjà ses blocs |
| A4+B1 | activé par défaut (`--no-grouped`) | fait | noyaux exécutés par l'interprète de Triton : mêmes logits en hi, lo et hobbit | **lo : 12,4 → 20,2 tok/s**, hobbit : 5,6 → 6,2 ; table d'adresses envoyée sans synchro depuis le 01/10 |
| A5 | couche sans synchro | en attente | — | la lecture du routage coûte 0,16-0,19 ms par couche (synchro comprise) ; après A6 |
| A6 | activé par défaut (`--no-cuda-graphs`) | fait | émulateur de graphs sur CPU : génération identique (OLMoE, Qwen3-Next) ; `check_setup.py` sur GPU | **lo : 20,9 → 41,9 tok/s**, hobbit : 6,8 → 8,0 ; partie dense 34 → 1,5 ms/token |
| B4 | noyaux groupés v2 / v3 / v4 (`grouped_kernels`, défaut 0 = choix par mesure) | fait | interprète de Triton : v1 à v4 = référence (chemins masqués et non masqués) | v2 : lo 41,8 → 48,4 tok/s ; v2/v3/v4 mesurés à ~70 Go/s mais la mesure était limitée par le CPU : remesure en graph |
| A8 | paliers de longueur du cache statique (`graph_buckets`) | fait | émulateur : passage de palier, logits identiques (OLMoE, Qwen3-Next) | **lo : 89,4 → 105,9 tok/s** |
| A9 | routage + prédiction du prefetch en une lecture (`graph_pack_routing`) | fait | émulateur : mêmes tokens, routeur jamais appelé en Python pendant la génération | **hobbit int8 : 35,1 → 39,4 tok/s** (benchmark 22,6 → 26,2) |
| D5 | partie dense en int8 (défaut depuis le 02/10, `--no-dense-int8`) | fait | QLinear ≈ Linear ; interprète : GEMV int8 = référence ; graphs émulés = passe normale | **lo 104,7 → 118,9 tok/s, hobbit 39,4 → 46,7 (profil)** ; KL hobbit 0,0079 → 0,0089 |
| E4 | int8 comme haute précision (store `--hi int8`) + noyaux groupés int8 | fait | interprète : int8 = référence ; store int8 + int4 en hobbit, graph = chemin normal | **hobbit : 5,6 → 22,6 tok/s (benchmark), 9,0 → 36,2 (profil)** ; noyau int8 221 Go/s |
| A6 bis | calcul groupé en graph (`moe_graphs`, avec les CUDA graphs) | fait | interprète : même génération en hi, lo, hobbit, avec transferts et branche CPU ; `check_setup.py` sur GPU | **lo : 47,0 → 89,6 tok/s**, hobbit : 7,9 → 8,6 |
| A7 | `torch.compile` de la partie dense | en attente | — | à envisager si A6 laisse du temps GPU dans la partie dense (fusion des normes) |
| B1 | activé par défaut si Triton est installé | fait | noyau exécuté par l'interprète de Triton : mêmes logits que PyTorch | **lo : 6,7 → 11,8 tok/s** ; calcul des experts 5,9 → 2,5 ms/couche |
| B2 | `--int4-unpacked` | rejeté | poids identiques bit à bit, mêmes logits | hit VRAM 100 % → 93 %, plus lent que B1 |
| B3 | branche CPU int4 | non retenu | — | sur ta machine, q* n'envoie jamais rien au CPU (B_Host < B_PCIe) : gain nul |
| C1 | copies groupées | non retenu | — | les copies sont déjà asynchrones sur un flux dédié ; à revoir si le profil montre le contraire |
| C2 | `--prefetch-depth 2` | rejeté | génération identique | hit VRAM 78 → 72 %, hobbit 6,8 → 5,0 tok/s |
| C3 | prefetch par précision | déjà le cas | — | le prefetch applique déjà HOBBIT aux poids prédits |
| D1 | activé par défaut (`--no-direct-io`) | fait | mêmes octets, repli si non aligné | hobbit : 6,2 → 6,8 tok/s (+9 %) |
| D2 | `--io-workers` (mesuré) | fait | mesure en test | `measure_bandwidth.py` mesure 1 à 16 lectures en parallèle et retient la meilleure |
| D3 | lectures triées | fait, activé | génération identique | ordre du fichier dans chaque couche |
| D4 | `--hot-store` + `build_hot_store.py` | fait | mêmes octets lus | experts les plus utilisés copiés sur le NVMe |
| E1 | `--t1-layers auto` | fait | décisions identiques en Python et en tenseurs | `measure_routing.py` calibre un t1 par couche (90 % de la vraie contribution en fp16) |
| E2 | `--t-mid` | fait | le niveau intermédiaire lit bien l'int8 | store à trois précisions (`build_store --hi fp16 --lo int4` + int8) |
| E3 | `--miss-t1` | rejeté | déclassements comptés | +6 % de vitesse pour une qualité nettement moins bonne (KL ×2,4, top-1 −3 points) |
| Q1 | int4 « mse » (`build_store --int4-scale`, `requant_store --method mse`) | fait, défaut des nouveaux stores | jamais pire que absmax groupe par groupe ; -8 utilisé ; réécriture en place = quantize | **`lo` KL 0,070 → 0,050, perplexité +10,2 → +5,0 %, top-1 89,1 → 93,4 % ; `hobbit` KL 0,0089 → 0,0065** ; vitesse inchangée |
| Q2 | GPTQ par expert (`requant_store --method gptq`) | fait, appliqué au store d'OLMoE | erreur de sortie réduite ; pipeline complet sur OLMoE miniature : logits `lo` plus proches de `hi` | par-dessus Q1 : `lo` KL 0,049, perplexité +3,4 % ; `hobbit` KL 0,0061 ; 60 min de calcul |
| S3 | `--cache-eviction stale`, `--prefetch-mass 0.9` | **rejeté** | génération identique ; masse faible = moins d'experts prédits, forte = plus | masse 0,9 : `hobbit` 49,8 → 20,7 tok/s (routeur d'OLMoE trop plat) ; stale : `hobbit` 51,5 → 43,1 |
| A5 | activé par défaut (`--no-full-graph`, `fullgraph.py`) | fait | émulateur + interprète : mêmes logits qu'en segments (lo, hobbit, upgrade), défaut forcé refait puis reprise, Qwen3-Next (états restaurés) ; `check_setup.py` sur GPU | **lo : 117,5 → 132,1 tok/s**, 100 % des tokens en un graph ; hobbit neutre (1re mesure faussée par un bug corrigé) |
| S5 | `measure_lookup.py` | **rejeté** | brouillon et simulation testés | brouillon juste pour 10-12 % des tokens : +13 % au mieux, vérification à 35 pas |
| M1 | réserve VRAM et `--graph-max-len` automatiques (`plan.py`) | fait, activé | tailles de cache KV vérifiées (OLMoE, Qwen3-Next, avec et sans `layer_types`) | Qwen3-Next `lo` : réserve 1,0 → 0,6 Gio, **5,93 → 6,81 tok/s** ; 8 192 tokens de conversation |
| P0-P8 | prefill (`profile_prefill.py`, `kv_reuse`, `prefill_*`) | P0, P1, P6 faits et activés (P1 : 3e message 14,0 → 6,0 s ; P6 : 5,95 → 3,76 s) ; P2-P4, P7, P8 rejetés ; P5 à trancher | P1 : mêmes logits qu'en recalculant tout (OLMoE, Qwen3-Next, historique modifié) ; P2-P4 mêmes logits ; P5 compté | à mesurer |

**Ce qu'a montré le modèle miniature sur CPU** (indicatif, le GPU dira le reste) :
une fois les décisions allégées, le calcul des experts domine, et en `lo` la
déquantification int4 coûte 2,6 fois plus que le calcul en `hobbit`. D'où B1 et B2,
deux réponses au même problème, à départager par la mesure.

## Mesurer

```powershell
python scripts\measure_bandwidth.py --store C:\moe-store\olmoe
python scripts\profile_moe.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe --ram-gb 5
```

`measure_bandwidth.py` mesure aussi le nombre idéal de lectures en parallèle (D2) et
vérifie que la lecture sans cache (D1) est possible. `profile_moe.py` compare par défaut,
pour `lo` et `hobbit@0.8`, les variantes `v30` (défaut du 30/09), `defaut` (actuel :
+ noyaux groupés + D1) et `cg` (+ CUDA graphs) ; `--policies` et `--variants` en
ajoutent d'autres (`base`, `a2`, `b1`, `b2`, `c2`, `grp`, `hi`...). Il donne aussi les
planchers physiques, le détail de chaque phase d'une couche en temps CPU (`async`) et
GPU compris (`sync`), le temps de la partie dense, et `torch.profiler` sur le décodage
seul. Fichiers : `results\profile_olmoe-hf_<date>.md`, `.json`, et une trace Chrome à
ouvrir sur https://ui.perfetto.dev.

## Les options de la politique (famille E)

- **E1** : `measure_routing.py` enregistre `<store>\t1_layers.json`, puis
  `--t1-layers auto` (chat) ou le réglage `hobbit_cal` (benchmark) l'utilisent.
- **E2** : pour trois paliers, construire un store fp16 + int8 + int4
  (`build_store.py --precisions fp16,int8,int4`, 21,4 Gio pour OLMoE), puis `--t-mid 0.8`.
- **E3** : `--miss-t1 0.3` (chat) ou le réglage `hobbit_miss` (benchmark).
