# MoE Offload Lab

Projet d'apprentissage : construire, à partir de zéro, un système d'inférence local
qui combine **quantification mixte** et **offloading intelligent** pour faire tourner
des modèles Mixture-of-Experts (MoE) sur du matériel grand public limité (RTX 2060
6 Go, 16 Go de RAM).

Le résultat est `moelab`, un petit moteur en Python/PyTorch, construit sur
transformers, qui garde les experts hors du GPU et décide à chaque token où les
calculer (VRAM, CPU) et en quelle précision (fp16, int4). Il implémente, à petite
échelle, les idées de quatre papiers récents (HOBBIT, DyMoE, MoEpic, FreeToken) et
d'un système de référence (Colibrì).

**Objectif final** : faire tourner un modèle MoE de 70B+ paramètres (Qwen3-Next-80B)
bien au-delà de ce qui tient en VRAM ou même en RAM, quitte à sacrifier la vitesse.
La faisabilité prime sur la performance.

## État du projet

| Étape | Contenu | État |
|---|---|---|
| 1 | Baseline Ollama sur un modèle dense (llama3:8b) | fait |
| 2 | Lecture de 4 papiers (HOBBIT, DyMoE, MoEpic, FreeToken) | fait — [doc 02](docs/02-research-notes.md) |
| 3 | Choix du modèle MoE : OLMoE-1B-7B-0125 | fait — [doc 03](docs/03-model-selection.md) |
| 4 | llama.cpp compilé avec CUDA, puis pivot vers PyTorch (GGUF fusionne les experts) | fait — [doc 05](docs/05-architecture-reality-check.md) |
| 5-6 | Checkpoint HF natif, accès individuel à chaque expert validé | fait |
| 7 | Scoring d'importance HOBBIT + pattern en U des couches | fait — [doc 06](docs/06-comprendre-le-scoring-hobbit.md), [doc 07](docs/07-pattern-en-u-des-couches.md) |
| 8 | Store d'experts sur disque, int4/int8 par groupes de 64, lecture des checkpoints FP8 | code testé — [doc 10](docs/10-store-et-quantification.md) |
| 9 | Précision par expert et par token (HOBBIT) + couches protégées | code testé — [doc 12](docs/12-politiques-hobbit-et-qstar.md) |
| 10 | Politique q* : transfert GPU ou calcul CPU en parallèle (FreeToken) + mesure des bandes passantes | code testé — [doc 12](docs/12-politiques-hobbit-et-qstar.md) |
| 11 | Caches VRAM/RAM : score LRFU, épinglage, anti-oscillation, historique persistant (Colibrì) | code testé — [doc 11](docs/11-cache-lrfu-persistance.md) |
| 12 | Prefetch de la couche suivante + mesures du routage | **mesuré** : prefetch 83%, localité 40% — [doc 13](docs/13-prefetch-et-mesures-routage.md), [doc 17](docs/17-resultats-olmoe.md) |
| 13 | Benchmark A/B : fp16 vs int4 statique vs HOBBIT | **fait**, balayage de t1 compris — [doc 14](docs/14-benchmark-ab.md), [doc 17](docs/17-resultats-olmoe.md) |
| 14 | Palier intermédiaire Qwen3-30B-A3B (optionnel) | prêt, même code — [doc 15](docs/15-palier-final-qwen3-next.md) |
| 15 | **Objectif final** : Qwen3-Next-80B-A3B, experts streamés depuis le disque | prêt, testé sur une version miniature, **à lancer** — [doc 15](docs/15-palier-final-qwen3-next.md) |
| 16 | Contribution à Colibrì (issue #249) | brouillon rempli avec les mesures, **à relire puis poster** — [doc 16](docs/16-contribution-colibri.md) |

"Code testé" = 79 tests automatiques : en haute précision, le modèle déchargé
donne **les mêmes logits que transformers** et génère **les mêmes tokens**, quel
que soit l'endroit où vivent les experts, pour OLMoE et Qwen3-Next miniatures. Le
chemin CUDA est vérifié par `scripts/check_setup.py` et `tests/test_cuda.py` sur la
vraie machine.

## Démarrage rapide

Tout est dans **[GUIDE.md](GUIDE.md)** (commandes prêtes à lancer, dans l'ordre). En bref :

```powershell
python scripts\check_setup.py --paths C:\ D:\
python scripts\build_store.py --model D:\models\olmoe-hf --out C:\moe-store\olmoe
python scripts\measure_bandwidth.py --store C:\moe-store\olmoe
python scripts\chat.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe
python scripts\benchmark.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe --repeats 2
```

## Comment ça marche

Le routeur d'origine choisit toujours ses experts. Seul le module qui *contient*
les experts est remplacé : il va les chercher dans une hiérarchie VRAM → RAM →
disque, et à chaque couche :

1. **HOBBIT** — chaque expert en fp16 s'il est important pour ce token, en int4 sinon ;
2. les experts déjà en **VRAM** sont calculés sur GPU ;
3. **q\*** — parmi les manquants, q\* sont copiés en VRAM, les autres calculés sur CPU en même temps ;
4. le **disque** est lu en arrière-plan pendant que le reste calcule ;
5. le **prefetch** applique le routeur de la couche suivante à l'état courant et commence à charger ses experts.

Détails : [docs/09-architecture-moelab.md](docs/09-architecture-moelab.md).

## Structure du repo

```
moe-offload-lab/
├── README.md, GUIDE.md, CHANGELOG.md, requirements.txt
├── moelab/                 le moteur
│   ├── quant.py            int4/int8 gs64, FP8 des checkpoints
│   ├── checkpoint.py       lecture des .safetensors tenseur par tenseur
│   ├── store.py            store d'experts (un record contigu par expert)
│   ├── cache.py            caches VRAM/RAM, LRFU, épinglage, usage.json
│   ├── hostpool.py         mémoire RAM (épinglée) en slots de taille fixe
│   ├── policy.py           HOBBIT et q*
│   ├── runtime.py          exécution d'une couche MoE + prefetch
│   ├── loader.py           chargement sans les experts, placement dense
│   ├── plan.py, hw.py      budgets mémoire, bandes passantes
│   ├── bench.py            qualité, vitesse, règle A/B
│   ├── profiling.py        chronomètres par phase, planchers physiques
│   ├── kernels.py          noyaux Triton int4 et groupés (optionnels)
│   ├── graphs.py           partie dense en CUDA graphs (+ émulateur pour les tests)
│   ├── generate.py         boucle de génération (graphs ou passe normale)
│   ├── options.py          catalogue de tous les réglages (effet mesuré, à chaud ou non)
│   ├── denseq.py           partie dense en int8 (--dense-int8)
│   ├── fullgraph.py        token entier en un graph, décisions sur le GPU (--full-graph)
│   ├── requant.py          int4 « mse » et GPTQ, réécrits en place dans un store
│   ├── reap.py             score REAP des experts (élagage), passes par lots, reprise (doc 22)
│   ├── prune_data.py       jeux de calibration et d'évaluation figés sur disque
│   ├── prune_eval.py       banc d'évaluation : vitesse, appels d'outils, perplexité
│   ├── agent.py            agent local : outils, appels lus dans le texte, boucle (doc 19)
│   ├── autopilot.py        le modèle optimise son propre moteur : mesure, garde-fous, profil
│   ├── standalone.py       store autonome : partie dense copiée, checkpoint supprimable
│   ├── testing.py          modèles miniatures pour les tests
│   └── app/                l'app : file d'attente, chat, agents, outils, MCP, mise en route (docs 20, 21)
├── scripts/                check_setup, build_store, measure_bandwidth, plan,
│                           chat, measure_routing, benchmark, profile_moe, build_hot_store,
│                           requant_store, measure_lookup, profile_prefill, agent, autopilot,
│                           make_standalone, app, eval_ppl, calibrate_reap,
│                           prune_eval, reap_pipeline
│                           (+ hobbit_scoring, étape 7)
├── tests/                  198 tests (pytest)
├── bench/                  texte d'évaluation, prompts du benchmark, questions d'outils
├── notebooks/              visite guidée du moteur
├── results/                résultats de benchmark (versionnés)
└── docs/                   01 à 21, une étape par document
```

## Papiers de recherche de référence

| Papier | Date | Ce qu'on en a pris |
|---|---|---|
| [HOBBIT](https://arxiv.org/abs/2411.01433) | nov 2024 | Précision par expert selon le poids du routeur (seuils T1/T2) |
| [DyMoE](https://arxiv.org/abs/2603.19172) | mars 2026 | Sensibilité selon la profondeur → couches protégées |
| [MoEpic](https://arxiv.org/abs/2509.08342) | sept 2025 | Cache combinant récence et fréquence |
| [FreeToken](https://arxiv.org/abs/2608.16157) | août 2026 | Politique q\* et exécution CPU/GPU en parallèle |

Système de référence : [Colibrì](https://github.com/JustVugg/colibri) (C pur, jusqu'à
2,8T paramètres depuis le disque) — records contigus, int4 gs64, LFRU, anti-oscillation,
historique persistant, prefetch mesuré. Notre benchmark répond à son issue ouverte
#249. Détails : [doc 08](docs/08-colibri-reference.md).

## Modèles

| Modèle | Rôle | Total / actifs | Store int4 |
|---|---|---|---|
| [OLMoE-1B-7B-0125-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) | modèle de travail | 6,9B / 1,3B | 3,2 Gio |
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | palier intermédiaire (optionnel) | 30,5B / 3,3B | 14,3 Gio |
| [Qwen3-Next-80B-A3B-FP8](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8) | objectif final | 79,7B / ~3B | 38,2 Gio |

## Matériel de référence

RTX 2060 6 Go (compute 7.5), ~16 Go de RAM, Windows, C: NVMe (~37 Go libres), D: HDD (~645 Go libres).
Détails : [doc 01](docs/01-hardware-baseline.md).

## Résultats (OLMoE, RTX 2060, 30 septembre 2026)

Analyse complète : [doc 17](docs/17-resultats-olmoe.md). Données brutes : `results/`.
Budget RAM fixé à 5 Gio, qualité de chaque règle mesurée seule, 2 répétitions.

| Réglage | tok/s | accord top-1 | écart perplexité | fp16 demandé | Mo lus sur disque / token |
|---|---|---|---|---|---|
| tout fp16 | 5,00 | 98,8% | +0,12% | 100% | 310 |
| tout int4 (statique) | **5,92** | 89,1% | +10,35% | 0% | 5 |
| HOBBIT t1=0,6 | 4,84 | 93,8% | +4,91% | 49% | 169 |
| HOBBIT t1=0,8 | 4,98 | 97,3% | **+0,83%** | 73% | 231 |
| HOBBIT t1=0,9 | 5,06 | **99,2%** | -0,17% | 87% | 270 |

- **La précision par token donne une courbe qualité / octets continue** : à t1=0,8,
  +0,8% de perplexité au lieu des +10,4% de l'int4 statique ; à t1=0,9, la qualité
  du fp16 avec 13% des experts en int4.
- **Mais pas de gain de vitesse sur ce moteur** : même tout en VRAM, une couche MoE
  coûte 9,5 ms (surcoût du moteur, probablement la boucle Python et les synchronisations), et le disque n'ajoute que ~25%.
  Les octets économisés compteront sur Qwen3-Next-80B, limité par le disque.
- Routage d'OLMoE : prefetch 83% (Colibrì : 71,6% sur GLM-5.2), localité 40% ; le
  poids du routeur prédit l'importance réelle moins bien que sur Mixtral (Pearson
  0,89 contre 0,99), d'où le t1 élevé nécessaire.
