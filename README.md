# MoE Offload Lab

Projet d'apprentissage : construire, à partir de zéro, un système d'inférence LLM local
qui combine **quantification mixte** et **offloading intelligent** pour faire tourner
des modèles Mixture-of-Experts (MoE) sur du matériel grand public limité.

## Objectif

Comprendre et implémenter (à un niveau simplifié) les techniques utilisées par des
systèmes de recherche récents comme HOBBIT, DyMoE, MoEpic et FreeToken — sans
réinventer un moteur d'inférence complet, en s'appuyant sur `llama.cpp` /
`llama-cpp-python` puis HuggingFace Transformers/PyTorch.

**Objectif final** : faire tourner un modèle MoE de 70B+ paramètres totaux — bien
au-delà de ce qui tient nativement en VRAM ou même en RAM sur ce matériel — quitte à
sacrifier la vitesse d'inférence pour la faisabilité. La performance n'est pas le
critère de succès ; la capacité à faire fonctionner le modèle du tout l'est.

## Matériel de référence

| Composant | Spec |
|---|---|
| GPU | NVIDIA RTX 2060, 6 Go VRAM |
| RAM | ~16 Go |
| OS | Windows |
| Driver CUDA | 13.2 (compatible CUDA Toolkit 12.x/13.x) |

Voir [`docs/01-hardware-baseline.md`](docs/01-hardware-baseline.md) pour le détail des mesures.

## Roadmap

- [x] Étape 1 — Observer le comportement d'offloading par défaut d'Ollama sur un modèle dense (llama3:8b)
- [x] Étape 2 — Lecture de fond : 4 papiers de recherche sur l'offloading MoE (HOBBIT, DyMoE, MoEpic, FreeToken)
- [x] Étape 3 — Choix et téléchargement d'un modèle MoE adapté au matériel (OLMoE-1B-7B-0125)
- [x] Étape 4 — Setup environnement de dev : `llama-cpp-python` compilé avec support CUDA
- [x] Étape 4b — **Pivot d'architecture** : GGUF/llama.cpp fusionne les experts en un
      seul tenseur (pas de contrôle par expert possible en Python pur) → bascule vers
      HuggingFace Transformers/PyTorch, qui expose chaque expert individuellement
      (voir [`docs/05-architecture-reality-check.md`](docs/05-architecture-reality-check.md))
- [ ] Étape 5 — Téléchargement du checkpoint HF natif (`.safetensors`) d'OLMoE + setup `transformers`/`accelerate`
- [x] Étape 6 — Inspection de la structure du modèle en PyTorch (confirmer l'accès individuel aux experts) ✅ validé
- [x] Étape 7 — Scoring d'importance des experts (HOBBIT) ✅ implémenté et testé — [`scripts/hobbit_scoring.py`](scripts/hobbit_scoring.py)
- [ ] Étape 8 — Placement manuel d'experts CPU/GPU basé sur le score d'importance
- [ ] Étape 8 — Quantification mixte : versions haute/basse précision par expert
- [ ] Étape 9 — Scoring d'importance des experts (inspiré de HOBBIT, basé sur la sortie du gating)
- [ ] Étape 10 — Politique q\* : répartition des cache miss entre transfert GPU et calcul CPU par bande passante mesurée (inspiré de FreeToken)
- [ ] Étape 11 — Cache d'experts avec politique de remplacement (LRU/LFU hybride, inspiré LCP de MoEpic)
- [ ] Étape 12 — Prefetching inter-couches (inspiré de la similarité d'activation, DyMoE/MoEpic)
- [ ] Étape 13 — Benchmark : comparer notre système vs chargement statique sur latence/mémoire
- [ ] Étape 14 — Montée en échelle vers Mixtral-8x7B (47B total, modèle de référence des papiers étudiés)
- [ ] Étape 15 — **Objectif final** : modèle 70B+ avec 3ᵉ niveau de hiérarchie mémoire (GPU → RAM → SSD),
      puisque 16 Go de RAM ne suffisent pas à héberger le pool complet d'experts d'un tel modèle,
      même quantifié — vitesse attendue : secondes/token plutôt qu'interactif, la faisabilité prime sur la performance

## Structure du repo

```
moe-offload-lab/
├── README.md                      # ce fichier
├── docs/
│   ├── 01-hardware-baseline.md    # specs matériel + mesures de référence
│   ├── 02-research-notes.md       # résumé des 4 papiers de recherche
│   ├── 03-model-selection.md      # comparatif et choix du modèle MoE
│   ├── 04-setup-olmoe.md          # setup pas-à-pas d'OLMoE dans Ollama
│   └── 05-architecture-reality-check.md  # GGUF vs PyTorch, validation experts
├── notebooks/                     # exploration interactive (Jupyter via VS Code)
│   └── 01_expert_exploration.ipynb
├── scripts/                       # scripts Python réutilisables (à venir)
└── CHANGELOG.md                   # journal chronologique des étapes réalisées
```

## Papiers de recherche de référence

| Papier | Date | Apport principal pour ce projet |
|---|---|---|
| [HOBBIT](https://arxiv.org/abs/2411.01433) | nov 2024 | Scoring d'importance d'expert via la sortie du gating |
| [DyMoE](https://arxiv.org/abs/2603.19172) | mars 2026 | Allocation de précision par profondeur de couche |
| [MoEpic](https://arxiv.org/abs/2509.08342) | sept 2025 | Split top/bottom d'expert, cache LRU/LFU/LHU pondéré |
| [FreeToken](https://arxiv.org/abs/2608.16157) | août 2026 | Politique q\* : répartition GPU/CPU par bande passante mesurée |

Détail complet : [`docs/02-research-notes.md`](docs/02-research-notes.md)

## Modèle utilisé

**OLMoE-1B-7B-0125-Instruct** (Allen AI) — Mixture-of-Experts 100% open source
(poids, code d'entraînement, données, papier de recherche).

- 6,9 milliards de paramètres au total
- 1,3 milliard de paramètres actifs par token
- Architecture Transformer MoE standard (pas d'architecture hybride exotique)
- Licence Apache 2.0
- Papier : [arXiv:2409.02060](https://arxiv.org/abs/2409.02060)