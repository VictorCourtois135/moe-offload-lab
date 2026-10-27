# 03 — Sélection du modèle MoE

## Pourquoi un modèle dense ne suffit pas

Les techniques visées (scoring d'importance d'expert, cache sélectif, split
haut/bas d'expert) n'ont de sens que sur une architecture **Mixture-of-Experts** :
un modèle dense active toutes ses couches pour chaque token, il n'y a rien à
sélectionner. `llama3:8b` (utilisé en baseline) reste donc hors scope pour la suite
du projet.

## Piste explorée et abandonnée — convertir llama3:8b en MoE

Techniquement possible via "sparse upcycling" (dupliquer la FFN en plusieurs
copies + routeur) ou "MoEfication" (clustering des neurones existants), mais :

- Nécessite un ré-entraînement significatif (plusieurs milliards de tokens) pour
  que le routeur et les experts se différencient réellement.
- Hors de portée d'un GPU grand public unique.
- N'aide pas au problème initial : le nombre total de paramètres augmente, sans
  réduire l'empreinte mémoire.

→ Abandonné, on part sur un modèle MoE pré-entraîné.

## Comparatif des candidats

| Modèle | Total / Actif | Architecture | Licence | Taille F16 | Verdict |
|---|---|---|---|---|---|
| Qwen1.5-MoE-A2.7B | 14,3B / 2,7B | Transformer MoE standard | Tongyi Qianwen | ~28,6 Go | Valide mais lourd |
| Qwen3.6-35B-A3B (le plus petit MoE Qwen actuel) | 35B / 3B | Hybride | Apache 2.0 | ~70 Go | Hors de portée pour l'étape d'apprentissage |
| LFM2.5-8B-A1B (Liquid AI, mai 2026) | 8,3B / 1,5B | **Hybride** (conv + attention + MoE) | LFM 1.0 (restreinte) | ~16,6 Go | Architecture non-standard |
| Mellum2-12B-A2.5B (JetBrains) | 12B / 2,5B | Transformer MoE standard | Apache 2.0 | ~24 Go | Moins documenté |
| **OLMoE-1B-7B-0125** (Allen AI) | 6,9B / 1,3B | Transformer MoE standard | **Apache 2.0** | **~13,8 Go** | **Retenu** |

## Critères de décision

1. **Architecture standard** : une architecture hybride complique l'application
   directe des concepts des papiers étudiés.
2. **Taille adaptée au matériel** : téléchargement F16 + conversion possibles sur
   16 Go de RAM.
3. **Documentation et ouverture** : OLMoE est publié avec son papier de recherche
   dédié (arXiv:2409.02060), son code d'entraînement et ses données.
4. **Licence sans restriction** : Apache 2.0.

## Modèle retenu

**OLMoE-1B-7B-0125-Instruct** (version de janvier 2025).

- Source GGUF : [huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct-GGUF](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct-GGUF)
- Source HF native (utilisée ensuite) : `allenai/OLMoE-1B-7B-0125-Instruct`
- Papier : Muennighoff et al., *"OLMoE: Open Mixture-of-Experts Language Models"*,
  [arXiv:2409.02060](https://arxiv.org/abs/2409.02060)

## Paliers suivants

- Palier intermédiaire possible (même code, aucune modification) :
  `Qwen/Qwen3-30B-A3B` (30B total / 3B actifs, 128 experts).
- Palier final (70B+) : **Qwen3-Next-80B-A3B-Instruct**, voir
  [`15-palier-final-qwen3-next.md`](15-palier-final-qwen3-next.md).
