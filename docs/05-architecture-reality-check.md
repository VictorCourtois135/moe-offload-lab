# 05 — Réalité technique : GGUF vs PyTorch pour le contrôle par expert

## Contexte

Après compilation réussie de `llama-cpp-python` avec support CUDA, premier
chargement du modèle OLMoE-1B-7B-0125 (F16) pour valider le pipeline.

## Commande de test

```powershell
python -c "from llama_cpp import Llama; llm = Llama(model_path='D:/models/olmoe/OLMoE-1B-7B-0125-Instruct-F16.gguf', n_gpu_layers=1, verbose=True)"
```

## Résultat — succès du chargement

- `ggml_cuda_init` détecte bien la RTX 2060 (compute capability 7.5)
- 16 couches, 64 experts/couche, 8 experts actifs/token, 6,92B paramètres au total
- Fichier F16 : 12,89 GiB, chargé via `mmap` (pas de duplication mémoire)
- `n_gpu_layers=1` → seule la couche de sortie placée sur GPU (196,51 MiB en VRAM)

## Découverte bloquante

Dans les logs de chargement :

```
create_tensor: loading tensor blk.0.ffn_gate_exps.weight
create_tensor: loading tensor blk.0.ffn_down_exps.weight
create_tensor: loading tensor blk.0.ffn_up_exps.weight
```

Le suffixe `_exps` indique que **les 64 experts d'une couche sont stockés comme un
seul tenseur 3D** dans le format GGUF, et le routage vers les experts actifs est
géré à l'intérieur des kernels de `llama.cpp`, de façon opaque depuis Python.

Les papiers de référence (HOBBIT, DyMoE, MoEpic) ont tous modifié le C++/CUDA de
`llama.cpp` pour atteindre ce niveau de contrôle (HOBBIT : +8000 lignes).

## Options envisagées

1. Réduire le scope à l'offloading niveau couche (Python pur via `n_gpu_layers`)
2. Patcher `llama.cpp` en C++/CUDA (fidèle aux papiers, complexité élevée)
3. **Explorer HuggingFace Transformers/PyTorch** — retenu

## Décision — Option 3 retenue

Les checkpoints HuggingFace natifs (`.safetensors`) stockent chaque expert comme un
tenseur séparé sur disque :

```
model.layers.3.mlp.experts.0.gate_proj.weight
model.layers.3.mlp.experts.0.up_proj.weight
model.layers.3.mlp.experts.0.down_proj.weight
```

**Précédent qui valide l'approche** : *"Dynamic Expert Quantization for Scalable
Mixture-of-Experts Inference"* (arXiv:2511.15015) implémente une quantification
dynamique par expert sur la stack Transformers, en ~3000 lignes Python.

**Compromis accepté** : inférence plus lente que `llama.cpp`, en échange d'un
contrôle total et lisible en Python.

## Résolution — la granularité par expert est bien accessible

Une fois chargé, `transformers` 5.15.1 fusionne les experts en mémoire en **un
tenseur PyTorch par couche** (et non plus en modules séparés) :

```python
experts.gate_up_proj.shape  # (64, 2048, 2048) — dimension 0 = index de l'expert
experts.down_proj.shape     # (64, 2048, 1024)
```

Contrairement au GGUF, c'est un **tenseur PyTorch standard** : l'indexation donne
un accès direct à chaque expert.

```python
expert_5 = experts.gate_up_proj[5].clone()   # extraction indépendante
expert_5_gpu = expert_5.to('cuda')            # déplacement GPU isolé — fonctionne (8 Mo)
```

**Mieux encore** : la fusion n'existe qu'en mémoire. Sur disque, le checkpoint
garde un tenseur par expert, et `transformers` les fusionne au chargement (règle
`qwen2_moe` de `conversion_mapping.py`, partagée par OLMoE et Qwen3-Next). Le
package `moelab` lit donc les experts directement depuis les fichiers
`.safetensors`, sans jamais charger le modèle complet (voir
[`10-store-et-quantification.md`](10-store-et-quantification.md)).

## Point de friction identifié (résolu depuis)

Chaque test via `python -c "..."` rechargeait le modèle complet (5 à 12 minutes).
Première parade : un notebook Jupyter qui garde le modèle en mémoire. Solution
définitive : le chargeur de `moelab` ne charge que la partie dense (~1 Go pour
OLMoE) et laisse les experts dans le store — le chargement prend quelques secondes.
