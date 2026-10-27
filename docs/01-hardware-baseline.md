# 01 — Matériel et baseline de référence

## Matériel

Relevé via `nvidia-smi` et `Get-ComputerInfo` (PowerShell) :

- **GPU** : NVIDIA GeForce RTX 2060, 6144 MiB VRAM (~1 Go déjà occupé par Windows/apps
  au repos, donc ~5 Go réellement disponibles pour l'inférence)
- **RAM** : ~17 016 107 008 octets ≈ 16 Go
- **Driver CUDA** : 13.2 (compatible avec CUDA Toolkit 12.x et 13.x)
- **CUDA Toolkit installé** : 13.3.1
- **Disques** : C: = SSD NVMe (~37 Go libres après nettoyage),
  D: = disque dur (HDD) SATA avec beaucoup d'espace

## Pourquoi cette étape compte

Avant de toucher au code, il faut savoir exactement avec quoi on travaille. Le ratio
VRAM/taille de modèle détermine tout : combien de couches peuvent tenir en GPU, à
partir de quelle taille de modèle l'offloading CPU devient obligatoire, etc.

## Baseline #1 — Modèle dense (llama3:8b, Q4, ~4.7 Go)

Commande de mesure :

```powershell
ollama run llama3:8b "Explique-moi la photosynthèse en une phrase"
ollama ps
```

Résultat observé :

```
NAME         ID              SIZE      PROCESSOR          CONTEXT    UNTIL
llama3:8b    365c0bd3c000    5.3 GB    22%/78% CPU/GPU    4096       4 minutes from now
```

**Interprétation** : le modèle (5,3 Go au total, avec le cache de contexte) ne tient
pas entièrement dans les ~5 Go de VRAM disponibles. Ollama offload automatiquement
22% du modèle sur CPU — une décision "statique" basée sur le remplissage maximal de
la VRAM, sans notion de priorité entre les couches.

Confirmé également via `nvidia-smi` pendant l'inférence : **5212 MiB / 6144 MiB**
utilisés — le modèle est collé au plafond de VRAM.

## Ce que cette baseline nous apprend

1. Même un modèle dense de taille modeste (8B) sature déjà la VRAM d'une carte grand
   public comme la RTX 2060.
2. L'offloading automatique d'Ollama est fonctionnel mais "naïf" — pas de notion
   d'importance différenciée entre les parties du modèle.
3. Un modèle dense n'a pas de structure "experts" à exploiter — pour appliquer les
   techniques de recherche visées (scoring d'importance, cache sélectif), il faut un
   modèle MoE (voir [`03-model-selection.md`](03-model-selection.md)).
4. Le type de disque compte autant que la VRAM dès qu'on stream des experts depuis
   le disque : un HDD lit ~100-200 Mo/s avec un temps d'accès de ~10 ms, un NVMe
   plusieurs Go/s. Voir [`15-palier-final-qwen3-next.md`](15-palier-final-qwen3-next.md).
