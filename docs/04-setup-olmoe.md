# 04 — Import d'OLMoE dans Ollama (tentative)

## Objectif de cette étape

Obtenir une deuxième baseline (après llama3:8b) sur un modèle MoE non quantifié
(F16), pour observer comment Ollama gère l'offloading d'un modèle structurellement
différent (experts vs couches denses).

## Téléchargement

```powershell
hf download allenai/OLMoE-1B-7B-0125-Instruct-GGUF --include "*F16*" --local-dir C:\Users\Home\models\olmoe
```

`hf download` est l'outil en ligne de commande du Hub Hugging Face (a remplacé
`huggingface-cli`). `--include "*F16*"` ne télécharge que le fichier en pleine
précision (16 bits par paramètre).

**Résultat** : `OLMoE-1B-7B-0125-Instruct-F16.gguf`, 13 844 566 016 octets,
déplacé ensuite sur `D:\models\olmoe\` faute de place sur C:.

## Le Modelfile — pourquoi c'est nécessaire

Un fichier `.gguf` contient les poids et quelques métadonnées, mais Ollama a besoin
d'un **Modelfile** pour savoir :

1. **`FROM`** — quel fichier de poids charger
2. **`PARAMETER`** — les hyperparamètres d'inférence par défaut
3. **`TEMPLATE`** — comment structurer le texte envoyé au modèle

Le template n'est **pas générique** : chaque famille de modèle est entraînée avec
ses propres tokens spéciaux. Se tromper de template ne provoque pas d'erreur
explicite, le modèle répond juste de façon dégradée.

```
FROM ./OLMoE-1B-7B-0125-Instruct-F16.gguf
PARAMETER temperature 0.7
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}
{{ end }}<|user|>
{{ .Prompt }}
<|assistant|>
"""
```

## Résultat — abandonné

```
Error: write C:\Users\Home\.ollama\models\blobs\sha256-...: There is not enough space on the disk.
```

`ollama create` **copie** le fichier GGUF dans son propre dossier de blobs : il faut
temporairement deux fois l'espace disque. Plutôt que de libérer 14 Go, on est passé
directement à `llama.cpp`, qui charge le GGUF par `mmap` sans copie (voir
[`05-architecture-reality-check.md`](05-architecture-reality-check.md)).
