# 10 — Le store d'experts et la quantification (étape 8)

## Pourquoi un store plutôt que le checkpoint directement ?

Dans le checkpoint Hugging Face, un expert = 3 tenseurs (`gate_proj`, `up_proj`,
`down_proj`), parfois dans des fichiers différents, en bf16. Charger un expert
coûte donc 3 lectures, et on n'a qu'une précision.

Le store réécrit chaque expert en un **record contigu** : une seule lecture disque
par expert (idée de Colibrì), et une version par précision.

```
C:\moe-store\olmoe\
    manifest.json          formes, précisions, position de chaque champ dans un record
    layer000.fp16.bin      64 records fp16 de 12 Mio (couche MoE 0)
    layer000.int4.bin      64 records int4 de 3,19 Mio
    ...
    usage.json             historique d'usage (créé à l'usage)
    bandwidth.json         bandes passantes mesurées (measure_bandwidth.py)
```

Un record est aligné sur 4096 octets. Une fois en RAM ou en VRAM, c'est un simple
tenseur `uint8` : `views()` en extrait des tenseurs typés **sans copie**, et
`materialize()` les déquantifie sur le device où il se trouve.

## Quantification int4 "gs64"

Pour chaque ligne d'une matrice, on découpe en groupes de 64 colonnes. Pour chaque
groupe : `échelle = max|w| / 7`, puis `q = arrondi(w / échelle)` dans [-8, 7].
Deux valeurs par octet (quartet bas = colonne paire). Coût réel : 4 bits + une
échelle fp16 pour 64 poids = **4,25 bits par poids**, soit 3,76 fois moins que fp16.

Pourquoi des groupes de 64 et pas une échelle par ligne ? Colibrì a mesuré sur
Qwen3.6 que les échelles par groupe réduisent l'erreur de quantification d'environ
44% (cosinus 0,98777 → 0,99313 contre la version int8). Une valeur extrême ne
dégrade que son groupe, pas toute la ligne.

L'erreur attendue sur des poids gaussiens est d'environ 10% relatif en int4 et
0,6% en int8 (`tests/test_units.py` le vérifie). Les vrais poids ont des queues plus
lourdes : c'est le benchmark de qualité qui dira ce que ça coûte vraiment.

## Tailles réelles

| Modèle | fp16 | int8 | int4 | lu par token si tout manque (int4) |
|---|---|---|---|---|
| OLMoE-1B-7B (1024 experts) | 12,0 Gio | 6,2 Gio | 3,2 Gio | 408 Mio |
| Qwen3-30B-A3B (6144 experts) | 54,0 Gio | 27,8 Gio | 14,3 Gio | 918 Mio |
| Qwen3-Next-80B-A3B (24576 experts) | 144,0 Gio | 74,2 Gio | 38,2 Gio | 765 Mio |

## Checkpoints FP8

Les versions `*-FP8` de Qwen stockent les poids en `float8_e4m3` avec une échelle
`weight_scale_inv` par bloc de 128x128 (et, malgré son nom, on la **multiplie**).
`checkpoint.py` les déquantifie à la lecture. Intérêt : télécharger 82 Go au lieu
de 160 Go pour Qwen3-Next. Contrepartie : les experts subissent deux
quantifications successives (FP8 puis int4) au lieu d'une.

## Construction

```powershell
python scripts\build_store.py --model D:\models\olmoe-hf --out C:\moe-store\olmoe
```

- Un expert à la fois en mémoire : la RAM nécessaire ne dépend pas de la taille du modèle.
- Vérifie l'espace libre avant de commencer.
- **Reprenable** : `progress.json` note les couches terminées ; relancer la même
  commande reprend où ça s'était arrêté.
- `--only lo` : un store int4 seul (pour Qwen3-Next, faute de place pour le fp16).
- `--hi int8` : haute précision en int8 au lieu de fp16 (store 2x plus petit).

Vérifié par les tests : un checkpoint "par expert" et le même au format "fusionné"
donnent des fichiers de store identiques octet pour octet.
