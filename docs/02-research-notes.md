# 02 — Notes de recherche : 4 systèmes d'offloading MoE

Quatre papiers ont servi de base théorique au projet. Aucun n'est implémenté à
l'identique (leurs systèmes complets font des milliers de lignes de C++/CUDA), mais
chacun apporte un concept réutilisable à petite échelle. Ce qui a été réellement
implémenté dans `moelab/` est indiqué à la fin.

## HOBBIT (Tang et al., 2024) — arXiv:2411.01433

**Idée centrale** : quand un expert manque en cache (cache miss), au lieu de le
charger à pleine précision (lent), charger une version basse précision si l'expert
est jugé peu important pour le token courant.

**Concept clé réutilisable — scoring d'importance sans coût additionnel** :
l'importance d'un expert `e_i` est approximée par la magnitude de sa sortie de
gating `||G(x)_ei||`, sans avoir besoin de calculer la sortie complète de l'expert
(corrélation de Pearson mesurée à 0.99 avec la vraie contribution sur Mixtral).

**Détail important (§3.2)** : les scores du routeur sont **normalisés** avant le
calcul, puis `s_i = somme des scores des experts mieux classés`. Deux seuils :
`s ≤ T1` → haute précision, `T1 < s ≤ T2` → basse précision, `s > T2` → l'expert
est sauté. Valeurs utilisées pour Mixtral-8x7B : `T1 = 0.6`, `T2 = 0.9`. Le premier
expert (`s = 0`) est toujours en haute précision.

**Politique de cache** : combinaison pondérée de LRU + LFU + LHU (Least
*High-precision* Frequently Used — une variante de LFU spécifique au cache miss
des versions haute précision) + FLD (distance à la couche courante).

## DyMoE (Huang et al., 2026) — arXiv:2603.19172

**Apporte par rapport à HOBBIT** : la sensibilité à la quantification dépend de la
**profondeur de la couche**, pas seulement de l'expert individuel.

- Couches peu profondes (proches de l'entrée) : très sensibles au bruit de
  quantification → garder en haute précision.
- Couches profondes : bien plus tolérantes → quantification agressive possible sans
  perte notable.

**Concept clé réutilisable — schedule cosinus par profondeur** :

```
r(l) = (1 - λ) · [cos(π · l/(L-1)) + 1] / 2 + λ
```

`r(l)` = ratio de rétention en haute précision à la couche `l`. Le cosinus reste
proche de 1 au début puis décroît en douceur.

**Distinction prefill / decode** : en prefill, l'importance d'un expert est mesurée
par le volume de tokens "importants" qu'il traite. En decode, on se base
directement sur le score de gating du token courant.

## MoEpic (Yan et al., 2025) — arXiv:2509.08342

**Idée centrale** : au lieu de mettre en cache un expert entier ou rien,
**découper chaque expert en deux segments** (haut / bas) et ne mettre en cache que
la partie haute des experts "chauds". Pour le même budget VRAM, on garde deux fois
plus d'experts partiellement en cache, et le calcul peut démarrer sur le segment
déjà présent pendant que le reste arrive.

**Complexité** : nécessite de résoudre un problème d'optimisation par couche
(budget VRAM + ratio de split) — hors scope de ce projet.

## FreeToken (Yang et al., 17 août 2026) — arXiv:2608.16157

**Le plus récent des quatre papiers, et le plus directement implémentable.**
FreeToken répond à la question : lors d'un cache miss, faut-il transférer l'expert
sur GPU, ou le calculer directement sur CPU ?

**La politique q\* (§3.2)** :

```
q* = m · (B_PCIe / B_Host)
```

- `m` = nombre d'experts manquants au layer courant
- `B_PCIe` = bande passante mesurée du transfert CPU→GPU
- `B_Host` = bande passante mesurée du calcul des experts sur CPU
- `q*` = nombre d'experts à transférer sur GPU ; le reste (`m - q*`) est calculé
  directement sur CPU, **en parallèle** du transfert

**D'où vient la formule** : le transfert PCIe et le calcul CPU lisent tous les deux
la RAM. Pendant que le lien PCIe est saturé, il ne reste au CPU que
`B_Host - B_PCIe` de bande passante. On égalise les deux branches :
`q·S / B_PCIe = (m - q)·S / (B_Host - B_PCIe)`, ce qui donne `q* = m·B_PCIe/B_Host`.
Si `B_Host ≤ B_PCIe`, tout est transféré (`q* = m`). FreeToken arrondit `q*` et
garde **au moins un transfert** pour que le cache continue de se réchauffer.

**Autres apports** : double-buffering par couche en prefill, cache LRU partagé
entre toutes les couches, redimensionnement du cache GPU à chaud.

## Synthèse — ce qui est implémenté dans `moelab/`

| Concept | Source | Où dans le code |
|---|---|---|
| Score d'importance via sortie du routeur, seuils T1/T2 | HOBBIT | `moelab/policy.py` |
| Couches protégées en haute précision (variante DyMoE / pattern en U) | DyMoE + doc 07 | `moelab/policy.py` |
| Cache partagé entre couches, score LRFU, épinglage, anti-thrashing | HOBBIT / MoEpic / Colibrì | `moelab/cache.py` |
| Politique q\* + exécution CPU/GPU en parallèle | FreeToken | `moelab/policy.py`, `moelab/experts.py` |
| Prefetch de la couche suivante avec son propre routeur | HOBBIT / DyMoE / Colibrì | `moelab/experts.py` |
| Split top/bottom d'expert | MoEpic | non implémenté (trop complexe pour le gain attendu ici) |
