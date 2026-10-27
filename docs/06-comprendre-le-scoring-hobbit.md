# 06 — Comprendre le scoring HOBBIT en profondeur

Notes de compréhension écrites avant l'implémentation du placement CPU/GPU.
Complète [`scripts/hobbit_scoring.py`](../scripts/hobbit_scoring.py). La version
utilisée par le système final est dans [`moelab/policy.py`](../moelab/policy.py).

## Le pipeline complet, du token au routeur

```
Embedding → Self-attention → Normalisation → Routeur (gate) → Top-8 experts
```

Chaque étape enrichit le vecteur du token avec du contexte. Le routeur a besoin de
ce vecteur *enrichi*, pas de l'embedding brut — voir "l'erreur de méthode"
ci-dessous pour la preuve empirique.

## Pourquoi le score du routeur plutôt que la vraie sortie de l'expert ?

Calculer `E_ei(x)` (la sortie réelle d'un expert) nécessite d'avoir déjà chargé et
exécuté cet expert — précisément le coût qu'on cherche à éviter. Le score du
routeur `G(x)_ei`, lui, est **déjà calculé de toute façon** : chaque token en a
besoin pour savoir quels experts activer. C'est un signal "gratuit". HOBBIT montre
que ce proxy est fortement corrélé à la vraie contribution (Pearson = 0.99 sur
Mixtral). Le script `scripts/measure_routing.py` mesure cette corrélation sur
OLMoE.

## Pourquoi cumsum et pas juste inverser l'ordre des scores ?

Exemple réel obtenu sur la phrase "Le chat mange une souris" (couche 0) :

```
Expert (trié) :  46      11      45      17      57      47      3       34
Score router  :  0.1345  0.1063  0.0921  0.0855  0.0829  0.0705  0.0682  0.0214
Cumsum exclu  :  0.0000  0.1345  0.2410  0.3330  0.4185  0.5015  0.5718  0.6396
```

Inverser le rang capture seulement la *position*, pas l'écart réel entre les
scores. Le cumsum répond à la question "quelle part de la confiance du routeur les
experts mieux classés que moi ont-ils déjà accaparée ?".

## Correction : HOBBIT normalise les scores d'abord

OLMoE a `norm_topk_prob = False` : les 8 scores ci-dessus ne somment pas à 1
(somme = 0,6614). HOBBIT normalise avant le cumsum (§3.2 du papier), ce que
`scripts/hobbit_scoring.py` ne faisait pas. Avec normalisation, le même exemple
donne :

```
Normalisé     :  0.203   0.161   0.139   0.129   0.125   0.107   0.103   0.032
Cumsum exclu  :  0.000   0.203   0.364   0.503   0.633   0.758   0.865   0.968
```

Avec les seuils du papier (`T1 = 0.6`, `T2 = 0.9`) : les 4 premiers experts en
haute précision, les 3 suivants en basse précision, le dernier (expert 34) serait
sauté si le saut est activé. Sans normalisation, les seuils n'ont pas le même sens
d'un modèle à l'autre. `moelab/policy.py` normalise toujours.

## L'erreur de méthode qu'on a faite (et pourquoi elle est instructive)

Premier test : envoyer l'embedding brut directement au routeur de la couche 0, en
sautant l'attention et la normalisation. Résultat :

```
Scores : [0.0165, 0.0163, 0.0162, 0.0161, 0.0161, 0.0161, 0.0159, 0.0159]
```

Tous proches de 1/64 ≈ 0.0156 : distribution quasi uniforme. Sans le contexte
apporté par l'attention, le signal reçu par le routeur est trop "brut".

Correction : un `forward_hook` PyTorch capture les scores réels pendant un vrai
passage complet du modèle, au lieu de reconstruire le calcul à la main.

## `register_forward_hook` vs `register_forward_pre_hook`

- **`register_forward_hook`** : s'exécute *après* le calcul du module. Reçoit
  `(module, input, output)` — permet d'observer.
- **`register_forward_pre_hook`** : s'exécute *avant* le calcul. Reçoit
  `(module, input)` — permet de modifier l'entrée.

Le système final n'utilise ni l'un ni l'autre pour intervenir : il **remplace** le
module `experts` de chaque couche par le sien (`OffloadedExperts`), qui reçoit
directement les décisions du routeur.
