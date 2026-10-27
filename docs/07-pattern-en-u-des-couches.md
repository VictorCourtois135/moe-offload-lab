# 07 — Bruit d'échantillon vs signal réel : le pattern en U des couches OLMoE

## Objectif de l'expérience

Vérifier si les variations de confiance du routeur observées par couche sont un
vrai pattern structurel du modèle, ou juste du bruit lié à un seul exemple.

## Méthode

Capture du meilleur score du routeur (dernier token) pour chaque couche (0-15), sur
20 phrases françaises variées, puis moyenne et écart-type par couche.

## Résultat

**Sur une seule phrase**, deux pics suspects apparaissaient aux couches 3 (0,1847)
et 13 (0,2472), et un creux à la couche 4 (0,0595).

**Sur la moyenne de 20 phrases**, ces pics s'effondrent (couche 3 : 0,114 ;
couche 13 : 0,111) : c'était du bruit d'échantillon.

**Un vrai pattern émerge, différent de ce qu'on attendait** : une forme en **U**,
pas une tendance monotone avec la profondeur.

| Zone | Couches | Score moyen | Interprétation |
|---|---|---|---|
| Début | 0-1 | 0,143 / 0,124 (élevé) | Routeur confiant |
| Milieu | 2-14 | ~0,09-0,11 (plateau bas) | Routeur plus "hésitant" |
| Fin | 15 | 0,160 (le plus élevé) | Routeur très confiant |

Critère de fiabilité : un pattern qui **survit** au moyennage (couche 15) est plus
crédible qu'un pic qui s'effondre (couches 3 et 13).

Moyennes brutes : `[0.143, 0.124, 0.108, 0.114, 0.099, 0.100, 0.088, 0.102, 0.093,
0.094, 0.099, 0.099, 0.097, 0.111, 0.110, 0.160]`, écarts-types de 0,011 à 0,044.

## Mise en perspective avec la littérature

Premières et dernières couches "spéciales", couches intermédiaires plus
interchangeables : résultat documenté par *"ShortGPT: Layers in Large Language
Models are More Redundant Than You Expect"* (Men et al., 2024), cité par HOBBIT.

## Limites

- 20 phrases seulement, une seule langue, un seul modèle.
- Métrique = confiance du routeur, pas la sensibilité à la quantification mesurée
  par DyMoE : lien plausible mais non prouvé.

## Ce qu'on en a fait

Plutôt que le schedule cosinus de DyMoE, `moelab/policy.py` propose l'option
`protect_layers` : des couches toujours calculées en haute précision. Le benchmark
compare `hobbit` et `hobbit + couches 0, 1, 15 protégées` : c'est ce test qui dira
si le pattern en U compte vraiment pour la qualité.
