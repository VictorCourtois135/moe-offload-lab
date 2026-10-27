# 08 — Colibrì : référence système pour la hiérarchie VRAM/RAM/SSD

Pas un papier de recherche, mais un vrai système open source
([JustVugg/colibri](https://github.com/JustVugg/colibri)) qui fait ce que vise
notre objectif final : faire tourner un modèle MoE bien plus gros que la mémoire
disponible, en traitant VRAM/RAM/disque comme une seule hiérarchie. Publié juillet
2026, ~27k étoiles. Écrit en C pur, un fichier par famille de modèle —
**supporte directement OLMoE**.

## Résultat le plus marquant

Fait tourner GLM-5.2 (744 milliards de paramètres) sur ~25 Go de RAM, sans GPU,
en streamant les experts depuis le disque. Vitesse honnêtement annoncée :
0,05-0,1 tok/s dans ce cas extrême.

## Politique de cache — proche de ce qu'on construit

**Score LFRU par couche** : fréquence d'usage à décroissance temporelle combinée à
la récence d'accès. Très proche du LCP de MoEpic.

**Cache d'apprentissage persistant entre sessions** (`.coli_usage`) : au démarrage,
le moteur épingle les experts historiquement les plus utilisés. Aucun des papiers
lus ne proposait cette persistance.

**Anti-oscillation** : lors de l'adaptation du placement (`--repin`), une
hystérésis de 25% et au plus 4 échanges par tour empêchent le cache de changer
constamment d'avis.

## Prefetch — validation empirique

Sur GLM-5.2 : appliquer le routeur de la couche L+1 à l'état de la couche L
retrouve **71,6%** du vrai top-8, contre **41,3%** pour "mêmes experts qu'au token
précédent".

## Philosophie sur le disque

"Le disque est une source de récupération immuable, pas une cible de décodage
normale" : les lectures froides passent par un pool d'I/O asynchrone pendant que les
experts déjà résidents calculent.

## Notre projet répond à une question ouverte chez eux

**Colibrì n'implémente pas de précision dynamique par expert.** Sa quantification
est statique, choisie à la conversion. L'issue ouverte **#249** demande précisément
une bascule dynamique de précision par expert. Leur méthode de contribution
correspond à la nôtre : mesures chiffrées, matériel exact, "a well-controlled
failure is more valuable here than an unexplained fast number".

## Idées de Colibrì transposées en Python

| Idée | Statut dans `moelab/` |
|---|---|
| Score LFRU | Implémenté (`cache.py`, formule LRFU à un paramètre) |
| Cache persistant entre sessions | Implémenté (`usage.json`, `cache.py`) |
| Anti-thrashing (hystérésis 25%, 4 échanges max) | Implémenté (`TierCache.repin`) |
| Auto-dimensionnement du cache | Implémenté (`plan.py`) |
| Prefetch inter-couches mesuré | Implémenté + script de mesure (`measure_routing.py`) |
| Transparence sur la dégradation | Implémenté : toute décision basse précision ou saut est comptée et affichée |
| Validation A/B (+3%, qualité bornée) | Implémenté (`bench.py`, `scripts/benchmark.py`) |
| Overlap I/O asynchrone | Implémenté (pool de threads pour les lectures disque, stream CUDA pour les copies) |
| Records d'expert contigus (une lecture par expert) | Implémenté (`store.py`) |
| int4 avec échelles par groupe de 64 (gs64) | Implémenté (`quant.py`) |

Détail de chaque brique : docs 09 à 16.
