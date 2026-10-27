# 13 — Prefetch et mesures du routage (étape 12)

## Le prefetch

Pendant la génération (1 token par passe), à la fin de la couche L, on applique le
**routeur de la couche L+1** à l'état qu'on vient de traiter à la couche L. On
obtient une prédiction des experts de L+1 (avec leurs poids, donc aussi leur
précision HOBBIT), et on commence à les charger :

- déjà en RAM → copie vers la VRAM lancée tout de suite, sur le stream de copie ;
- sur disque → lecture lancée dans le pool de threads (RAM), sans attendre.

Pourquoi ça marche : l'état caché change peu d'une couche à la suivante (connexion
résiduelle), donc le routeur de L+1 "voit" presque la même chose. Colibrì mesure
71,6% du vrai top-8 retrouvé sur GLM-5.2, contre 41,3% pour "les mêmes experts
qu'au token précédent".

Le système mesure en continu la qualité de sa prédiction : `prefetch_recall` =
part des experts réellement utilisés qui avaient été prédits.

## Trois mesures à faire sur OLMoE (`scripts\measure_routing.py`)

| Mesure | Question | Référence publiée |
|---|---|---|
| Proxy d'importance | Le poids du routeur prédit-il la vraie contribution `‖w·E(x)‖` ? | HOBBIT : Pearson 0,99 (Mixtral) |
| Prefetch | Le routeur de L+1 appliqué à l'état de L retrouve-t-il le top-k de L+1 ? | Colibrì : 71,6% (GLM-5.2) |
| Localité temporelle | Un token réutilise-t-il les experts du précédent ? | Colibrì : 41,3% (GLM-5.2) |

Les deux premières valident (ou non) les hypothèses sur lesquelles reposent HOBBIT
et le prefetch pour **ce** modèle. Une corrélation faible sur OLMoE voudrait dire
que les seuils HOBBIT y décident moins bien que sur Mixtral : c'est un résultat
intéressant en soi, à documenter tel quel.

Le script mesure la vraie contribution de chaque expert sur les 32 premiers tokens
(il faut calculer tous les experts choisis, c'est coûteux) et le prefetch sur 256
tokens. Résultats dans `results\routing_<modèle>_<date>.json`.

Note : sur les modèles miniatures aléatoires des tests, ces chiffres ne veulent rien
dire (un routeur aléatoire n'a pas de structure). Seuls les chiffres du vrai OLMoE
comptent.

## Résultats sur OLMoE

Proxy HOBBIT : Pearson 0,888 (contre 0,99 sur Mixtral), plus gros poids = plus
grosse contribution dans 67% des cas. Prefetch : 83,3% (56% de la couche 0 vers la
1, puis 85-90% dans les couches profondes). Localité : 39,9%. Analyse : [doc 17](17-resultats-olmoe.md).
