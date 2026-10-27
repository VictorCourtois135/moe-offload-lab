# 22. Élaguer des experts avec REAP : banc d'évaluation et calibration

## Pourquoi (08/10)

Qwen3-Next-80B a 512 experts par couche et en choisit 10 par token. Si une partie de ces
experts sert rarement, ou peu, les retirer réduit le store (38 Gio en int4) sans
ré-entraînement. Avec l'offloading, le gain est double : moins de disque, et un cache RAM/VRAM
de même taille qui couvre une plus grande part du modèle, donc un meilleur taux de hit.

Méthode : **REAP** (Router-weighted Expert Activation Pruning, Cerebras, arXiv 2510.13999).
Pour la couche l et l'expert j, sur les tokens x que le routeur lui envoie :

    S[l, j] = moyenne de  g_j(x) · ||f_j(x)||₂

- g_j(x) : poids du routeur après top-k et normalisation (Qwen3-Next : `norm_topk_prob`) ;
- f_j(x) : sortie de l'expert avant pondération.

Un expert choisi souvent mais avec des poids faibles et des sorties petites compte peu ; un
expert rare mais décisif quand il sert compte beaucoup. L'élagage retire, dans chaque couche,
les experts au score le plus bas. L'expert partagé (toujours actif) n'est jamais concerné.

## Plan (étapes 1 et 2 faites le 08/10)

| Étape | Quoi | État |
|---|---|---|
| 1 | banc d'évaluation : vitesse, appels d'outils, perplexité → `results/reap/baseline.json` | fait |
| 2 | calibration REAP → `results/reap/reap_stats.pt`, `reap_summary.json` | fait |
| 3 | élagage virtuel : masque avant le top-k, `--prune-ratio`, sans toucher au store | fait (13/10) |
| 4 | réglage dans l'autopilot (taille ↔ qualité mesurées) | à faire |
| — | réécriture du store | seulement si la place disque est nécessaire, avec sauvegarde |
| — | élagage de couches (ShortGPT) | écarté pour l'instant : plus risqué, motif hybride |

## Le banc d'évaluation (`scripts/eval_ppl.py`)

Trois mesures, enregistrées dans un JSON par essai (`results/reap/<nom>.json`), toutes sur des
données figées : le jeu d'évaluation est construit une fois puis gardé en tokens
(`eval_ids.pt`). Une mesure d'un modèle élagué compare donc exactement les mêmes extraits.

| Mesure | Contenu | Pourquoi |
|---|---|---|
| vitesse | 3 invites de `bench/prompts.json`, 64 tokens gloutons | tok/s, 1er token, hit VRAM, lectures disque/token |
| outils | `bench/tool_tasks.json` : 20 questions en un tour | l'usage réel : le bon outil avec les bons arguments, ou une réponse directe juste |
| perplexité | 50 extraits WikiText-2 test + 20 extraits Wikipédia FR, 512 tokens | la qualité de prédiction ; + top-1 gardé pour l'accord avec la référence |

Pourquoi les outils : le papier REAP montre que l'élagage abîme d'abord la **génération**,
alors que la perplexité bouge peu. Les 20 questions sont notées par le code (pas par le
modèle) : nom d'outil, morceaux attendus dans les arguments (« 3,35 » ou « 3.35 »), ou réponse
directe qui contient ce qu'il faut sans appeler d'outil. Le système et les outils sont fixes
(pas de date du jour) : deux mesures à des jours différents restent comparables.

Les passes de perplexité se font par lots (`--batch 8` : 8 extraits, 4096 tokens par passe) :
512 tokens choisissent déjà presque tous les experts de chaque couche, donc une passe lit
presque tout le store ; un lot de 8 divise ces lectures par 8. Si la VRAM manque, le lot est
coupé en deux automatiquement. La tête de sortie est appliquée par morceaux de 128 positions :
jamais de tenseur 4096 × 151 936.

## La calibration (`scripts/calibrate_reap.py`)

**Les données.** 256 échantillons de 512 tokens, graine 0, figés dans `calib_ids.pt` :

| Source | Part | Jeu |
|---|---|---|
| tes données | 10 % | conversations et runs d'agents de l'app (`app_data`), rendus avec le modèle de chat, appels d'outils compris |
| conversation | 15 % | UltraChat 200k |
| code | 20 % | HumanEval + MBPP |
| maths | 15 % | GSM8K |
| français | 20 % | Wikipédia FR (après les 2 000 premiers articles : l'évaluation est prise dans les 400 premiers) |
| texte général | 20 % | WikiText-2 train (l'évaluation est sur test) |

REAP montre que le domaine de calibration décide de ce qui est préservé (calibrer sur du code
garde le code). D'où tes conversations et tes runs d'agents : français, appels d'outils,
raisonnement en plusieurs étapes. Une source indisponible passe sa part aux suivantes ; le
script refuse de continuer si un échantillon de calibration est identique à un extrait
d'évaluation.

**Le calcul.** Le runtime a un nouveau crochet, `expert_hook`, appelé pour chaque expert
calculé avec (couche, expert, poids du routeur, sortie avant pondération), sur le chemin GPU
comme sur la branche CPU. Quand il est actif, les noyaux groupés (qui ne rendent que la somme
des experts) et le graph complet sont coupés ; il ne sert qu'aux passes de calibration. Pendant
ces passes : q* coupé (tous les experts sur le GPU), aucun saut d'expert (P5, t2).

**Reprise.** Les accumulateurs d'une passe vont dans des tampons temporaires, ajoutés aux
totaux seulement si la passe se termine : une passe interrompue (Ctrl+C, mémoire pleine) ne
compte jamais à moitié. Sauvegarde atomique tous les 4 lots ; relancer la même commande reprend
là où elle s'était arrêtée, si le store et les échantillons sont les mêmes (empreinte vérifiée).

**Ce qui est enregistré** par (couche, expert) : somme de g·||f|| (REAP), somme de ||f||
(variante EAN), somme de g (masse du routeur), nombre de tokens. La fréquence d'activation en
découle (utile aussi pour le prefetch).

**Le résumé** (`reap_summary.json`) dit, pour 25 / 40 / 50 % : combien d'experts partent par
couche, quelle part des choix du routeur ils recevaient (en moyenne et dans la pire couche), et
la taille du store obtenu. C'est une première idée du coût ; seule l'étape 3 le mesure.

## Vérifié (08/10, modèle miniature Qwen3-Next sur CPU)

`tests/test_reap.py`, 10 tests :
- scores REAP du crochet = calcul direct (entrées de chaque couche, poids lus dans le store), à
  1e-5 près, sur le chemin « VRAM » simulé et sur la branche CPU ; réglages remis après ;
- interruption au milieu d'un lot puis reprise = calibration d'une traite ; autres échantillons :
  ancien fichier ignoré ;
- lots coupés en deux quand la mémoire manque ; autre erreur : remontée ;
- masque d'élagage (au moins top-k experts gardés, couches choisies), résumé ;
- données : découpage, parts redistribuées si une source tombe, même graine = mêmes échantillons,
  jeu figé relu sans être reconstruit ;
- notation des 20 questions ; perplexité par lots = modèle Hugging Face de référence ; mesure
  complète écrite puis accord top-1 de 100 % avec elle-même.

Pas encore lancé sur la RTX 2060 avec Qwen3-Next : c'est la prochaine mesure.

## Référence mesurée (13/10, RTX 2060, Qwen3-Next int4, RAM 5 Gio, VRAM 2,1 Gio)

| Mesure | Valeur |
|---|---|
| durée totale | 24,9 min (vitesse ~1 min, outils 12,3 min, perplexité 11 min) |
| génération | 5,1 tok/s, 1er token 4,7 s ; hit VRAM 58 %, hit RAM 29 %, **142,6 lectures disque par token** |
| appels d'outils | 20/20 (le banc détecte une régression, pas un progrès : il est déjà au maximum) |
| perplexité WikiText-2 | 6,910 (50 × 512 tokens) |
| perplexité français | 4,370 (20 × 512 tokens) |

Passe de perplexité : ~70 s pour 8 extraits (4 096 tokens), soit environ une lecture complète du store.
La calibration REAP (256 échantillons, lots de 8) devrait donc prendre de l'ordre de 40 minutes. Les 142
lectures disque par token disent où l'élagage peut aider : à cache égal, moins d'experts = plus de hits.


## Essai de calibration (13/10, 10 échantillons)

1,6 min pour 10 échantillons (~60 s par lot de 8) : les 256 échantillons prendront ~35 min. Le jeu de
calibration : 26 échantillons tirés de tes conversations et runs d'agents, 38 conversations, 51 code,
38 maths, 51 Wikipédia FR, 52 WikiText. Sur 10 échantillons seulement, 1 579 experts n'ont jamais été
choisis : ce chiffre baissera beaucoup avec 256 ; les taux ne sont à lire qu'après la vraie calibration.

## Étape 3 : élagage virtuel (`scripts/prune_eval.py`)

Le routeur de chaque couche reçoit un biais : 0 pour les experts gardés, −inf pour les experts élagués,
ajouté aux logits **avant** le softmax et le top-k. Le routeur choisit donc ses k experts parmi ceux qui
restent, avec la même normalisation. C'est le même module qu'appellent la passe normale, le prefetch (le
routeur de la couche suivante) et les CUDA graphs : un expert masqué n'est jamais choisi, jamais prédit,
jamais chargé. Le préchargement au démarrage (experts chauds d'usage.json) les saute aussi.

Le masque se pose après le chargement, avant le premier message ; rien n'est écrit dans le store. Au moins
top_k experts restent toujours par couche. `--layers 10-20` limite l'élagage à ces couches (validation du
mécanisme), `--score ean` ou `frequence` compare avec d'autres critères que REAP.

Vérifié (modèle miniature) : masque nul = modèle inchangé ; masque = même résultat que le modèle Hugging Face
de référence auquel on applique le même masque ; aucun expert masqué choisi ni présent dans les caches ;
masque retiré = modèle d'origine.


## Une commande pour tout (13/10)

`scripts/reap_pipeline.py --store … --ram-gb 5` enchaîne référence, calibration, essai sur les couches 10 à
20, élagage à 25 / 40 / 50 %, puis écrit `comparaison.md`. Chaque étape tourne dans son propre processus
(mémoire GPU neuve) ; une étape déjà faite est sautée, la calibration reprend au milieu. Les résultats vont
dans `<store>\reap` au lieu de `results\reap` du repo : remplacer le repo par une nouvelle version ne les
efface plus ; un ancien `results\reap` est recopié là au premier lancement.
