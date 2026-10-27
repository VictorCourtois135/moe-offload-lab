# 17 — Premiers résultats sur OLMoE (29 septembre 2026)

Mesures faites sur la machine de référence : RTX 2060 6 Go, 16 Go de RAM, store sur
le NVMe (C:). Budgets : 2,98 Gio de VRAM et 6,65 Gio de RAM pour les experts. Les
fichiers bruts sont dans `results/` :

- `routing_olmoe-hf_2026-09-29.json` (`measure_routing.py`)
- `benchmark_olmoe-hf_2026-09-29_2034.md` et `.json` (`benchmark.py --repeats 2`)

## 1. Le routage d'OLMoE

| Mesure | OLMoE | Référence publiée |
|---|---|---|
| Proxy HOBBIT : Pearson entre poids du routeur et vraie contribution `‖w·E(x)‖` | **0,888** | 0,99 (Mixtral, papier HOBBIT) |
| Le plus gros poids du routeur = la plus grosse contribution | 67% des cas | — |
| Prefetch : top-8 de la couche L+1 retrouvé depuis la couche L | **83,3%** | 71,6% (Colibrì, GLM-5.2) |
| Localité : experts du token précédent réutilisés | **39,9%** | 41,3% (Colibrì, GLM-5.2) |

**Ce que ça veut dire :**

- **Le poids du routeur est un indicateur d'importance moins fiable sur OLMoE que
  sur Mixtral.** Dans un tiers des cas, l'expert au plus gros poids n'est pas celui
  qui contribue le plus. HOBBIT décide avec ce poids : ses décisions sont donc
  moins justes ici. Explication probable : Mixtral choisit 2 experts sur 8, avec
  un gagnant net ; OLMoE en choisit 8 sur 64, avec des poids beaucoup plus étalés.
- **Le prefetch marche très bien** : 83% en moyenne. Il est faible à l'entrée
  (56% de la couche 0 vers la 1, là où la représentation change le plus) et monte
  à 85-90% dans les couches profondes, où l'état caché évolue peu d'une couche à
  l'autre (connexion résiduelle).
- **La localité temporelle est la même que sur GLM-5.2** (40%) : d'un token à
  l'autre, un peu moins de la moitié des experts sont réutilisés. C'est ce qui
  fait marcher le cache LRFU.

## 2. Le benchmark A/B

8 prompts x 64 tokens gloutons, 2 répétitions en ordre alterné. Qualité : 256
tokens de texte en teacher forcing, comparés au modèle tout en fp16.

| Réglage | tok/s (médiane) | accord top-1 | écart perplexité | KL | fp16 demandé / utilisé | disque/token | Mo lus sur disque / token |
|---|---|---|---|---|---|---|---|
| hi (tout fp16) | 3,68 | 99,6% | -0,15% | 0,001 | 100% / 100% | 36,1 | 455 |
| lo (tout int4) | **6,22** | 88,3% | **+10,4%** | 0,070 | 0% / 0% | 0 | 0 |
| hobbit t1=0,6 | 4,10 | 95,3% | **+2,7%** | 0,016 | 49% / 77% | 15,7 | 197 |
| hobbit + couches 0, 1, 15 en fp16 | 4,26 | 94,1% | +2,7% | 0,014 | 59% / 81% | 18,0 | 227 |

**Qualité :**

- **L'int4 statique coûte cher sur OLMoE** : +10,4% de perplexité et un token sur
  neuf qui change. Pourquoi autant ? Ce n'est pas encore mesuré. Une piste :
  l'erreur de chaque expert s'additionne sur 8 experts par couche et 16 couches.
- **HOBBIT récupère environ les trois quarts de la perte** (+10,4% → +2,7%) en
  demandant le fp16 pour la moitié des experts. C'est la réponse à l'issue #249
  pour ce modèle : la précision dynamique fait nettement mieux que l'int4 uniforme.
  Elle n'atteint pas pour autant le seuil d'acceptation (+1% de perplexité au plus).
- **Protéger les couches du bord n'apporte rien ici** (+2,75% contre +2,71%) : le
  pattern en U mesuré au doc 07 ne se traduit pas par un gain de qualité à t1=0,6.
- Le « bruit » du fp16 lui-même : `hi` n'est pas identique à la référence (99,6%,
  soit 1 token sur 256) parce que l'ordre des additions change selon où sont les
  experts. Le seuil de 99% est donc juste au-dessus du bruit.

**Vitesse :**

- **lo est le plus rapide** : tout le pool int4 tient en RAM (préchargé) et 98% des
  experts sont déjà en VRAM. Aucune lecture disque.
- **hi et hobbit sont limités par le disque et le PCIe** : 757 Mo copiés vers la
  VRAM par token en hi, 654 Mo en hobbit.
- **hobbit ne transfère que 14% de moins que hi**, alors qu'il demande 2 fois moins
  de fp16. En effet, un fp16 déjà en cache sert aussi les demandes int4 (77%
  utilisés pour 49% demandés) : le cache se remplit de fp16 et ne garde pas les
  versions int4, plus légères.

## 3. Ce qui fausse encore ces chiffres

1. **La mémoire de Windows saturait.** Pendant la référence de qualité, Windows a
   refusé d'épingler plus de 7,19 Gio. Ensuite, deux passages ont des TTFT de 30 à
   48 s et des vitesses divisées par 3 (hobbit, 1re répétition : 1,61 tok/s contre
   5,17 à la 2e). Le plus probable : le reste du système se mettait à paginer.
   Correctif : le budget RAM automatique est plafonné à 35% de la RAM totale avec
   un GPU (`PINNED_MAX_FRACTION` dans `plan.py`, 5,5 Gio sur ta machine). Les
   vitesses de ce benchmark sont à refaire.
2. **La qualité de hobbit dépend du contenu du cache.** Elle est mesurée juste
   après les prompts, et un fp16 resté en cache remplace une demande int4. Le
   chiffre mesuré est celui du système réel, pas celui de la règle HOBBIT seule.
3. Un seul texte d'évaluation (256 tokens), 8 prompts, une machine.

## 4. Balayage de t1 (30 septembre)

`--configs hi,lo,hobbit@0.2,hobbit@0.4,hobbit@0.6,hobbit@0.8 --repeats 2`.
Fichiers : `results/benchmark_olmoe-hf_2026-09-30_0923.md` et `.json`.

| Réglage | tok/s (répétition 1 / 2) | accord top-1 | écart perplexité | KL |
|---|---|---|---|---|
| hi | 2,70 / 4,03 | 99,6% | +0,38% | 0,0011 |
| lo | 5,64 / 5,68 | 88,3% | +10,4% | 0,0705 |
| hobbit t1=0,2 | 3,78 / 4,06 | 90,6% | +3,3% | 0,0357 |
| hobbit t1=0,4 | 3,75 / 3,92 | 93,0% | +4,7% | 0,0298 |
| hobbit t1=0,6 | 3,98 / 3,78 | 94,5% | +3,4% | 0,0191 |
| hobbit t1=0,8 | 4,14 / 3,78 | **97,3%** | **+1,5%** | **0,0058** |

**Ce qu'on peut en tirer :**

- **La qualité suit bien t1.** Le KL et l'accord top-1 s'améliorent régulièrement de
  t1=0,2 à 0,8. La perplexité, elle, n'est pas monotone (+3,3% à 0,2 mais +4,7% à
  0,4) : sur 256 tokens, elle est trop bruitée pour départager des réglages
  proches. **Le KL est la mesure fiable ici.**
- **t1=0,8 est le réglage le plus proche de l'acceptation** : +1,5% de perplexité,
  97,3% d'accord, KL divisé par 12 par rapport à l'int4. Il échoue de peu (seuils :
  +1% et 99%).
- ~~**Même t1=0,2 fait beaucoup mieux que l'int4 pur** (+3,3% contre +10,4%).~~
  **Démenti par la mesure propre de la section 5** : c'était l'effet des fp16 laissés en cache. À 0,2,
  seul le premier expert (parfois le deuxième) est demandé en fp16. Garder
  **l'expert principal** de chaque token en fp16 récupère donc déjà l'essentiel.
  Réserve : ces chiffres de qualité incluent les fp16 gratuits pris dans le cache
  (voir plus bas), c'est à confirmer.
- **La vitesse ne dépend presque pas de t1** (3,8 à 4,1 tok/s pour tous les
  hobbit). Le cache se remplit de fp16 de toute façon (demandes fp16, plus
  surclassements), et le coût est dominé par les transferts.

**Ce qui a faussé ce benchmark :**

1. **Budget RAM de 1,87 Gio seulement** (mémoire disponible au lancement : ~4,9 Gio).
   Le pool int4 (3,19 Gio) n'a pas pu être préchargé. Les vitesses ne sont pas
   comparables à celles du 29/09 (6,65 Gio de RAM).
2. **Bug : les compteurs n'étaient pas remis à zéro entre les réglages** quand il
   n'y avait rien à précharger. Les colonnes VRAM hit, fp16 demandé/utilisé et
   disque/token du fichier `.md` cumulent tous les réglages précédents : **elles
   sont fausses** (d'où « lo : fp16 demandé 40% »). Les tok/s, le TTFT et la qualité,
   mesurés séparément, sont justes. Corrigé.
3. **Dérive entre répétitions** : hi passe de 2,70 à 4,03 tok/s entre le début et la
   fin. Le plus probable : le cache de fichiers de Windows. Après une heure de
   benchmark, une partie du store est en RAM côté système, et les « lectures
   disque » deviennent des copies mémoire. L'ordre alterné compense en moyenne,
   mais la dispersion reste de ±20%.

**Corrigé depuis :**

- compteurs remis à zéro à chaque réglage ;
- **qualité mesurée sur la règle seule** : caches vidés et surclassement interdit
  pendant la mesure (`PrecisionPolicy(upgrade=False)`). Le résultat ne dépend plus
  de ce que les prompts ont laissé en cache ;
- avertissement au lancement du benchmark si le budget RAM empêche le préchargement.

## 5. Le balayage propre (30 septembre, 9 h 56) — les chiffres de référence

Mêmes réglages, après les corrections : RAM libérée (10,1 Gio disponibles), budget
fixé (`--ram-gb 5`), compteurs remis à zéro, **qualité de la règle seule** (caches
vidés, pas de fp16 gratuit). Fichiers : `results/benchmark_olmoe-hf_2026-09-30_0956.*`.

| Réglage | tok/s | accord top-1 | écart perplexité | KL | fp16 demandé | Mo lus sur disque / token | Mo copiés vers la VRAM / token |
|---|---|---|---|---|---|---|---|
| hi (tout fp16) | 5,00 | 98,8% | +0,12% | 0,0019 | 100% | 310 | 832 |
| lo (tout int4) | **5,92** | 89,1% | +10,35% | 0,0698 | 0% | 5 | 10 |
| hobbit t1=0,2 | 4,73 | 88,7% | +10,73% | 0,0648 | 16% | 71 | 519 |
| hobbit t1=0,6 | 4,84 | 93,8% | +4,91% | 0,0305 | 49% | 169 | 717 |
| hobbit t1=0,8 | 4,98 | 97,3% | **+0,83%** | 0,0066 | 73% | 231 | 779 |
| hobbit t1=0,9 | 5,06 | **99,2%** | -0,17% | **0,0019** | 87% | 270 | 812 |

### Qualité : une courbe nette

- **La qualité suit t1 de façon régulière** sur les trois mesures (accord, perplexité,
  KL). Le bruit de la veille venait des fp16 gratuits pris dans le cache.
- **t1=0,9 = la qualité du fp16** : son KL (0,0019) est celui de `hi` contre la
  référence, c'est-à-dire le bruit du calcul fp16 lui-même. Avec 13% d'experts en int4.
- **t1=0,8 passe le seuil de perplexité** (+0,83%, sous le +1%) avec 27% d'experts en
  int4 ; seul l'accord top-1 (97,3% contre 99%) le fait rejeter.
- **t1=0,2 ≈ int4 pur** (+10,7%). Garder seulement l'expert principal en fp16 ne
  suffit pas : l'hypothèse de la section 4 était un artefact du cache.
- **Pourquoi HOBBIT a besoin de beaucoup de fp16 sur OLMoE** : le poids du routeur
  prédit mal la contribution réelle (Pearson 0,89, section 1), et 8 experts aux
  poids étalés se partagent chaque token. Il faut un t1 élevé pour ne pas rater les
  experts qui comptent.

### Vitesse : les octets économisés ne se voient pas

Tous les réglages hobbit tournent entre 4,7 et 5,1 tok/s, comme `hi` (5,00). Même
`lo`, qui ne lit rien sur le disque et trouve 97% des experts en VRAM, n'est que 18%
plus rapide. En effet :

| | ms par couche MoE | dont attente disque |
|---|---|---|
| lo (tout en VRAM) | 9,5 | 0 |
| hi | 12,0 | ~3 |

**Le moteur est limité par son propre surcoût, pas par la mémoire.** Même quand tout
est en VRAM, une couche MoE coûte 9,5 ms, soit environ 1,2 ms par expert, pour un
calcul GPU de quelques dizaines de microsecondes. Ce temps n'est pas encore profilé ; les suspects : la boucle Python
sur les experts, la synchronisation GPU → CPU à chaque couche (le routage est lu sur
CPU pour décider des précisions et des transferts) et la décompression int4 refaite
à chaque utilisation. Les transferts et le disque n'ajoutent que ~25%.

HOBBIT réduit bien les octets déplacés : -25% de lectures disque et -6% de copies
vers la VRAM à t1=0,8, par rapport à hi. Mais ces octets ne sont pas le goulot
d'étranglement ici, donc la vitesse ne bouge pas.

**Ce que ça change pour la suite :** sur Qwen3-Next-80B, c'est l'inverse. 38 Gio
d'experts int4 ne tiennent ni en VRAM ni en RAM, et chaque token lira des centaines
d'experts sur le disque. Le goulot sera le disque, et les octets économisés
compteront directement.

### Réponse à l'issue Colibrì #249, pour OLMoE

La précision par token donne une courbe qualité / octets continue et bien réglable,
que l'int4 statique n'offre pas. À t1=0,8 : +0,8% de perplexité au lieu de +10,4%,
avec 27% des experts en int4. Sur ce moteur et ce modèle, qui tiennent presque en
mémoire, ça ne se traduit pas en vitesse. Le gain attendu concerne les
configurations limitées par le disque, celles de Colibrì.

## 6. Pistes suivantes

1. **Qwen3-Next-80B** (doc 15) : le cas limité par le disque, où les octets comptent.
2. **Réduire le surcoût par expert** : grouper les experts d'une couche en un seul
   appel, éviter la synchronisation GPU → CPU à chaque couche, garder en VRAM les
   experts int4 déjà décompressés quand la place le permet.
