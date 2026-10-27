# 19. moelab autopilot : le modèle optimise son propre moteur

## L'idée (05/10)

Tu choisis le modèle MoE qui te plaît ; moelab s'optimise tout seul pour ton PC, au mieux,
sans dégrader la qualité, et explique ce qu'il a trouvé. Le cerveau de l'agent est le modèle
local lui-même (Qwen3-Next-80B sur la RTX 2060) : rien ne passe par internet, rien n'est payé.

C'est l'automatisation de ce qu'on a fait à la main du 29/09 au 05/10 (doc 18) : mesurer,
diagnostiquer, essayer, garder ou rejeter. Qwen3-Next est passé de 3,35 à 6,8 tok/s et de
6,7 à 4,7 s avant le premier token. L'épreuve finale : repartir des réglages du 02/10 et
voir si l'agent retrouve seul ces gains.

## Le principe : le code garantit, le modèle propose

| Le code (déterministe) | Le modèle |
|---|---|
| seuil de bruit (vitesse : ~5 % ; KL : ~0,0001 sur prompt court, 0,002-0,003 sur 64 tokens) | hypothèses (« le prefill attend le fil Python ») |
| garde-fou qualité : un réglage hors du bruit de KL est refusé | choix des expériences |
| répétitions, budget de temps, réglages interdits | lecture des phases d'un profil |
| journal des expériences | rapport final |

Un modèle local peut se tromper : au pire il perd du temps, il ne peut pas rendre le moteur
plus mauvais.

## Le défi : le cerveau et le sujet sont le même moteur

Une carte, un modèle chargé. La boucle alterne donc : **réfléchir** (meilleure config
connue) → **mesurer** (réglages appliqués à chaud dans le même programme, caches vidés ;
déchargement / rechargement pour les budgets mémoire) → **analyser**. Une réflexion de 300
tokens coûte ~45 s, une mesure plusieurs minutes : la lenteur du cerveau compte peu.

## Les étapes

1. **Appels d'outils dans moelab** (fait le 05/10) : `moelab/agent.py`, `scripts/agent.py`.
2. L'outil `mesurer` et la boucle autopilot (fait le 05/10) : `moelab/autopilot.py`,
   `scripts/autopilot.py`. Les outils restent dans le programme ; un serveur MCP viendra
   quand un autre programme devra piloter moelab.
3. L'épreuve : retrouver seul les gains du doc 18 ; puis un autre modèle MoE.
4. L'interface.

## Étape 1 : le modèle appelle des outils

`moelab/agent.py` :

- `@tool` transforme une fonction Python en outil : nom, description et schéma JSON des
  arguments tirés des annotations et de la docstring (section `Args:`).
- `parse_tool_calls` lit les appels au format Qwen / Hermes,
  `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`, tolère une balise non fermée
  (génération coupée) et un bloc ```json ; un appel illisible devient une erreur rendue au
  modèle.
- `Agent` : le modèle répond ; s'il appelle des outils, le code les exécute et lui renvoie
  les résultats ; on recommence jusqu'à une réponse sans appel (`max_steps` tours au plus).
  Outil inconnu, mauvais arguments, exception : un message d'erreur, jamais un plantage.
- `moelab_generate_fn` branche la boucle sur un modèle moelab. Le modèle de chat de Qwen3
  sait décrire les outils ; pour les autres (OLMoE), moelab les décrit lui-même dans le
  message système, au même format. Les messages de l'assistant gardent le texte brut du
  modèle : d'un tour à l'autre le début du prompt est identique et **P1 reprend tout
  l'historique du cache**. Seuls le résultat de l'outil et la suite sont calculés.

`scripts/agent.py` : un chat dont le modèle a 4 outils sur moelab lui-même : `etat`,
`reglages`, `changer_reglage` (réglages à chaud seulement ; ceux qui demandent un
rechargement sont refusés avec une explication), `calculer`.

```powershell
python scripts\agent.py --model D:\models\qwen3-next-80b-fp8 --store C:\moe-store\qwen3-next --ram-gb 5
```

À surveiller : le 1er tour calcule le message système et la description des outils (plusieurs
centaines de tokens, au-delà du seuil de P6), donc nettement plus lent ; les tours suivants
reprennent tout du cache.

### Premier essai (05/10, 11 h) : ça marche

Qwen3-Next appelle le bon outil avec les bons arguments, du premier coup, sur les 3
questions (`etat`, `reglages(groupe="prefill")`, `changer_reglage("t1", "0.7")`), et résume
correctement les réglages rejetés. Trois problèmes visibles :

| Problème | Mesuré | Correction |
|---|---|---|
| 1er tour lent | 33,7 s : message système + outils = 575 tokens, au-delà du seuil de P6 | **P9** : ce début fixe est calculé une fois, gardé en mémoire et sur disque (`<store>/prefix_cache`), rechargé ensuite, même après un redémarrage |
| Tours après un gros résultat d'outil | 14 s (220 tokens repris de `etat`), 22 s (518 tokens de `reglages`) | résultats d'outils plus courts ; seuil de P6 à mesurer au-delà de 64 tokens (`profile_prefill.py --long-tokens 400`, variantes `p6_256`, `p6_512`) |
| Statistiques mal lues | « 29,8 Mo/tok » pour 29,8 lectures disque par token ; statistiques de la lecture du prompt prises pour celles d'une réponse | `etat` renvoie des clés explicites avec leur unité (`experts_lus_sur_disque_par_token`) et précise qu'il compte la question en cours, prompt compris |

Découverte au passage : `etat` montrait « 1091 tampons hors pool ». Pendant un long prefill,
le pool de RAM épinglée est épuisé et des tampons ordinaires prennent le relais (copies
vers la VRAM plus lentes). Une piste pour l'autopilot lui-même.

### P9 : le début fixe d'un prompt gardé sur disque

`GraphDecoder.prefill(ids, reuse=True, prefix_len=n)` : les n premiers tokens (message
système + outils, trouvés par `agent.prefix_length`) sont calculés à part, puis leurs K/V
(attention) et états (attention linéaire) sont gardés sur CPU et écrits dans
`<store>/prefix_cache/prefix_<hash>.pt` (~90 Mo pour Qwen3-Next et 575 tokens). La clé
contient le store, la politique et le modèle : changer de politique recalcule. Au prochain
prompt qui commence pareil (nouvelle question après `/reset`, ou nouveau lancement), le
début est rechargé au lieu d'être recalculé. Exact (test : mêmes logits, OLMoE et Qwen3-Next).

### Deuxième essai (05/10, 11 h 40)

| Tour | 1er token | Lecture |
|---|---|---|
| 1er lancement, 1re question (système + outils calculés et gardés) | 47,5 s | 581 tokens, au-delà de P6 |
| 2e lancement, 1re question (début fixe **rechargé du disque**, 563 tokens) | 14,8 s | seulement 27 tokens à calculer : on attendait ~4 s |
| 2e lancement, question suivante (27 tokens nouveaux) | **4,6 s** | ce qu'on attendait |
| après un résultat d'outil de ~200 tokens | 15 s | au-delà du seuil de P6 (64 à ce moment) |

- P9 marche (« début fixe rechargé », 563 tokens repris).
- Les 14,8 s du 1er message après un lancement : même prompt court que les 4,6 s suivants.
  Hypothèse : la 1re lecture de prompt du programme compile les noyaux Triton et
  initialise cuBLAS. D'où un **échauffement au chargement** (`generate.warmup`, dans
  `chat.py` et `agent.py`, `--no-warmup` pour s'en passer) : le temps passe dans
  « Chargement », pas dans la 1re réponse. À vérifier.
- Résultats d'outils de ~200 tokens : le seuil de P6 passe de 64 à 192 (doc 18).
- Qualité des réponses : le modèle lit bien les nouvelles clés (lectures disque **par
  token**) et signale de lui-même les tampons hors pool.

### Vérifié (05/10, 11 h 50)

Échauffement au chargement : 19,2 s, et le 1er message passe de **14,8 à 3,9 s** (début
fixe rechargé, 17 tokens à calculer). Le tour qui suit un résultat d'outil de 192 tokens
reste à 15,4 s : ~250 experts distincts par couche, presque tous lus sur le disque. Ce
n'est plus le calcul mais le disque : la piste est de garder les résultats d'outils courts.

## Étape 2 : la boucle autopilot (05/10)

`moelab/autopilot.py` :

- **`Bench`** : une mesure = caches vidés et préchargés comme au démarrage, un prompt
  d'échauffement, puis 3 prompts : lecture chronométrée, puis 24 tokens imposés (ceux de la
  référence) chronométrés, dont les probabilités sont comparées à la référence (KL, top-1).
  ~1 min sur Qwen3-Next. La référence est la configuration de départ.
- **Score** : durée d'une réponse type = 1er token + 200 tokens / (tok/s), en secondes.
- **Espace de recherche** : les réglages qui changent à chaud, avec leurs valeurs permises :
  `read_parts`, `prefill_grouped`, `cache_eviction`, `prefetch`, `prefetch_depth`,
  `prefetch_mass`, `prefetch_disk_ahead`, `full_graph`, `prefill_skip`, `ram_gb`,
  `vram_gb` (borné par la VRAM libre), et `policy` / `t1` si le store a deux précisions.
  Depuis le 05/10, `vram_gb`, `ram_gb` et `read_parts` changent sans recharger le modèle
  (`MoERuntime.set_budgets`).
- **Garde-fous (le code)** : valeur hors de l'espace, configuration déjà mesurée, budget
  d'essais ou de temps épuisé : refusé sans mesurer. KL > 0,005 ou top-1 < 90 % : refusé
  (qualité). Gain < 5 % : « pas mieux ». Gain ≥ 5 % : 2e mesure ; gardé seulement si elle
  gagne encore au moins 2,5 %. Une mesure qui plante (mémoire GPU) : « échec ». Après chaque
  essai, la meilleure configuration est remise avant que le modèle reprenne la parole.
- **Outils du modèle** : `machine`, `espace`, `mesurer(changements)`, `journal`,
  `terminer(rapport)`. `terminer` arrête la boucle ; un modèle qui s'arrête sans terminer
  est relancé (3 fois au plus).
- **Sorties** : `<store>/autopilot_profile.json` (appliqué au démarrage par `chat.py` et
  `agent.py`, `--no-profile` pour s'en passer, les valeurs données en ligne de commande
  priment) et `results/autopilot_<modèle>_<date>.md` (rapport du modèle, journal).

```powershell
# optimiser à partir des réglages actuels (écrit le profil si un gain est gardé)
python scripts\autopilot.py --model D:\models\qwen3-next-80b-fp8 --store C:\moe-store\qwen3-next --ram-gb 5
# l'épreuve : départ = réglages du 02/10, effets déjà mesurés cachés au modèle, pas de profil écrit
python scripts\autopilot.py --model D:\models\qwen3-next-80b-fp8 --store C:\moe-store\qwen3-next --ram-gb 5 --start 02-10 --blind
```

### L'épreuve

Départ « 02/10 » (`start_0210`) : `read_parts` 1 au lieu de 4, `prefill_grouped` 0 (sans
P6), budget VRAM réduit de 0,27 Gio (réserve 1,0 au lieu de 0,74). C'est ce qu'on a corrigé
à la main du 03 au 05/10 : lectures découpées (+59 % au profil), réserve VRAM (+15 %), P6
(1er token −37 %). L'éviction en tas, un changement de code, reste active : le départ sera
plus rapide qu'au vrai 02/10. Avec `--blind`, `espace` ne montre pas au modèle l'effet déjà
mesuré des réglages : il doit le découvrir. Réussite = retrouver les trois corrections, ou
mieux.

### Premier essai de l'épreuve (05/10, 12 h 20)

`results/autopilot_qwen3-next-80b-fp8_2026-10-05_1220.md` :

| essai | changement (choisi par le modèle) | réponse type | tok/s | 1er token | verdict (code) |
|---|---|---|---|---|---|
| 0 | départ 02/10 | 53,1 s | 4,29 | 6,58 s | |
| 1 | `read_parts` 1 → 2 | **34,2 s** | 7,06 | 5,85 s | GARDÉ, +35,7 % (confirmé : 34,1) |
| 2 | `prefetch_depth` 1 → 2 | 44,3 s | 5,20 | 5,91 s | pas mieux (−29,7 %) |

Ce qui marche : à l'aveugle, le modèle a lu `machine` et `espace`, formulé une hypothèse
juste (« le SSD ne donne son débit qu'avec plusieurs lectures en vol ») et trouvé du
premier coup la plus grosse correction de la semaine. La mesure est stable (34,2 puis
34,1 s) et le KL nul. Il a aussi redécouvert seul que `prefetch_depth` 2 nuit (C2, rejeté
sur OLMoE le 30/09).

Ce qui a cassé : au 3e tour, il a écrit « Mesurons. » sans appeler l'outil, et la boucle
s'est arrêtée (6 min, 2 essais sur 12). Corrections :

- **Relances** : jusqu'à 5, avec un message explicite (« tu n'as appelé aucun outil ; si
  tu as annoncé une mesure, appelle mesurer »), affichées dans le terminal et comptées
  dans le rapport. Le message système impose qu'un message se termine par un appel d'outil.
- **La conversation du cerveau survit aux mesures** : `Bench` utilise le même décodeur et
  écrasait son cache ; chaque tour suivant recalculait toute la conversation (15-25 s).
  `GraphDecoder.export_state()` / `import_state()` la copient sur CPU avant la mesure et la
  remettent après : seul le résultat de l'outil est calculé.

### Deuxième essai de l'épreuve (05/10, 12 h 58) : 11 essais, terminé par le modèle

`results/autopilot_qwen3-next-80b-fp8_2026-10-05_1258.md` — 17,5 min, aucune relance,
rapport écrit par le modèle. Réponse type 52,8 → 34,2 s (+35 %).

| essai | changement | réponse type | tok/s | 1er token | verdict |
|---|---|---|---|---|---|
| 1 | `read_parts` 2 | 34,5 s | 7,02 | 6,00 s | GARDÉ (+35 %) |
| 2 | `prefetch_depth` 2 | 44,3 | 5,22 | 5,99 | pas mieux |
| 3 | `prefetch_mass` 0,8 | 52,0 | 4,34 | 5,89 | pas mieux |
| 4 | `cache_eviction` stale | 71,0 | 3,19 | 8,33 | pas mieux |
| 5 | `full_graph` off | 33,9 | 7,11 | 5,72 | pas mieux (+0,9 %) |
| 6 | `prefetch` off | 34,9 | 6,89 | 5,83 | pas mieux |
| 7 | `ram_gb` 3 | 39,3 | 6,05 | 6,18 | pas mieux |
| 8 | `vram_gb` 1,74 | 35,1 | 6,88 | 6,02 | pas mieux |
| 9 | `read_parts` 4 | 35,9 | 6,71 | 6,06 | pas mieux (−5 %) |
| 10 | `prefill_grouped` 64 | 32,6 | 6,95 | **3,86** | **refusé à tort** (+4,7 %) |
| 11 | `prefill_skip` 0,05 | 33,1 | 7,22 | 5,42 | pas mieux (+3,2 %) |

Le modèle a bien joué : un changement à la fois, des hypothèses lisibles, il redécouvre
seul ce qu'on avait rejeté à la main (C2, S3 : `prefetch_mass` et `stale`), il signale une
incohérence (« max 1,74 mais actuel 1,84 → anomalie ») et s'arrête de lui-même avec un
rapport. `read_parts` 2 vaut 4 sur ce disque (34,2 contre 35,9 s) : une vraie découverte,
qu'on n'avait pas mesurée.

Les deux corrections manquées sont des **défauts du code**, pas du modèle :

- **P6 refusé** : le 1er token passe de 6,0 à 3,86 s (−36 %), mais il pèse peu dans la
  réponse type (200 tokens à ~7 tok/s = 28 s). Nouvelle règle (`Autopilot._better`) : gardé si
  la réponse type gagne 5 %, **ou** si le 1er token gagne 10 % (ou les tok/s 5 %) sans que
  l'autre perde plus que le bruit ; la 2e mesure doit confirmer à moitié de ces seuils. Le
  modèle reçoit aussi les écarts en % pour chacun des trois chiffres.
- **VRAM bornée trop bas** : le maximum ne comptait pas la VRAM gardée par l'allocateur de
  PyTorch après la baisse de budget du départ (elle n'est pas rendue au pilote), et gardait
  0,4 Gio de marge. Il compte maintenant `memory_reserved - memory_allocated`, marge 0,25 Gio.

### Troisième essai (05/10, 13 h 26) : les mesures dérivent

`results/autopilot_qwen3-next-80b-fp8_2026-10-05_1326.md` — 10 essais, 20 min, réponse type
52,7 → 31,0 s. Gardés : `read_parts` 2, `prefill_skip` 0,05 (1er token +11 %), puis
`vram_gb` 3,0. Le modèle va au bout et conclut seul, mais les chiffres ne tiennent pas :

- **Dérive** : des réglages qui ne touchent pas la génération la font varier de 4,9 à 7,7
  tok/s (`prefill_grouped` 128 : tok/s −30 % alors que P6 n'agit que sur le prompt).
  Comparer un candidat à une mesure de la meilleure config faite 10 minutes plus tôt ne
  vaut rien quand la machine change d'état (disque qui chauffe, Windows, mémoire). Ainsi
  P6 (1er token +37 %) a encore été refusé, et `vram_gb` 2,5 jugé −24 %.
- **`vram_gb` 3,0 sur une carte de 6 Go** : le plafond comptait deux fois la VRAM rendue à
  l'allocateur de PyTorch quand les caches sont vidés. Sous Windows, le pilote NVIDIA
  déborde alors en silence dans la RAM (« mémoire GPU partagée ») : ça tourne, mais ce
  n'est plus la même machine.

Corrections :

- **Mesures appariées** : dès qu'un candidat donne un résultat net (gain, ou écart de plus
  de 5 % dans un sens ou l'autre), la meilleure config est **remesurée juste à côté**, et le
  candidat est jugé contre cette mesure fraîche. La dérive est signalée au modèle
  (`derive_pourcent`) et la référence suit les conditions du moment. Un gain doit ensuite se
  répéter à une 2e mesure du candidat.
- **Vérification finale** : départ et meilleure config mesurés l'un après l'autre à la fin ;
  c'est ce gain-là que le rapport annonce.
- **Plafond VRAM** = VRAM totale − ce qui n'est pas à PyTorch − ce que PyTorch utilise hors
  experts − 0,55 Gio d'activations. Plus de débordement dans la RAM.
- « déjà celles de la meilleure configuration » au lieu de « déjà mesurée » quand le modèle
  redemande les valeurs actuelles (4 appels perdus ainsi).

### Quatrième essai (05/10, 14 h 31) : épreuve réussie

`results/autopilot_qwen3-next-80b-fp8_2026-10-05_1431.md` — 11 essais, 24 min, une seule
dérive détectée (5,5 %), conclu par le modèle.

**Vérification finale, côte à côte : réponse type 52,1 → 32,3 s (+38 %), 4,38 → 7,01 tok/s,
1er token 6,51 → 3,80 s.**

| correction de la semaine | retrouvée ? |
|---|---|
| lectures découpées (`read_parts`) | oui, essai 1 (+33 %). Et mieux : 2 morceaux battent 4 sur ce disque (−9,9 % pour 4, mesurés côte à côte) |
| P6 (`prefill_grouped`) | oui, essai 9 : 1er token −37 % ; 128 n'apporte rien de plus que 64 |
| plus de VRAM pour les experts | essayée au plafond (2,06 Gio) : +2 %, dans le bruit. Cohérent avec la mesure à la main (+15 % pour 1,84 → 2,24 Gio, au-delà du plafond sûr) |

Il redécouvre aussi seul que `prefetch_depth` 2, `prefetch_mass` 0,8, `stale` et une RAM
réduite font perdre, et que `full_graph` et `prefill_skip` 0,05 sont dans le bruit.
Le résultat égale le réglage fait à la main en une semaine (6,8 tok/s, 1er token 4,7 s au
benchmark), en 24 minutes, sans aide.

Découverte : `read_parts` 2 pourrait être meilleur que le 4 choisi par
`measure_bandwidth.py`. À confirmer par un autopilot parti des réglages actuels (qui, lui,
écrit le profil).

### Premier vrai run, depuis les réglages actuels (05/10, 15 h 01)

`results/autopilot_qwen3-next-80b-fp8_2026-10-05_1501.md` — 11 essais, 20 min, rien de
gardé : **les réglages actuels sont au plus haut** (réponse type 32,8 s, 6,94 tok/s, 1er token
3,95 s). Tous les essais perdent ou restent dans le bruit (`read_parts` 8 −12 %,
`prefetch_disk_ahead` 1, `prefill_grouped` 256, `ram_gb` 5,5, `prefill_skip` 0,02...).

Deux remarques :

- Il n'a pas essayé `read_parts` 2 (meilleur que 4 dans l'épreuve) : l'effet déjà mesuré
  de 1 → 4, qu'il voyait cette fois (pas `--blind`), l'a orienté vers 8. La question 2 ou 4
  reste ouverte.
- Le plafond VRAM (1,96 Gio) était sous le budget en cours (2,11), qui tourne très bien : le
  modèle s'en est étonné. Le plafond ne descend plus sous le budget du chargement ni sous le
  budget en cours.
