# Journal du projet

Format : le plus récent en haut.

## 2026-10-13 (8) — REAP en une commande, résultats à côté du store

- `scripts/reap_pipeline.py` : référence, calibration, essai, élagages, tableau ; reprenable.
- Résultats dans `<store>\reap` (un ancien `results\reap` y est recopié) : une nouvelle version du repo ne
  les efface plus.

## 2026-10-13 (7) — REAP : référence mesurée, élagage virtuel

- Référence : 5,1 tok/s, 142,6 lectures disque/token, outils 20/20, perplexité 6,91 (WikiText) / 4,37 (FR).
- Essai de calibration : 10 échantillons en 1,6 min (~35 min pour 256).
- `scripts/prune_eval.py` : masque −inf sur les logits du routeur avant le top-k (experts jamais choisis,
  prédits ni chargés), même mesure que la référence, comparaison affichée.

## 2026-10-13 (6) — Chat : contrôle des faits à l'écriture, dates des pages avec leur texte

- 9 tours, fichier écrit ; dates fausses de quelques jours.
- Écriture refusée une fois si ses dates ou chiffres ne sont dans aucune page lue (comme les agents).
- Dates des balises `time` : texte affiché + date complète (« 02 Sep (2026-09-02) »).
- Chat avec outils : réponses jusqu'à 1 536 tokens ; consigne de ne pas recopier le contenu.

## 2026-10-13 (5) — Chat : le fichier demandé est écrit, dates complètes des pages

- Contenu juste mais recopié dans la réponse : un bloc unique pour un fichier demandé et non écrit est écrit
  par moelab (mêmes contrôles, même accord).
- `lire_page` garde les dates complètes des balises `time` (GitHub : « 08 Jul » -> 2026-07-08).
- `chercher` : plusieurs termes en un appel, séparés par `|`.

## 2026-10-13 (4) — Chat : archive des résultats, recherche dans une page jamais vide

- Chat de 17 tours sans fichier : titres devinés absents des pages (9 tours), dates perdues au nettoyage,
  refus d'écrire au dernier tour.
- `chercher` : mots un par un si l'expression manque, sinon le début de la page.
- Archive [rN] et `relire` dans le chat, gardées d'un message à l'autre ; écrire un résultat partiel.

## 2026-10-13 (3) — Chat avec outils : plus de « CUDA out of memory »

- Huit pages lues dans le chat : contexte au-delà du cache statique, mémoire GPU pleine.
- Même garde que les agents : nettoyage à 80 %, filet de sécurité, un nouvel essai après avoir libéré la
  mémoire ; la conversation enregistrée garde les résultats complets.

## 2026-10-13 (2) — Dates du futur contrôlées à l'écriture

- Chat avec outils : 10 tours, ~10 min, fichier écrit ; mais 2.15 (prévue le 28/10) comptée parmi les
  « dernières versions ».
- Écriture de fichier avec une date postérieure à aujourd'hui : refusée une fois, avec la question ; réécrite
  telle quelle, elle passe. Règle de date : « dernier » = déjà paru.

## 2026-10-13 — Agents : critères sans exigences inventées ; chat qui conclut

- Run agents de 62,5 min terminé en « Échec » avec un fichier juste : critères inventés (« GitHub seulement,
  pas de blog »), fausses absences du vérificateur, orchestrateur sans lire_page.
- Critères : seulement ce que l'utilisateur demande. Vérificateur : « n'apparaît pas » rejeté si le fait est
  dans les sources. Orchestrateur : outils de lecture (pages, recherche, calcul).
- Chat avec outils : 16 tours au lieu de 8, puis un dernier tour pour conclure ; consignes d'économie de tours.

## 2026-10-08 (6) — Agents : moins de générations perdues

- Run de 67,9 min, résultat juste : la tâche en cours d'un agent était effacée au 2e nettoyage (17 min
  perdues) ; le rédacteur n'avait pas lire_fichier ; un rédacteur créé pour recopier.
- Tâche en cours protégée du nettoyage ; outils de lecture accordés au premier appel ; l'orchestrateur
  écrit lui-même un fichier dont il a le contenu ; contenu jamais recopié dans le texte.
- Vérification sautée (limite de temps) affichée comme telle.

## 2026-10-08 (5) — Agents : le vrai problème du vérificateur n'est plus jeté

- Run de 68,7 min : 2.14.0 manquée, l'hypothèse de l'orchestrateur (« la version avant 2.12.1 ») propagée
  aux agents ; le contrôle final avait trouvé 2.14.0 mais sans citation au bon format : ignoré.
- Remarque gardée si elle apporte des faits lus dans les sources et absents du livrable.
- Sources du vérificateur = résultats d'outils seulement ; nom « verificateur » réservé au contrôle final.
- Consignes : redonner la demande d'origine sans hypothèse ; signaler ce qui contredit la tâche.

## 2026-10-08 (4) — Élagage REAP : banc d'évaluation et calibration (doc 22)

- `scripts/eval_ppl.py` : vitesse, 20 questions d'appels d'outils notées par le code, perplexité
  WikiText-2 + Wikipédia FR ; jeu figé sur disque, top-1 gardé pour comparer les essais suivants.
- `scripts/calibrate_reap.py` : score REAP (g · ||f||) de chaque expert, par lots de 8 échantillons,
  reprise automatique, résumé du coût de 25 / 40 / 50 % d'élagage.
- Calibration mélangée : texte, français, code, maths, conversation, et les conversations et runs
  d'agents de l'app.
- Runtime : crochet `expert_hook` (sortie de chaque expert), inactif hors calibration.

## 2026-10-08 (3) — Vérificateur : livrable réel, citations contrôlées

- Run de 62 min : faux problèmes du vérificateur (il ne voyait pas le fichier), orchestrateur perdu.
- Le vérificateur reçoit la réponse et les fichiers écrits, sans bloc-notes ; chaque problème cite
  le livrable ou une source, citation vérifiée par le code (sinon ignoré).
- 4 tours de correction au plus, puis la dernière réponse proposée avec les remarques.
- Comptes rendus du bloc-notes coupés en fin de phrase, texte complet par `relire`.

## 2026-10-08 (2) — Agents : critères, vérificateur final, leçons

- Run du 08/10 : faits justes, 22,5 min, 82 % du prompt repris du cache ; 2.14 oubliée (compréhension).
- Critères de réussite au départ, corrigeables dans l'écran Agents (90 s puis le run continue).
- Vérificateur indépendant, une seule fois avant la réponse finale (auto / toujours / jamais).
- Mémoire de leçons : « Corriger » une réponse ou garder un problème du vérificateur ; leçons proches
  réutilisées aux runs suivants.

## 2026-10-08 — Agents : contexte optimisé (DeepSeek, MiMo)

- Repli par tâche ; grand nettoyage à 80 % (une coupe, puis la conversation ne fait que s'allonger) ;
  archive des longs résultats et outil `relire`.
- Même message système et mêmes outils pour tous les sous-agents (un seul cache de préfixe) ; rôle et
  outils permis dans le 1er message ; outil non reçu refusé sans s'exécuter.
- Plusieurs appels par message ; `lire_fichier(chercher=…)` ; statistiques de contexte du run.
- Pas encore : cache int8, décodage spéculatif (à mesurer sur la carte, doc 21).

## 2026-10-06 (8) — Agents : contexte sous contrôle, outils compagnons

- 7e run : vérification des faits efficace, mais `chercher_web` sans `lire_page`, contexte au-delà
  du cache statique (« CUDA out of memory »), prompt affiché après l'erreur.
- `compact_messages` : chaque prompt tient dans le cache statique ; bloc-notes joint limité ;
  erreur mémoire -> on libère, on raccourcit, on réessaie une fois.
- Outils compagnons (`chercher_web` -> `lire_page`) ; affichage remis à zéro avant chaque génération.

## 2026-10-06 (7) — Agents : vérification automatique des faits

- 6e run : dates 2026 mais fausses (jour, mois), nouveautés inventées par le rédacteur, fichier
  « vérifié » par celui qui l'avait écrit.
- `factcheck.py` : dates et chiffres comparés à ce qui a été lu ; compte rendu, fichier et réponse
  finale refusés une fois, puis signalés.
- Comptes rendus ajoutés au bloc-notes ; l'orchestrateur relit lui-même les fichiers écrits.
- `lire_page` : `chercher` (passages autour d'un mot) et `debut` (morceaux de 4 000 caractères).

## 2026-10-06 (6) — Agents : dernier tour pour rendre, erreurs qui guident

- 5e run : dates en 2026 et sourcées, mais le chercheur a épuisé ses tours sans rendre.
- Dernier tour forcé (`rendre` seul) ; notes de l'agent jointes si pas de compte rendu ;
  10 tours par tâche par défaut.
- Outil des agents appelé par l'orchestrateur : l'erreur dit à qui confier ; « existe déjà »
  donne les outils de l'agent.

## 2026-10-06 (5) — Agents : date du jour, règle des sources

- 4e run (recherche web) : dates de PyTorch 2.12-2.14 inventées (2024 au lieu de 2026).
- Date du jour dans les consignes (orchestrateur, agents, chat) ; règle des sources pour les
  agents (« non trouvé » plutôt qu'inventer) ; l'orchestrateur fait vérifier.
- Appel au nom tronqué (`rende` + JSON) lu si les clés désignent un seul outil.
- Cache de préfixe limité aux 24 plus récents.

## 2026-10-06 (4) — Agents : 1er run réussi, appels style Python, ajouter_fichier

- 3e vrai run réussi : `resume.md` juste pour les 3 fichiers, ~15 min.
- Appels `nom(arg="...")` lus (ast, valeurs littérales seulement).
- Outil `ajouter_fichier` : écrire à la fin d'un fichier sans le relire.
- Corrigé (vu par le test lancé sous Windows) : un chemin `C:\...` dans un appel style Python
  faisait garder des « \n » littéraux dans le contenu ; seuls les chemins restent bruts.

## 2026-10-06 (3) — Agents : 2e run, dossiers autorisés

- Appels `nom("texte")` reconnus (chaîne -> 1er argument obligatoire).
- Dossiers autorisés dans les consignes de l'orchestrateur et des agents ; aucun dossier ->
  terminer tout de suite. Avertissement sur l'écran Agents.

## 2026-10-06 (2) — Agents : 1er vrai run, trois défauts corrigés

- Appels écrits sans balises (`terminer({...})`) reconnus pour les outils connus (agents, chat,
  autopilot) ; format exact rappelé dans les consignes et la relance.
- `lire_fichier` sur un dossier : message clair (sous Windows, `PermissionError` était pris pour
  un refus) ; agents de fichiers toujours équipés de `lister_dossier` et `chercher_fichiers`.
- L'orchestrateur peut regarder (lecture seule) pour écrire des tâches précises ; une erreur
  d'arguments liste les arguments attendus.

## 2026-10-06 — Agents : un orchestrateur qui crée ses propres agents

- Écran Agents (doc 21) : un objectif ; l'orchestrateur crée des agents (rôle + outils permis),
  leur confie des tâches, assemble les comptes rendus. Un seul niveau, limites fixées par le
  code, relances, accords pour les outils « demander ».
- Un seul modèle : les agents parlent l'un après l'autre ; la conversation de chacun est mise
  de côté puis remise (pas de recalcul). Le chat passe entre deux tours (`serve_waiting`).
- Bibliothèque d'agents (garder un agent créé pendant un run), runs enregistrés et relisibles.
- Nouveaux outils : `chercher_web` (DuckDuckGo), `executer_python` (accord à chaque appel).

## 2026-10-05 (11) — L'app : six écrans, un seul modèle, une file d'attente

- `python scripts\app.py` : Chat, Mise en route, Autopilot, Outils et MCP, Moteur, Tâches,
  servis sur 127.0.0.1 (doc 20). Un seul fil de travail à priorités (chat avant tâches de
  fond, tâche de fond découpée par fichier).
- Mise en route : plan de place par précision et par disque, téléchargement, store, mesure
  du disque, bibliothèque. Store autonome (`make_standalone.py`, par défaut dans
  `build_store.py`) : le checkpoint téléchargé devient supprimable.
- Outils avec permissions et dossiers autorisés ; moelab client et serveur MCP.
- Qwen3-30B-A3B (Qwen3-MoE) pris en charge sans adaptateur (testé sur miniature).
- Vérifié dans Chromium sur un modèle miniature ; corrigés : tokens `<pad>` dans les
  conversations enregistrées, message d'un réglage de politique qui ne disait pas quoi.

## 2026-10-05 (10) — 1er vrai run : les réglages actuels sont au plus haut

- 11 essais, rien de gardé (32,8 s par réponse type, 6,94 tok/s, 1er token 3,95 s).
- Plafond VRAM jamais sous le budget du chargement ni sous le budget en cours.

## 2026-10-05 (9) — Épreuve réussie : l'autopilot égale une semaine de réglages à la main

- À l'aveugle, depuis les réglages du 02/10 : `read_parts` 2 et P6 retrouvés, VRAM essayée au
  plafond (dans le bruit). Vérification côte à côte : 52,1 → 32,3 s par réponse type (+38 %),
  4,38 → 7,01 tok/s, 1er token 6,51 → 3,80 s. 24 min, une dérive détectée.
- `read_parts` 2 bat 4 sur ce disque (−9,9 % pour 4) : à confirmer par un autopilot « actuel ».

## 2026-10-05 (8) — Autopilot : mesures appariées, vérification finale, plafond VRAM juste

- 3e essai : même config entre 4,9 et 7,7 tok/s selon le moment ; `vram_gb` 3,0 accepté sur
  une carte de 6 Go (débordement dans la RAM sous Windows).
- Dès qu'un résultat est net, la meilleure config est remesurée à côté (dérive signalée,
  référence mise à jour) ; vérification finale départ / meilleur côte à côte ; plafond VRAM
  calculé sur la VRAM réellement disponible pour les experts. Test de dérive ajouté.

## 2026-10-05 (7) — Épreuve, 2e essai : 11 essais menés et conclus par le modèle

- `read_parts` 2 gardé (+35 %, vaut 4 sur ce disque) ; C2, `prefetch_mass`, `stale` redécouverts
  comme mauvais ; rapport final écrit par le modèle ; aucune relance.
- Défauts du code corrigés : P6 refusé (1er token −36 % noyé dans la réponse type) → règle de
  décision à deux chiffres (`_better`) ; plafond VRAM qui oubliait la VRAM gardée par PyTorch.

## 2026-10-05 (6) — Épreuve, 1er essai : read_parts retrouvé à l'aveugle (+35,7 %)

- Le modèle trouve seul `read_parts` 2 (53,1 → 34,2 s par réponse type) et rejette
  `prefetch_depth` 2, puis s'arrête en écrivant « Mesurons. » sans appeler l'outil.
- Relances (5 au plus, message explicite, comptées) ; règle « un appel d'outil par message ».
- `GraphDecoder.export_state` / `import_state` : la conversation du cerveau survit aux mesures.
- Tests ajoutés (3).

## 2026-10-05 (5) — Autopilot, étape 2 : la boucle

- `moelab/autopilot.py` : `Bench` (mesure ~1 min, KL contre la référence), score « réponse
  type » (1er token + 200 tokens), espace de recherche, garde-fous (qualité, bruit avec
  2e mesure, budget, échec), outils `machine`, `espace`, `mesurer`, `journal`, `terminer`,
  profil machine et rapport.
- `scripts/autopilot.py` ; `--start 02-10 --blind` pour l'épreuve.
- `vram_gb`, `ram_gb`, `read_parts` changent à chaud (`MoERuntime.set_budgets`).
- `chat.py` et `agent.py` appliquent `<store>/autopilot_profile.json` (`--no-profile`).
- `tests/test_autopilot.py` (5).

## 2026-10-05 (4) — P6 jusqu'à 192 tokens ; échauffement au chargement

- Seuil de P6 mesuré : 93 tokens 14,3 → 10,5 s ; 407 tokens 22,5 → 27,9 s (perd). Défaut 64 → 192.
- P9 vérifié : début fixe rechargé du disque au 2e lancement. Mais 1er message après un
  lancement à 14,8 s (4,6 s ensuite) : `generate.warmup` compile les noyaux et capture les
  graphs pendant le chargement (`chat.py`, `agent.py`, `--no-warmup`). Vérifié : 1er message
  14,8 → 3,9 s, échauffement 19,2 s au chargement.

## 2026-10-05 (3) — Agent : 1er essai réussi ; P9, début fixe du prompt gardé sur disque

- Qwen3-Next appelle les bons outils du premier coup. 1er tour 33,7 s (575 tokens de système
  et d'outils) : **P9** les calcule une fois et les garde (mémoire + `<store>/prefix_cache`).
- `etat` et `reglages` : sorties plus courtes, clés avec unités (le modèle avait lu des
  lectures par token comme des Mo).
- `profile_prefill.py --long-tokens 400`, variantes `p6_256` et `p6_512` (seuil de P6).

## 2026-10-05 (2) — Autopilot, étape 1 : le modèle local appelle des outils

- Projet (doc 19) : moelab s'optimise tout seul, avec le modèle local comme cerveau.
- `moelab/agent.py` : outils depuis des fonctions Python, appels `<tool_call>` lus dans le
  texte, boucle robuste, description des outils par le modèle de chat ou par moelab.
- `scripts/agent.py` : chat avec 4 outils sur moelab (etat, reglages, changer_reglage, calculer).
- `tests/test_agent.py` (6).

## 2026-10-05 — P6 activé : premier token 5,95 → 3,76 s ; le prefill est limité par le disque

- **P6** (`prefill_grouped`, 64 tokens, activé) : noyaux groupés token par token au prefill.
  Qwen3-Next : 5,95 → 3,76 s (17 tokens), 11,66 → 7,90 s (64), disque à 1,20 Go/s (son
  maximum), exact. Benchmark : TTFT 6,71 → 4,70 s, génération 6,80 tok/s (inchangée).
- P7 (5,95 → 6,31 s) et P8 (→ 12,2 s, génération ÷2) rejetés.
- Seule piste restante : lire moins (P5, seuil 0,05 ou 0,10), à décider.
- `profile_prefill.py` : variantes par défaut `defaut,sans_p6,skip05,skip10` ; `p6_256`.

## 2026-10-03 (6) — Prefill : mesuré ; P6 à P8 pour alléger le fil principal

- Mesuré : P1 en conversation, 1er token du 3e message 14,0 → 6,0 s. Le prefill n'attend
  presque pas le disque (0,1 s sur 5,8) : c'est le fil principal, ralenti par les fils de
  lecture (GIL), qui limite. P2, P3, P4 rejetés ; P5 à 0,10 : −40 % mais KL ×4.
- P6 (`prefill_grouped`), P7 (`prefill_split`), P8 (`gil_switch_ms`) : à mesurer avec
  `profile_prefill.py` (nouvelles variantes par défaut). Tests ajoutés.

## 2026-10-03 (5) — Prefill : la conversation n'est plus recalculée à chaque message

- **P1** (`kv_reuse`, activé) : `chat.py` recalculait toute la conversation à chaque message.
  Le cache garde maintenant l'historique : seuls les nouveaux tokens passent dans le modèle.
  Exact (testé sur OLMoE et Qwen3-Next miniatures, y compris historique modifié).
- **P0** : copie O(n²) de `protect` à chaque insertion en VRAM, coûteuse au prefill. Corrigé.
- P2 à P5 (`prefill_order`, `prefill_prefetch`, `prefill_cache`, `prefill_skip`) : options
  désactivées, à mesurer avec `scripts/profile_prefill.py` (nouveau).
- `chat.py` : tok/s de génération seule, temps avant le premier token, tokens repris du cache.
  Tests : `tests/test_prefill.py` (17).

## 2026-10-03 (4) — Réserve VRAM et longueur max automatiques ; budgets mesurés

- Benchmarks Qwen3-Next `lo` : RAM 4 → 6 Gio, lectures disque 140 → 109 par token mais
  vitesse dans le bruit (6,03 / 5,93 / 6,38) ; réserve VRAM 1,0 → 0,6 Gio : **6,81 tok/s**
  (+15 %, hit 58 → 61 %).
- `--vram-reserve-gb` : défaut automatique, 0,55 Gio + cache KV statique (OLMoE 0,80,
  Qwen3-Next 0,74 Gio). `--graph-max-len` : défaut automatique, cache KV ≤ 256 Mio
  (OLMoE 2 048, Qwen3-Next 8 192 tokens). `plan.py` suit les mêmes règles. Test ajouté.
- Vérifié sur la RTX 2060 : Qwen3-Next 6,76 tok/s avec 8 192 tokens de cache ; OLMoE `lo`
  116,2 tok/s, `hobbit` 29,4, pas de manque de VRAM avec la réserve de 0,80 Gio.

## 2026-10-03 (3) — Lectures découpées : Qwen3-Next 6,82 → 10,85 tok/s

- `split4` : attente disque/copies 1,82 → 0,65 ms par couche, 10,85 tok/s au profil.
- `measure_bandwidth.py` choisit `read_parts` d'après la courbe des lectures en parallèle
  (≥ +20 % → autant de morceaux, au plus 4) ; la mesure directe, trop bruitée, n'est plus
  qu'indicative. Relancer `measure_bandwidth.py` pour mettre à jour bandwidth.json.

## 2026-10-03 (2) — Éviction en tas : Qwen3-Next ×2 (3,35 → 6,82 tok/s) ; lectures découpées

- Mesuré : attente disque/copies 3,06 → 1,78 ms par couche, prefetch 1,06 → 0,28, routage
  1,38 → 0,56 ; partie « dense » 25 → 10,6 ms par token. 6,82 tok/s au profil.
- `store.read` peut lire un expert en N morceaux simultanés (`read_parts`) : un expert
  attendu seul profite du débit du SSD à plusieurs lectures en vol. `measure_bandwidth.py`
  mesure 1/2/4/8 morceaux et enregistre le meilleur (`read_parts` dans bandwidth.json),
  `--read-parts` pour forcer. Variantes `split1` / `split4` de `profile_moe.py`. Test ajouté.

## 2026-10-03 — Qwen3-Next : lectures à l'avance rejetées ; éviction du cache en O(log n)

- `prefetch_disk_ahead` mesuré : 3,35 → 3,01 (ahead1) → 2,83 tok/s (ahead2). L'attente vient
  des experts non prévus (16 %), qu'une prédiction plus lointaine ne prévoit pas. Rejeté.
- `cache.py` : la victime d'éviction était trouvée en triant toutes les entrées (0,26 à
  0,54 ms avec les 1 200 à 2 500 entrées de Qwen3-Next, plusieurs fois par couche). Tas avec
  invalidation paresseuse, même ordre LRFU : 0,01 ms. Test d'équivalence ajouté.

## 2026-10-02 (11) — Qwen3-Next tourne ; 1er profil ; lectures disque à l'avance

- Qwen3-Next-80B (store int4 sur C:, D5, split) : 1,5-1,9 tok/s en chat avec `--ram-gb 4`,
  3,49 tok/s au profil (experts chauds préchargés). Temps par couche : attente disque /
  copies 2,9 ms, lecture du routage 1,4 ms, prefetch 1,0 ms.
- `--prefetch-disk-ahead N` : experts prédits de L+2.. lus du disque vers la RAM à
  l'avance, prédiction calculée dans le graph. Variantes `ahead1`, `ahead2`. Tests ajoutés.

## 2026-10-02 (10) — Lecteur safetensors maison (plantage Windows sur Qwen3-Next)

- Le chargement de Qwen3-Next s'arrêtait sans message (« Windows fatal exception: access
  violation », code -1073741819) dans la bibliothèque safetensors, qui projette chaque
  fichier en mémoire. `checkpoint.py` lit maintenant l'en-tête JSON lui-même, puis chaque
  tenseur par seek + read (`SafeFile`) : plus de projection mémoire, mêmes octets (types
  F32/F16/BF16/F8/entiers, lecture d'une seule ligne pour les experts fusionnés).

## 2026-10-02 (9) — Qwen3-Next : plan avec D5, placement split automatique

- `plan.py` tient compte de D5 (partie dense Qwen3-Next : 4,41 → 2,65 Gio), option `--no-dense-int8`.
- Placement automatique : avec D5 (tête en int8 sur le GPU) et des embeddings d'au moins
  256 Mio, `split` plutôt que `gpu` : les embeddings restent sur CPU (une ligne par token),
  la VRAM libérée (~0,58 Gio sur Qwen3-Next, ~370 experts int4) va au cache d'experts.
  OLMoE (embeddings 0,2 Gio) garde `gpu`.

## 2026-10-02 (8) — `build_store.py` : la reprise comptait deux fois la place déjà écrite

- Lors d'une reprise, la vérification de place exigeait la taille du store ENTIER en plus
  des couches déjà écrites (Qwen3-Next, 25 couches sur 48 faites : « pas assez de place »
  avec 38,3 Gio libres pour 17,5 Gio restants). Seule la place des fichiers absents compte
  maintenant ; le message dit combien il manque.

## 2026-10-02 (7) — Correctif du pool vérifié ; fin de l'optimisation sur OLMoE

- `lo` puis `hobbit@0.8` dans le même benchmark, réglages par défaut : `lo` 117,2 tok/s
  (KL 0,0495), `hobbit@0.8` 30,0 tok/s (KL 0,0062) : le ralentissement a disparu.
- Bilan final dans le doc 18. Prochaine étape : Qwen3-Next-80B.

## 2026-10-02 (6) — `hobbit` au benchmark : le pool de RAM épinglée recycle ses plaques

- `hobbit@0.8` seul : 28,9 tok/s avec D5 (KL 0,0058), 24,8 sans : D5 = +16 %, la mesure de
  qualité ne change rien. Le ralentissement à ~19 tok/s venait de l'ordre des réglages :
  après `lo`, le pool de RAM épinglée était plein de plaques int4, et l'int8 de `hobbit`
  passait par de la RAM ordinaire.
- `hostpool.py` : une plaque entièrement libre est redécoupée pour la précision qui
  manque de place (`reclaims`, affiché par `describe()`). Test ajouté.
- **D5 activé par défaut** dans les scripts (`--no-dense-int8` pour le couper).

## 2026-10-02 (5) — A5 adopté (lo 132 tok/s), least-stale rejeté

- **A5 mesuré** : `lo` 117,5 → 132,1 tok/s (100 % des tokens en un graph, 0 refait) ;
  **activé par défaut** (`--no-full-graph` pour le couper). `hobbit` : neutre.
- **Least-stale rejeté** : `hobbit` 51,5 → 43,1 tok/s (plus d'experts chauds évincés).
  S3 entièrement rejeté sur OLMoE.
- Hypothèse « VRAM trop pleine » réfutée : `--vram-gb 3.0` ralentit (`hobbit` 16,1, hit 79 %).
  Reste ouvert : `hobbit` ~19 tok/s au benchmark avec D5 + qualité contre 26 sans.
- `profile_moe.py` : `defaut` inclut A5, variante `sans_a5` ; catalogue mis à jour.

## 2026-10-02 (4) — Mesures des pistes : Q1/Q2 adoptés, S3 (masse) et S5 rejetés, A5 corrigé

- **Q1 + Q2** sur le store d'OLMoE (contre fp16) : `lo` KL 0,070 → 0,049, perplexité
  +10,2 → +3,4 %, accord top-1 89,1 → 93,4 % ; `hobbit@0.8` KL 0,0089 → 0,0061. Vitesse inchangée.
- **A5** : bug corrigé — le compteur de défauts n'était jamais remis à zéro entre deux
  replays (un défaut faisait échouer tous les tokens suivants : 0 % en A5). Test : défaut
  forcé puis reprise du chemin A5.
- **S3** : `prefetch_mass=0.9` rejeté (`hobbit` 49,8 → 20,7 tok/s, routeur d'OLMoE trop plat).
- **S5** : rejeté (brouillon juste pour 10-12 % des tokens). `measure_lookup.py` : `crop` négatif.
- Point ouvert : `hobbit` 18,8 tok/s au benchmark avec D5 (26,1 sans) ; hypothèse de
  VRAM trop pleine, à vérifier avec `--vram-gb 3.0`.

## 2026-10-02 (3) — Les pistes de la littérature récente : Q1, Q2, S3, A5, S5

Recherche bibliographique (2025-2026), puis tout ce qui restait de prometteur, codé,
testé sur CPU et désactivable (doc 18, « Les pistes du 02/10 ») :

- **Q1** `quant.py` : échelle int4 « mse » (32 candidats par groupe, le plus grand poids
  sur -8). Défaut des nouveaux stores (`build_store.py --int4-scale`), réécriture en place
  d'un store existant : `scripts/requant_store.py --method mse` (`moelab/requant.py`).
- **Q2** GPTQ par expert : calibration (texte écrit par le modèle, ou `--calib-file`),
  entrées de chaque couche MoE enregistrées, Hessiennes pondérées par le poids du routeur,
  `requant_store.py --method gptq`.
- **S3** `cache.py` : éviction least-stale (`--cache-eviction stale`) ; runtime : prefetch
  selon la masse de probabilité du routeur (`--prefetch-mass 0.9`), aussi dans le graph (A9).
- **A5** `moelab/fullgraph.py` : token entier en un graph quand tout est en VRAM, décisions
  HOBBIT et adresses des experts sur le GPU, défaut de cache refait par le chemin normal
  (états du cache restaurés) ; `--full-graph`. Émulateur : appels opaques (noyaux Triton).
- **S5** `scripts/measure_lookup.py` : mesure le gain attendu du décodage spéculatif
  « prompt lookup » avant de le construire.
- `profile_moe.py` : variantes `a5`, `stale`, `mass`, `s3`, `tout` (défaut : `defaut,a5,s3`) ;
  `check_setup.py` : A5 sur la carte (lo, hobbit). Catalogue : `full_graph`,
  `prefetch_mass`, `cache_eviction`, `int4_quant`.
- 91 tests (89 + 2 sur GPU).

## 2026-10-02 (2) — D5 mesuré ; plancher de la mesure de qualité

- D5 (`--dense-int8`) : `lo` 104,7 → 118,9 tok/s, `hobbit@0.8` 39,4 → 46,7 au profil ;
  benchmark `lo` 86,9 → 109,1. Qualité : nulle en `lo`, KL 0,0079 → 0,0089 en `hobbit`.
  Benchmark `hobbit` revérifié en 2 passes : 26,1 → 26,8 tok/s (le 17,5 était aberrant).
- Plancher de la mesure de qualité : `hi` fp16 token par token contre la référence en une
  passe = KL 0,0006 ; perte propre à l'int8 : ~0,002.
- Catalogue des réglages mis à jour avec ces mesures.

## 2026-10-02 — A9 mesuré ; qualité int8 contre fp16 ; D5 partie dense en int8

- Profil (store int8) : A9 fait passer `hobbit@0.8` de 35,1 à 39,4 tok/s.
- Benchmark avec la référence fp16 : `hobbit@0.8` sur store int8 = même qualité que sur
  store fp16 (KL 0,0079 contre 0,0074), 4,7 fois plus vite ; int8 seul : KL 0,0027.
- **D5, `--dense-int8`** (`moelab/denseq.py`) : couches Linear de la partie dense en int8
  (GEMV Triton en génération, capturable ; déquantification par blocs en prefill), tête
  de sortie gardée sur GPU en placement split, placement automatique calculé sur la
  taille int8. `check_setup.py` : vérification et chrono sur une tête 50304 x 2048.
- 79 tests (77 + 2 sur GPU).

## 2026-10-01 (9) — A9 : prédiction du prefetch dans le graph

- Le graph de chaque segment calcule aussi le routeur de la couche suivante et rassemble
  routage + prédiction en un tenseur : une lecture GPU → CPU par couche au lieu de deux,
  plus de routeur lancé en Python pour le prefetch (`graph_pack_routing`, défaut).
- `profile_moe.py` : variantes `sans_a9`, `defaut`. Catalogue : réglage `graph_pack_routing`.
- 76 tests (74 + 2 sur GPU).

## 2026-10-01 (8) — Catalogue des réglages

- `moelab/options.py` : tous les réglages (politique, seuils, mémoire, optimisations GPU,
  transferts, store) avec leur explication, leurs valeurs possibles, leur effet mesuré
  (doc 18) et leur mode d'application (à chaud, caches vidés, rechargement, autre store).
  `apply()` les change à chaud avec les effets de bord nécessaires ; `current()`,
  `describe_json()` et des préréglages (vitesse, équilibre, qualité, référence) pour l'app.
- `chat.py` : `/options`, `/set <réglage> <valeur>`, `/preset <nom>`.
- 74 tests (72 + 2 sur GPU).

## 2026-10-01 (7) — int8 comme haute précision : hobbit ×4 ; bilan de l'optimisation

- Store int8 + int4 (`C:\moe-store\olmoe-i8`) : `hobbit@0.8` 5,6 → **22,6 tok/s** au
  benchmark (9,0 → 36,2 au profil), 130 Mo copiés par token au lieu de 680, qualité du
  même ordre (KL 0,0066 contre la référence int8). Noyau int8 groupé : 221 Go/s.
- `benchmark.py --save-ref / --ref` : référence de qualité enregistrée sur un store
  (fp16) et réutilisée sur un autre (int8), pour chiffrer la perte de la haute précision.
- Colonnes « haute précision demandée / utilisée » (au lieu de « fp16 », faux avec un store int8).
- Doc 18 : bilan de l'optimisation (lo ×16, hobbit ×9). GUIDE : section 8 ter (store int8).

## 2026-10-01 (6) — Paliers : 105,9 tok/s en lo ; E3 rejeté ; int8 comme haute précision

- Profil de 11h55 : paliers de longueur `lo` 89,4 → **105,9 tok/s**. Noyaux mesurés en
  graph : int4 v3 176 Go/s, v2 167, v4 121 (v3 ajouté au choix automatique).
- Benchmark de 12h03 (qualité token par token) : E3 (`hobbit_miss`) +6 % de vitesse,
  KL ×2,4 : rejeté. `hobbit@0.8` : ~680 Mo copiés RAM → VRAM par token.
- **Noyaux groupés int8** (et repli PyTorch pour les groupes hors noyau) : un store
  int8 + int4 (`build_store --hi int8`) prend l'int8 comme haute précision, 2 fois plus
  petit que le fp16. Calcul groupé en graph pour int8, testé sous l'interprète.
- Microbenchmark : int8 et fp16 en graph. `check_setup.py` : int8 vérifié sur la carte.
- 71 tests (69 + 2 sur GPU).

## 2026-10-01 (5) — Traces lo et hobbit ; paliers de longueur ; qualité token par token

- Profils de 11h33/11h37 : `lo` 89,1 tok/s, `hobbit` 8,5-10,4. v4 = v2 = v3 (~70 Go/s),
  mais cette mesure était limitée par le lancement Python : les noyaux sont maintenant
  chronométrés en graph (microbenchmark et choix automatique v2/v4).
- Trace `hobbit` : ~49 ms de copies RAM → VRAM par token (46 experts fp16) sur ~96.
- **Paliers de longueur** (`graph_buckets`, défaut) : un jeu de graphs par palier (128...
  max_len) sur une vue du cache statique ; l'attention ne lit plus les 2 048 positions.
- `benchmark.py --quality-mode decode` (défaut) : qualité mesurée token par token avec
  le décodeur de génération ; seul mode où E3 (`hobbit_miss`) agit.
- `profile_moe.py` : transferts par token (à la demande, prefetch, Mo RAM → VRAM),
  précision et rappel du prefetch ; variantes `sans_paliers`, `defaut`.
- 70 tests (68 + 2 sur GPU).

## 2026-10-01 (4) — Calcul groupé en graph : 89,6 tok/s en lo ; noyaux v4, choix par mesure

- Profil de 11h16 : A6 bis fait passer `lo` de 47,0 à **89,6 tok/s** (`hobbit` 8,6). v3
  ne gagne rien (68 Go/s) : la conversion des quartets n'était pas le frein.
- **v4** : noyaux int4 groupés sur le modèle du noyau fp16 (blocs 2D d'un groupe,
  accumulateur 2D, une seule réduction). `grouped_kernels = 0` (défaut) : v2 et v4
  chronométrés au démarrage sur la carte, le plus rapide est gardé.
- `profile_moe.py` : v4 et la version choisie dans les planchers ; variantes `g2`, `g4`,
  `defaut` ; `--trace-policy` pour tracer `hobbit`.
- `check_setup.py` : v4 vérifié ; deux tokens d'échauffement avant de chronométrer les graphs.

## 2026-10-01 (3) — Noyaux v2 mesurés (48,4 tok/s) ; v3 et calcul groupé en graph

- Profil (`results/profile_olmoe-hf_2026-10-01_1054.md`) : int4 groupé v2 à 69 Go/s (v1 :
  53), `lo` 41,8 → **48,4 tok/s**. Le noyau fp16 atteint 263 Go/s : l'int4 est limité
  par le calcul.
- **v3** (défaut) : conversion des quartets par astuce de bits au lieu de l'instruction
  entier → flottant. Identique au bit près.
- **A6 bis** (`moe_graphs`, défaut avec les CUDA graphs) : le calcul groupé de chaque
  couche en un CUDA graph (table épinglée de taille fixe, poids 0 pour les lignes vides),
  au lieu de ~0,7 ms de Python par couche.
- `profile_moe.py` : v1/v2/v3 et appel normal contre graph dans les planchers ; variantes
  `cg_v2`, `cg_v3`, `defaut`. `check_setup.py` : v3, et les graphs en `lo` et `hobbit`.
- 67 tests (65 + 2 sur GPU).

## 2026-10-01 (2) — CUDA graphs mesurés : 41,9 tok/s en lo ; noyaux groupés v2

- Profil (`results/profile_olmoe-hf_2026-10-01_1026.md`) : avec les CUDA graphs, la partie
  dense passe de 34 à 1,5 ms par token ; `lo` 20,9 → **41,9 tok/s**, `hobbit` 6,8 → 8,0.
  **CUDA graphs activés par défaut** (`--no-cuda-graphs`).
- Noyaux groupés v2 (`kernels.py`) : la trace montrait la v1 à ~35 Go/s effectifs.
  Alignement déclaré, chemin sans masques, échelle par groupe, noyau B réparti sur les
  experts (addition atomique en float32), `triton.autotune`. La v1 reste disponible
  (`RuntimeConfig.grouped_kernels = 1`, variante `cg_v1`) pour comparer.
- `profile_moe.py` : microbenchmark v1 / v2 (Go/s effectifs), variantes par défaut
  `v0110,cg_v1,defaut`, trace faite avec des caches remis à zéro.
- `check_setup.py` : v1 et v2 vérifiés et chronométrés sur la carte.
- `tests/test_kernels.py` : v1 et v2 contre la référence sous l'interprète de Triton.

## 2026-10-01 — Noyaux groupés par défaut ; CUDA graphs pour la partie dense (A6)

- Profil du 01/10 (`results/profile_olmoe-hf_2026-10-01_0951.md`) : les noyaux groupés
  font passer `lo` de 12,4 à **20,2 tok/s** et `hobbit` de 5,6 à 6,2 (6,8 avec D1). C2
  est rejeté (hit VRAM 78 → 72 %). La partie dense pèse 35 ms sur 50 en `lo`.
- **Activés par défaut** : noyaux groupés (`--no-grouped` pour les couper) et lecture
  sans cache D1 (`--no-direct-io`). Les anciennes options `--grouped` et `--direct-io`
  disparaissent.
- Table d'adresses des noyaux groupés envoyée par un anneau de tampons épinglés : la
  copie depuis la mémoire non épinglée attendait la fin du travail GPU à chaque couche.
- **A6, CUDA graphs** (`--cuda-graphs`, `moelab/graphs.py`) : la passe de génération est
  capturée une fois et découpée aux couches MoE (17 graphs pour OLMoE), cache KV
  statique (`--graph-max-len`, 2 048 par défaut). Générique (aucun code propre au
  modèle), testé par un émulateur de graphs sur CPU (OLMoE et Qwen3-Next miniatures) et
  par `check_setup.py` sur GPU. Repli automatique si la capture échoue.
- `moelab/generate.py` : une seule boucle de génération pour `chat.py`, `benchmark.py`
  et `profile_moe.py` (graphs ou passe normale). Le benchmark compte maintenant le 1er
  token de décodage (capture des graphs) dans le TTFT.
- `profile_moe.py` : variantes `v30`, `defaut`, `cg` par défaut, politiques `lo` et
  `hobbit@0.8` (`hi` en option), temps de la partie dense, `torch.profiler` sur le
  décodage seul (le prefill gonflait le nombre de lancements par token).
- 65 tests (63 + 2 sur GPU).

## 2026-09-30 — Profil GPU : les lancements de noyaux dominent ; noyaux groupés

- Profil sur la RTX 2060 (`results/profile_olmoe-hf_2026-09-30_1616.md`) : un lancement de
  noyau coûte ~52 µs sous Windows, ~3 500 lancements par token. Le noyau Triton int4
  (B1) fait passer `lo` de 6,7 à 11,8 tok/s ; A2 + prefetch adaptatif : +22 % (lo), +35 %
  (hobbit) ; B2 rejeté. Doc 18 mis à jour.
- **Activés par défaut** : `fast_decode`, `prefetch_adaptive`, `triton_int4` (repli
  automatique sans Triton). Les options de ligne de commande deviennent `--no-...`.
- **A4+B1, noyaux groupés** (`--grouped`) : tous les experts GPU d'une couche en 2 noyaux
  Triton par précision (table d'adresses, pas de recopie des poids), en int4 et en
  flottant. Testé par l'interprète de Triton en hi, lo et hobbit ; vérifié sur GPU par
  `check_setup.py`. `profile_moe.py` compare base / défaut / groupé / + D1 / + C2.

## 2026-09-30 — Optimisation : disque, prefetch, int4, politique, noyau Triton

Toutes les optimisations du doc 18 qui se testent sans GPU sont faites, chacune en
option (désactivée tant qu'elle n'est pas mesurée), chacune avec son test :

- **Disque** : lectures sans cache du système (`--direct-io`, D1) ; nombre de lectures
  en parallèle mesuré par `measure_bandwidth.py` (D2) ; lectures triées dans l'ordre
  du fichier (D3, activé) ; store chaud sur le NVMe (`build_hot_store.py` +
  `--hot-store`, D4).
- **Prefetch** : deux couches à l'avance (`--prefetch-depth 2`, C2).
- **Int4** : noyau Triton qui lit l'int4 sans le décompresser (`--triton-int4`, B1,
  vérifié par l'interprète de Triton, repli automatique) ; int4 déballé en int8 dans
  la VRAM (`--int4-unpacked`, B2, identique bit à bit).
- **Politique** : précisions en niveaux ; un t1 par couche calibré par
  `measure_routing.py` (`--t1-layers auto`, E1) ; trois paliers fp16 / int8 / int4
  (`--t-mid`, `build_store.py --precisions`, E2) ; basse précision pour les experts
  absents de la VRAM (`--miss-t1`, E3). Réglages `hobbit_cal` et `hobbit_miss` au
  benchmark.
- `check_setup.py` teste le noyau Triton sur le GPU s'il est installé.
- Non retenus, avec la raison dans le doc 18 : A3, B3, C1. En attente du profil GPU :
  A4 à A7.

## 2026-09-30 — Optimisation A2 : décisions de génération allégées

- `RuntimeConfig.fast_decode` : en génération (1 token), routage et poids lus en UNE
  copie GPU -> CPU, décisions HOBBIT en Python pur (`PrecisionPolicy.decide_one`),
  accumulation sans indexation. Même chose pour le prefetch. Modèle miniature sur
  CPU : 2,03 -> 1,12 ms par couche MoE.
- `RuntimeConfig.prefetch_adaptive` : pas de prefetch tant que tout est résident.
- `profile_moe.py --variants` compare la version de base et chaque optimisation.
- Tests : génération identique à transformers avec chaque variante (branche CPU et
  transferts exercés), `decide_one` identique à `decide`.

## 2026-09-30 — Optimisation, phase 0 : mesurer

- Doc 18 : le plan d'optimisation complet (surcoût du moteur, calcul des experts,
  transferts, disque, politique), avec un tableau de suivi mesure par mesure.
- `moelab/profiling.py` : chronomètre par phase de `run_layer` (temps CPU ou GPU
  compris) et microbenchmarks des planchers physiques (lancement de noyau, synchro,
  8 experts un par un ou groupés, déquantification int4).
- `scripts/profile_moe.py` : planchers, décodage détaillé par phase et par politique,
  `torch.profiler` avec le nombre de lancements de noyaux par token.

## 2026-09-30 — Balayage propre de t1 : les chiffres de référence

- RAM fixée à 5 Gio, qualité de chaque règle mesurée seule. Qualité régulière en t1 :
  t1=0,8 → +0,83% de perplexité (sous le seuil de +1%), t1=0,9 → qualité du fp16
  (KL 0,0019) avec 13% d'experts en int4 ; t1=0,2 ≈ int4 pur (+10,7%), ce qui
  dément la conclusion provisoire de la veille (artefact du cache).
- Vitesse : aucun gain de hobbit sur hi. Le moteur est limité par son surcoût
  (9,5 ms par couche MoE même tout en VRAM), pas par la mémoire.
- Doc 17 (sections 5 et 6), README et brouillon Colibrì (doc 16) mis à jour avec
  ces chiffres. Message du benchmark sur le préchargement reformulé.

## 2026-09-30 — Balayage de t1 et deux corrections du benchmark

- Balayage t1 = 0,2 / 0,4 / 0,6 / 0,8 : la qualité (KL, accord top-1) s'améliore
  régulièrement avec t1 ; t1=0,8 : +1,5% de perplexité, 97,3% d'accord. Doc 17,
  section 4 ; fichiers dans `results/`.
- **Bug** : `warm_start` ne remettait pas les compteurs à zéro quand il n'y avait
  rien à précharger (budget RAM trop petit, historique vierge). Dans le benchmark,
  les colonnes hit / fp16 / disque cumulaient donc tous les réglages précédents.
  Test ajouté (il échoue sans le correctif).
- **Qualité mesurée sur la règle seule** : caches vidés et `upgrade=False` pendant
  le teacher forcing. Avant, un fp16 laissé en cache par les prompts améliorait la
  qualité mesurée de hobbit, et le résultat dépendait de l'ordre des réglages.
- Le benchmark avertit si le budget RAM empêche le préchargement du pool int4.

## 2026-09-29 — Premiers résultats sur OLMoE

- `measure_routing` : proxy HOBBIT 0,888 (0,99 sur Mixtral), prefetch 83,3%,
  localité 39,9%. `benchmark --repeats 2` : int4 statique +10,4% de perplexité,
  HOBBIT t1=0,6 +2,7% avec 49% de fp16 demandés, 57% de lectures disque en moins
  que le tout-fp16. Analyse : doc 17 ; README, docs 13, 14 et 16 mis à jour ;
  fichiers bruts dans `results/`.
- **Budget RAM plafonné à 35% de la RAM totale avec un GPU** (`PINNED_MAX_FRACTION`) :
  Windows a refusé d'épingler plus de 7,19 Gio pendant le benchmark, et deux
  passages ont eu des TTFT de 30 à 48 s (pagination probable).
- **Balayage de t1 dans un seul benchmark** : `--configs hobbit@0.2,hobbit@0.4,...`.

## 2026-09-29 — Statistiques : fp16 demandé / utilisé

Au chat, `/t1 0.25` affichait encore 64% de « haute précision ». Ce chiffre était
la part **calculée** en fp16 : en mode hobbit, un expert fp16 déjà en VRAM sert
aussi une demande int4 (qualité en plus, sans transfert). La ligne du chat et le
tableau du benchmark affichent maintenant les deux : **demandé** (la décision
HOBBIT, qui dépend de t1) et **utilisé** (ce que le cache a permis).

## 2026-09-29 — Correctif : mémoire épinglée épuisée (pool de slots)

Chat sur la vraie machine, après le correctif du cache KV : plantage au bout de
quelques tokens, `CUDA_ERROR_OUT_OF_MEMORY` dans `cuMemHostAlloc` (allocation de
mémoire épinglée), avec un budget RAM de 5,08 Gio.

- **Cause** : un `torch.empty(..., pin_memory=True)` par lecture disque. L'allocateur
  de PyTorch arrondit chaque tampon à la puissance de 2 (12 -> 16 Mio, 3,19 -> 4 Mio)
  et garde la mémoire libérée par taille : la mémoire épinglée réelle dépassait
  largement le budget, jusqu'au refus de Windows.
- **Correctif** : `moelab/hostpool.py`, un pool de slots de taille exacte taillés
  dans des plaques puissance de 2, plafonné au budget RAM + une marge, avec repli
  sur de la RAM ordinaire au lieu d'une erreur. Libération différée (événement CUDA
  + couche suivante) pour ne jamais écraser un tampon en cours de copie. Doc 11
  complétée. 2 tests (36 au total).

## 2026-09-29 — Correctif : cache KV désactivé par le config.json d'OLMoE

Premier chat sur la vraie machine : 0,38 token/s, 0% de hits VRAM, pas de
statistique de prefetch. Trois symptômes, une cause.

- **Cause** : le `config.json` d'OLMoE contient `"use_cache": false`. Sans cache
  KV, `generate()` recalcule **toute la séquence** à chaque nouveau token : chaque
  couche redemande presque ses 64 experts à chaque pas, bien plus que la VRAM
  n'en contient (les caches tournent en rond, 0% de hits), le prefetch (réservé
  aux passes d'un seul token) ne se déclenche jamais, et la vitesse baisse avec
  la longueur de la réponse. Le benchmark n'était pas touché (il passe le cache
  explicitement).
- **Correctif** (`loader.py`) : cache KV forcé, et `generation_config.json` du
  modèle chargé (tokens de fin de réponse corrects). Test ajouté : il échoue sans
  le correctif (`[7, 8, 9, 10, 11]` tokens par passe au lieu de `[7, 1, 1, 1, 1]`).
- **chat.py** : une commande tapée à la fin d'un message (`qu'est-ce que X /policy lo`)
  était envoyée au modèle comme du texte (d'où une réponse sur « The Hobbit »). Elle
  est maintenant refusée avec un message ; les commandes inconnues aussi.
- **plan.py / --preload-lo** : `plan.py` annonçait « préchargement conseillé » dès
  que le pool int4 tenait en RAM, alors que le chat ne précharge que s'il prend au
  plus 60% du budget RAM. Une seule règle (`preload_advised`) pour les deux.

## 2026-09-29 — Correctif : placement `split` sur GPU

Trouvé sur la vraie machine (RTX 2060) par `check_setup.py` : en placement `split`,
`generate()` plantait (« Expected all tensors to be on the same device »).

- **Cause** : transformers déduit `model.device` du *premier paramètre* du modèle.
  En `split`, c'était la table d'embeddings, restée en RAM : `generate()` en
  concluait que le modèle était sur CPU, y déplaçait les entrées, et les couches
  (sur GPU) recevaient des tenseurs CPU.
- **Correctif** : `CpuEmbedding` et `CpuLinear` gardent leurs poids comme simples
  tenseurs (plus des `nn.Parameter`), donc `model.device` vaut `cuda` ; et
  `CpuEmbedding` renvoie toujours son résultat sur le device des couches.
- Tests ajoutés : `model.device` vérifié dans `test_cuda.py`, et un test CPU qui
  monte les mêmes modules et vérifie que `generate()` reste exact (33 tests).

## 2026-09-29 — Le moteur `moelab` : étapes 8 à 16

Construction du système complet, testé sur des versions miniatures d'OLMoE et de
Qwen3-Next (mêmes formats de checkpoint, dimensions minuscules).

- **Store d'experts** (`moelab/store.py`) : un record contigu par expert, une
  lecture disque par expert, fp16/int8/int4, construction reprenable, lecture des
  checkpoints par expert, fusionnés et FP8 par blocs.
- **Quantification** int4/int8 symétrique par groupes de 64 (4,25 bits par poids).
- **Chargeur** : squelette du modèle sur le device `meta`, seuls les poids denses
  sont lus (0,89 Gio pour OLMoE au lieu de 13 Go) ; placements `gpu`, `cpu`, `split`.
- **HOBBIT** : scores normalisés comme dans le papier (correction de l'étape 7,
  voir doc 06), seuils T1/T2, couches protégées.
- **Problème trouvé par un test** : en prefill, un expert partagé par beaucoup de
  tokens finissait presque toujours en fp16 (il suffisait d'un token). Ajout d'un
  vote par expert (`hi_vote`, 50%) ; en génération c'est exactement HOBBIT.
- **q\* de FreeToken** : formule vérifiée dans le papier (§3.2, figure 2), branche
  CPU exécutée pendant les copies RAM → VRAM sur un stream CUDA dédié ; bandes
  passantes mesurées par `measure_bandwidth.py`.
- **Caches** : score LRFU, épinglage, anti-oscillation (hystérésis 25%, 4 échanges),
  historique persistant `usage.json`, préchargement du pool int4.
- **Prefetch** de la couche suivante avec son propre routeur, rappel mesuré en continu.
- **Deux bugs trouvés en testant les scripts** : le mode `lo` utilisait des experts
  fp16 déjà en cache (il doit reproduire une quantification statique exacte), et
  l'historique d'usage d'un réglage biaisait le suivant dans le benchmark.
- **Benchmark A/B** (`hi`, `lo`, `hobbit`, `hobbit_edges`) avec règle d'acceptation
  adaptée de Colibrì, et **mesures du routage** (proxy HOBBIT, prefetch, localité).
- **Palier final** : choix de Qwen3-Next-80B-A3B (version FP8, 82 Go), store int4 de
  38,2 Gio ; le disque D: étant un HDD, vitesse attendue ~0,1 à 0,3 token/s.
- Brouillon de contribution pour l'issue Colibrì #249 (à compléter avec les mesures).
- Docs 09 à 16, `GUIDE.md`, 57 tests (dont 2 pour CUDA).
- Repo reconstruit : les docs 01 à 08 ont été réécrites à l'identique sur le fond,
  avec deux corrections (normalisation HOBBIT dans la doc 06, dérivation complète de
  q* dans la doc 02) et une réorganisation de la doc 05.

## 2026-09-12 — Découverte de Colibrì, référence système pour l'objectif final

- Projet open source (JustVugg/colibri, C pur, ~27k étoiles) qui fait tourner des
  modèles MoE jusqu'à 2,8T paramètres en traitant VRAM/RAM/disque comme une seule
  hiérarchie, et qui supporte OLMoE.
- Concepts retenus : score LFRU, historique persistant, anti-oscillation, prefetch
  mesuré à 71,6% sur GLM-5.2.
- Découverte : Colibrì n'a pas de précision dynamique par expert (issue ouverte #249).
- Détail : [doc 08](docs/08-colibri-reference.md)

## 2026-08-21 — Pattern en U de la confiance du routeur

- Test sur 20 phrases : les pics observés sur une seule phrase disparaissent en
  moyenne ; il reste une forme en U (couches 0-1 et 15 plus confiantes).
- Détail : [doc 07](docs/07-pattern-en-u-des-couches.md)

## 2026-08-21 — Première brique fonctionnelle : scoring d'importance HOBBIT

- Structure du routeur OLMoE : `OlmoeTopKRouter`, linear → softmax → top-k.
- Erreur de méthode corrigée : envoyer l'embedding brut au routeur donnait une
  distribution quasi uniforme ; un forward hook capture les vrais scores.
- `scripts/hobbit_scoring.py`. Détail : [doc 06](docs/06-comprendre-le-scoring-hobbit.md)

## 2026-08-21 — Validation de l'accès individuel aux experts (PyTorch)

- transformers 5.15.1 fusionne les experts en un tenseur par couche
  (`gate_up_proj` (64, 2048, 2048)), mais chaque expert reste accessible par
  indexation : `experts.gate_up_proj[5].to('cuda')` fonctionne.
- Objectif long terme ajouté : un modèle 70B+ avec hiérarchie GPU → RAM → disque.
- FreeToken ajouté aux papiers de référence.

## 2026-08-21 — Pivot : GGUF → HuggingFace Transformers

- `llama-cpp-python` compilé avec CUDA, OLMoE F16 chargé.
- Découverte bloquante : GGUF stocke les 64 experts d'une couche en un seul tenseur,
  inaccessible depuis Python. Bascule vers les checkpoints `.safetensors`.
- Détail : [doc 05](docs/05-architecture-reality-check.md)

## 2026-08-21 — Téléchargement et import d'OLMoE

- `OLMoE-1B-7B-0125-Instruct-F16.gguf` (13,8 Go) téléchargé, import Ollama
  abandonné faute de place (Ollama copie le fichier).
- Détail : [doc 04](docs/04-setup-olmoe.md)

## 2026-08-21 — Sélection du modèle MoE

- Comparatif de 5 candidats, OLMoE-1B-7B-0125-Instruct retenu.
- Détail : [doc 03](docs/03-model-selection.md)

## 2026-08-21 — Lecture de fond : papiers de recherche

- HOBBIT, DyMoE, MoEpic. Synthèse : [doc 02](docs/02-research-notes.md)

## 2026-08-21 — Baseline matériel + llama3:8b

- RTX 2060 (6 Go VRAM), ~16 Go RAM ; `llama3:8b` : 22%/78% CPU/GPU dans Ollama.
- Détail : [doc 01](docs/01-hardware-baseline.md)
