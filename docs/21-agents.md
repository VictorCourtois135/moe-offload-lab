# 21. Agents : un orchestrateur qui crée ses propres agents

## L'idée (06/10)

Tu donnes un objectif ; l'orchestrateur le découpe, crée lui-même des agents spécialisés
(un lecteur, un chercheur, un rédacteur…), leur confie des tâches précises et assemble leurs
comptes rendus en une réponse. Tout tourne sur le modèle local, rien ne passe par internet
(sauf les outils web, si tu les permets).

Objectif volontairement général : les agents reçoivent les outils que tu permets, donc
l'orchestrateur peut lire et écrire des fichiers, chercher sur le web, exécuter du Python,
consulter le moteur et utiliser les serveurs MCP branchés.

## Un seul modèle en mémoire

Les agents ne tournent pas en parallèle : ils parlent l'un après l'autre sur le même modèle.
Chacun garde quand même son rôle (message système), ses outils et sa conversation pendant le
run.

| Problème | Ce que fait moelab |
|---|---|
| L'orchestrateur perd son contexte pendant qu'un agent travaille | sa conversation (K/V, états) est mise de côté sur CPU puis remise : il reprend sans recalculer son historique |
| Un agent qu'on relance repart de zéro | sa conversation est gardée et remise de la même façon |
| Le chat est bloqué pendant un run de 20 minutes | entre deux tours de l'orchestrateur, les messages de chat en attente passent (`Engine.serve_waiting`), puis le run reprend |
| Le message système et les outils de chaque agent coûtent un prefill | cache de préfixe (P9) : calculés une fois par agent |

Ordre de grandeur sur la RTX 2060 avec Qwen3-Next (~7 tok/s, ~4 s avant le premier token) :
environ 30 s par tour d'agent, donc 10 à 30 minutes pour un objectif à plusieurs étapes.

## Les outils

**L'orchestrateur** ne travaille pas lui-même :

| Outil | Effet |
|---|---|
| `creer_agent(nom, role, outils)` | crée un agent avec ses consignes et une partie des outils permis |
| `confier(agent, tache)` | l'agent travaille, puis son compte rendu revient comme résultat |
| `lister_agents()` | agents du run et agents enregistrés |
| `noter(texte)`, `lire_notes()` | bloc-notes partagé par tous les agents du run |
| `terminer(reponse)` | réponse finale |

**Un sous-agent** reçoit ses outils, le bloc-notes et `rendre(compte_rendu)`. Il ne voit ni
l'objectif ni les autres agents : seulement sa tâche et le bloc-notes. L'orchestrateur doit
donc écrire des tâches autonomes (chemins, adresses, contraintes).

**Nouveaux outils** (aussi disponibles dans le chat) :

| Outil | Permission par défaut | Note |
|---|---|---|
| `chercher_web(requete)` | demander | DuckDuckGo, sans clé ; lire ensuite avec `lire_page` |
| `executer_python(code)` | demander | sous-processus Python, 60 s par défaut, dans `app_data\espace_agents`. Ce n'est pas un bac à sable : le script a les droits de ton compte, d'où l'accord demandé à chaque appel |

## Les garde-fous

Le code garantit, le modèle propose (comme l'autopilot) :

- **un seul niveau** : un sous-agent ne crée pas d'agents ;
- **limites** : nombre d'agents (5), tours de l'orchestrateur (14), tours par tâche (8),
  minutes (60), réglables au lancement. Budget épuisé : les outils répondent « appelle
  terminer » ;
- **jamais plus que ce que tu permets** : un agent ne reçoit que des outils cochés au lancement
  et non « jamais » ; un outil « demander » attend ton accord, quel que soit l'agent ;
- **relances** : si l'orchestrateur s'arrête sans appeler d'outil, il est relancé (3 fois au
  plus) ; s'il n'appelle jamais `terminer`, la réponse rassemble ce qui a été produit ;
- **erreurs rendues au modèle** : nom pris, outil inconnu, agent inexistant deviennent un
  message qu'il peut corriger, jamais un plantage.

## La page Agents

- **Lancer** : objectif, outils permis aux agents, limites.
- **Déroulé en direct** : une colonne pour l'orchestrateur (ce qu'il pense, ses appels) ;
  chaque agent créé et chaque tâche confiée s'y branchent, à la couleur de l'agent, avec ses
  appels d'outils, ses notes et son compte rendu. Les demandes d'accord s'affichent en haut.
- **Réponse** en tête une fois le run fini, avec « Copier ».
- **Garder un agent** : un agent créé pendant un run peut être enregistré. L'orchestrateur voit
  les agents enregistrés et peut leur confier une tâche sans les recréer.
- **Agents enregistrés** : créer, modifier, supprimer à la main.
- **Runs précédents** : chaque run est enregistré (`app_data\agent_runs\<id>.json`) et se
  relit dans le même affichage.

## Fichiers

| Fichier | Rôle |
|---|---|
| `moelab/app/agents.py` | `Orchestrator` (indépendant du moteur, testable), bibliothèque d'agents, job de l'app |
| `moelab/app/engine.py` | `serve_waiting` : le chat passe pendant un long job |
| `moelab/app/tools.py` | `chercher_web`, `executer_python` |
| `moelab/app/server.py` | `/api/agents`, `/api/agents/run`, `/api/agents/runs`, `/garder` |
| `tests/test_orchestrator.py` | 7 tests avec une fausse génération scriptée par agent |

## Vérifié (06/10)

- `tests/test_orchestrator.py` : création, délégation, bloc-notes, limites, relances,
  bibliothèque, conversation d'un agent gardée d'une tâche à l'autre, ordre des mises de côté
  de l'état de décodage, budget de temps, chat servi pendant un long job (et jamais un
  chargement de modèle).
- `tests/test_app.py` : run complet par l'API, run enregistré, agent gardé, bibliothèque.
- Navigateur (Chromium), avec un faux modèle scripté sur le store miniature : run complet en
  direct (lecteur puis rédacteur, accord pour `ecrire_fichier`, fichier écrit), un message de
  chat servi entre deux tours, relecture d'un run, affichage téléphone et sombre.
- Pas encore lancé avec Qwen3-Next : reste à voir si le modèle local orchestre bien (créer
  des agents utiles, écrire des tâches autonomes). Les relances et les messages d'erreur sont
  là pour l'aider ; les consignes se règlent dans `ORCH_SYSTEM` et `SUB_SYSTEM`.

## Premier vrai run (06/10, Qwen3-Next) : trois défauts corrigés

Objectif : résumer les fichiers texte d'un dossier dans `resume.md`. Le run a échoué, à cause
de trois défauts de moelab, pas du modèle :

| Ce qui s'est passé | Cause | Correction |
|---|---|---|
| L'orchestrateur appelle `lister_dossier` puis `chercher_fichiers` lui-même : refusés, il confie la tâche à l'aveugle | il n'avait aucun outil pour regarder ; il a aussi deviné de mauvais noms d'arguments (`chemin`, `pattern`) | l'orchestrateur reçoit `lister_dossier` et `chercher_fichiers` (lecture seule, si permis) ; une erreur d'arguments liste les arguments attendus |
| L'agent lit le dossier avec `lire_fichier` et conclut « dossier non autorisé » | sous Windows, ouvrir un dossier comme un fichier donne `PermissionError` ; l'agent n'avait pas `lister_dossier` | `lire_fichier` sur un dossier : « c'est un dossier, utilise lister_dossier » ; un agent qui lit ou écrit des fichiers reçoit toujours `lister_dossier` et `chercher_fichiers` |
| L'orchestrateur écrit `terminer({"reponse": ...})` deux fois, relancé en boucle | appel écrit en texte simple, sans les balises `<tool_call>` | les appels `nom({...})` d'un outil connu sont reconnus (aussi dans le chat et l'autopilot) ; le format exact est rappelé dans les consignes et la relance |

Rejoué tel quel dans `tests/test_orchestrator.py::test_run_of_06_10_replayed`.

## Deuxième run (06/10) : le dossier n'était pas autorisé

Encore lancé avec l'ancien code (l'agent n'avait que `lire_fichier`). Le message rapporté par les
agents, « aucun dossier autorisé n'est configuré », vient du cas où la liste des dossiers
autorisés est vide : le refus était réel. Le 1er run avait très probablement la même cause ; la
piste `PermissionError` de Windows reste corrigée, mais elle n'était sans doute pas en jeu.

Ce qui reste corrigé :

- le modèle a écrit `terminer("texte")` et `noter("texte")` (une chaîne, sans JSON) : reconnu
  maintenant, la chaîne va au 1er argument obligatoire ;
- les dossiers autorisés sont écrits dans les consignes de l'orchestrateur et de chaque agent ;
  s'il n'y en a aucun, l'orchestrateur sait qu'il doit terminer tout de suite (le run avait
  tourné 7,6 min à vérifier l'accès) ;
- l'écran Agents affiche un avertissement quand aucun dossier n'est autorisé.

## Troisième run (06/10) : réussi

Dossier de test créé (le premier chemin n'existait pas : les Documents de ce PC sont sur D:).
L'orchestrateur regarde le dossier, crée un seul agent `redacteur` (rôle et format précis), lui
confie les trois fichiers un par un, termine. `resume.md` est juste (5 lignes par fichier,
fidèles). Environ 15 minutes.

Deux défauts vus dans le déroulé, corrigés :

| Défaut | Correction |
|---|---|
| `rendre(compte_rendu="...")` en style Python, 3e forme d'appel sans balises : non lue, le compte rendu passait en texte brut | appels `nom(arg=valeur, ...)` lus avec `ast` (valeurs littérales seulement, rien n'est exécuté ; chemins Windows à `\` simple acceptés) |
| le rédacteur relisait puis réécrivait tout `resume.md` à chaque fichier : le contexte grossit à chaque tour | nouvel outil `ajouter_fichier` (écrit à la fin, sans relire), permission « demander » |

## Quatrième run (06/10) : recherche web, dates inventées

Objectif : les trois dernières versions de PyTorch, leur date et deux nouveautés, dans
`pytorch.md`. Le run a l'air réussi (11,8 min, versions 2.12 à 2.14 et liens justes), mais
**les trois dates sont fausses** : 27 juin, 29 août et 28 octobre 2024, au lieu de mai, juillet
et septembre 2026 (calendrier officiel, `RELEASE.md` du dépôt PyTorch). Le modèle ne savait pas
quel jour on est (ses connaissances s'arrêtent avant) et a complété de mémoire ; rien ne
l'obligeait à citer la source d'une date.

| Cause | Correction |
|---|---|
| le modèle ignore la date du jour | « Nous sommes le <date>… vérifie avec les outils au lieu de répondre de mémoire » dans les consignes de l'orchestrateur, des agents et du chat |
| rien n'impose une source | règle des sources pour les agents : chaque date, chiffre, nom vient d'une page ou d'un fichier lu, avec sa source, sinon « non trouvé » ; l'orchestrateur fait vérifier ce qui manque ou paraît incohérent |
| `rende` puis `{"compte_rendu": ...}` (nom tronqué) non lu | dernier recours : un objet JSON qui termine le message et dont les clés correspondent aux arguments d'un seul outil (ambigu = rien) |

Effet de bord traité : la date change le message système chaque jour, donc la clé du cache de
préfixe (P9). Le dossier `<store>\prefix_cache` garde désormais les 24 préfixes utilisés le plus
récemment.

La leçon : un résultat propre, avec des liens, peut contenir un chiffre inventé. C'est ce
qu'il faut relire en premier.

## Cinquième run (06/10) : bonnes années, mais le chercheur n'a pas rendu

Même objectif, avec la date du jour et la règle des sources. Progrès : les dates sont en 2026,
chacune notée avec sa source, et le chercheur lit la page des releases GitHub. Mais :

| Ce qui s'est passé | Correction |
|---|---|
| le chercheur atteint sa limite de 8 tours sans appeler `rendre` : compte rendu vide, alors que 2.14 et 2.13 étaient notées | à la limite, un dernier tour où seul `rendre` est permis ; sans compte rendu, les notes de l'agent pendant la tâche sont ajoutées ; 10 tours par tâche par défaut |
| l'orchestrateur appelle `chercher_web` lui-même, lit « outil inconnu » et conclut que la recherche web n'existe pas | « c'est un outil des agents, pas le tien : confie la tâche à un agent qui l'a (chercheur) » |
| il veut recréer le chercheur : « existe déjà » | le message donne les outils de l'agent et dit d'utiliser confier, ou un autre nom pour d'autres outils |

## Sixième run (06/10) : deux inventions dans un résultat qui a l'air parfait

Dates en 2026, sources citées, réponse finale propre. Pourtant, vérifié sur les pages de release
GitHub : 2.12 le 13 mai, 2.13 le 8 juillet, 2.14 le 2 septembre ; le modèle avait écrit 17
juillet, 22 août, 18 septembre, avec pour « source » des billets de blog qui ne donnent pas la
date. Et `pytorch.md` contenait des nouveautés inventées (« GPU Hopper », « CUDA 12.4 et cuDNN
8.9 », « 20 % ») : le rédacteur ne voyait pas le compte rendu du chercheur et a comblé le vide.
L'orchestrateur a ensuite demandé au rédacteur de vérifier son propre fichier (« correspond
exactement ») ; la réponse finale montrait les vraies nouveautés, le fichier les fausses.

Leçon : les consignes ne suffisent pas, il faut des garde-fous dans le code.

| Garde-fou | Ce qu'il fait |
|---|---|
| vérification des faits (`moelab/app/factcheck.py`) | dates (jour, mois ; français, anglais, ISO, jj/mm/aaaa), décimaux, pourcentages et multiplicateurs d'une note, d'un compte rendu, d'un fichier écrit ou de la réponse finale, comparés à ce qui a été réellement lu pendant le run (résultats des outils, objectif, date du jour). Un fichier écrit pendant le run n'est pas une source. Compte rendu, fichier et réponse finale : refusés une fois avec la liste des éléments non retrouvés ; ensuite acceptés mais signalés (et visibles dans le déroulé, en rouge) |
| comptes rendus dans le bloc-notes | chaque compte rendu y est ajouté : un rédacteur a ce qu'ont trouvé les autres |
| l'orchestrateur vérifie lui-même | il reçoit `lire_fichier` (lecture seule) ; consigne : relire tout fichier écrit et le comparer aux comptes rendus, ne jamais faire vérifier un agent par lui-même |
| `lire_page` | morceaux de 4 000 caractères (ce que l'agent voit d'un résultat), `debut` pour la suite, `chercher` pour ne recevoir que les passages autour d'un mot (« released » donne « released this 02 Sep ») |

Limite connue : la vérification porte sur les dates et les chiffres, pas sur le sens d'une
phrase. Une nouveauté inventée sans chiffre passe ; c'est le rôle de la relecture par
l'orchestrateur et des comptes rendus dans le bloc-notes.

## Septième run (06/10) : la vérification marche, le contexte déborde

La vérification automatique a fait son travail : « 5 août », « 7 juillet », « 17 juin » signalés
(faux : 2.13 le 8 juillet, 2.12 le 13 mai), « 2 septembre » accepté (juste). Mais 39,7 minutes et
aucun fichier :

| Ce qui s'est passé | Correction |
|---|---|
| chercheur créé avec `chercher_web` seul : chaque `lire_page` refusé, il n'avait que les extraits de recherche | outils compagnons ajoutés d'office : `chercher_web` donne `lire_page` (comme les outils de fichiers donnent la navigation) |
| « CUDA out of memory » deux fois dans `confier` : le contexte (comptes rendus, bloc-notes recopié dans chaque tâche, relances) a dépassé le cache statique ; au-delà, passe normale, qui ne tient pas dans 6 Go | chaque prompt est ramené sous le cache statique moins la place de la réponse : anciens résultats raccourcis, puis anciens échanges retirés (système et 1re demande gardés) ; bloc-notes joint limité aux notes récentes (3 000 caractères) ; en cas d'erreur mémoire : mémoire libérée, conversation raccourcie à 60 %, un nouvel essai |
| après l'erreur, le prompt entier de l'orchestrateur s'est affiché dans le déroulé | l'affichage en direct est remis à zéro avant chaque génération |
| 14 tours épuisés à revérifier, pas de rédacteur | conséquence des trois défauts ci-dessus |

## Le contexte des agents (08/10), d'après DeepSeek et Xiaomi MiMo

Sur la RTX 2060, le contexte coûte deux fois : la place (cache statique en VRAM, 8 192 tokens) et
le calcul (chaque token de prompt passe par les experts, souvent lus sur le disque : ~4 s avant
le premier token). Ce qui a été fait :

| Mécanisme | Inspiré de | Ce que ça change |
|---|---|---|
| **Repli par tâche** : entre deux tâches, un agent ne garde que les demandes et ses comptes rendus ; le raisonnement et les résultats d'outils de la tâche finie disparaissent | règles de contexte de DeepSeek-V3.2 (raisonnement gardé tant que seuls des résultats d'outils arrivent, jeté à la demande suivante) | contexte court d'une tâche à l'autre ; plus d'état de décodage par agent à garder en RAM |
| **Grand nettoyage à 80 %** : une seule coupe jusqu'à 45 % (résultats -> référence d'archive, comptes rendus -> bloc-notes, texte de l'assistant -> ses appels), puis la conversation ne fait que s'allonger | DeepSeek-V3.2 sur BrowseComp : effacer l'historique des outils (67,6) a battu le résumé (60,2) et rien (51,4) | le cache (P1) reprend la conversation à chaque tour ; avant, raccourcir un peu à chaque tour forçait à tout recalculer à chaque tour |
| **Archive et `relire`** : tout résultat de plus de 600 caractères reçoit une référence [r1], [r2]… ; `relire(ref, chercher=…, debut=…)` le retrouve même après un nettoyage | « just-in-time retrieval » : garder des identifiants, recharger à la demande | un nettoyage ne perd rien : les pages restent consultables, par passages |
| **Même début de prompt pour tous les sous-agents** : message système commun, mêmes outils dans le même ordre ; le nom, le rôle et les outils permis sont dans le 1er message ; un outil non reçu répond « Refusé » sans s'exécuter | stabilité du préfixe (cache de contexte de DeepSeek, cache par requête de MiMo) | un seul cache de préfixe P9 pour tous les agents : ~1 500 tokens de prefill évités à chaque nouvel agent |
| **Plusieurs appels par message** : consigne explicite (lire trois pages en un message) | — | moins de tours, donc moins de fois les ~4 s d'attente |
| **`lire_fichier(chercher=…)`** comme `lire_page` | — | seulement les passages utiles dans le contexte |
| **Statistiques de contexte** du run : générations, tokens de prompt, part reprise du cache, nettoyages, replis, résultats archivés (affichées à la fin du run) | — | mesurer l'effet sur un vrai run |

### Pas encore fait, et pourquoi

| Idée | Pourquoi pas maintenant |
|---|---|
| **Cache de contexte en int8** (2× plus de tokens pour la même VRAM) | Il faut un calcul d'attention qui lise directement l'int8 : décompresser tout le cache à chaque token demanderait un tampon fp16 de la même taille, donc aucun gain de place. C'est un noyau Triton à écrire puis à mesurer sur la carte (qualité par KL, vitesse) ; sans la RTX 2060 ici, impossible à valider. Plus simple d'abord : `graph_max_len` à 16 384 dans l'écran Moteur (~190 Mo de VRAM en plus pour Qwen3-Next, 24 Ko par token) et mesurer avec l'autopilot |
| **Décodage spéculatif** (par recopie du prompt, ou tête MTP de Qwen3-Next) | Vérifier plusieurs tokens d'un coup charge l'union de leurs experts : avec des experts lus sur le disque, le gain peut s'annuler. Et revenir en arrière dans l'attention linéaire de Qwen3-Next demande un instantané des états à chaque essai. À prototyper et mesurer sur la carte |
| **Consignes internes en anglais** (−20 % de tokens environ) | risque sur la qualité des réponses en français ; à tester par un run comparé |

## Critères, vérificateur final, leçons (08/10)

Run du 08/10 (contexte optimisé) : toutes les dates et nouveautés justes, 82 % du prompt repris du
cache, 22,5 min au lieu de 39,7 ; mais PyTorch 2.14, la plus récente, oubliée au profit de 2.11. Une
erreur de compréhension, pas une invention : la vérification des faits ne peut pas l'attraper.
Plutôt qu'une règle pour « dernières » (il y aura d'autres mots mal compris), trois mécanismes
généraux, sans vérification après chaque agent :

| Mécanisme | Quand | Coût | Rôle |
|---|---|---|---|
| **Critères de réussite** | une fois au départ | une génération courte (~1 min) | l'objectif devient 3 à 6 critères vérifiables, avec la date du jour ; affichés dans l'écran Agents, corrigeables pendant 90 s (réglable, 0 = ne pas attendre), puis le run continue |
| **Vérificateur final** | une fois, avant d'accepter `terminer` | ~1 à 3 min | agent neuf (même préfixe commun, en cache) : objectif, critères, réponse, références des résultats archivés, sans l'historique du run ; `relire` pour contrôler ; « OK » ou « PROBLÈMES : … » ; un problème renvoie l'orchestrateur corriger une fois. Mode automatique : seulement si la réponse contient des dates ou chiffres, ou si l'objectif demande un choix (dernier, meilleur, liste, version, prix…) ; ou toujours, ou jamais |
| **Leçons** (`moelab/app/lessons.py`, `app_data\lessons.json`) | ajoutées au départ | quasi nul (choix par mots communs, dans le code) | « Corriger » sous une réponse, ou « Garder comme leçon » sur un problème du vérificateur, enregistre une leçon ; les leçons proches d'un nouvel objectif sont données aux critères, à l'orchestrateur et au vérificateur. Le système s'améliore avec l'usage |

Autres changements : tout résultat d'outil est archivé (r1, r2…), même court, pour que le vérificateur
voie tout ce qui a été lu ; temps des critères et de la vérification affichés en fin de run.

### Run du 08/10 midi : 62 minutes, à cause du vérificateur

Critères justes (« les trois versions stables les plus récentes publiées avant le 8 octobre 2026 »),
dates inventées par le chercheur repérées puis corrigées par l'orchestrateur, fichier écrit avec les
bonnes dates. Puis le vérificateur a affirmé que le fichier contenait 2026-09-10 et 2026-05-15 :
faux. Il ne voyait pas le fichier mais le résumé de l'orchestrateur, et le bloc-notes, où restait le
premier compte rendu (périmé) ; il a aussi pris pour une erreur un compte rendu coupé au milieu d'un
mot dans le bloc-notes (« torch.ac »). L'orchestrateur s'est perdu 30 minutes à corriger ces faux
problèmes sans jamais rappeler `terminer`.

| Correction | Détail |
|---|---|
| le vérificateur juge le **livrable réel** | la réponse proposée + le contenu des fichiers écrits pendant le run ; plus de bloc-notes |
| **chaque problème cite** le livrable ou une source, **le code vérifie la citation** | « passage du livrable » : problème ; ou MANQUE : … [rN] « passage de la source ». Citation introuvable = problème ignoré (affiché : « N remarques ignorées ») ; les trois faux problèmes de ce run l'auraient été |
| **4 tours** pour corriger après un refus | au-delà, ou si `terminer` n'est pas rappelé : la dernière réponse proposée, avec les remarques du vérificateur, au lieu d'un vrac de comptes rendus |
| bloc-notes : comptes rendus **coupés en fin de phrase** | avec un renvoi `relire('rN')` vers le texte complet |
| le compte rendu du vérificateur échappe au contrôle des faits | il cite les affirmations douteuses pour les contester |

### Run du 08/10 après-midi : 68,7 minutes, livrable faux accepté

Fichier écrit avec 2.13.0 et 2.12.1 (un correctif), « troisième version non trouvée ». Les vraies :
2.14.0 (2 septembre), 2.13.0 (8 juillet), 2.12.0 (13 mai), plus le correctif 2.14.1. Ce qui s'est passé :

1. le chercheur a lu la page des releases mais n'a retenu que 2.13.0 et 2.12.1 ;
2. l'orchestrateur en a conclu que 2.13.0 était la plus récente et a demandé à un 2e agent « la
   troisième version **après 2.12.1** ». Cet agent a ouvert la page de 2.14.0… et ne l'a pas signalée,
   puisque sa tâche demandait une version plus ancienne. L'hypothèse de l'orchestrateur s'est
   propagée de tâche en tâche ;
3. ce 2e agent s'appelait « verificateur » : le contrôle final a **réutilisé cet agent** (ses outils,
   son historique) au lieu d'un agent neuf ;
4. le contrôle final a pourtant trouvé 2.14.0, sortie le 2026-09-02, avec deux nouveautés. Mais il l'a
   écrit en texte libre, sans le format « MANQUE : … [rN] « citation » » : remarque ignorée, livrable
   accepté (« tous les critères sont remplis »). La règle de citation du run précédent avait jeté un
   vrai problème ;
5. les comptes rendus des agents étaient listés au vérificateur comme « sources » (r8, r13…).

| Correction | Détail |
|---|---|
| une remarque sans citation est **gardée si elle apporte des faits** | versions x.y.z, dates, chiffres présents dans une source lue (résultat d'outil) et absents du livrable : « MANQUE (relevé dans les sources, absent du livrable : 2.14.0, 2026-09-02) ». Un fait inventé par le vérificateur reste ignoré : il n'est dans aucune source |
| **sources = résultats d'outils** seulement | les comptes rendus restent relisibles (`relire`) mais ne prouvent rien |
| nom **« verificateur » réservé** au contrôle final | l'orchestrateur reçoit une erreur et choisit un autre nom (ex. controleur) |
| orchestrateur : **redonner la demande d'origine** | quand un compte rendu est incomplet, sans y mettre sa propre hypothèse (« la version avant X ») |
| sous-agents : **signaler ce qui contredit la tâche** | en tête du compte rendu (« un élément plus récent que ce qu'on m'a dit ») |

Le temps (68,7 min, dont 12 min de vérification) vient surtout des allers-retours dus à l'erreur de
départ ; avec la remarque gardée, l'orchestrateur a 4 tours pour faire corriger le fichier.

### Run du 08/10, 17 h : bon résultat, mais 67,9 minutes

Fichier juste (2.14.1, 2.14.0, 2.13.0 avec leurs dates et deux nouveautés). Vérification finale sautée :
la limite de 60 minutes était dépassée. 39 générations, ~1,7 min chacune : sur cette machine, le temps
d'un run est d'abord le **nombre de générations**. Une vingtaine ont été perdues :

| Perte | Cause | Correction |
|---|---|---|
| ~17 min : le chercheur lit 2.12, 2.11, 2.10 au lieu de chercher une 2e nouveauté de 2.13.0 | au 2e grand nettoyage de son contexte, **sa tâche en cours a été retirée** (seuls le système et le 1er message étaient protégés) | la tâche en cours est marquée « à garder » : ni réduite ni retirée par le nettoyage ou le filet de sécurité |
| 3 générations : « lire_fichier n'est pas autorisé » | le rédacteur n'avait reçu que ecrire_fichier (+ lister, chercher) | **les outils de lecture** (lire_fichier, lister_dossier, chercher_fichiers, lire_page, chercher_web, calculer, etat, reglages) sont accordés au premier appel ; écrire, ajouter, exécuter du Python restent à donner explicitement. Les permissions de l'écran Outils s'appliquent toujours |
| ~6 générations : un rédacteur créé pour recopier un contenu déjà prêt | l'orchestrateur ne pouvait pas écrire | **l'orchestrateur écrit lui-même** un fichier dont il a tout le contenu (même contrôle des faits) |
| des centaines de tokens par message | le rédacteur recopiait le fichier dans son texte, puis dans l'appel | consigne : le contenu une seule fois, dans l'appel |

Ce même run, sans ces pertes : ~18 générations, soit ~30 min. Pour une tâche de ce genre (une
recherche, un fichier), le chat avec outils (un seul agent) reste plus rapide que l'orchestrateur.

### 13/10 : agents « Échec » avec un fichier juste, chat arrêté après 8 tours

**Agents (62,5 min).** Le fichier écrit était juste pour 2.13.0 et 2.12.0 (2.14.0 avec la date du blog,
17/09, au lieu de la sortie, 02/09). La réponse finale a pourtant été « Échec », pour trois raisons :

1. les **critères avaient inventé des exigences** : « GitHub Releases ou pytorch.org seulement, aucun blog »,
   « vérifier le chemin, sinon échouer ». L'utilisateur n'avait rien demandé de tel ; le blog officiel est
   une source officielle. L'orchestrateur a fini par refuser son propre travail ;
2. le vérificateur a affirmé que 2026-07-08 et 2026-05-13 n'étaient dans aucune source : faux, les pages lues
   les donnaient (« 08 Jul », « May 13, 2026 ») ;
3. l'orchestrateur a voulu vérifier une date lui-même avec lire_page (refusé : un tour perdu), puis le temps
   était écoulé.

| Correction | Détail |
|---|---|
| critères : **rien d'inventé** | 2 à 5 critères qui reformulent la demande ; aucune exigence non exprimée (sources imposées, format, vérifications, cas d'échec) ; site, blog et dépôt officiels également valables |
| vérificateur : **une absence affirmée est contrôlée** | « « X » n'apparaît pas / non trouvé… » est rejeté si les dates et chiffres de X sont dans les sources lues, sauf si la remarque apporte une autre valeur lue (« la sortie est September 2 ») : celle-là est gardée |
| orchestrateur : **outils de lecture** | lire_page, chercher_web, calculer… en plus des fichiers : vérifier une date ne demande plus un agent entier |

**Chat avec outils.** Après 8 tours (une recherche, sept pages lues), le chat s'arrêtait sans réponse ni
fichier. Maintenant 16 tours, puis un dernier tour « termine avec ce que tu as » (écrire le fichier, dire ce
qui manque). Consigne ajoutée : plusieurs appels par message, `chercher=…` dans les longues pages, écrire
dès que l'essentiel est trouvé.

Vitesse mesurée dans ce chat : 2,1 tok/s et 139 s avant le premier token (prompt avec la description des
outils). À ce rythme, une tâche web de ce genre prend 15 à 30 minutes même sans erreur.

### 13/10, chat avec outils : 10 tours, environ 10 minutes

Même demande dans le chat (un seul agent) : 10 tours, 954 tokens à 4,8 tok/s, 118 s avant le premier
token ; le fichier est écrit. Six fois plus rapide que l'orchestrateur sur cette tâche. Deux erreurs :

- **2.15 comptée parmi les « dernières versions »** alors qu'elle est annoncée pour le 28 octobre (pas sortie) ;
  ses « nouveautés » sont tirées d'une page de calendrier ;
- une nouveauté de 2.14 (NCCL2) vient probablement de la page de 2.15 : mélange entre deux pages.

Correction générale : toute **écriture de fichier qui cite une date postérieure à aujourd'hui** est refusée une
fois, avec la question « prévu ou déjà sorti ? » ; réécrit tel quel, le contenu passe (un planning peut contenir
des dates futures). Le contrôle se fait avant ta demande d'accord, dans le chat comme chez les agents. La
règle de date le dit aussi : « dernier » veut dire déjà paru. Le mélange entre pages, lui, n'est attrapé que
par le vérificateur des agents.
