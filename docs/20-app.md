# 20. L'app moelab : un seul endroit pour tout faire en local

## L'idée (05/10)

Après l'autopilot (doc 19), tout se faisait en lignes de commande. L'app rassemble le tout
dans une page web servie par ton PC (`127.0.0.1` seulement : rien ne sort) et vise un but :
**quelqu'un qui télécharge le repo doit pouvoir refaire la même chose que nous**, sans
connaître les scripts. Choisir un modèle, vérifier la place, télécharger, construire le
store, mesurer le disque, discuter, laisser l'autopilot régler le moteur.

```powershell
pip install -r requirements.txt
python scripts\app.py
```

Le navigateur s'ouvre sur `http://127.0.0.1:8000`. Avec `--store <dossier>` le modèle se
charge au démarrage ; sans, l'app recharge le dernier modèle utilisé.

## Les écrans (six au 05/10, sept avec Agents le 06/10)

| Écran | Ce qu'on y fait |
|---|---|
| Agents (06/10, doc 21) | un objectif, un orchestrateur qui crée ses agents et leur confie le travail, déroulé en direct |
| Chat | conversations enregistrées (`app_data\conversations`), réponse en direct, chiffres sous chaque réponse (tok/s, 1er token). Case « avec outils » : le modèle peut appeler les outils de l'écran Outils ; chaque appel s'affiche en carte, ceux en « demander » attendent ton accord. |
| Mise en route | ta machine (GPU, RAM, disques), catalogue (OLMoE, Qwen3-30B-A3B, Qwen3-Next-80B), plan de place par précision, installation suivie en direct, bibliothèque des stores (charger, recharger avec d'autres budgets, retirer). |
| Autopilot | même boucle que `scripts\autopilot.py` : graphique « réponse type » par essai, journal, raisonnement du modèle en direct, rapports. |
| Outils et MCP | permission de chaque outil (toujours / demander / jamais), dossiers autorisés, serveurs MCP à brancher, moelab comme serveur MCP. |
| Moteur | jauges VRAM / RAM / disque, dernière réponse, tous les réglages de `options.py` avec leur effet mesuré ; « à chaud » s'applique tout de suite. |
| Tâches | file d'attente, tâche de fond « un dossier de fichiers → une instruction → un fichier Markdown ». |

Le bandeau en bas de la barre de gauche montre en permanence le modèle chargé, le remplissage
des trois étages (VRAM, RAM, disque) et ce que fait le moteur.

## Un seul modèle en mémoire : la file d'attente

Sur 6 Go de VRAM et 16 Go de RAM, un seul modèle tient. Tout ce qui touche au modèle passe
donc par **un seul fil de travail** et une file à priorités :

| Priorité | Travail |
|---|---|
| 0 | changer un réglage |
| 1 | message du chat |
| 2 | charger un modèle |
| 3 | autopilot |
| 4 | agents (06/10 : laisse passer le chat entre deux tours) |
| 5 | tâche de fond |
| 9 | installation |

Une tâche de fond se découpe elle-même : un fichier par passage, puis elle se remet dans la
file. Un message du chat passe donc entre deux fichiers, au lieu d'attendre la fin du dossier.
Le téléchargement d'un modèle tourne dans son propre fil (il ne touche pas au GPU) ; seule la
construction du store prend le fil de travail, décharge le modèle et le recharge ensuite.
Chaque travail peut être arrêté (bouton « Arrêter ») ; le chat s'arrête au token suivant.

## Mise en route : la place d'abord

Le plan d'installation compte, pour chaque précision, ce que prennent le téléchargement et le
store, sur quels disques, avec la place libre réelle (même disque : les deux s'additionnent).
Il signale aussi l'absence de GPU NVIDIA et le manque de RAM.

| Modèle | Téléchargement | Store int4 | Store int8 + int4 |
|---|---|---|---|
| OLMoE 1B-7B | ~13,8 Go | 3,2 Gio | 9,4 Gio |
| Qwen3-30B-A3B (FP8) | ~31 Go | 14,3 Gio | 42,2 Gio |
| Qwen3-Next-80B (FP8) | ~82 Go | 38,2 Gio | 112,5 Gio |

Installation : téléchargement (progression d'après la taille sur disque) → store → store
autonome → mesure des bandes passantes → bibliothèque → suppression du checkpoint si tu ne
coches pas « garder ».

### Le store autonome

Avant, il fallait garder le checkpoint téléchargé (82 Go pour Qwen3-Next) à côté du store :
la partie dense (attention, embeddings, tête) était relue dedans à chaque chargement.
`make_standalone` copie cette partie dense, la config et le tokenizer dans `<store>\model`
(morceaux de 1 Go au plus). Le checkpoint peut ensuite être supprimé ; `--model` devient
facultatif partout.

```powershell
python scripts\make_standalone.py --model D:\models\qwen3-next-80b-fp8 --store C:\moe-store\qwen3-next
```

`build_store.py` le fait maintenant par défaut. Pour Qwen3-30B-A3B : ~31 Go au pic du
téléchargement plus le store qui se construit (~46 Go au total), ~16,5 Go une fois le
checkpoint supprimé.

### Qwen3-30B-A3B

La famille Qwen3-MoE (128 experts, 8 actifs, 48 couches) passe par le même chemin que
OLMoE : pas d'adaptateur à écrire. Test sur un modèle miniature (`tests/test_qwen3moe.py`).
Pas encore mesuré sur la RTX 2060.

## Outils et MCP

**Les outils du chat** : `etat`, `reglages`, `changer_reglage` (moelab lui-même), `calculer`,
`lister_dossier`, `chercher_fichiers`, `lire_fichier` (texte et PDF), `ecrire_fichier`,
`lire_page`. Les outils de fichiers ne voient que les dossiers autorisés. Par défaut, ceux
qui changent quelque chose (`changer_reglage`, `ecrire_fichier`) demandent ton accord.

**moelab client MCP** : tu ajoutes un serveur (nom, commande, arguments), moelab le lance et
ses outils apparaissent sous `serveur__outil`. Exemple, le serveur de fichiers officiel :
nom `fichiers`, commande `npx`, arguments `-y @modelcontextprotocol/server-filesystem
C:\Users\Home\Documents`.

**moelab serveur MCP** : une case à cocher. D'autres applications (Claude Desktop, VS Code…)
peuvent alors demander au modèle local via `http://127.0.0.1:8000/mcp/` : `demander`,
`etat_moteur`, `lancer_autopilot`. Leurs demandes passent par la même file d'attente que le
chat. L'écran donne les deux configurations à copier (VS Code directement ; Claude Desktop via
`mcp-remote`).

## Comment c'est construit

| Fichier | Rôle |
|---|---|
| `moelab/app/engine.py` | le modèle chargé, le fil de travail, la file à priorités, les événements de chaque travail, les demandes d'accord |
| `moelab/app/chat.py` | conversations, réponse avec ou sans outils (agent du doc 19) |
| `moelab/app/tools.py` | outils, permissions, dossiers, client MCP |
| `moelab/app/onboarding.py` | machine, catalogue, plan de place, installation, bibliothèque |
| `moelab/app/jobs.py` | autopilot et tâche de fond |
| `moelab/app/mcp_server.py` | moelab serveur MCP |
| `moelab/app/server.py` | API (FastAPI) et flux en direct (SSE) |
| `moelab/app/static/` | les six écrans, sans framework |
| `scripts/app.py` | lancement |

Pourquoi un serveur web : le navigateur est l'interface la plus simple à faire marcher sous
Windows sans rien installer de plus, et la même API sert l'écran, le serveur MCP et, plus
tard, d'autres clients.

## Vérifié (05/10)

- `tests/test_app.py` (8 tests, modèle miniature sur CPU) : chargement, réponse en direct,
  outil avec accord, réglages, tâche de fond découpée, mise en route, bibliothèque, MCP.
- Les six écrans ouverts dans Chromium sur le modèle miniature : aucune erreur JavaScript,
  conversation envoyée et reçue en direct, plan de place, éditeur MCP, réglage appliqué à chaud,
  affichage téléphone. Corrigés au passage : un « null » affiché sur l'écran Moteur, des
  tokens `<pad>` dans les conversations rechargées, boutons « Appliquer » qui passaient à la
  ligne.
- Pas encore lancé sur la RTX 2060 avec Qwen3-Next : c'est le prochain test.

## Chat avec outils : contexte tenu en mémoire (13/10)

Une recherche web dans le chat (huit pages lues, ~1 000 tokens chacune, plus la description des outils)
dépassait le cache statique de 8 192 tokens : passe normale, puis « CUDA out of memory ». Le chat a
maintenant la même garde que les agents (doc 21) : grand nettoyage à 80 % de la fenêtre (anciens résultats
d'outils réduits à quelques lignes), filet de sécurité qui raccourcit si besoin, et un nouvel essai après
avoir libéré la mémoire si elle déborde quand même. La conversation enregistrée garde les résultats
complets ; seul ce que voit le modèle est réduit. Limite : 16 tours, puis un dernier tour pour conclure.

### 13/10, midi : plus de mémoire pleine, mais 17 tours et pas de fichier

Premier token en 37 s (1 769 tokens sur 1 830 repris du cache de préfixe), plus d'erreur de mémoire. Mais :

- **9 tours perdus à deviner des titres** (« New Features », « Highlights », « Performance ») absents des
  pages : chaque essai rendait seulement « n'apparaît pas ». Maintenant `chercher` essaie aussi les mots de
  l'expression un par un, et sinon rend le début de la page : un tour n'est jamais vide ;
- **les dates lues au début ont disparu au nettoyage** : le chat n'avait pas d'archive (les agents si). Les
  résultats de lecture reçoivent une référence [r1], [r2]… et l'outil `relire` les retrouve, même au message
  suivant (texte complet enregistré dans la conversation) ;
- **au dernier tour, il a refusé d'écrire** (« données incomplètes »). Consigne : un résultat partiel vaut
  mieux que rien, écrire le fichier avec « non trouvé » pour ce qui manque.

### 13/10, 13 h 50 : contenu juste dans la réponse, mais pas de fichier

17 tours, 5 tok/s, premier token en 50 s. Les nouveautés sont exactes (tirées des notes officielles), mais :

- **le fichier n'a pas été écrit** : au dernier tour, le modèle a recopié le contenu dans sa réponse au lieu
  d'appeler `ecrire_fichier`. Maintenant, si la demande nomme un fichier, qu'il n'a pas été écrit, et que la
  réponse contient un seul bloc de code, moelab écrit ce bloc dans le fichier, en passant par les mêmes
  contrôles et la même demande d'accord qu'un appel du modèle ;
- **dates « non trouvé » ou fausses** (2.13.0 « 8 juillet 2023 ») : GitHub affiche « 08 Jul » sans l'année ;
  la date complète est dans le code de la page (`datetime="2026-07-08…"`). `lire_page` la garde désormais
  (balises `time`, `relative-time`, `local-time`) ;
- **un mot par appel** (15 `lire_page` sur deux pages) : `chercher` accepte plusieurs termes en un appel,
  séparés par `|` (ex. `2.14.0 | 2.13.0 | 2.12.0`), et le résultat archivé se relit avec `relire`.

### 13/10, 14 h : fichier écrit en 9 tours, mais dates fausses de quelques jours

9 tours, 5,1 tok/s, premier token en 44 s, fichier écrit. Les nouveautés sont justes, les liens aussi ; les
dates non (2.14.0 « 2026-08-27 » au lieu du 2 septembre, 2.13.0 « 07-03 » au lieu du 8 juillet, 2.12.0
« 05-11 » au lieu du 13 mai). Deux causes possibles, deux corrections :

- ma conversion des balises de date (13 h 50) remplaçait le texte affiché (« released this 02 Sep ») par la
  seule date du code : si la page contient d'autres dates (étiquette, commit), on ne savait plus laquelle est
  la sortie. Maintenant les deux restent : « released this 02 Sep (2026-09-02) » ;
- le chat n'avait pas le contrôle des faits des agents. Maintenant, une écriture dont les dates ou chiffres
  ne sont dans aucune page ni aucun fichier lus est refusée une fois, avec la liste ; réécrite, elle passe.

Aussi : le premier appel d'écriture était coupé (le contenu recopié dans le texte, puis dans l'appel, au-delà
de 1 024 tokens). Réponses du chat avec outils jusqu'à 1 536 tokens, et consigne de ne pas recopier.
