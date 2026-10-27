# Guide d'exécution (Windows, PowerShell)

Toutes les commandes se lancent **depuis la racine du repo** :

```powershell
cd C:\Users\Home\becode_projects\moe-offload-lab
```

Le modèle OLMoE au format Hugging Face est déjà dans `D:\models\olmoe-hf`. Le store
d'experts (15,2 Gio en fp16 + int4) va sur C:, le NVMe (37 Go libres) : les experts
fp16 manquants y sont lus bien plus vite que sur le HDD.

---

## 1. Dépendances (5 min)

torch doit être la version **CUDA** (`+cu130`). Depuis PyPI, pip installe la version
CPU sous Windows : on l'installe donc à part, puis le reste.

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available())"
```

Si ça n'affiche pas `+cu130 True` :

```powershell
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Puis :

```powershell
pip install -r requirements.txt
```

Ferme aussi tout noyau Jupyter qui garde l'ancien modèle chargé en entier (VS Code :
"Restart"/"Close" sur le notebook) : il occupe ~13 Go de RAM.

## 2. Vérifier la machine et le code (1 min)

```powershell
python scripts\check_setup.py --paths C:\ D:\
python -m pytest tests -q
```

Attendu : des lignes `[OK]`, dont les trois auto-tests (CPU seul, GPU avec q* et
prefetch, placement split), puis `Tout est bon.`, et `79 passed` pour les tests (`77 passed, 2 skipped` sans GPU).
Si le script affiche "À corriger avant de continuer", suis ce qu'il indique
(torch CPU au lieu de CUDA, ou pas assez de RAM libre).
C'est la première fois que le chemin CUDA tourne sur une vraie carte : **si une
ligne est en ÉCHEC, envoie-moi la sortie complète avant d'aller plus loin.**

## 3. Construire le store d'OLMoE (5 à 20 min selon le disque)

```powershell
python scripts\build_store.py --model D:\models\olmoe-hf --out C:\moe-store\olmoe
```

Si c'est interrompu, relance la même commande : elle reprend.

## 4. Mesurer les bandes passantes (1 min)

```powershell
python scripts\measure_bandwidth.py --store C:\moe-store\olmoe
```

Enregistre B_PCIe et B_Host dans `bandwidth.json`, utilisés par q*.

## 5. Voir le plan de placement (quelques secondes)

```powershell
python scripts\plan.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe
```

## 6. Discuter avec le modèle

```powershell
python scripts\chat.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe
```

Le chargement prend moins d'une minute (au lieu de 12). Après chaque réponse, une
ligne de statistiques : tokens/s de la génération seule (depuis le 03/10 ; avant, le
temps de lecture du prompt était compté dedans), temps avant le premier token, tokens du
prompt repris du cache (P1 : seul le nouveau message est calculé), hit VRAM, part de fp16 demandée / utilisée, part CPU,
lectures disque par token, rappel du prefetch. Les commandes se tapent **seules,
sur leur propre ligne**, avant la question suivante : `/policy lo`, `/policy hi`,
`/t1 0.4`, pour sentir les différences. En quittant, l'historique
d'usage est sauvegardé : le lancement suivant précharge les experts chauds.

## 7. Mesurer le routage d'OLMoE (2 à 5 min)

```powershell
python scripts\measure_routing.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe
```

Trois chiffres à comparer aux papiers : proxy HOBBIT (0,99 sur Mixtral), prefetch
(71,6% sur GLM-5.2), localité temporelle (41,3%).

## 8. Le benchmark A/B (10 à 40 min)

```powershell
python scripts\benchmark.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe --repeats 2
```

Résultat : `results\benchmark_olmoe-hf_<date>.md` et `.json`.

Puis le balayage du seuil HOBBIT (20 à 60 min) :

```powershell
python scripts\benchmark.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe --configs hi,lo,hobbit@0.2,hobbit@0.4,hobbit@0.6,hobbit@0.8 --repeats 2
```

**Envoie-moi ces deux fichiers + celui de l'étape 7** : on met à jour le README
avec les vrais chiffres, et on remplit le brouillon pour Colibrì (docs/16).

## 8 bis. Optimiser (doc 18)

Installer Triton (optionnel, pour le noyau int4) puis mesurer toutes les variantes :

```powershell
pip install triton-windows
python scripts\check_setup.py --paths C:\ D:\
python scripts\measure_bandwidth.py --store C:\moe-store\olmoe
python scripts\measure_routing.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe
python scripts\profile_moe.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe --ram-gb 5
```

`check_setup` vérifie les noyaux Triton et les CUDA graphs sur ta carte,
`measure_bandwidth` mesure les lectures en parallèle, `measure_routing` calibre les
seuils par couche, et `profile_moe` compare les dernières variantes (environ 10 minutes).

Les CUDA graphs (A6) sont activés par défaut (`--no-cuda-graphs` pour les couper) :
cache KV statique de 2 048 tokens pour OLMoE et 8 192 pour Qwen3-Next (automatique d'après
la taille du cache KV du modèle), `--graph-max-len` pour changer. La réserve VRAM
(`--vram-reserve-gb`) est aussi calculée automatiquement : activations + ce cache KV.

## 8 ter. Le store recommandé pour hobbit : int8 + int4

Depuis le 01/10, la haute précision recommandée est l'int8 (2 fois plus petit que le
fp16, quasi sans perte) : `hobbit` passe de 5,6 à 22,6 tok/s au benchmark.

```powershell
python scripts\build_store.py --model D:\models\olmoe-hf --out C:\moe-store\olmoe-i8 --hi int8 --lo int4
python scripts\measure_bandwidth.py --store C:\moe-store\olmoe-i8
python scripts\chat.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --policy hobbit --t1 0.8
```

Mesurer la perte int8 → fp16 (référence fp16 enregistrée, puis réutilisée) :

```powershell
python scripts\benchmark.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe --ram-gb 5 --configs lo --prompts 1 --no-dense-int8 --save-ref results\ref_fp16.pt
python scripts\benchmark.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --ram-gb 5 --configs hi,lo,hobbit@0.8 --ref results\ref_fp16.pt
```

## 8 quater. Jouer avec les réglages

Tous les réglages sont décrits dans `moelab/options.py` : ce qu'ils font, leurs valeurs,
l'effet mesuré sur ta machine, et s'ils se changent à chaud, en rechargeant le modèle,
ou avec un autre store. Dans `chat.py` :

```
/options                 réglages principaux et valeurs actuelles (/options all : tous)
/set t1 0.5              changer un réglage à chaud
/set cuda_graphs off     couper une optimisation pour voir ce qu'elle apporte
/preset vitesse          vitesse (lo), équilibre (hobbit 0,5), qualité (hobbit 0,8), référence (hi)
```

La vitesse s'affiche après chaque réponse. Pour la qualité, le benchmark avec la
référence fp16 (section 8 ter). C'est ce même catalogue que l'app affichera.

## 8 quinquies. Les pistes du 02/10 (à mesurer)

Cinq optimisations trouvées dans les papiers récents (doc 18, « Les pistes du 02/10 »).
Dans l'ordre, environ 1 h en tout :

```powershell
python -m pytest -q
python scripts\check_setup.py --paths C:\ D:\

# vitesse : A5 (token entier en un graph) et S3 (éviction least-stale + prefetch selon le score)
python scripts\profile_moe.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --ram-gb 5 --dense-int8 --variants defaut,a5,s3,tout

# qualité Q1 : meilleure échelle int4, réécrite en place (~5 min), puis qualité contre le fp16
python scripts\requant_store.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --ram-gb 5 --method mse
python scripts\benchmark.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --ram-gb 5 --dense-int8 --configs lo,hobbit@0.8 --ref results\ref_fp16.pt

# qualité Q2 : GPTQ (calibration ~10 min + ~10 min de réécriture, ~2 Gio temporaires sur D:)
python scripts\requant_store.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --ram-gb 5 --method gptq --work-dir D:\moe-calib
python scripts\benchmark.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --ram-gb 5 --dense-int8 --configs lo,hobbit@0.8 --ref results\ref_fp16.pt

# S5 : le décodage spéculatif vaudrait-il le coup ?
python scripts\measure_lookup.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --ram-gb 5 --dense-int8 --policy lo
```

`requant_store.py --method absmax` remet l'ancienne quantification si besoin.

Mesures du 02/10 : Q1 + Q2, A5 et D5 adoptés (A5 actif par défaut : `lo` 132 tok/s ;
`hobbit` ~29 tok/s au benchmark), S3 et S5 rejetés (doc 18). Vérification du correctif du
pool de RAM (`hobbit` après `lo` doit rester à ~29 tok/s) :

```powershell
python scripts\benchmark.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe-i8 --ram-gb 5 --dense-int8 --configs lo,hobbit@0.8 --ref results\ref_fp16.pt
``` Dans
`chat.py` : `/set full_graph on`, `/set cache_eviction stale`, `/set prefetch_mass 0.9`.

## 8 sexies. Le temps avant le premier token (03/10)

```powershell
python scripts\profile_prefill.py --model D:\models\qwen3-next-80b-fp8 --store C:\moe-store\qwen3-next --ram-gb 5
```

Environ 20 à 30 minutes sur Qwen3-Next. Pour chaque variante (P2 à P5, doc 18) : temps du
prefill, part MoE / dense / attente disque, experts par couche et d'où ils viennent, débit
disque obtenu, vitesse et qualité des tokens qui suivent ; puis une conversation de 3
messages avec et sans P1. Résultat : `results\prefill_<modèle>_<date>.md`.

## 8 septies. L'agent local (doc 19)

```powershell
python scripts\agent.py --model D:\models\qwen3-next-80b-fp8 --store C:\moe-store\qwen3-next --ram-gb 5
```

Le modèle appelle des outils sur moelab lui-même (état, réglages, changer un réglage,
calculer). Les appels et leurs résultats s'affichent sous `[outil]`. Exemples : « Dans quel
état est le moteur ? », « Quels réglages touchent au prefill ? », « Passe en hobbit avec
t1 = 0,7 ». Le 1er message est plus lent (description des outils à calculer).

## 8 octies. L'autopilot (doc 19)

```powershell
python scripts\autopilot.py --model D:\models\qwen3-next-80b-fp8 --store C:\moe-store\qwen3-next --ram-gb 5 --start 02-10 --blind
```

Le modèle mène lui-même les essais (30 à 45 min pour 12 essais). Chaque essai s'affiche
avec son verdict (GARDÉ, pas mieux, qualité dégradée, bruit, échec). Sans `--start 02-10
--blind`, il part des réglages actuels et écrit `<store>\autopilot_profile.json`, que le
chat applique ensuite.

## 8 nonies. L'app (doc 20)

```powershell
pip install -r requirements.txt
python scripts\make_standalone.py --model D:\models\qwen3-next-80b-fp8 --store C:\moe-store\qwen3-next
python scripts\app.py --store C:\moe-store\qwen3-next --ram-gb 5
```

La 2e ligne (une fois, quelques minutes) copie la partie dense dans le store : ensuite
`--model` n'est plus utile et le checkpoint FP8 de D: peut être supprimé. Le navigateur
s'ouvre sur `http://127.0.0.1:8000` ; les fois suivantes, `python scripts\app.py` suffit
(il recharge le dernier modèle). Ctrl+C pour arrêter.

## 8 decies. Les agents (doc 21)

Dans l'app, écran **Agents** : écris un objectif, coche les outils permis, « Lancer
l'orchestrateur ». Premier essai conseillé, court et vérifiable :

> Lis les fichiers texte de C:\Users\Home\Documents\test, puis écris dans
> C:\Users\Home\Documents\test\resume.md un résumé de 5 lignes par fichier.

Ajoute d'abord ce dossier dans Outils et MCP (dossiers autorisés). Compte 10 à 20 minutes ;
tu peux continuer à utiliser le chat pendant ce temps.

## 8 undecies. Élaguer des experts avec REAP (doc 22)

Une seule commande, l'app fermée (il faut une fois `pip install datasets` et
`pip install "huggingface-hub>=1.31,<2.0"`) :

```powershell
python scripts\reap_pipeline.py --store C:\moe-store\qwen3-next --ram-gb 5
```

Elle fait la référence, la calibration, l'essai sur les couches 10 à 20, puis l'élagage à 25, 40 et 50 %,
et écrit un tableau comparatif. Ce qui est déjà fait est sauté : Ctrl+C à tout moment, la même commande
reprend. Les résultats sont dans `C:\moe-store\qwen3-next\reap` (à côté du store, pas dans le repo :
copier une nouvelle version du repo ne les efface pas). Rien n'est écrit dans le store lui-même.

## 9. Le palier final : Qwen3-Next-80B

Tout est dans [`docs/15-palier-final-qwen3-next.md`](docs/15-palier-final-qwen3-next.md).
En résumé : `hf auth login`, télécharger la version FP8 (82 Go), construire un store
int4 seul (38,2 Gio), puis `chat.py --dense split`. Sur le HDD, compte ~0,1 à 0,3
token/s : c'est l'objectif "faisabilité avant vitesse".

## Nettoyage possible

Le GGUF d'OLMoE (`D:\models\olmoe\OLMoE-1B-7B-0125-Instruct-F16.gguf`, 13,8 Go) ne
sert plus depuis le passage à PyTorch : tu peux le supprimer.
