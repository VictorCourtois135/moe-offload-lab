# 15 — Paliers suivants : Qwen3-30B-A3B et Qwen3-Next-80B-A3B (étapes 14 et 15)

Le code ne change pas : même store, mêmes caches, mêmes politiques. Seuls changent
le modèle, la taille du store et l'endroit où se trouve le goulot d'étranglement.

## Le modèle final : Qwen3-Next-80B-A3B-Instruct

| | |
|---|---|
| Paramètres | 79,7B au total, ~3B actifs par token |
| Architecture | 48 couches hybrides (3 Gated DeltaNet pour 1 attention), toutes MoE |
| Experts | 512 par couche, 10 actifs par token + 1 expert partagé toujours actif |
| Partie dense (fp16) | 4,41 Gio, dont 1,16 Gio d'embeddings + tête |
| Store int4 | 38,2 Gio (1,59 Mio par expert) |
| Lu par token si tout manque | 765 Mio |

Vérifié par les tests sur un Qwen3-Next miniature : l'expert partagé et les couches
DeltaNet restent dans la partie dense et marchent tels quels ; seuls les 512 experts
routés passent par le store. Sans les noyaux optimisés (`fla`, `causal-conv1d`),
transformers utilise sa version PyTorch des couches DeltaNet : plus lente en
prefill, correcte.

## Ce qui va coincer sur ta machine : le disque

| Ressource | Disponible | Besoin |
|---|---|---|
| VRAM | ~5 Go | 3,25 Gio pour les couches denses (placement `split`) → reste ~0,5-1 Gio pour les experts |
| RAM | 16 Go | 1,16 Gio embeddings + tête, ~6-7 Gio de cache d'experts (~18% des experts) |
| Disque | C: NVMe ~22 Go libres une fois le store d'OLMoE créé, D: HDD 644 Go | 82 Go de checkpoint FP8 + 38,2 Gio de store |

Chaque token utilise 480 experts (48 couches x 10). Même avec 50 à 70% trouvés en
cache (localité temporelle + experts chauds), il reste ~150 à 250 lectures d'experts
par token.

- Sur le **HDD** (D:) : ~10 ms de positionnement + ~10 ms de lecture par expert
  → 3 à 5 secondes par token, soit **~0,1 à 0,3 token/s**.
- Sur le **NVMe** (C:) : ~1 ms par expert → **~1 à 3 tokens/s**.

Ce sont des estimations à partir d'hypothèses de hit rate, pas des mesures : le
chat et le benchmark afficheront les vrais chiffres (lectures disque par token,
hit rates). Colibrì annonce 0,05 à 0,1 tok/s pour son cas extrême (744B sur 25 Go
de RAM) : l'objectif du projet est la faisabilité, pas la vitesse.

Le store se déplace librement (un simple copier-coller du dossier) : on peut le
construire sur D: puis le copier sur C: si tu y libères ~40 Go (par exemple en supprimant le store d'OLMoE une fois ses mesures faites).

## Marche à suivre

**1. Se connecter à Hugging Face** (sinon le téléchargement est limité à ~1 Mo/s,
soit plus de 20 heures pour 82 Go) :

```powershell
hf auth login
```

Colle un token "read" créé sur https://huggingface.co/settings/tokens

**2. Télécharger la version FP8** (82 Go au lieu de 160 Go en bf16) :

```powershell
hf download Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 --local-dir D:\models\qwen3-next-80b-fp8
```

**3. Construire le store int4 seul** (38,2 Gio, reprenable, compte 1 à 2 heures sur HDD) :

```powershell
python scripts\build_store.py --model D:\models\qwen3-next-80b-fp8 --out D:\moe-store\qwen3-next --only lo
```

**4. Mesurer, planifier, discuter** :

```powershell
python scripts\measure_bandwidth.py --store D:\moe-store\qwen3-next
python scripts\plan.py --model D:\models\qwen3-next-80b-fp8 --store D:\moe-store\qwen3-next
python scripts\chat.py --model D:\models\qwen3-next-80b-fp8 --store D:\moe-store\qwen3-next --dense split --max-new-tokens 64
```

Si les réponses sont vides ou incohérentes dès le premier token, c'est
probablement un dépassement de capacité du fp16 dans les couches denses (les
modèles Qwen ont des activations très grandes). Relance avec
`--dtype bfloat16` (`check_setup.py` indique si le bf16 est natif ou émulé sur
ta carte).

## Palier intermédiaire optionnel : Qwen3-30B-A3B

Même format d'experts qu'OLMoE, 30,5B paramètres, store int4 de 14,3 Gio (qui ne
tient pas en RAM : bon test du niveau disque sans les 82 Go de Qwen3-Next).

```powershell
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir D:\models\qwen3-30b-a3b
python scripts\build_store.py --model D:\models\qwen3-30b-a3b --out D:\moe-store\qwen3-30b --only lo
python scripts\chat.py --model D:\models\qwen3-30b-a3b --store D:\moe-store\qwen3-30b
```
