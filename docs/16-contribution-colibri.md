# 16 — Contribution à Colibrì (issue #249)

## Ce que demande l'issue

L'issue [JustVugg/colibri#249](https://github.com/JustVugg/colibri/issues/249)
propose une quantification à précision mixte : garder les couches partagées en
haute précision et descendre les experts chargés dynamiquement en très basse
précision, pour réduire les octets lus par token.

Notre benchmark répond à une variante plus fine de la question : faut-il une
précision **fixe** par expert (statique), ou une précision choisie **par token**
selon l'importance de l'expert (HOBBIT) ?

## Ce que notre expérience peut apporter

Sur un même moteur, un même modèle (OLMoE, le plus petit de leur liste) et une même
machine, pour chaque politique : qualité (accord top-1, Δ perplexité, KL), vitesse
(tok/s, TTFT), et surtout **octets lus par token** et part de haute précision.

## Honnêteté sur les limites

- Notre moteur est en Python/PyTorch, pas Colibrì : les tok/s absolus ne se
  comparent pas aux leurs. Ce qui compte, ce sont les **écarts relatifs** entre
  politiques sur le même moteur, et les octets déplacés.
- Un seul modèle, une seule machine, un seul texte d'évaluation.
- Leur règle : "a well-controlled failure is more valuable here than an unexplained
  fast number". Un résultat négatif se publie aussi.

## Avant de poster : la liste de Colibrì

Leur README demande pour chaque expérience : matériel, commit, modèle/conteneur,
commande exacte, prompt, état du cache, débit, TTFT, taux de hit, octets lus,
contrôle de qualité, un seul paramètre changé à la fois, logs bruts. Le JSON de
`results\benchmark_*.json` contient tout ça sauf le commit : ajoute le hash avec
`git rev-parse --short HEAD`.

1. `python scripts\benchmark.py ... --repeats 2`
2. `python scripts\measure_routing.py ...` (valide les hypothèses de HOBBIT sur OLMoE)
3. Relis les chiffres, puis remplis le brouillon ci-dessous avec ceux de
   `results\benchmark_*.md`.
4. Poste en commentaire sur l'issue #249, avec le JSON en pièce jointe.

## Brouillon du commentaire (en anglais, à compléter)

Rempli avec le balayage propre du 30 septembre (doc 17, section 5). Restent : ton compte GitHub et le hash du commit, une fois le projet poussé. Joins `results\benchmark_olmoe-hf_2026-09-30_0956.json` et `results\routing_olmoe-hf_2026-09-29.json`.

```markdown
**Data point for #249: per-token dynamic expert precision (HOBBIT-style) vs static int4, on OLMoE-1B-7B**

I tested a related variant of this idea in a small Python/PyTorch offloading engine
(not colibri, so absolute tok/s are not comparable; the relative numbers are the point):
https://github.com/{ton-compte}/moe-offload-lab (commit {hash})

Setup: OLMoE-1B-7B-0125-Instruct, experts stored as fp16 + int4 (group size 64,
4.25 bits/weight), RTX 2060 6 GB, 16 GB RAM, store on NVMe.
Same memory budgets for all policies (VRAM 2.98 GiB, RAM fixed at 5 GiB), caches and
usage history reset between runs, 8 prompts x 64 greedy tokens, 2 repeats in
alternating order. Quality by teacher forcing on 256 tokens against the all-fp16
model, with caches cleared and no opportunistic fp16 reuse, so each row measures the
policy alone. Raw JSON attached.

| policy | tok/s | top-1 agreement | Δ ppl | KL | experts requested in fp16 | MB read from disk / token |
|---|---|---|---|---|---|---|
| all fp16 | 5.00 | 98.8% (fp16 noise) | +0.12% | 0.0019 | 100% | 310 |
| static int4 | 5.92 | 89.1% | +10.35% | 0.0698 | 0% | 5 |
| per-token, t1=0.2 | 4.73 | 88.7% | +10.73% | 0.0648 | 16% | 71 |
| per-token, t1=0.6 | 4.84 | 93.8% | +4.91% | 0.0305 | 49% | 169 |
| per-token, t1=0.8 | 4.98 | 97.3% | +0.83% | 0.0066 | 73% | 231 |
| per-token, t1=0.9 | 5.06 | 99.2% | -0.17% | 0.0019 | 87% | 270 |

Router checks on OLMoE: Pearson between normalized gate weight and ||w·E(x)|| = 0.888
(HOBBIT reports 0.99 on Mixtral; the top gate weight is the top contributor only 67%
of the time); next-layer router lookahead recalls 83.3% of the true top-8 (56% for
layer 0->1, 85-90% in deeper layers; vs 71.6% you measured on GLM-5.2); 39.9% of
experts are reused from the previous token (vs 41.3%).

Takeaway: on OLMoE, uniform int4 costs +10.4% perplexity. Choosing precision per
token gives a smooth quality/bytes curve: +0.8% at t1=0.8 with 27% of experts in
int4, and fp16-level quality at t1=0.9 with 13% in int4. Because the router weight
is a weaker importance proxy on this model, a high threshold is needed; keeping only
the top-1 expert in fp16 (t1=0.2) recovers almost nothing. Negative result on
speed: my engine is overhead-bound at this scale (~9.5 ms per MoE layer even with
every expert already in VRAM), so the saved bytes don't show up as tok/s. The
savings should matter in disk-bound setups like colibri's; I plan to check that on
Qwen3-Next-80B next.
```
