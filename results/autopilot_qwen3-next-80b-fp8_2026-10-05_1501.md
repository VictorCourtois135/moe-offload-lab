# Autopilot qwen3-next-80b-fp8 — 2026-10-05_1501

Réponse type (1er token + 200 tokens) : **32.8 → 32.8 s** (+0.0 %), 11 essais, 19.8 min

## Rapport du modèle

Meilleure configuration conservée : read_parts=4, prefill_grouped=192, prefetch=true, cache_eviction=lrfu, full_graph=true, ram_gb=5.0, vram_gb=2.105. Tous les autres réglages testés ont dégradé les performances. Pas d'amélioration possible avec les 12 essais.

## Journal

| essai | changements | réponse type (s) | tok/s | 1er token (s) | KL | verdict |
|---|---|---|---|---|---|---|
| 0 | départ | 32.8 | 6.94 | 3.95 | 0.0 | départ |
| 1 | {'read_parts': 8} | 36.0 | 6.26 | 4.07 | 0.0001 | pas mieux (réponse type -12.1 %, 1er token -5.4 %, tok/s -11.7 %, meilleure config remesurée à côté) |
| 2 | {'prefetch_disk_ahead': 1} | 33.1 | 6.85 | 3.88 | 0.0002 | pas mieux (réponse type -3.1 %, 1er token -0.5 %, tok/s -3.4 %) |
| 3 | {'prefill_skip': 0.02} | 32.1 | 7.08 | 3.86 | 0.0001 | pas mieux (réponse type +0.0 %, 1er token +0.0 %, tok/s -0.1 %) |
| 4 | {'vram_gb': 1.96} | 33.0 | 6.87 | 3.93 | 0.0002 | pas mieux (réponse type -2.8 %, 1er token -1.8 %, tok/s -3.1 %) |
| 5 | {'prefetch': False} | 33.4 | 6.77 | 3.89 | 0.0 | pas mieux (réponse type -4.0 %, 1er token -0.8 %, tok/s -4.5 %) |
| 6 | {'full_graph': False} | 32.9 | 6.92 | 3.99 | 0.0001 | pas mieux (réponse type -2.5 %, 1er token -3.4 %, tok/s -2.4 %) |
| 7 | {'cache_eviction': 'stale'} | 72.1 | 3.05 | 6.64 | 0.0001 | pas mieux (réponse type -125.3 %, 1er token -72.0 %, tok/s -57.1 %, meilleure config remesurée à côté) |
| 8 | {'prefetch_depth': 2} | 43.5 | 5.05 | 3.9 | 0.0 | pas mieux (réponse type -34.7 %, 1er token -2.1 %, tok/s -28.2 %, meilleure config remesurée à côté) |
| 9 | {'ram_gb': 5.5} | 31.9 | 7.15 | 3.94 | 0.0001 | pas mieux (réponse type +1.2 %, 1er token -3.1 %, tok/s +1.7 %) |
| 10 | {'prefill_grouped': 256} | 33.0 | 6.91 | 4.03 | 0.0001 | pas mieux (réponse type -2.2 %, 1er token -5.5 %, tok/s -1.7 %) |
| 11 | {'prefetch_mass': 0.8} | 53.9 | 4.01 | 4.03 | 0.0002 | pas mieux (réponse type -64.3 %, 1er token -2.8 %, tok/s -42.1 %, meilleure config remesurée à côté) |

## Meilleure configuration

```json
{
 "read_parts": 4,
 "prefill_grouped": 192,
 "cache_eviction": "lrfu",
 "prefetch": true,
 "prefetch_depth": 1,
 "prefetch_mass": null,
 "prefetch_disk_ahead": 0,
 "full_graph": true,
 "prefill_skip": null,
 "ram_gb": 5.0,
 "vram_gb": 2.105
}
```
