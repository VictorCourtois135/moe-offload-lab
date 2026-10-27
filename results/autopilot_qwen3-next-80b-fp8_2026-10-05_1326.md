# Autopilot qwen3-next-80b-fp8 — 2026-10-05_1326

Réponse type (1er token + 200 tokens) : **52.7 → 31.0 s** (+41.2 %), 10 essais, 20.0 min ; épreuve à l'aveugle

## Rapport du modèle

Meilleure configuration : vram_gb=3.0, prefill_skip=0.05, read_parts=2. Réduction de 10.8 % de la réponse type, gain de 3.6 % sur le premier token et +13.3 % de tok/s. Aucun autre réglage n'apporte de gain significatif.

## Journal

| essai | changements | réponse type (s) | tok/s | 1er token (s) | KL | verdict |
|---|---|---|---|---|---|---|
| 0 | départ | 52.7 | 4.34 | 6.68 | 0.0 | départ |
| 1 | {'read_parts': 2} | 34.6 | 7.0 | 6.05 | 0.0 | GARDÉ : réponse type +34.3 % (confirmé : réponse type +35.1 %) |
| 2 | {'read_parts': 4} | 35.3 | 6.82 | 5.96 | 0.0 | pas mieux (réponse type -2.6 %, 1er token +1.0 %, tok/s -3.3 %) |
| 3 | {'prefetch_depth': 2} | 43.5 | 5.29 | 5.7 | 0.0001 | pas mieux (réponse type -26.5 %, 1er token +5.3 %, tok/s -25.0 %) |
| 4 | {'cache_eviction': 'stale'} | 70.9 | 3.2 | 8.27 | 0.0001 | pas mieux (réponse type -106.1 %, 1er token -37.4 %, tok/s -54.6 %) |
| 5 | {'vram_gb': 2.5} | 43.2 | 5.36 | 5.9 | 0.0 | pas mieux (réponse type -25.6 %, 1er token +2.0 %, tok/s -24.0 %) |
| 6 | {'full_graph': True, 'prefill_grouped': 128} | 44.5 | 4.91 | 3.81 | 0.0001 | pas mieux (réponse type -29.4 %, 1er token +36.7 %, tok/s -30.4 %) |
| 7 | {'ram_gb': 3.0} | 50.1 | 4.55 | 6.1 | 0.0 | pas mieux (réponse type -45.6 %, 1er token -1.3 %, tok/s -35.5 %) |
| 8 | {'prefetch_mass': 0.8} | 56.9 | 3.94 | 6.17 | 0.0001 | pas mieux (réponse type -65.4 %, 1er token -2.5 %, tok/s -44.1 %) |
| 9 | {'prefill_skip': 0.05} | 34.6 | 6.83 | 5.36 | 0.0001 | GARDÉ : 1er token +11.0 %, tok/s -3.1 % (confirmé : 1er token +9.1 %, tok/s -3.1 %) |
| 10 | {'vram_gb': 3.0, 'prefill_skip': 0.05} | 31.1 | 7.74 | 5.22 | 0.0001 | GARDÉ : réponse type +10.2 % (confirmé : réponse type +10.8 %) |

## Meilleure configuration

```json
{
 "read_parts": 2,
 "prefill_grouped": 0,
 "cache_eviction": "lrfu",
 "prefetch": true,
 "prefetch_depth": 1,
 "prefetch_mass": null,
 "prefetch_disk_ahead": 0,
 "full_graph": true,
 "prefill_skip": 0.05,
 "ram_gb": 5.0,
 "vram_gb": 3.0
}
```
