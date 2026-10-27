# Autopilot qwen3-next-80b-fp8 — 2026-10-05_1220

Réponse type (1er token + 200 tokens) : **53.1 → 34.2 s** (+35.7 %), 2 essais, 6.2 min ; épreuve à l'aveugle

## Rapport du modèle

(pas de rapport)

## Journal

| essai | changements | réponse type (s) | tok/s | 1er token (s) | KL | verdict |
|---|---|---|---|---|---|---|
| 0 | départ | 53.1 | 4.29 | 6.58 | 0.0 | départ |
| 1 | {'read_parts': 2} | 34.2 | 7.06 | 5.85 | 0.0 | GARDÉ : +35.7 % (confirmé) |
| 2 | {'prefetch_depth': 2} | 44.3 | 5.2 | 5.91 | 0.0001 | pas mieux (-29.7 %, seuil 5 %) |

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
 "prefill_skip": null,
 "ram_gb": 5.0,
 "vram_gb": 1.84
}
```
