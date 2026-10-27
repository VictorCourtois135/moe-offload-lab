# Autopilot qwen3-next-80b-fp8 — 2026-10-05_1258

Réponse type (1er token + 200 tokens) : **52.8 → 34.2 s** (+35.2 %), 11 essais, 17.5 min ; épreuve à l'aveugle

## Rapport du modèle

Meilleure configuration : read_parts=2. Tous les autres changements testés n'ont pas amélioré la réponse type de plus de 5 %. Le SSD est mieux exploité avec 2 lectures simultanées, et le cache VRAM/RAM est déjà optimisé. Aucune autre variable n'apporte de gain significatif.

## Journal

| essai | changements | réponse type (s) | tok/s | 1er token (s) | KL | verdict |
|---|---|---|---|---|---|---|
| 0 | départ | 52.8 | 4.33 | 6.56 | 0.0 | départ |
| 1 | {'read_parts': 2} | 34.5 | 7.02 | 6.0 | 0.0 | GARDÉ : +35.2 % (confirmé) |
| 2 | {'prefetch_depth': 2} | 44.3 | 5.22 | 5.99 | 0.0001 | pas mieux (-29.5 %, seuil 5 %) |
| 3 | {'prefetch_mass': 0.8} | 52.0 | 4.34 | 5.89 | 0.0 | pas mieux (-52.0 %, seuil 5 %) |
| 4 | {'cache_eviction': 'stale'} | 71.0 | 3.19 | 8.33 | 0.0 | pas mieux (-107.6 %, seuil 5 %) |
| 5 | {'full_graph': False} | 33.9 | 7.11 | 5.72 | 0.0 | pas mieux (+0.9 %, seuil 5 %) |
| 6 | {'prefetch': False} | 34.9 | 6.89 | 5.83 | 0.0 | pas mieux (-2.0 %, seuil 5 %) |
| 7 | {'ram_gb': 3.0} | 39.3 | 6.05 | 6.18 | 0.0001 | pas mieux (-14.9 %, seuil 5 %) |
| 8 | {'vram_gb': 1.74} | 35.1 | 6.88 | 6.02 | 0.0 | pas mieux (-2.6 %, seuil 5 %) |
| 9 | {'read_parts': 4} | 35.9 | 6.71 | 6.06 | 0.0 | pas mieux (-5.0 %, seuil 5 %) |
| 10 | {'prefill_grouped': 64} | 32.6 | 6.95 | 3.86 | 0.0 | pas mieux (+4.7 %, seuil 5 %) |
| 11 | {'prefill_skip': 0.05} | 33.1 | 7.22 | 5.42 | 0.0002 | pas mieux (+3.2 %, seuil 5 %) |

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
