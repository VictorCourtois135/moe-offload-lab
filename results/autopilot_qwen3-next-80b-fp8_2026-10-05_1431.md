# Autopilot qwen3-next-80b-fp8 — 2026-10-05_1431

Réponse type (1er token + 200 tokens) : **54.2 → 32.5 s** (+40.1 %), 11 essais, 24.4 min ; épreuve à l'aveugle

Vérification finale (mesurés l'un après l'autre) : départ {'reponse_type_s': 52.1, 'premier_token_s': 6.51, 'tok_s': 4.38, 'kl': 0.0} → meilleur {'reponse_type_s': 32.3, 'premier_token_s': 3.8, 'tok_s': 7.01, 'kl': 0.0001} : **+38.0 %**

## Rapport du modèle

Meilleure configuration : read_parts=2 (meilleur débit SSD), prefill_grouped=64 (accélère le prefill). Gain : premier token -36.6 %, réponse type -5.6 %. Toutes les autres modifications ont dégradé ou n'ont pas amélioré significativement les performances.

## Journal

| essai | changements | réponse type (s) | tok/s | 1er token (s) | KL | verdict |
|---|---|---|---|---|---|---|
| 0 | départ | 54.2 | 4.21 | 6.65 | 0.0 | départ |
| 1 | {'read_parts': 2} | 35.6 | 6.76 | 6.04 | 0.0 | GARDÉ : réponse type +32.8 % (confirmé : réponse type +32.5 %) |
| 2 | {'read_parts': 4} | 38.9 | 6.13 | 6.25 | 0.0 | pas mieux (réponse type -9.9 %, 1er token -4.9 %, tok/s -9.7 %, meilleure config remesurée à côté) |
| 3 | {'prefetch_depth': 2} | 46.5 | 4.93 | 5.9 | 0.0001 | pas mieux (réponse type -31.4 %, 1er token +1.0 %, tok/s -27.4 %, meilleure config remesurée à côté) |
| 4 | {'cache_eviction': 'stale'} | 77.4 | 2.9 | 8.39 | 0.0001 | pas mieux (réponse type -113.8 %, 1er token -39.6 %, tok/s -56.3 %, meilleure config remesurée à côté) |
| 5 | {'vram_gb': 2.06} | 33.5 | 7.24 | 5.89 | 0.0001 | pas mieux (réponse type +2.0 %, 1er token +1.8 %, tok/s +2.0 %, meilleure config remesurée à côté) |
| 6 | {'full_graph': False} | 34.1 | 7.14 | 6.13 | 0.0 | pas mieux (réponse type +0.3 %, 1er token -2.2 %, tok/s +0.6 %) |
| 7 | {'prefetch_mass': 0.8} | 53.0 | 4.27 | 6.14 | 0.0001 | pas mieux (réponse type -55.4 %, 1er token -3.7 %, tok/s -39.9 %, meilleure config remesurée à côté) |
| 8 | {'ram_gb': 3.0} | 39.2 | 6.06 | 6.23 | 0.0 | pas mieux (réponse type -13.6 %, 1er token -4.0 %, tok/s -13.6 %, meilleure config remesurée à côté) |
| 9 | {'prefill_grouped': 64} | 32.6 | 6.95 | 3.84 | 0.0001 | GARDÉ : 1er token +36.6 %, tok/s -2.3 % (confirmé : réponse type +5.6 %) |
| 10 | {'prefill_grouped': 128} | 32.2 | 7.05 | 3.86 | 0.0001 | pas mieux (réponse type +0.8 %, 1er token -0.3 %, tok/s +0.8 %) |
| 11 | {'prefill_skip': 0.05} | 31.7 | 7.13 | 3.64 | 0.0002 | pas mieux (réponse type +2.3 %, 1er token +5.5 %, tok/s +1.9 %) |

## Meilleure configuration

```json
{
 "read_parts": 2,
 "prefill_grouped": 64,
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
