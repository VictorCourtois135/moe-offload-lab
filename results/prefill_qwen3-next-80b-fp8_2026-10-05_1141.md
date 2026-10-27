# Prefill qwen3-next-80b-fp8 — 2026-10-05_1141

Machine : NVIDIA GeForce RTX 2060 ; politique lo ; 48 couches MoE ; disque mesuré 1.14 Go/s au mieux

| Variante | Prompt | tokens | prefill (s) | MoE (s) | dense (s) | attente disque (s) | experts/couche | VRAM | RAM | disque | Go lus | Go/s | sautés | tok/s ensuite | KL | top-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| defaut | court 1 | 17 | 4.01 | 2.89 | 1.12 | 1.27 | 66 | 375 | 93 | 2703 | 4.52 | 1.13 | 0 | 8.84 | - | - |
| defaut | long | 93 | 14.33 | 13.41 | 0.92 | 0.41 | 183 | 782 | 895 | 7097 | 11.86 | 0.83 | 0 | 6.30 | - | - |
| defaut | tres_long | 407 | 22.53 | 21.60 | 0.93 | 0.40 | 281 | 971 | 1644 | 10874 | 18.17 | 0.81 | 0 | 8.31 | - | - |
| p6_256 | court 1 | 17 | 3.67 | 2.83 | 0.85 | 1.26 | 66 | 375 | 93 | 2703 | 4.52 | 1.23 | 0 | 8.92 | 0.0000 | 100% |
| p6_256 | long | 93 | 10.51 | 9.66 | 0.86 | 3.17 | 183 | 782 | 896 | 7102 | 11.87 | 1.13 | 0 | 6.19 | 0.0012 | 100% |
| p6_256 | tres_long | 407 | 23.00 | 22.03 | 0.96 | 0.41 | 281 | 971 | 1632 | 10862 | 18.15 | 0.79 | 0 | 8.58 | 0.0010 | 100% |
| p6_512 | court 1 | 17 | 3.85 | 2.86 | 0.99 | 1.25 | 66 | 375 | 93 | 2703 | 4.52 | 1.17 | 0 | 8.79 | 0.0000 | 100% |
| p6_512 | long | 93 | 10.52 | 9.63 | 0.90 | 3.17 | 183 | 782 | 896 | 7102 | 11.87 | 1.13 | 0 | 6.31 | 0.0012 | 100% |
| p6_512 | tres_long | 407 | 27.88 | 26.95 | 0.93 | 4.79 | 281 | 971 | 1633 | 10891 | 18.20 | 0.65 | 0 | 8.70 | 0.0007 | 100% |

### Phases d'une couche MoE au prefill, defaut (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.38 |
| 2 décisions HOBBIT + regroupement | 8.77 |
| 3 recherche caches + q* + lectures lancées | 3.65 |
| 4 calcul des experts déjà en VRAM | 65.02 |
| 5 copies lancées + branche CPU | 4.10 |
| 6 attente disque/copies + calcul des manquants | 180.92 |
| 7 prefetch | 0.02 |

### Phases d'une couche MoE au prefill, p6_256 (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.37 |
| 2 décisions HOBBIT + regroupement | 7.91 |
| 3 recherche caches + q* + lectures lancées | 3.47 |
| 4 calcul des experts déjà en VRAM | 42.92 |
| 5 copies lancées + branche CPU | 6.03 |
| 6 attente disque/copies + calcul des manquants | 178.31 |
| 7 prefetch | 0.02 |

### Phases d'une couche MoE au prefill, p6_512 (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.40 |
| 2 décisions HOBBIT + regroupement | 7.84 |
| 3 recherche caches + q* + lectures lancées | 3.31 |
| 4 calcul des experts déjà en VRAM | 0.18 |
| 5 copies lancées + branche CPU | 10.32 |
| 6 attente disque/copies + calcul des manquants | 248.39 |
| 7 prefetch | 0.02 |

