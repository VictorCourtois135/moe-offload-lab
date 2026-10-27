# Prefill qwen3-next-80b-fp8 — 2026-10-05_0925

Machine : NVIDIA GeForce RTX 2060 ; politique lo ; 48 couches MoE ; disque mesuré 1.14 Go/s au mieux

| Variante | Prompt | tokens | prefill (s) | MoE (s) | dense (s) | attente disque (s) | experts/couche | VRAM | RAM | disque | Go lus | Go/s | sautés | tok/s ensuite | KL | top-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| defaut | court 1 | 17 | 5.95 | 5.05 | 0.90 | 0.35 | 66 | 374 | 94 | 2705 | 4.52 | 0.76 | 0 | 8.13 | - | - |
| defaut | court 2 | 22 | 6.53 | 5.55 | 0.99 | 0.11 | 77 | 462 | 504 | 2743 | 4.58 | 0.70 | 0 | 6.57 | - | - |
| defaut | court 3 | 22 | 5.93 | 5.07 | 0.86 | 0.03 | 73 | 583 | 630 | 2294 | 3.83 | 0.65 | 0 | 5.24 | - | - |
| defaut | long | 64 | 11.66 | 10.72 | 0.94 | 0.17 | 153 | 735 | 1109 | 5512 | 9.21 | 0.79 | 0 | 4.64 | - | - |
| p6 | court 1 | 17 | 3.76 | 2.87 | 0.89 | 1.27 | 66 | 375 | 93 | 2703 | 4.52 | 1.20 | 0 | 8.88 | 0.0003 | 100% |
| p6 | court 2 | 22 | 4.02 | 3.18 | 0.84 | 1.16 | 77 | 462 | 502 | 2748 | 4.59 | 1.14 | 0 | 6.64 | 0.0000 | 100% |
| p6 | court 3 | 22 | 3.78 | 2.92 | 0.85 | 0.81 | 73 | 590 | 634 | 2283 | 3.82 | 1.01 | 0 | 7.05 | 0.0001 | 100% |
| p6 | long | 64 | 7.90 | 7.06 | 0.85 | 2.23 | 153 | 734 | 1109 | 5506 | 9.20 | 1.16 | 0 | 5.86 | 0.0023 | 92% |
| p7 | court 1 | 17 | 6.31 | 5.49 | 0.82 | 1.67 | 66 | 374 | 94 | 2705 | 4.52 | 0.72 | 0 | 8.54 | 0.0000 | 100% |
| p7 | court 2 | 22 | 6.61 | 5.83 | 0.78 | 1.63 | 77 | 462 | 504 | 2743 | 4.58 | 0.69 | 0 | 6.72 | 0.0000 | 100% |
| p7 | court 3 | 22 | 6.15 | 5.34 | 0.81 | 1.24 | 73 | 583 | 630 | 2294 | 3.83 | 0.62 | 0 | 7.07 | 0.0000 | 100% |
| p7 | long | 64 | 12.83 | 12.02 | 0.81 | 3.20 | 153 | 735 | 1109 | 5512 | 9.21 | 0.72 | 0 | 5.30 | 0.0000 | 100% |
| p8 | court 1 | 17 | 12.20 | 11.33 | 0.88 | 0.85 | 66 | 374 | 94 | 2705 | 4.52 | 0.37 | 0 | 4.02 | 0.0000 | 100% |
| p8 | court 2 | 22 | 12.66 | 11.82 | 0.84 | 0.41 | 77 | 462 | 504 | 2743 | 4.58 | 0.36 | 0 | 2.72 | 0.0000 | 100% |
| p8 | court 3 | 22 | 11.22 | 10.39 | 0.83 | 0.19 | 73 | 583 | 630 | 2294 | 3.83 | 0.34 | 0 | 2.92 | 0.0000 | 100% |
| p8 | long | 64 | 22.68 | 21.88 | 0.80 | 1.01 | 153 | 735 | 1109 | 5512 | 9.21 | 0.41 | 0 | 2.39 | 0.0020 | 92% |
| p678 | court 1 | 17 | 4.79 | 4.01 | 0.78 | 2.21 | 66 | 376 | 94 | 2701 | 4.51 | 0.94 | 0 | 4.45 | 0.0003 | 100% |
| p678 | court 2 | 22 | 5.08 | 4.28 | 0.80 | 2.27 | 77 | 462 | 505 | 2745 | 4.59 | 0.90 | 0 | 3.04 | 0.0000 | 100% |
| p678 | court 3 | 22 | 4.89 | 4.03 | 0.86 | 2.03 | 73 | 593 | 631 | 2283 | 3.82 | 0.78 | 0 | 3.05 | 0.0001 | 100% |
| p678 | long | 64 | 9.25 | 8.42 | 0.83 | 3.75 | 153 | 733 | 1106 | 5516 | 9.22 | 1.00 | 0 | 2.47 | 0.0039 | 92% |
| p678_skip05 | court 1 | 17 | 4.37 | 3.55 | 0.82 | 1.86 | 66 | 392 | 124 | 2298 | 3.84 | 0.88 | 362 | 4.69 | 0.0003 | 100% |
| p678_skip05 | court 2 | 22 | 4.78 | 3.95 | 0.83 | 2.01 | 77 | 463 | 506 | 2353 | 3.93 | 0.82 | 394 | 3.11 | 0.0000 | 100% |
| p678_skip05 | court 3 | 22 | 4.55 | 3.68 | 0.87 | 1.65 | 73 | 574 | 690 | 1877 | 3.14 | 0.69 | 368 | 3.06 | 0.0000 | 100% |
| p678_skip05 | long | 64 | 8.54 | 7.75 | 0.79 | 3.40 | 153 | 737 | 1151 | 4969 | 8.30 | 0.97 | 501 | 2.41 | 0.0046 | 92% |
| p678_skip10 | court 1 | 17 | 2.48 | 1.67 | 0.81 | 0.41 | 66 | 391 | 926 | 439 | 0.73 | 0.30 | 1435 | 3.94 | 0.0030 | 100% |
| p678_skip10 | court 2 | 22 | 2.97 | 2.10 | 0.87 | 0.56 | 78 | 505 | 1094 | 593 | 0.99 | 0.33 | 1543 | 3.01 | 0.0042 | 100% |
| p678_skip10 | court 3 | 22 | 2.73 | 1.86 | 0.87 | 0.37 | 73 | 609 | 1086 | 445 | 0.74 | 0.27 | 1358 | 2.92 | 0.0009 | 100% |
| p678_skip10 | long | 64 | 6.45 | 5.64 | 0.81 | 1.82 | 154 | 785 | 1503 | 2154 | 3.60 | 0.56 | 2936 | 2.10 | 0.0122 | 88% |

### Phases d'une couche MoE au prefill, defaut (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.25 |
| 2 décisions HOBBIT + regroupement | 4.15 |
| 3 recherche caches + q* + lectures lancées | 1.22 |
| 4 calcul des experts déjà en VRAM | 37.78 |
| 5 copies lancées + branche CPU | 2.39 |
| 6 attente disque/copies + calcul des manquants | 91.47 |
| 7 prefetch | 0.02 |

### Phases d'une couche MoE au prefill, p6 (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.26 |
| 2 décisions HOBBIT + regroupement | 4.20 |
| 3 recherche caches + q* + lectures lancées | 1.19 |
| 4 calcul des experts déjà en VRAM | 0.07 |
| 5 copies lancées + branche CPU | 7.12 |
| 6 attente disque/copies + calcul des manquants | 70.27 |
| 7 prefetch | 0.02 |

### Phases d'une couche MoE au prefill, p7 (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.26 |
| 2 décisions HOBBIT + regroupement | 4.19 |
| 3 recherche caches + q* + lectures lancées | 1.25 |
| 4 calcul des experts déjà en VRAM | 12.04 |
| 5 copies lancées + branche CPU | 1.83 |
| 6 attente disque/copies + calcul des manquants | 129.65 |
| 7 prefetch | 0.02 |

### Phases d'une couche MoE au prefill, p8 (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.29 |
| 2 décisions HOBBIT + regroupement | 4.43 |
| 3 recherche caches + q* + lectures lancées | 24.98 |
| 4 calcul des experts déjà en VRAM | 103.43 |
| 5 copies lancées + branche CPU | 23.07 |
| 6 attente disque/copies + calcul des manquants | 131.50 |
| 7 prefetch | 0.02 |

### Phases d'une couche MoE au prefill, p678 (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.28 |
| 2 décisions HOBBIT + regroupement | 5.15 |
| 3 recherche caches + q* + lectures lancées | 4.67 |
| 4 calcul des experts déjà en VRAM | 0.04 |
| 5 copies lancées + branche CPU | 2.10 |
| 6 attente disque/copies + calcul des manquants | 94.09 |
| 7 prefetch | 0.02 |

### Phases d'une couche MoE au prefill, p678_skip05 (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.26 |
| 2 décisions HOBBIT + regroupement | 4.32 |
| 3 recherche caches + q* + lectures lancées | 5.02 |
| 4 calcul des experts déjà en VRAM | 0.04 |
| 5 copies lancées + branche CPU | 2.22 |
| 6 attente disque/copies + calcul des manquants | 85.31 |
| 7 prefetch | 0.02 |

### Phases d'une couche MoE au prefill, p678_skip10 (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.21 |
| 2 décisions HOBBIT + regroupement | 4.29 |
| 3 recherche caches + q* + lectures lancées | 3.71 |
| 4 calcul des experts déjà en VRAM | 0.05 |
| 5 copies lancées + branche CPU | 3.58 |
| 6 attente disque/copies + calcul des manquants | 46.15 |
| 7 prefetch | 0.02 |

