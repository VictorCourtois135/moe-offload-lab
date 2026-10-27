# Prefill qwen3-next-80b-fp8 — 2026-10-03_1613

Machine : NVIDIA GeForce RTX 2060 ; politique lo ; 48 couches MoE ; disque mesuré 1.14 Go/s au mieux

| Variante | Prompt | tokens | prefill (s) | MoE (s) | dense (s) | attente disque (s) | experts/couche | VRAM | RAM | disque | Go lus | Go/s | sautés | tok/s ensuite | KL | top-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| defaut | court 1 | 17 | 5.83 | 4.88 | 0.95 | 0.10 | 66 | 374 | 94 | 2705 | 4.52 | 0.77 | 0 | 8.95 | - | - |
| defaut | court 2 | 22 | 6.25 | 5.37 | 0.88 | 0.12 | 77 | 462 | 504 | 2743 | 4.58 | 0.73 | 0 | 6.68 | - | - |
| defaut | court 3 | 22 | 5.96 | 5.12 | 0.83 | 0.04 | 73 | 583 | 630 | 2294 | 3.83 | 0.64 | 0 | 7.09 | - | - |
| defaut | long | 64 | 11.42 | 10.56 | 0.85 | 0.19 | 153 | 735 | 1109 | 5512 | 9.21 | 0.81 | 0 | 5.93 | - | - |
| p2 | court 1 | 17 | 5.88 | 5.01 | 0.86 | 0.03 | 66 | 375 | 93 | 2706 | 4.52 | 0.77 | 0 | 8.88 | 0.0001 | 100% |
| p2 | court 2 | 22 | 6.47 | 5.58 | 0.89 | 0.03 | 77 | 463 | 502 | 2745 | 4.59 | 0.71 | 0 | 6.71 | 0.0000 | 100% |
| p2 | court 3 | 22 | 6.05 | 5.15 | 0.90 | 0.02 | 73 | 587 | 629 | 2291 | 3.83 | 0.63 | 0 | 7.11 | 0.0001 | 100% |
| p2 | long | 64 | 11.63 | 10.82 | 0.80 | 0.05 | 153 | 742 | 1097 | 5511 | 9.21 | 0.79 | 0 | 5.47 | 0.0016 | 96% |
| p3 | court 1 | 17 | 6.25 | 4.04 | 2.22 | 0.01 | 66 | 369 | 174 | 2885 | 4.82 | 0.77 | 0 | 8.26 | 0.0000 | 100% |
| p3 | court 2 | 22 | 7.05 | 4.80 | 2.25 | 0.01 | 77 | 469 | 570 | 2992 | 5.00 | 0.71 | 0 | 6.07 | 0.0000 | 100% |
| p3 | court 3 | 22 | 6.93 | 4.79 | 2.13 | 0.01 | 73 | 585 | 671 | 2593 | 4.33 | 0.63 | 0 | 6.42 | 0.0000 | 100% |
| p3 | long | 64 | 13.64 | 9.58 | 4.06 | 0.02 | 153 | 732 | 1120 | 6106 | 10.20 | 0.75 | 0 | 5.39 | 0.0030 | 92% |
| p4 | court 1 | 17 | 5.82 | 4.94 | 0.88 | 0.04 | 66 | 421 | 62 | 2688 | 4.49 | 0.77 | 0 | 8.38 | 0.0001 | 100% |
| p4 | court 2 | 22 | 6.20 | 5.37 | 0.82 | 0.07 | 77 | 482 | 485 | 2745 | 4.59 | 0.74 | 0 | 6.05 | 0.0000 | 100% |
| p4 | court 3 | 22 | 6.12 | 5.25 | 0.87 | 0.01 | 73 | 601 | 634 | 2272 | 3.80 | 0.62 | 0 | 6.46 | 0.0000 | 100% |
| p4 | long | 64 | 11.80 | 10.96 | 0.84 | 0.06 | 153 | 863 | 1087 | 5397 | 9.02 | 0.76 | 0 | 4.88 | 0.0024 | 92% |
| p23 | court 1 | 17 | 6.25 | 4.08 | 2.17 | 0.02 | 66 | 368 | 169 | 2903 | 4.85 | 0.78 | 0 | 8.19 | 0.0001 | 100% |
| p23 | court 2 | 22 | 7.12 | 4.89 | 2.23 | 0.03 | 77 | 470 | 566 | 2990 | 5.00 | 0.70 | 0 | 6.05 | 0.0000 | 100% |
| p23 | court 3 | 22 | 6.68 | 4.66 | 2.02 | 0.02 | 73 | 585 | 667 | 2593 | 4.33 | 0.65 | 0 | 6.36 | 0.0000 | 100% |
| p23 | long | 64 | 13.44 | 9.73 | 3.71 | 0.05 | 153 | 734 | 1125 | 6110 | 10.21 | 0.76 | 0 | 5.36 | 0.0028 | 92% |
| skip05 | court 1 | 17 | 5.52 | 4.62 | 0.90 | 0.03 | 66 | 392 | 118 | 2312 | 3.86 | 0.70 | 361 | 8.43 | 0.0002 | 100% |
| skip05 | court 2 | 22 | 5.90 | 5.07 | 0.83 | 0.05 | 77 | 464 | 500 | 2358 | 3.94 | 0.67 | 395 | 6.19 | 0.0001 | 100% |
| skip05 | court 3 | 22 | 5.79 | 4.91 | 0.89 | 0.01 | 73 | 577 | 675 | 1891 | 3.16 | 0.55 | 364 | 6.48 | 0.0000 | 100% |
| skip05 | long | 64 | 11.46 | 10.62 | 0.84 | 0.07 | 153 | 745 | 1128 | 4976 | 8.32 | 0.73 | 503 | 5.25 | 0.0036 | 92% |
| skip10 | court 1 | 17 | 3.42 | 2.48 | 0.93 | 0.00 | 66 | 394 | 921 | 439 | 0.73 | 0.21 | 1436 | 7.90 | 0.0033 | 100% |
| skip10 | court 2 | 22 | 3.81 | 2.93 | 0.88 | 0.00 | 78 | 507 | 1096 | 589 | 0.98 | 0.26 | 1540 | 6.06 | 0.0047 | 100% |
| skip10 | court 3 | 22 | 3.73 | 2.83 | 0.90 | 0.00 | 73 | 610 | 1087 | 439 | 0.73 | 0.20 | 1357 | 6.10 | 0.0006 | 100% |
| skip10 | long | 64 | 7.29 | 6.41 | 0.88 | 0.01 | 154 | 784 | 1507 | 2146 | 3.59 | 0.49 | 2934 | 4.89 | 0.0126 | 88% |

### Phases d'une couche MoE au prefill, defaut (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.24 |
| 2 décisions HOBBIT + regroupement | 4.29 |
| 3 recherche caches + q* + lectures lancées | 1.20 |
| 4 calcul des experts déjà en VRAM | 39.55 |
| 5 copies lancées + branche CPU | 2.26 |
| 6 attente disque/copies + calcul des manquants | 87.41 |
| 7 prefetch | 0.02 |

### Phases d'une couche MoE au prefill, p2 (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.25 |
| 2 décisions HOBBIT + regroupement | 4.06 |
| 3 recherche caches + q* + lectures lancées | 1.95 |
| 4 calcul des experts déjà en VRAM | 38.37 |
| 5 copies lancées + branche CPU | 2.41 |
| 6 attente disque/copies + calcul des manquants | 91.14 |
| 7 prefetch | 0.02 |

### Phases d'une couche MoE au prefill, p3 (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.58 |
| 2 décisions HOBBIT + regroupement | 4.79 |
| 3 recherche caches + q* + lectures lancées | 0.57 |
| 4 calcul des experts déjà en VRAM | 16.57 |
| 5 copies lancées + branche CPU | 1.92 |
| 6 attente disque/copies + calcul des manquants | 89.32 |
| 7 prefetch | 6.70 |

### Phases d'une couche MoE au prefill, p4 (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.24 |
| 2 décisions HOBBIT + regroupement | 4.06 |
| 3 recherche caches + q* + lectures lancées | 1.19 |
| 4 calcul des experts déjà en VRAM | 48.77 |
| 5 copies lancées + branche CPU | 1.78 |
| 6 attente disque/copies + calcul des manquants | 81.92 |
| 7 prefetch | 0.02 |

### Phases d'une couche MoE au prefill, p23 (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.64 |
| 2 décisions HOBBIT + regroupement | 5.40 |
| 3 recherche caches + q* + lectures lancées | 0.59 |
| 4 calcul des experts déjà en VRAM | 16.56 |
| 5 copies lancées + branche CPU | 1.93 |
| 6 attente disque/copies + calcul des manquants | 89.45 |
| 7 prefetch | 6.66 |

### Phases d'une couche MoE au prefill, skip05 (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.25 |
| 2 décisions HOBBIT + regroupement | 4.09 |
| 3 recherche caches + q* + lectures lancées | 1.79 |
| 4 calcul des experts déjà en VRAM | 43.56 |
| 5 copies lancées + branche CPU | 2.00 |
| 6 attente disque/copies + calcul des manquants | 79.51 |
| 7 prefetch | 0.02 |

### Phases d'une couche MoE au prefill, skip10 (ms)

| Phase | ms |
|---|---|
| 1 routage lu sur CPU (synchro) | 0.22 |
| 2 décisions HOBBIT + regroupement | 4.11 |
| 3 recherche caches + q* + lectures lancées | 1.14 |
| 4 calcul des experts déjà en VRAM | 22.30 |
| 5 copies lancées + branche CPU | 3.37 |
| 6 attente disque/copies + calcul des manquants | 45.08 |
| 7 prefetch | 0.02 |

## Conversation (P1)

| | message | tokens du prompt | repris du cache | 1er token (s) |
|---|---|---|---|---|
| sans P1 | 1 | 17 | 0 | 6.51 |
| sans P1 | 2 | 73 | 0 | 10.20 |
| sans P1 | 3 | 129 | 0 | 14.03 |
| avec P1 | 1 | 17 | 0 | 6.00 |
| avec P1 | 2 | 73 | 48 | 6.50 |
| avec P1 | 3 | 129 | 104 | 5.97 |

