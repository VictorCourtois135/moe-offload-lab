# Profil olmoe-hf — 2026-09-30_1616

Machine : NVIDIA GeForce RTX 2060, k = 8

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 51.765 |
| sync_us | 52.978 |
| floor_loop_gpu_ms | 1.648 |
| floor_loop_wall_ms | 1.815 |
| floor_bmm_gpu_ms | 0.352 |
| floor_bmm_wall_ms | 0.354 |
| vram_bandwidth_gbs | 286.176 |
| dequant_int4_expert_gpu_ms | 0.560 |
| dequant_int4_expert_wall_ms | 0.562 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | ms par couche MoE | hit VRAM |
|---|---|---|---|---|---|
| lo / base | async | 148.5 | 6.73 | 7.17 | 100% |
| lo / base | sync | 149.8 | 6.67 | 7.27 | 100% |
| lo / a2 | async | 140.5 | 7.12 | 6.44 | 100% |
| lo / a2 | sync | 133.8 | 7.47 | 6.19 | 100% |
| lo / a2+adapt | async | 121.5 | 8.23 | 5.49 | 100% |
| lo / a2+adapt | sync | 123.6 | 8.09 | 5.62 | 100% |
| lo / b2 | async | 142.8 | 7.00 | 6.77 | 93% |
| lo / b2 | sync | 148.6 | 6.73 | 7.08 | 93% |
| lo / b1 | async | 84.6 | 11.82 | 3.16 | 100% |
| lo / b1 | sync | 87.9 | 11.38 | 3.33 | 100% |
| lo / tout_b2 | async | 123.0 | 8.13 | 5.49 | 93% |
| lo / tout_b2 | sync | 127.2 | 7.86 | 5.71 | 93% |
| lo / tout_b1 | async | 84.8 | 11.79 | 3.02 | 100% |
| lo / tout_b1 | sync | 82.5 | 12.12 | 2.98 | 100% |
| hi / base | async | 752.7 | 1.33 | 43.83 | 77% |
| hi / base | sync | 391.6 | 2.55 | 21.12 | 78% |
| hi / a2 | async | 6361.7 | 0.16 | 394.74 | 78% |
| hi / a2 | sync | 2151.9 | 0.46 | 131.33 | 78% |
| hi / a2+adapt | async | 253.8 | 3.94 | 11.63 | 78% |
| hi / a2+adapt | sync | 245.3 | 4.08 | 12.01 | 78% |
| hi / b2 | async | 310.7 | 3.22 | 12.66 | 78% |
| hi / b2 | sync | 265.4 | 3.77 | 12.57 | 78% |
| hi / b1 | async | 271.9 | 3.68 | 13.60 | 78% |
| hi / b1 | sync | 255.5 | 3.91 | 11.32 | 78% |
| hi / tout_b2 | async | 199.1 | 5.02 | 9.59 | 78% |
| hi / tout_b2 | sync | 206.8 | 4.84 | 10.21 | 78% |
| hi / tout_b1 | async | 193.0 | 5.18 | 9.34 | 78% |
| hi / tout_b1 | sync | 216.5 | 4.62 | 10.82 | 78% |
| hobbit@0.8 / base | async | 242.8 | 4.12 | 12.31 | 78% |
| hobbit@0.8 / base | sync | 221.4 | 4.52 | 11.10 | 78% |
| hobbit@0.8 / a2 | async | 190.4 | 5.25 | 8.51 | 78% |
| hobbit@0.8 / a2 | sync | 206.6 | 4.84 | 10.09 | 78% |
| hobbit@0.8 / a2+adapt | async | 179.3 | 5.58 | 8.41 | 78% |
| hobbit@0.8 / a2+adapt | sync | 205.3 | 4.87 | 9.40 | 78% |
| hobbit@0.8 / b2 | async | 216.1 | 4.63 | 10.74 | 78% |
| hobbit@0.8 / b2 | sync | 226.6 | 4.41 | 10.94 | 78% |
| hobbit@0.8 / b1 | async | 184.9 | 5.41 | 8.34 | 78% |
| hobbit@0.8 / b1 | sync | 196.7 | 5.08 | 9.39 | 78% |
| hobbit@0.8 / tout_b2 | async | 169.6 | 5.90 | 8.04 | 78% |
| hobbit@0.8 / tout_b2 | sync | 200.1 | 5.00 | 9.70 | 78% |
| hobbit@0.8 / tout_b1 | async | 170.5 | 5.87 | 8.04 | 78% |
| hobbit@0.8 / tout_b1 | sync | 184.7 | 5.41 | 8.95 | 78% |

### Phases, lo / base (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.098 | 0.119 |
| 2 décisions HOBBIT + regroupement | 0.682 | 0.696 |
| 3 recherche caches + q* + lectures lancées | 0.017 | 0.029 |
| 4 calcul des experts déjà en VRAM | 5.917 | 5.918 |
| 5 copies lancées + branche CPU | 0.001 | 0.012 |
| 6 attente disque/copies + calcul des manquants | 0.001 | 0.009 |
| 7 prefetch | 0.446 | 0.461 |

### Phases, lo / a2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.167 | 0.191 |
| 2 décisions HOBBIT + regroupement | 0.101 | 0.103 |
| 3 recherche caches + q* + lectures lancées | 0.019 | 0.029 |
| 4 calcul des experts déjà en VRAM | 5.760 | 5.430 |
| 5 copies lancées + branche CPU | 0.001 | 0.012 |
| 6 attente disque/copies + calcul des manquants | 0.001 | 0.010 |
| 7 prefetch | 0.384 | 0.394 |

### Phases, lo / a2+adapt (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.161 | 0.191 |
| 2 décisions HOBBIT + regroupement | 0.098 | 0.095 |
| 3 recherche caches + q* + lectures lancées | 0.019 | 0.030 |
| 4 calcul des experts déjà en VRAM | 5.202 | 5.256 |
| 5 copies lancées + branche CPU | 0.001 | 0.012 |
| 6 attente disque/copies + calcul des manquants | 0.001 | 0.009 |
| 7 prefetch | 0.007 | 0.008 |

### Phases, lo / b2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.093 | 0.124 |
| 2 décisions HOBBIT + regroupement | 0.692 | 0.728 |
| 3 recherche caches + q* + lectures lancées | 0.020 | 0.032 |
| 4 calcul des experts déjà en VRAM | 3.750 | 3.863 |
| 5 copies lancées + branche CPU | 0.415 | 0.434 |
| 6 attente disque/copies + calcul des manquants | 0.447 | 0.464 |
| 7 prefetch | 1.345 | 1.407 |

### Phases, lo / b1 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.155 | 0.187 |
| 2 décisions HOBBIT + regroupement | 0.095 | 0.097 |
| 3 recherche caches + q* + lectures lancées | 0.017 | 0.029 |
| 4 calcul des experts déjà en VRAM | 2.527 | 2.586 |
| 5 copies lancées + branche CPU | 0.001 | 0.012 |
| 6 attente disque/copies + calcul des manquants | 0.001 | 0.009 |
| 7 prefetch | 0.359 | 0.384 |

### Phases, lo / tout_b2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.163 | 0.191 |
| 2 décisions HOBBIT + regroupement | 0.096 | 0.100 |
| 3 recherche caches + q* + lectures lancées | 0.020 | 0.031 |
| 4 calcul des experts déjà en VRAM | 3.096 | 3.181 |
| 5 copies lancées + branche CPU | 0.421 | 0.438 |
| 6 attente disque/copies + calcul des manquants | 0.393 | 0.415 |
| 7 prefetch | 1.300 | 1.330 |

### Phases, lo / tout_b1 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.164 | 0.187 |
| 2 décisions HOBBIT + regroupement | 0.081 | 0.095 |
| 3 recherche caches + q* + lectures lancées | 0.020 | 0.031 |
| 4 calcul des experts déjà en VRAM | 2.742 | 2.618 |
| 5 copies lancées + branche CPU | 0.001 | 0.012 |
| 6 attente disque/copies + calcul des manquants | 0.001 | 0.009 |
| 7 prefetch | 0.007 | 0.009 |

### Phases, hi / base (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.461 | 0.235 |
| 2 décisions HOBBIT + regroupement | 0.759 | 0.721 |
| 3 recherche caches + q* + lectures lancées | 0.042 | 0.057 |
| 4 calcul des experts déjà en VRAM | 2.532 | 2.616 |
| 5 copies lancées + branche CPU | 1.437 | 1.791 |
| 6 attente disque/copies + calcul des manquants | 23.179 | 11.634 |
| 7 prefetch | 14.955 | 3.755 |

### Phases, hi / a2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 1.884 | 0.304 |
| 2 décisions HOBBIT + regroupement | 0.135 | 0.252 |
| 3 recherche caches + q* + lectures lancées | 0.043 | 0.101 |
| 4 calcul des experts déjà en VRAM | 2.114 | 3.088 |
| 5 copies lancées + branche CPU | 1.936 | 8.810 |
| 6 attente disque/copies + calcul des manquants | 379.744 | 102.907 |
| 7 prefetch | 8.844 | 15.057 |

### Phases, hi / a2+adapt (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.223 | 0.240 |
| 2 décisions HOBBIT + regroupement | 0.164 | 0.122 |
| 3 recherche caches + q* + lectures lancées | 0.039 | 0.056 |
| 4 calcul des experts déjà en VRAM | 1.930 | 2.060 |
| 5 copies lancées + branche CPU | 1.466 | 1.868 |
| 6 attente disque/copies + calcul des manquants | 4.124 | 3.475 |
| 7 prefetch | 3.669 | 3.809 |

### Phases, hi / b2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.170 | 0.151 |
| 2 décisions HOBBIT + regroupement | 0.686 | 0.712 |
| 3 recherche caches + q* + lectures lancées | 0.040 | 0.055 |
| 4 calcul des experts déjà en VRAM | 2.922 | 2.651 |
| 5 copies lancées + branche CPU | 1.268 | 1.778 |
| 6 attente disque/copies + calcul des manquants | 3.681 | 3.115 |
| 7 prefetch | 3.874 | 3.488 |

### Phases, hi / b1 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 1.715 | 0.227 |
| 2 décisions HOBBIT + regroupement | 0.101 | 0.119 |
| 3 recherche caches + q* + lectures lancées | 0.039 | 0.053 |
| 4 calcul des experts déjà en VRAM | 1.859 | 1.905 |
| 5 copies lancées + branche CPU | 1.285 | 1.804 |
| 6 attente disque/copies + calcul des manquants | 3.263 | 3.259 |
| 7 prefetch | 5.327 | 3.903 |

### Phases, hi / tout_b2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.207 | 0.232 |
| 2 décisions HOBBIT + regroupement | 0.104 | 0.118 |
| 3 recherche caches + q* + lectures lancées | 0.037 | 0.053 |
| 4 calcul des experts déjà en VRAM | 1.992 | 1.890 |
| 5 copies lancées + branche CPU | 1.177 | 1.654 |
| 6 attente disque/copies + calcul des manquants | 3.045 | 2.825 |
| 7 prefetch | 3.015 | 3.346 |

### Phases, hi / tout_b1 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.201 | 0.227 |
| 2 décisions HOBBIT + regroupement | 0.101 | 0.115 |
| 3 recherche caches + q* + lectures lancées | 0.037 | 0.050 |
| 4 calcul des experts déjà en VRAM | 1.887 | 1.865 |
| 5 copies lancées + branche CPU | 1.157 | 1.672 |
| 6 attente disque/copies + calcul des manquants | 2.960 | 3.479 |
| 7 prefetch | 2.989 | 3.381 |

### Phases, hobbit@0.8 / base (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.862 | 0.147 |
| 2 décisions HOBBIT + regroupement | 0.805 | 0.856 |
| 3 recherche caches + q* + lectures lancées | 0.042 | 0.056 |
| 4 calcul des experts déjà en VRAM | 2.702 | 2.655 |
| 5 copies lancées + branche CPU | 1.204 | 1.585 |
| 6 attente disque/copies + calcul des manquants | 2.859 | 2.445 |
| 7 prefetch | 3.827 | 3.325 |

### Phases, hobbit@0.8 / a2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.202 | 0.235 |
| 2 décisions HOBBIT + regroupement | 0.108 | 0.131 |
| 3 recherche caches + q* + lectures lancées | 0.040 | 0.059 |
| 4 calcul des experts déjà en VRAM | 2.160 | 2.054 |
| 5 copies lancées + branche CPU | 0.972 | 1.630 |
| 6 attente disque/copies + calcul des manquants | 2.326 | 2.664 |
| 7 prefetch | 2.692 | 3.279 |

### Phases, hobbit@0.8 / a2+adapt (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.199 | 0.226 |
| 2 décisions HOBBIT + regroupement | 0.110 | 0.126 |
| 3 recherche caches + q* + lectures lancées | 0.040 | 0.055 |
| 4 calcul des experts déjà en VRAM | 1.915 | 1.932 |
| 5 copies lancées + branche CPU | 0.961 | 1.590 |
| 6 attente disque/copies + calcul des manquants | 2.476 | 2.296 |
| 7 prefetch | 2.695 | 3.145 |

### Phases, hobbit@0.8 / b2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.135 | 0.142 |
| 2 décisions HOBBIT + regroupement | 0.819 | 0.798 |
| 3 recherche caches + q* + lectures lancées | 0.043 | 0.055 |
| 4 calcul des experts déjà en VRAM | 2.890 | 2.518 |
| 5 copies lancées + branche CPU | 1.083 | 1.586 |
| 6 attente disque/copies + calcul des manquants | 2.709 | 2.519 |
| 7 prefetch | 3.053 | 3.289 |

### Phases, hobbit@0.8 / b1 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.551 | 0.239 |
| 2 décisions HOBBIT + regroupement | 0.107 | 0.129 |
| 3 recherche caches + q* + lectures lancées | 0.040 | 0.057 |
| 4 calcul des experts déjà en VRAM | 1.850 | 1.974 |
| 5 copies lancées + branche CPU | 0.992 | 1.612 |
| 6 attente disque/copies + calcul des manquants | 2.164 | 2.096 |
| 7 prefetch | 2.630 | 3.248 |

### Phases, hobbit@0.8 / tout_b2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.191 | 0.241 |
| 2 décisions HOBBIT + regroupement | 0.108 | 0.133 |
| 3 recherche caches + q* + lectures lancées | 0.038 | 0.056 |
| 4 calcul des experts déjà en VRAM | 1.828 | 1.990 |
| 5 copies lancées + branche CPU | 1.015 | 1.643 |
| 6 attente disque/copies + calcul des manquants | 2.163 | 2.206 |
| 7 prefetch | 2.688 | 3.392 |

### Phases, hobbit@0.8 / tout_b1 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.197 | 0.262 |
| 2 décisions HOBBIT + regroupement | 0.109 | 0.123 |
| 3 recherche caches + q* + lectures lancées | 0.039 | 0.053 |
| 4 calcul des experts déjà en VRAM | 1.856 | 1.849 |
| 5 copies lancées + branche CPU | 0.984 | 1.536 |
| 6 attente disque/copies + calcul des manquants | 2.133 | 1.966 |
| 7 prefetch | 2.705 | 3.125 |

## torch.profiler

Lancements de noyaux par token : 3471

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaLaunchKernel | 26527 | 2798.8 |
| cudaMemcpyAsync | 2068 | 502.6 |
| aten::mul | 5864 | 247.0 |
| aten::index | 897 | 183.9 |
| aten::embedding | 9 | 156.8 |
| aten::empty_strided | 7059 | 155.5 |
| cudaMemsetAsync | 77 | 141.8 |
| aten::slice | 15434 | 81.2 |
| aten::mm | 2387 | 78.8 |
| aten::cat | 2371 | 75.9 |
| aten::expand | 813 | 70.2 |
| Activity Buffer Request | 7 | 69.4 |
| aten::copy_ | 8751 | 68.4 |
| aten::empty | 6192 | 66.4 |
| aten::view | 21540 | 62.2 |
| aten::bitwise_and | 1538 | 58.9 |
| aten::zeros | 304 | 57.6 |
| aten::sort | 16 | 48.8 |
| aten::silu | 1793 | 48.7 |
| cuLaunchKernelEx | 2048 | 44.4 |
