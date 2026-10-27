# Profil olmoe-hf — 2026-10-01_0951

Machine : NVIDIA GeForce RTX 2060, k = 8

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 18.890 |
| sync_us | 24.616 |
| floor_loop_gpu_ms | 1.743 |
| floor_loop_wall_ms | 1.820 |
| floor_bmm_gpu_ms | 0.352 |
| floor_bmm_wall_ms | 0.355 |
| vram_bandwidth_gbs | 285.797 |
| dequant_int4_expert_gpu_ms | 0.567 |
| dequant_int4_expert_wall_ms | 0.561 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | ms par couche MoE | hit VRAM |
|---|---|---|---|---|---|
| lo / base | async | 159.0 | 6.29 | 7.69 | 100% |
| lo / base | sync | 160.9 | 6.21 | 7.82 | 100% |
| lo / defaut | async | 80.5 | 12.42 | 2.88 | 100% |
| lo / defaut | sync | 81.2 | 12.31 | 2.95 | 100% |
| lo / grp | async | 49.6 | 20.16 | 0.89 | 100% |
| lo / grp | sync | 57.2 | 17.48 | 1.38 | 100% |
| lo / grp+d1 | async | 50.0 | 20.01 | 0.91 | 100% |
| lo / grp+d1 | sync | 58.1 | 17.21 | 1.39 | 100% |
| lo / grp+d1+c2 | async | 55.3 | 18.09 | 1.24 | 100% |
| lo / grp+d1+c2 | sync | 63.9 | 15.66 | 1.76 | 100% |
| hi / base | async | 1592.6 | 0.63 | 94.25 | 78% |
| hi / base | sync | 1430.6 | 0.70 | 83.84 | 78% |
| hi / defaut | async | 2534.8 | 0.39 | 154.53 | 78% |
| hi / defaut | sync | 560.4 | 1.78 | 31.84 | 78% |
| hi / grp | async | 1282.5 | 0.78 | 75.35 | 78% |
| hi / grp | sync | 2798.2 | 0.36 | 170.41 | 78% |
| hi / grp+d1 | async | 937.3 | 1.07 | 55.63 | 78% |
| hi / grp+d1 | sync | 192.5 | 5.19 | 9.07 | 78% |
| hi / grp+d1+c2 | async | 226.1 | 4.42 | 11.22 | 72% |
| hi / grp+d1+c2 | sync | 238.6 | 4.19 | 12.14 | 72% |
| hobbit@0.8 / base | async | 210.8 | 4.74 | 10.47 | 78% |
| hobbit@0.8 / base | sync | 223.0 | 4.48 | 11.24 | 78% |
| hobbit@0.8 / defaut | async | 179.8 | 5.56 | 8.25 | 78% |
| hobbit@0.8 / defaut | sync | 201.4 | 4.96 | 9.51 | 78% |
| hobbit@0.8 / grp | async | 161.0 | 6.21 | 7.36 | 78% |
| hobbit@0.8 / grp | sync | 181.8 | 5.50 | 8.58 | 78% |
| hobbit@0.8 / grp+d1 | async | 147.2 | 6.80 | 6.68 | 78% |
| hobbit@0.8 / grp+d1 | sync | 171.6 | 5.83 | 8.11 | 78% |
| hobbit@0.8 / grp+d1+c2 | async | 199.5 | 5.01 | 9.82 | 72% |
| hobbit@0.8 / grp+d1+c2 | sync | 226.1 | 4.42 | 11.30 | 72% |

### Phases, lo / base (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.096 | 0.116 |
| 2 décisions HOBBIT + regroupement | 0.773 | 0.804 |
| 3 recherche caches + q* + lectures lancées | 0.019 | 0.032 |
| 4 calcul des experts déjà en VRAM | 6.346 | 6.343 |
| 5 copies lancées + branche CPU | 0.002 | 0.013 |
| 6 attente disque/copies + calcul des manquants | 0.001 | 0.010 |
| 7 prefetch | 0.450 | 0.478 |

### Phases, lo / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.161 | 0.180 |
| 2 décisions HOBBIT + regroupement | 0.081 | 0.092 |
| 3 recherche caches + q* + lectures lancées | 0.020 | 0.031 |
| 4 calcul des experts déjà en VRAM | 2.612 | 2.595 |
| 5 copies lancées + branche CPU | 0.001 | 0.012 |
| 6 attente disque/copies + calcul des manquants | 0.001 | 0.009 |
| 7 prefetch | 0.007 | 0.009 |

### Phases, lo / grp (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.162 | 0.188 |
| 2 décisions HOBBIT + regroupement | 0.082 | 0.093 |
| 3 recherche caches + q* + lectures lancées | 0.020 | 0.030 |
| 4 calcul des experts déjà en VRAM | 0.007 | 0.017 |
| 5 copies lancées + branche CPU | 0.001 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.608 | 1.011 |
| 7 prefetch | 0.008 | 0.012 |

### Phases, lo / grp+d1 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.166 | 0.189 |
| 2 décisions HOBBIT + regroupement | 0.082 | 0.096 |
| 3 recherche caches + q* + lectures lancées | 0.020 | 0.031 |
| 4 calcul des experts déjà en VRAM | 0.007 | 0.017 |
| 5 copies lancées + branche CPU | 0.001 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.621 | 1.017 |
| 7 prefetch | 0.008 | 0.012 |

### Phases, lo / grp+d1+c2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.159 | 0.185 |
| 2 décisions HOBBIT + regroupement | 0.079 | 0.096 |
| 3 recherche caches + q* + lectures lancées | 0.019 | 0.031 |
| 4 calcul des experts déjà en VRAM | 0.007 | 0.017 |
| 5 copies lancées + branche CPU | 0.008 | 0.020 |
| 6 attente disque/copies + calcul des manquants | 0.635 | 1.083 |
| 7 prefetch | 0.333 | 0.302 |

### Phases, hi / base (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.638 | 0.153 |
| 2 décisions HOBBIT + regroupement | 0.940 | 0.832 |
| 3 recherche caches + q* + lectures lancées | 0.266 | 0.055 |
| 4 calcul des experts déjà en VRAM | 2.711 | 2.896 |
| 5 copies lancées + branche CPU | 8.689 | 4.859 |
| 6 attente disque/copies + calcul des manquants | 59.095 | 60.636 |
| 7 prefetch | 18.987 | 14.057 |

### Phases, hi / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 1.137 | 0.353 |
| 2 décisions HOBBIT + regroupement | 0.108 | 0.148 |
| 3 recherche caches + q* + lectures lancées | 0.042 | 0.057 |
| 4 calcul des experts déjà en VRAM | 1.925 | 2.448 |
| 5 copies lancées + branche CPU | 2.469 | 1.919 |
| 6 attente disque/copies + calcul des manquants | 128.298 | 19.111 |
| 7 prefetch | 20.138 | 7.457 |

### Phases, hi / grp (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.588 | 0.282 |
| 2 décisions HOBBIT + regroupement | 0.108 | 0.306 |
| 3 recherche caches + q* + lectures lancées | 0.255 | 0.197 |
| 4 calcul des experts déjà en VRAM | 0.084 | 0.061 |
| 5 copies lancées + branche CPU | 9.954 | 30.392 |
| 6 attente disque/copies + calcul des manquants | 48.110 | 23.568 |
| 7 prefetch | 16.233 | 114.449 |

### Phases, hi / grp+d1 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.496 | 0.228 |
| 2 décisions HOBBIT + regroupement | 0.116 | 0.118 |
| 3 recherche caches + q* + lectures lancées | 0.077 | 0.053 |
| 4 calcul des experts déjà en VRAM | 0.269 | 0.057 |
| 5 copies lancées + branche CPU | 1.322 | 1.616 |
| 6 attente disque/copies + calcul des manquants | 49.524 | 3.619 |
| 7 prefetch | 3.806 | 3.308 |

### Phases, hi / grp+d1+c2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.198 | 0.223 |
| 2 décisions HOBBIT + regroupement | 0.099 | 0.117 |
| 3 recherche caches + q* + lectures lancées | 0.041 | 0.054 |
| 4 calcul des experts déjà en VRAM | 0.028 | 0.039 |
| 5 copies lancées + branche CPU | 1.641 | 2.312 |
| 6 attente disque/copies + calcul des manquants | 3.634 | 3.259 |
| 7 prefetch | 5.566 | 6.096 |

### Phases, hobbit@0.8 / base (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.131 | 0.147 |
| 2 décisions HOBBIT + regroupement | 0.885 | 0.942 |
| 3 recherche caches + q* + lectures lancées | 0.045 | 0.059 |
| 4 calcul des experts déjà en VRAM | 2.656 | 2.738 |
| 5 copies lancées + branche CPU | 1.082 | 1.595 |
| 6 attente disque/copies + calcul des manquants | 2.491 | 2.365 |
| 7 prefetch | 3.170 | 3.356 |

### Phases, hobbit@0.8 / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.205 | 0.230 |
| 2 décisions HOBBIT + regroupement | 0.109 | 0.127 |
| 3 recherche caches + q* + lectures lancées | 0.041 | 0.056 |
| 4 calcul des experts déjà en VRAM | 1.903 | 2.017 |
| 5 copies lancées + branche CPU | 0.994 | 1.604 |
| 6 attente disque/copies + calcul des manquants | 2.302 | 2.245 |
| 7 prefetch | 2.682 | 3.198 |

### Phases, hobbit@0.8 / grp (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.198 | 0.229 |
| 2 décisions HOBBIT + regroupement | 0.106 | 0.127 |
| 3 recherche caches + q* + lectures lancées | 0.040 | 0.056 |
| 4 calcul des experts déjà en VRAM | 0.046 | 0.061 |
| 5 copies lancées + branche CPU | 0.980 | 1.590 |
| 6 attente disque/copies + calcul des manquants | 3.400 | 3.290 |
| 7 prefetch | 2.584 | 3.193 |

### Phases, hobbit@0.8 / grp+d1 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.184 | 0.217 |
| 2 décisions HOBBIT + regroupement | 0.104 | 0.121 |
| 3 recherche caches + q* + lectures lancées | 0.037 | 0.054 |
| 4 calcul des experts déjà en VRAM | 0.041 | 0.058 |
| 5 copies lancées + branche CPU | 0.895 | 1.558 |
| 6 attente disque/copies + calcul des manquants | 2.975 | 2.895 |
| 7 prefetch | 2.430 | 3.173 |

### Phases, hobbit@0.8 / grp+d1+c2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.188 | 0.230 |
| 2 décisions HOBBIT + regroupement | 0.106 | 0.126 |
| 3 recherche caches + q* + lectures lancées | 0.039 | 0.058 |
| 4 calcul des experts déjà en VRAM | 0.026 | 0.042 |
| 5 copies lancées + branche CPU | 1.569 | 2.299 |
| 6 attente disque/copies + calcul des manquants | 2.775 | 2.677 |
| 7 prefetch | 5.112 | 5.835 |

## torch.profiler

Lancements de noyaux par token : 3199

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaLaunchKernel | 24351 | 480.2 |
| aten::_to_copy | 7491 | 216.7 |
| aten::sum | 1554 | 145.8 |
| cudaMemcpyAsync | 2568 | 119.9 |
| aten::mul | 4840 | 84.8 |
| aten::mm | 2499 | 82.0 |
| aten::copy_ | 9843 | 77.9 |
| aten::slice | 12874 | 67.8 |
| Activity Buffer Request | 7 | 60.4 |
| aten::view | 20596 | 57.5 |
| aten::cat | 2483 | 44.9 |
| aten::bitwise_and | 1538 | 43.7 |
| aten::empty | 4916 | 39.6 |
| aten::empty_strided | 7635 | 38.8 |
| aten::__rshift__ | 1538 | 30.8 |
| aten::unsqueeze | 5707 | 28.2 |
| aten::as_strided | 27118 | 27.1 |
| aten::index | 897 | 26.4 |
| aten::sub | 1538 | 26.1 |
| cuLaunchKernel | 1242 | 22.5 |
