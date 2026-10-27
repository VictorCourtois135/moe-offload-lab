# Profil olmoe-hf — 2026-10-01_1026

Machine : NVIDIA GeForce RTX 2060, k = 8

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 20.706 |
| sync_us | 22.481 |
| floor_loop_gpu_ms | 1.800 |
| floor_loop_wall_ms | 1.609 |
| floor_bmm_gpu_ms | 0.426 |
| floor_bmm_wall_ms | 0.357 |
| vram_bandwidth_gbs | 236.575 |
| dequant_int4_expert_gpu_ms | 0.562 |
| dequant_int4_expert_wall_ms | 0.564 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token |
|---|---|---|---|---|---|---|---|---|
| lo / v30 | async | 79.2 | 12.62 | 45.2 | 34.0 | 2.83 | 100% | 0.0 |
| lo / v30 | sync | 80.2 | 12.48 | 46.6 | 33.5 | 2.91 | 100% | 0.0 |
| lo / defaut | async | 47.9 | 20.86 | 14.2 | 33.7 | 0.89 | 100% | 0.0 |
| lo / defaut | sync | 56.9 | 17.58 | 22.5 | 34.4 | 1.41 | 100% | 0.0 |
| lo / cg | async | 23.9 | 41.88 | 22.4 | 1.5 | 1.40 | 100% | 0.0 |
| lo / cg | sync | 27.9 | 35.86 | 26.2 | 1.7 | 1.63 | 100% | 0.0 |
| hobbit@0.8 / v30 | async | 377.1 | 2.65 | 328.2 | 49.0 | 20.51 | 78% | 11.0 |
| hobbit@0.8 / v30 | sync | 252.1 | 3.97 | 175.7 | 76.3 | 10.98 | 78% | 10.9 |
| hobbit@0.8 / defaut | async | 148.2 | 6.75 | 105.2 | 43.0 | 6.57 | 78% | 11.0 |
| hobbit@0.8 / defaut | sync | 177.7 | 5.63 | 131.9 | 45.7 | 8.25 | 78% | 10.8 |
| hobbit@0.8 / cg | async | 125.3 | 7.98 | 122.3 | 3.0 | 7.65 | 78% | 11.0 |
| hobbit@0.8 / cg | sync | 140.1 | 7.14 | 137.2 | 3.0 | 8.57 | 78% | 10.8 |

- lo / cg : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens
- hobbit@0.8 / cg : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens

### Phases, lo / v30 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.165 | 0.196 |
| 2 décisions HOBBIT + regroupement | 0.091 | 0.100 |
| 3 recherche caches + q* + lectures lancées | 0.019 | 0.029 |
| 4 calcul des experts déjà en VRAM | 2.547 | 2.536 |
| 5 copies lancées + branche CPU | 0.002 | 0.012 |
| 6 attente disque/copies + calcul des manquants | 0.001 | 0.010 |
| 7 prefetch | 0.001 | 0.009 |

### Phases, lo / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.157 | 0.189 |
| 2 décisions HOBBIT + regroupement | 0.084 | 0.094 |
| 3 recherche caches + q* + lectures lancées | 0.018 | 0.029 |
| 4 calcul des experts déjà en VRAM | 0.007 | 0.016 |
| 5 copies lancées + branche CPU | 0.001 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.621 | 1.034 |
| 7 prefetch | 0.002 | 0.012 |

### Phases, lo / cg (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.693 | 0.194 |
| 2 décisions HOBBIT + regroupement | 0.068 | 0.083 |
| 3 recherche caches + q* + lectures lancées | 0.018 | 0.029 |
| 4 calcul des experts déjà en VRAM | 0.007 | 0.016 |
| 5 copies lancées + branche CPU | 0.001 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.608 | 0.935 |
| 7 prefetch | 0.002 | 0.013 |

### Phases, hobbit@0.8 / v30 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.300 | 0.230 |
| 2 décisions HOBBIT + regroupement | 0.111 | 0.129 |
| 3 recherche caches + q* + lectures lancées | 0.040 | 0.059 |
| 4 calcul des experts déjà en VRAM | 1.913 | 1.946 |
| 5 copies lancées + branche CPU | 0.994 | 1.815 |
| 6 attente disque/copies + calcul des manquants | 14.387 | 3.045 |
| 7 prefetch | 2.757 | 3.713 |

### Phases, hobbit@0.8 / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.192 | 0.218 |
| 2 décisions HOBBIT + regroupement | 0.104 | 0.121 |
| 3 recherche caches + q* + lectures lancées | 0.038 | 0.054 |
| 4 calcul des experts déjà en VRAM | 0.044 | 0.057 |
| 5 copies lancées + branche CPU | 0.921 | 1.602 |
| 6 attente disque/copies + calcul des manquants | 2.407 | 3.099 |
| 7 prefetch | 2.860 | 3.061 |

### Phases, hobbit@0.8 / cg (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.518 | 0.233 |
| 2 décisions HOBBIT + regroupement | 0.098 | 0.108 |
| 3 recherche caches + q* + lectures lancées | 0.042 | 0.051 |
| 4 calcul des experts déjà en VRAM | 0.042 | 0.052 |
| 5 copies lancées + branche CPU | 0.975 | 1.549 |
| 6 attente disque/copies + calcul des manquants | 2.997 | 3.219 |
| 7 prefetch | 2.966 | 3.036 |

## torch.profiler (décodage seul, variante cg)

Lancements de noyaux par token : 187 (+ 17 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaMemcpyAsync | 1410 | 202.2 |
| cudaGraphLaunch | 136 | 43.5 |
| aten::_to_copy | 864 | 32.3 |
| cudaLaunchKernel | 1496 | 27.7 |
| aten::copy_ | 2410 | 24.0 |
| aten::view | 7688 | 15.7 |
| aten::slice | 4736 | 14.9 |
| aten::empty | 1674 | 14.2 |
| aten::linear | 128 | 9.8 |
| aten::embedding | 8 | 7.7 |
| aten::empty_strided | 864 | 6.1 |
| cuLaunchKernelEx | 256 | 5.6 |
| aten::mm | 128 | 5.1 |
| aten::cat | 248 | 4.9 |
| aten::as_strided | 4872 | 3.7 |
| aten::topk | 120 | 3.3 |
| cudaStreamIsCapturing | 3642 | 3.0 |
| aten::record_stream | 760 | 2.7 |
| Activity Buffer Request | 2 | 2.6 |
| aten::reshape | 1016 | 2.5 |
