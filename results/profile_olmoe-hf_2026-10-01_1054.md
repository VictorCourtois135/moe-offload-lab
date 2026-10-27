# Profil olmoe-hf — 2026-10-01_1054

Machine : NVIDIA GeForce RTX 2060, k = 8

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 20.030 |
| sync_us | 23.371 |
| floor_loop_gpu_ms | 1.957 |
| floor_loop_wall_ms | 1.939 |
| floor_bmm_gpu_ms | 0.352 |
| floor_bmm_wall_ms | 0.356 |
| vram_bandwidth_gbs | 285.652 |
| dequant_int4_expert_gpu_ms | 0.562 |
| dequant_int4_expert_wall_ms | 0.563 |
| grouped_int4_v1_gpu_ms | 0.501 |
| grouped_int4_v1_gbs | 53.369 |
| grouped_int4_v2_gpu_ms | 0.387 |
| grouped_int4_v2_gbs | 69.093 |
| grouped_int4_floor_ms | 0.094 |
| grouped_fp16_gpu_ms | 0.382 |
| grouped_fp16_gbs | 263.465 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token |
|---|---|---|---|---|---|---|---|---|
| lo / v0110 | async | 49.5 | 20.21 | 15.8 | 33.6 | 0.99 | 100% | 0.0 |
| lo / v0110 | sync | 53.0 | 18.87 | 18.2 | 34.8 | 1.14 | 100% | 0.0 |
| lo / cg_v1 | async | 23.9 | 41.80 | 22.4 | 1.5 | 1.40 | 100% | 0.0 |
| lo / cg_v1 | sync | 28.5 | 35.05 | 26.7 | 1.8 | 1.67 | 100% | 0.0 |
| lo / defaut | async | 20.7 | 48.41 | 19.1 | 1.6 | 1.19 | 100% | 0.0 |
| lo / defaut | sync | 24.6 | 40.69 | 22.9 | 1.7 | 1.43 | 100% | 0.0 |
| hobbit@0.8 / v0110 | async | 396.2 | 2.52 | 332.0 | 64.2 | 20.75 | 78% | 11.0 |
| hobbit@0.8 / v0110 | sync | 901.7 | 1.11 | 861.0 | 40.7 | 53.81 | 78% | 10.7 |
| hobbit@0.8 / cg_v1 | async | 476.0 | 2.10 | 472.9 | 3.1 | 29.55 | 78% | 11.2 |
| hobbit@0.8 / cg_v1 | sync | 157.9 | 6.33 | 154.8 | 3.2 | 9.67 | 78% | 10.7 |
| hobbit@0.8 / defaut | async | 131.2 | 7.62 | 128.2 | 3.0 | 8.01 | 78% | 11.0 |
| hobbit@0.8 / defaut | sync | 148.5 | 6.74 | 145.5 | 2.9 | 9.10 | 78% | 10.7 |

- lo / cg_v1 : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens
- lo / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens
- hobbit@0.8 / cg_v1 : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens
- hobbit@0.8 / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens

### Phases, lo / v0110 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.155 | 0.189 |
| 2 décisions HOBBIT + regroupement | 0.084 | 0.094 |
| 3 recherche caches + q* + lectures lancées | 0.018 | 0.030 |
| 4 calcul des experts déjà en VRAM | 0.007 | 0.016 |
| 5 copies lancées + branche CPU | 0.001 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.723 | 0.764 |
| 7 prefetch | 0.002 | 0.012 |

### Phases, lo / cg_v1 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.688 | 0.201 |
| 2 décisions HOBBIT + regroupement | 0.070 | 0.086 |
| 3 recherche caches + q* + lectures lancées | 0.017 | 0.028 |
| 4 calcul des experts déjà en VRAM | 0.006 | 0.016 |
| 5 copies lancées + branche CPU | 0.000 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.617 | 0.964 |
| 7 prefetch | 0.002 | 0.013 |

### Phases, lo / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.390 | 0.184 |
| 2 décisions HOBBIT + regroupement | 0.063 | 0.083 |
| 3 recherche caches + q* + lectures lancées | 0.017 | 0.030 |
| 4 calcul des experts déjà en VRAM | 0.006 | 0.016 |
| 5 copies lancées + branche CPU | 0.001 | 0.009 |
| 6 attente disque/copies + calcul des manquants | 0.712 | 0.743 |
| 7 prefetch | 0.002 | 0.012 |

### Phases, hobbit@0.8 / v0110 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.277 | 0.224 |
| 2 décisions HOBBIT + regroupement | 0.103 | 0.158 |
| 3 recherche caches + q* + lectures lancées | 0.038 | 0.053 |
| 4 calcul des experts déjà en VRAM | 0.039 | 0.054 |
| 5 copies lancées + branche CPU | 1.112 | 1.853 |
| 6 attente disque/copies + calcul des manquants | 15.773 | 47.616 |
| 7 prefetch | 3.400 | 3.682 |

### Phases, hobbit@0.8 / cg_v1 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.878 | 0.255 |
| 2 décisions HOBBIT + regroupement | 0.099 | 0.114 |
| 3 recherche caches + q* + lectures lancées | 0.041 | 0.052 |
| 4 calcul des experts déjà en VRAM | 0.040 | 0.053 |
| 5 copies lancées + branche CPU | 1.075 | 1.715 |
| 6 attente disque/copies + calcul des manquants | 23.762 | 3.859 |
| 7 prefetch | 3.647 | 3.305 |

### Phases, hobbit@0.8 / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.513 | 0.236 |
| 2 décisions HOBBIT + regroupement | 0.096 | 0.108 |
| 3 recherche caches + q* + lectures lancées | 0.041 | 0.051 |
| 4 calcul des experts déjà en VRAM | 0.040 | 0.052 |
| 5 copies lancées + branche CPU | 1.057 | 1.612 |
| 6 attente disque/copies + calcul des manquants | 2.992 | 3.601 |
| 7 prefetch | 3.265 | 3.114 |

## torch.profiler (décodage seul, variante defaut)

Lancements de noyaux par token : 83 (+ 17 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaMemcpyAsync | 264 | 435.3 |
| cudaLaunchKernel | 664 | 86.3 |
| aten::embedding | 8 | 27.0 |
| cudaGraphLaunch | 136 | 21.2 |
| aten::view | 7456 | 15.2 |
| aten::slice | 4736 | 14.7 |
| aten::linear | 8 | 11.6 |
| cuLaunchKernelEx | 256 | 8.3 |
| aten::copy_ | 904 | 6.9 |
| aten::empty | 776 | 4.2 |
| aten::as_strided | 4752 | 3.4 |
| aten::_to_copy | 384 | 3.1 |
| aten::empty_strided | 384 | 3.0 |
| aten::cat | 128 | 2.3 |
| aten::add_ | 128 | 2.2 |
| aten::index_select | 8 | 1.5 |
| Activity Buffer Request | 1 | 1.4 |
| aten::reshape | 656 | 1.3 |
| aten::fill_ | 128 | 1.2 |
| aten::to | 640 | 1.2 |
