# Profil olmoe-hf — 2026-10-02_1242

Machine : NVIDIA GeForce RTX 2060, k = 8

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 17.593 |
| sync_us | 24.627 |
| floor_loop_gpu_ms | 1.790 |
| floor_loop_wall_ms | 1.689 |
| floor_bmm_gpu_ms | 0.353 |
| floor_bmm_wall_ms | 0.359 |
| vram_bandwidth_gbs | 285.173 |
| dequant_int4_expert_gpu_ms | 0.570 |
| dequant_int4_expert_wall_ms | 0.568 |
| grouped_int4_v1_gpu_ms | 0.748 |
| grouped_int4_v1_gbs | 35.748 |
| grouped_int4_v2_gpu_ms | 0.408 |
| grouped_int4_v2_gbs | 65.529 |
| grouped_int4_v3_gpu_ms | 0.373 |
| grouped_int4_v3_gbs | 71.762 |
| grouped_int4_v4_gpu_ms | 0.385 |
| grouped_int4_v4_gbs | 69.446 |
| grouped_int4_v3_wall_ms | 0.381 |
| grouped_int4_auto_version | 3.000 |
| grouped_int4_graph_wall_ms | 0.156 |
| grouped_int4_v2_kernel_ms | 0.158 |
| grouped_int4_v2_kernel_gbs | 168.944 |
| grouped_int4_v3_kernel_ms | 0.148 |
| grouped_int4_v3_kernel_gbs | 180.188 |
| grouped_int4_v4_kernel_ms | 0.216 |
| grouped_int4_v4_kernel_gbs | 123.570 |
| grouped_int8_kernel_ms | 0.246 |
| grouped_int8_kernel_gbs | 210.974 |
| grouped_fp16_kernel_ms | 0.403 |
| grouped_fp16_kernel_gbs | 249.630 |
| grouped_int4_floor_ms | 0.094 |
| grouped_fp16_gpu_ms | 0.376 |
| grouped_fp16_gbs | 267.612 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token | Mo RAM→VRAM/token | prefetch précision | A5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lo / defaut | async | 8.5 | 117.48 | 6.8 | 1.7 | 0.42 | 100% | 0.0 | 0 | - | 0% |
| lo / defaut | sync | 10.6 | 94.78 | 8.7 | 1.8 | 0.55 | 100% | 0.0 | 0 | - | 0% |
| lo / a5 | async | 7.6 | 132.08 | 0.0 | 7.6 | 0.00 | 100% | 0.0 | 0 | - | 100% |
| lo / a5 | sync | 7.6 | 132.09 | 0.0 | 7.6 | 0.00 | 100% | 0.0 | 0 | - | 100% |
| lo / stale | async | 8.4 | 118.35 | 6.8 | 1.6 | 0.43 | 100% | 0.0 | 0 | - | 0% |
| lo / stale | sync | 10.6 | 94.48 | 8.8 | 1.8 | 0.55 | 100% | 0.0 | 0 | - | 0% |
| hobbit@0.8 / defaut | async | 19.4 | 51.52 | 17.4 | 2.0 | 1.09 | 96% | 0.1 | 88 | 84% | 0% |
| hobbit@0.8 / defaut | sync | 24.6 | 40.70 | 22.5 | 2.1 | 1.41 | 96% | 0.0 | 84 | 84% | 0% |
| hobbit@0.8 / a5 | async | 21.9 | 45.71 | 17.5 | 4.4 | 1.09 | 96% | 0.1 | 89 | 84% | 0% |
| hobbit@0.8 / a5 | sync | 24.4 | 40.92 | 22.3 | 2.2 | 1.39 | 96% | 0.0 | 83 | 84% | 0% |
| hobbit@0.8 / stale | async | 23.2 | 43.11 | 21.0 | 2.1 | 1.32 | 97% | 0.0 | 99 | 84% | 0% |
| hobbit@0.8 / stale | sync | 29.4 | 34.05 | 27.0 | 2.3 | 1.69 | 96% | 0.0 | 99 | 84% | 0% |

- lo / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens
- lo / a5 : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 48 tokens, 0 refaits
- lo / stale : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 0 refaits
- hobbit@0.8 / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 0 refaits
- hobbit@0.8 / a5 : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 1 refaits
- hobbit@0.8 / stale : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 0 refaits

### Phases, lo / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.294 | 0.063 |
| 2 décisions HOBBIT + regroupement | 0.034 | 0.048 |
| 3 recherche caches + q* + lectures lancées | 0.015 | 0.026 |
| 4 calcul des experts déjà en VRAM | 0.006 | 0.015 |
| 5 copies lancées + branche CPU | 0.000 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.073 | 0.177 |
| 7 prefetch | 0.001 | 0.010 |

### Phases, lo / a5 (ms par couche)

| Phase | async | sync |
|---|---|---|

### Phases, lo / stale (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.302 | 0.062 |
| 2 décisions HOBBIT + regroupement | 0.032 | 0.048 |
| 3 recherche caches + q* + lectures lancées | 0.015 | 0.026 |
| 4 calcul des experts déjà en VRAM | 0.005 | 0.015 |
| 5 copies lancées + branche CPU | 0.000 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.070 | 0.177 |
| 7 prefetch | 0.001 | 0.010 |

### Phases, hobbit@0.8 / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.302 | 0.075 |
| 2 décisions HOBBIT + regroupement | 0.049 | 0.064 |
| 3 recherche caches + q* + lectures lancées | 0.023 | 0.036 |
| 4 calcul des experts déjà en VRAM | 0.012 | 0.023 |
| 5 copies lancées + branche CPU | 0.096 | 0.205 |
| 6 attente disque/copies + calcul des manquants | 0.211 | 0.371 |
| 7 prefetch | 0.395 | 0.440 |

### Phases, hobbit@0.8 / a5 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.304 | 0.071 |
| 2 décisions HOBBIT + regroupement | 0.052 | 0.062 |
| 3 recherche caches + q* + lectures lancées | 0.025 | 0.035 |
| 4 calcul des experts déjà en VRAM | 0.013 | 0.023 |
| 5 copies lancées + branche CPU | 0.096 | 0.201 |
| 6 attente disque/copies + calcul des manquants | 0.209 | 0.368 |
| 7 prefetch | 0.394 | 0.435 |

### Phases, hobbit@0.8 / stale (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.270 | 0.084 |
| 2 décisions HOBBIT + regroupement | 0.057 | 0.070 |
| 3 recherche caches + q* + lectures lancées | 0.026 | 0.039 |
| 4 calcul des experts déjà en VRAM | 0.020 | 0.031 |
| 5 copies lancées + branche CPU | 0.067 | 0.184 |
| 6 attente disque/copies + calcul des manquants | 0.210 | 0.407 |
| 7 prefetch | 0.663 | 0.691 |

## torch.profiler (décodage seul, variante stale)

Lancements de noyaux par token : 2 (+ 33 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaMemcpyAsync | 136 | 25.9 |
| cudaGraphLaunch | 264 | 25.9 |
| cudaLaunchKernel | 16 | 2.0 |
| aten::copy_ | 136 | 1.5 |
| Activity Buffer Request | 1 | 1.1 |
| aten::index_select | 8 | 1.1 |
| aten::_to_copy | 128 | 0.9 |
| aten::view | 168 | 0.7 |
| aten::embedding | 8 | 0.7 |
| aten::to | 128 | 0.4 |
| cudaStreamSynchronize | 128 | 0.4 |
| aten::reshape | 152 | 0.3 |
| aten::empty_strided | 128 | 0.3 |
| aten::resize_ | 8 | 0.2 |
| cuLaunchKernelEx | 8 | 0.2 |
| aten::argmax | 8 | 0.2 |
| cudaDeviceSynchronize | 2 | 0.2 |
| aten::empty | 16 | 0.1 |
| cudaStreamIsCapturing | 128 | 0.1 |
| aten::resolve_conj | 128 | 0.1 |
