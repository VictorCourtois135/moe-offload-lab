# Profil olmoe-hf — 2026-10-02_1104

Machine : NVIDIA GeForce RTX 2060, k = 8

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 17.401 |
| sync_us | 23.642 |
| floor_loop_gpu_ms | 1.650 |
| floor_loop_wall_ms | 1.957 |
| floor_bmm_gpu_ms | 0.356 |
| floor_bmm_wall_ms | 0.365 |
| vram_bandwidth_gbs | 282.393 |
| dequant_int4_expert_gpu_ms | 0.571 |
| dequant_int4_expert_wall_ms | 0.568 |
| grouped_int4_v1_gpu_ms | 1.260 |
| grouped_int4_v1_gbs | 21.217 |
| grouped_int4_v2_gpu_ms | 0.387 |
| grouped_int4_v2_gbs | 69.047 |
| grouped_int4_v3_gpu_ms | 0.383 |
| grouped_int4_v3_gbs | 69.840 |
| grouped_int4_v4_gpu_ms | 0.369 |
| grouped_int4_v4_gbs | 72.549 |
| grouped_int4_v3_wall_ms | 0.372 |
| grouped_int4_auto_version | 3.000 |
| grouped_int4_graph_wall_ms | 0.154 |
| grouped_int4_v2_kernel_ms | 0.156 |
| grouped_int4_v2_kernel_gbs | 171.024 |
| grouped_int4_v3_kernel_ms | 0.152 |
| grouped_int4_v3_kernel_gbs | 176.444 |
| grouped_int4_v4_kernel_ms | 0.217 |
| grouped_int4_v4_kernel_gbs | 123.103 |
| grouped_int8_kernel_ms | 0.270 |
| grouped_int8_kernel_gbs | 192.242 |
| grouped_fp16_kernel_ms | 0.422 |
| grouped_fp16_kernel_gbs | 238.306 |
| grouped_int4_floor_ms | 0.095 |
| grouped_fp16_gpu_ms | 0.392 |
| grouped_fp16_gbs | 256.948 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token | Mo RAM→VRAM/token | prefetch précision | A5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lo / defaut | async | 8.5 | 118.34 | 6.8 | 1.7 | 0.42 | 100% | 0.0 | 0 | - | 0% |
| lo / defaut | sync | 10.7 | 93.50 | 8.8 | 1.9 | 0.55 | 100% | 0.0 | 0 | - | 0% |
| lo / a5 | async | 15.8 | 63.09 | 6.7 | 9.1 | 0.42 | 100% | 0.1 | 1 | 83% | 0% |
| lo / a5 | sync | 18.7 | 53.62 | 9.3 | 9.4 | 0.58 | 100% | 0.0 | 0 | 83% | 0% |
| lo / s3 | async | 8.9 | 112.01 | 7.1 | 1.8 | 0.45 | 100% | 0.0 | 0 | - | 0% |
| lo / s3 | sync | 11.0 | 91.03 | 9.0 | 1.9 | 0.57 | 100% | 0.0 | 0 | - | 0% |
| lo / tout | async | 16.8 | 59.47 | 7.5 | 9.3 | 0.47 | 100% | 0.1 | 1 | 49% | 0% |
| lo / tout | sync | 19.2 | 52.10 | 9.6 | 9.6 | 0.60 | 100% | 0.0 | 0 | 49% | 0% |
| hobbit@0.8 / defaut | async | 20.1 | 49.81 | 18.1 | 2.0 | 1.13 | 96% | 0.1 | 89 | 84% | 0% |
| hobbit@0.8 / defaut | sync | 26.0 | 38.48 | 23.6 | 2.4 | 1.47 | 96% | 0.0 | 84 | 84% | 0% |
| hobbit@0.8 / a5 | async | 22.3 | 44.74 | 17.9 | 4.5 | 1.12 | 96% | 0.1 | 89 | 84% | 0% |
| hobbit@0.8 / a5 | sync | 25.7 | 38.90 | 22.9 | 2.8 | 1.43 | 96% | 0.0 | 84 | 84% | 0% |
| hobbit@0.8 / s3 | async | 48.3 | 20.72 | 45.7 | 2.5 | 2.86 | 98% | 0.6 | 286 | 49% | 0% |
| hobbit@0.8 / s3 | sync | 54.6 | 18.33 | 52.2 | 2.4 | 3.26 | 98% | 0.1 | 286 | 49% | 0% |
| hobbit@0.8 / tout | async | 49.1 | 20.39 | 46.5 | 2.6 | 2.90 | 98% | 0.6 | 286 | 49% | 0% |
| hobbit@0.8 / tout | sync | 54.4 | 18.37 | 52.1 | 2.4 | 3.25 | 98% | 0.1 | 286 | 49% | 0% |

- lo / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens
- lo / a5 : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 48 refaits
- lo / s3 : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 0 refaits
- lo / tout : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 48 refaits
- hobbit@0.8 / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 0 refaits
- hobbit@0.8 / a5 : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 1 refaits
- hobbit@0.8 / s3 : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 0 refaits
- hobbit@0.8 / tout : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 0 refaits

### Phases, lo / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.298 | 0.063 |
| 2 décisions HOBBIT + regroupement | 0.032 | 0.050 |
| 3 recherche caches + q* + lectures lancées | 0.015 | 0.027 |
| 4 calcul des experts déjà en VRAM | 0.006 | 0.016 |
| 5 copies lancées + branche CPU | 0.000 | 0.009 |
| 6 attente disque/copies + calcul des manquants | 0.070 | 0.178 |
| 7 prefetch | 0.001 | 0.011 |

### Phases, lo / a5 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.258 | 0.067 |
| 2 décisions HOBBIT + regroupement | 0.036 | 0.052 |
| 3 recherche caches + q* + lectures lancées | 0.014 | 0.026 |
| 4 calcul des experts déjà en VRAM | 0.005 | 0.015 |
| 5 copies lancées + branche CPU | 0.000 | 0.009 |
| 6 attente disque/copies + calcul des manquants | 0.073 | 0.184 |
| 7 prefetch | 0.031 | 0.044 |

### Phases, lo / s3 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.320 | 0.062 |
| 2 décisions HOBBIT + regroupement | 0.032 | 0.047 |
| 3 recherche caches + q* + lectures lancées | 0.015 | 0.027 |
| 4 calcul des experts déjà en VRAM | 0.005 | 0.015 |
| 5 copies lancées + branche CPU | 0.000 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.071 | 0.178 |
| 7 prefetch | 0.001 | 0.010 |

### Phases, lo / tout (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.292 | 0.065 |
| 2 décisions HOBBIT + regroupement | 0.037 | 0.051 |
| 3 recherche caches + q* + lectures lancées | 0.015 | 0.026 |
| 4 calcul des experts déjà en VRAM | 0.006 | 0.016 |
| 5 copies lancées + branche CPU | 0.000 | 0.009 |
| 6 attente disque/copies + calcul des manquants | 0.076 | 0.185 |
| 7 prefetch | 0.043 | 0.056 |

### Phases, hobbit@0.8 / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.299 | 0.083 |
| 2 décisions HOBBIT + regroupement | 0.051 | 0.070 |
| 3 recherche caches + q* + lectures lancées | 0.024 | 0.039 |
| 4 calcul des experts déjà en VRAM | 0.013 | 0.025 |
| 5 copies lancées + branche CPU | 0.107 | 0.209 |
| 6 attente disque/copies + calcul des manquants | 0.229 | 0.392 |
| 7 prefetch | 0.403 | 0.475 |

### Phases, hobbit@0.8 / a5 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.302 | 0.081 |
| 2 décisions HOBBIT + regroupement | 0.051 | 0.066 |
| 3 recherche caches + q* + lectures lancées | 0.024 | 0.037 |
| 4 calcul des experts déjà en VRAM | 0.013 | 0.024 |
| 5 copies lancées + branche CPU | 0.107 | 0.206 |
| 6 attente disque/copies + calcul des manquants | 0.224 | 0.389 |
| 7 prefetch | 0.397 | 0.459 |

### Phases, hobbit@0.8 / s3 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.262 | 0.089 |
| 2 décisions HOBBIT + regroupement | 0.061 | 0.070 |
| 3 recherche caches + q* + lectures lancées | 0.024 | 0.037 |
| 4 calcul des experts déjà en VRAM | 0.022 | 0.034 |
| 5 copies lancées + branche CPU | 0.031 | 0.110 |
| 6 attente disque/copies + calcul des manquants | 0.209 | 0.383 |
| 7 prefetch | 2.248 | 2.331 |

### Phases, hobbit@0.8 / tout (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.259 | 0.089 |
| 2 décisions HOBBIT + regroupement | 0.057 | 0.069 |
| 3 recherche caches + q* + lectures lancées | 0.024 | 0.036 |
| 4 calcul des experts déjà en VRAM | 0.023 | 0.034 |
| 5 copies lancées + branche CPU | 0.030 | 0.110 |
| 6 attente disque/copies + calcul des manquants | 0.207 | 0.391 |
| 7 prefetch | 2.302 | 2.317 |

## torch.profiler (décodage seul, variante tout)

Lancements de noyaux par token : 4 (+ 34 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaGraphLaunch | 272 | 58.5 |
| cudaMemcpyAsync | 290 | 56.4 |
| cudaLaunchKernel | 32 | 2.9 |
| aten::copy_ | 290 | 2.9 |
| Activity Buffer Request | 3 | 2.3 |
| Buffer Flush | 1 | 1.9 |
| aten::index_select | 16 | 1.2 |
| aten::_to_copy | 136 | 0.9 |
| aten::embedding | 16 | 0.8 |
| aten::view | 192 | 0.8 |
| aten::to | 136 | 0.5 |
| cudaStreamSynchronize | 136 | 0.4 |
| cudaEventRecordWithFlags | 10 | 0.4 |
| aten::reshape | 168 | 0.3 |
| aten::empty_strided | 136 | 0.3 |
| aten::empty | 31 | 0.3 |
| aten::resize_ | 16 | 0.3 |
| cudaDeviceSynchronize | 2 | 0.2 |
| aten::cat | 8 | 0.2 |
| aten::argmax | 8 | 0.2 |
