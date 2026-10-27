# Profil olmoe-hf — 2026-10-02_0923

Machine : NVIDIA GeForce RTX 2060, k = 8

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 20.398 |
| sync_us | 30.703 |
| floor_loop_gpu_ms | 2.601 |
| floor_loop_wall_ms | 1.664 |
| floor_bmm_gpu_ms | 0.401 |
| floor_bmm_wall_ms | 0.386 |
| vram_bandwidth_gbs | 251.027 |
| dequant_int4_expert_gpu_ms | 0.562 |
| dequant_int4_expert_wall_ms | 0.565 |
| grouped_int4_v1_gpu_ms | 0.516 |
| grouped_int4_v1_gbs | 51.774 |
| grouped_int4_v2_gpu_ms | 0.390 |
| grouped_int4_v2_gbs | 68.527 |
| grouped_int4_v3_gpu_ms | 0.383 |
| grouped_int4_v3_gbs | 69.803 |
| grouped_int4_v4_gpu_ms | 0.385 |
| grouped_int4_v4_gbs | 69.397 |
| grouped_int4_v3_wall_ms | 0.385 |
| grouped_int4_auto_version | 3.000 |
| grouped_int4_graph_wall_ms | 0.159 |
| grouped_int4_v2_kernel_ms | 0.157 |
| grouped_int4_v2_kernel_gbs | 170.849 |
| grouped_int4_v3_kernel_ms | 0.151 |
| grouped_int4_v3_kernel_gbs | 177.214 |
| grouped_int4_v4_kernel_ms | 0.212 |
| grouped_int4_v4_kernel_gbs | 125.852 |
| grouped_int8_kernel_ms | 0.262 |
| grouped_int8_kernel_gbs | 198.427 |
| grouped_fp16_kernel_ms | 0.430 |
| grouped_fp16_kernel_gbs | 234.086 |
| grouped_int4_floor_ms | 0.107 |
| grouped_fp16_gpu_ms | 0.391 |
| grouped_fp16_gbs | 257.152 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token | Mo RAM→VRAM/token | prefetch précision |
|---|---|---|---|---|---|---|---|---|---|---|
| lo / sans_a9 | async | 9.7 | 102.74 | 8.2 | 1.6 | 0.51 | 100% | 0.0 | 0 | - |
| lo / sans_a9 | sync | 13.2 | 75.57 | 11.5 | 1.8 | 0.72 | 100% | 0.0 | 0 | - |
| lo / defaut | async | 9.5 | 104.72 | 7.9 | 1.6 | 0.49 | 100% | 0.0 | 0 | - |
| lo / defaut | sync | 11.5 | 86.87 | 9.8 | 1.7 | 0.61 | 100% | 0.0 | 0 | - |
| hobbit@0.8 / sans_a9 | async | 28.5 | 35.12 | 26.5 | 1.9 | 1.66 | 94% | 0.1 | 131 | 84% |
| hobbit@0.8 / sans_a9 | sync | 38.1 | 26.25 | 36.1 | 2.0 | 2.26 | 94% | 0.0 | 127 | 84% |
| hobbit@0.8 / defaut | async | 25.4 | 39.39 | 23.3 | 2.1 | 1.45 | 94% | 0.1 | 130 | 84% |
| hobbit@0.8 / defaut | sync | 33.4 | 29.91 | 31.1 | 2.3 | 1.94 | 94% | 0.0 | 127 | 84% |

- lo / sans_a9 : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens
- lo / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens
- hobbit@0.8 / sans_a9 : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens
- hobbit@0.8 / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens

### Phases, lo / sans_a9 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.380 | 0.167 |
| 2 décisions HOBBIT + regroupement | 0.034 | 0.049 |
| 3 recherche caches + q* + lectures lancées | 0.015 | 0.027 |
| 4 calcul des experts déjà en VRAM | 0.006 | 0.016 |
| 5 copies lancées + branche CPU | 0.000 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.071 | 0.178 |
| 7 prefetch | 0.001 | 0.010 |

### Phases, lo / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.367 | 0.061 |
| 2 décisions HOBBIT + regroupement | 0.033 | 0.048 |
| 3 recherche caches + q* + lectures lancées | 0.015 | 0.026 |
| 4 calcul des experts déjà en VRAM | 0.005 | 0.015 |
| 5 copies lancées + branche CPU | 0.000 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.071 | 0.177 |
| 7 prefetch | 0.001 | 0.010 |

### Phases, hobbit@0.8 / sans_a9 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.318 | 0.185 |
| 2 décisions HOBBIT + regroupement | 0.052 | 0.068 |
| 3 recherche caches + q* + lectures lancées | 0.026 | 0.039 |
| 4 calcul des experts déjà en VRAM | 0.016 | 0.028 |
| 5 copies lancées + branche CPU | 0.144 | 0.297 |
| 6 attente disque/copies + calcul des manquants | 0.211 | 0.399 |
| 7 prefetch | 0.890 | 0.985 |

### Phases, hobbit@0.8 / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.367 | 0.092 |
| 2 décisions HOBBIT + regroupement | 0.057 | 0.077 |
| 3 recherche caches + q* + lectures lancées | 0.028 | 0.043 |
| 4 calcul des experts déjà en VRAM | 0.017 | 0.031 |
| 5 copies lancées + branche CPU | 0.159 | 0.319 |
| 6 attente disque/copies + calcul des manquants | 0.241 | 0.437 |
| 7 prefetch | 0.584 | 0.698 |

## torch.profiler (décodage seul, variante defaut)

Lancements de noyaux par token : 3 (+ 33 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaMemcpyAsync | 136 | 33.0 |
| cudaGraphLaunch | 264 | 27.6 |
| cudaLaunchKernel | 24 | 2.0 |
| Activity Buffer Request | 1 | 1.7 |
| aten::copy_ | 136 | 1.6 |
| aten::index_select | 8 | 1.2 |
| aten::_to_copy | 128 | 0.8 |
| aten::embedding | 8 | 0.7 |
| aten::view | 160 | 0.7 |
| cudaDeviceSynchronize | 2 | 0.5 |
| aten::mm | 8 | 0.5 |
| aten::to | 128 | 0.4 |
| cudaStreamSynchronize | 128 | 0.4 |
| aten::empty_strided | 128 | 0.3 |
| aten::reshape | 144 | 0.3 |
| aten::argmax | 8 | 0.2 |
| aten::resize_ | 8 | 0.2 |
| aten::matmul | 8 | 0.2 |
| cudaStreamIsCapturing | 128 | 0.1 |
| aten::t | 8 | 0.1 |
