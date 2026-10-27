# Profil olmoe-hf — 2026-10-01_1227

Machine : NVIDIA GeForce RTX 2060, k = 8

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 19.764 |
| sync_us | 29.309 |
| floor_loop_gpu_ms | 2.366 |
| floor_loop_wall_ms | 1.645 |
| floor_bmm_gpu_ms | 0.388 |
| floor_bmm_wall_ms | 0.371 |
| vram_bandwidth_gbs | 259.286 |
| dequant_int4_expert_gpu_ms | 0.562 |
| dequant_int4_expert_wall_ms | 0.564 |
| grouped_int4_v1_gpu_ms | 2.137 |
| grouped_int4_v1_gbs | 12.513 |
| grouped_int4_v2_gpu_ms | 0.386 |
| grouped_int4_v2_gbs | 69.350 |
| grouped_int4_v3_gpu_ms | 0.386 |
| grouped_int4_v3_gbs | 69.254 |
| grouped_int4_v4_gpu_ms | 0.400 |
| grouped_int4_v4_gbs | 66.922 |
| grouped_int4_v3_wall_ms | 0.377 |
| grouped_int4_auto_version | 3.000 |
| grouped_int4_graph_wall_ms | 0.159 |
| grouped_int4_v2_kernel_ms | 0.160 |
| grouped_int4_v2_kernel_gbs | 167.126 |
| grouped_int4_v3_kernel_ms | 0.152 |
| grouped_int4_v3_kernel_gbs | 175.890 |
| grouped_int4_v4_kernel_ms | 0.238 |
| grouped_int4_v4_kernel_gbs | 112.239 |
| grouped_int8_kernel_ms | 0.235 |
| grouped_int8_kernel_gbs | 221.190 |
| grouped_fp16_kernel_ms | 0.425 |
| grouped_fp16_kernel_gbs | 236.939 |
| grouped_int4_floor_ms | 0.103 |
| grouped_fp16_gpu_ms | 0.382 |
| grouped_fp16_gbs | 263.586 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token | Mo RAM→VRAM/token | prefetch précision |
|---|---|---|---|---|---|---|---|---|---|---|
| lo / defaut | async | 9.6 | 104.36 | 8.1 | 1.5 | 0.51 | 100% | 0.0 | 0 | - |
| lo / defaut | sync | 13.7 | 73.02 | 11.9 | 1.8 | 0.74 | 100% | 0.0 | 0 | - |
| hobbit@0.8 / defaut | async | 27.7 | 36.17 | 25.9 | 1.7 | 1.62 | 94% | 0.1 | 130 | 84% |
| hobbit@0.8 / defaut | sync | 36.9 | 27.11 | 35.1 | 1.8 | 2.19 | 94% | 0.0 | 127 | 84% |

- lo / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens
- hobbit@0.8 / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens

### Phases, lo / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.366 | 0.199 |
| 2 décisions HOBBIT + regroupement | 0.037 | 0.049 |
| 3 recherche caches + q* + lectures lancées | 0.015 | 0.026 |
| 4 calcul des experts déjà en VRAM | 0.006 | 0.015 |
| 5 copies lancées + branche CPU | 0.000 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.079 | 0.194 |
| 7 prefetch | 0.001 | 0.011 |

### Phases, hobbit@0.8 / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.301 | 0.184 |
| 2 décisions HOBBIT + regroupement | 0.051 | 0.064 |
| 3 recherche caches + q* + lectures lancées | 0.024 | 0.036 |
| 4 calcul des experts déjà en VRAM | 0.015 | 0.025 |
| 5 copies lancées + branche CPU | 0.147 | 0.299 |
| 6 attente disque/copies + calcul des manquants | 0.211 | 0.391 |
| 7 prefetch | 0.868 | 0.953 |

## torch.profiler (décodage seul, variante defaut)

Lancements de noyaux par token : 51 (+ 33 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaGraphLaunch | 264 | 22.7 |
| cudaMemcpyAsync | 136 | 14.4 |
| cudaLaunchKernel | 408 | 8.0 |
| aten::copy_ | 392 | 3.8 |
| aten::empty_strided | 384 | 2.3 |
| aten::_to_copy | 384 | 2.3 |
| aten::cat | 128 | 2.2 |
| aten::view | 416 | 1.6 |
| Activity Buffer Request | 1 | 1.4 |
| aten::to | 384 | 1.0 |
| aten::index_select | 8 | 0.8 |
| aten::reshape | 400 | 0.8 |
| aten::embedding | 8 | 0.6 |
| cudaDeviceSynchronize | 2 | 0.5 |
| aten::mm | 8 | 0.3 |
| cudaStreamSynchronize | 128 | 0.3 |
| aten::resize_ | 8 | 0.2 |
| aten::argmax | 8 | 0.2 |
| cudaStreamIsCapturing | 128 | 0.1 |
| aten::matmul | 8 | 0.1 |
