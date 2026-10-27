# Profil qwen3-next-80b-fp8 — 2026-10-03_1413

Machine : NVIDIA GeForce RTX 2060, k = 10

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 22.307 |
| sync_us | 30.694 |
| floor_loop_gpu_ms | 2.338 |
| floor_loop_wall_ms | 2.265 |
| floor_bmm_gpu_ms | 0.356 |
| floor_bmm_wall_ms | 0.358 |
| vram_bandwidth_gbs | 176.578 |
| dequant_int4_expert_gpu_ms | 0.391 |
| dequant_int4_expert_wall_ms | 0.418 |
| grouped_int4_v1_gpu_ms | 1.350 |
| grouped_int4_v1_gbs | 12.379 |
| grouped_int4_v2_gpu_ms | 0.407 |
| grouped_int4_v2_gbs | 41.082 |
| grouped_int4_v3_gpu_ms | 0.407 |
| grouped_int4_v3_gbs | 41.023 |
| grouped_int4_v4_gpu_ms | 0.388 |
| grouped_int4_v4_gbs | 43.029 |
| grouped_int4_v3_wall_ms | 0.395 |
| grouped_int4_auto_version | 3.000 |
| grouped_int4_graph_wall_ms | 0.119 |
| grouped_int4_v2_kernel_ms | 0.120 |
| grouped_int4_v2_kernel_gbs | 139.782 |
| grouped_int4_v3_kernel_ms | 0.115 |
| grouped_int4_v3_kernel_gbs | 145.340 |
| grouped_int4_v4_kernel_ms | 0.150 |
| grouped_int4_v4_kernel_gbs | 111.110 |
| grouped_int8_kernel_ms | 0.179 |
| grouped_int8_kernel_gbs | 181.061 |
| grouped_fp16_kernel_ms | 0.277 |
| grouped_fp16_kernel_gbs | 227.338 |
| grouped_int4_floor_ms | 0.095 |
| grouped_fp16_gpu_ms | 0.377 |
| grouped_fp16_gbs | 166.758 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token | Mo RAM→VRAM/token | prefetch précision | A5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lo / defaut | async | 298.5 | 3.35 | 273.2 | 25.2 | 5.69 | 84% | 60.9 | 227 | 84% | 0% |
| lo / defaut | sync | 307.5 | 3.25 | 283.0 | 24.4 | 5.90 | 85% | 56.7 | 226 | 84% | 0% |
| lo / ahead1 | async | 332.4 | 3.01 | 287.3 | 45.2 | 5.98 | 84% | 71.4 | 228 | 84% | 0% |
| lo / ahead1 | sync | 336.4 | 2.97 | 294.9 | 41.5 | 6.14 | 85% | 66.7 | 228 | 84% | 0% |
| lo / ahead2 | async | 352.9 | 2.83 | 292.6 | 60.3 | 6.10 | 84% | 83.2 | 230 | 84% | 0% |
| lo / ahead2 | sync | 341.5 | 2.93 | 286.3 | 55.2 | 5.97 | 84% | 78.9 | 230 | 84% | 0% |

- lo / defaut : CUDA graphs : 49 segments, 48 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 0 refaits
- lo / ahead1 : CUDA graphs : 49 segments, 48 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 0 refaits
- lo / ahead2 : CUDA graphs : 49 segments, 48 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 0 refaits

### Phases, lo / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 1.377 | 0.111 |
| 2 décisions HOBBIT + regroupement | 0.061 | 0.075 |
| 3 recherche caches + q* + lectures lancées | 0.029 | 0.045 |
| 4 calcul des experts déjà en VRAM | 0.032 | 0.050 |
| 5 copies lancées + branche CPU | 0.063 | 0.136 |
| 6 attente disque/copies + calcul des manquants | 3.058 | 3.054 |
| 7 prefetch | 1.058 | 1.089 |

### Phases, lo / ahead1 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 1.521 | 0.125 |
| 2 décisions HOBBIT + regroupement | 0.065 | 0.079 |
| 3 recherche caches + q* + lectures lancées | 0.032 | 0.048 |
| 4 calcul des experts déjà en VRAM | 0.036 | 0.050 |
| 5 copies lancées + branche CPU | 0.071 | 0.144 |
| 6 attente disque/copies + calcul des manquants | 3.046 | 3.047 |
| 7 prefetch | 1.195 | 1.203 |

### Phases, lo / ahead2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 1.618 | 0.112 |
| 2 décisions HOBBIT + regroupement | 0.064 | 0.073 |
| 3 recherche caches + q* + lectures lancées | 0.032 | 0.045 |
| 4 calcul des experts déjà en VRAM | 0.036 | 0.047 |
| 5 copies lancées + branche CPU | 0.072 | 0.144 |
| 6 attente disque/copies + calcul des manquants | 3.041 | 2.870 |
| 7 prefetch | 1.213 | 1.184 |

## torch.profiler (décodage seul, variante ahead2)

Lancements de noyaux par token : 1 (+ 97 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaMemcpyAsync | 667 | 273.2 |
| cudaGraphLaunch | 388 | 61.6 |
| aten::_to_copy | 200 | 12.8 |
| aten::embedding | 4 | 8.5 |
| aten::copy_ | 667 | 8.2 |
| aten::empty | 471 | 5.2 |
| Activity Buffer Request | 5 | 4.4 |
| cudaStreamSynchronize | 200 | 1.6 |
| cudaDeviceSynchronize | 2 | 1.5 |
| aten::record_stream | 332 | 1.3 |
| cudaStreamIsCapturing | 1658 | 1.2 |
| cudaEventRecordWithFlags | 530 | 1.1 |
| aten::view | 212 | 1.1 |
| cudaEventQuery | 1027 | 1.0 |
| aten::to | 200 | 0.9 |
| aten::empty_strided | 200 | 0.8 |
| cudaEventRecord | 330 | 0.8 |
| cudaStreamWaitEvent | 465 | 0.7 |
| aten::index_select | 4 | 0.5 |
| aten::reshape | 204 | 0.4 |
