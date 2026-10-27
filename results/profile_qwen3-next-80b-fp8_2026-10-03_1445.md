# Profil qwen3-next-80b-fp8 — 2026-10-03_1445

Machine : NVIDIA GeForce RTX 2060, k = 10

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 17.770 |
| sync_us | 23.533 |
| floor_loop_gpu_ms | 2.126 |
| floor_loop_wall_ms | 2.145 |
| floor_bmm_gpu_ms | 0.276 |
| floor_bmm_wall_ms | 0.279 |
| vram_bandwidth_gbs | 227.935 |
| dequant_int4_expert_gpu_ms | 0.393 |
| dequant_int4_expert_wall_ms | 0.394 |
| grouped_int4_v1_gpu_ms | 1.097 |
| grouped_int4_v1_gbs | 15.240 |
| grouped_int4_v2_gpu_ms | 0.428 |
| grouped_int4_v2_gbs | 39.003 |
| grouped_int4_v3_gpu_ms | 0.390 |
| grouped_int4_v3_gbs | 42.816 |
| grouped_int4_v4_gpu_ms | 0.388 |
| grouped_int4_v4_gbs | 43.110 |
| grouped_int4_v3_wall_ms | 0.388 |
| grouped_int4_auto_version | 3.000 |
| grouped_int4_graph_wall_ms | 0.120 |
| grouped_int4_v2_kernel_ms | 0.119 |
| grouped_int4_v2_kernel_gbs | 140.163 |
| grouped_int4_v3_kernel_ms | 0.120 |
| grouped_int4_v3_kernel_gbs | 139.351 |
| grouped_int4_v4_kernel_ms | 0.151 |
| grouped_int4_v4_kernel_gbs | 110.513 |
| grouped_int8_kernel_ms | 0.213 |
| grouped_int8_kernel_gbs | 152.579 |
| grouped_fp16_kernel_ms | 0.292 |
| grouped_fp16_kernel_gbs | 215.804 |
| grouped_int4_floor_ms | 0.073 |
| grouped_fp16_gpu_ms | 0.379 |
| grouped_fp16_gbs | 165.951 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token | Mo RAM→VRAM/token | prefetch précision | A5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lo / split1 | async | 146.5 | 6.82 | 136.1 | 10.5 | 2.83 | 84% | 60.7 | 226 | 84% | 0% |
| lo / split1 | sync | 151.1 | 6.62 | 141.4 | 9.7 | 2.95 | 84% | 57.1 | 225 | 84% | 0% |
| lo / split4 | async | 92.2 | 10.85 | 78.9 | 13.3 | 1.64 | 84% | 60.7 | 226 | 84% | 0% |
| lo / split4 | sync | 103.9 | 9.63 | 91.3 | 12.5 | 1.90 | 85% | 57.0 | 225 | 84% | 0% |

- lo / split1 : CUDA graphs : 49 segments, 48 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 0 refaits
- lo / split4 : CUDA graphs : 49 segments, 48 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 0 refaits

### Phases, lo / split1 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.520 | 0.098 |
| 2 décisions HOBBIT + regroupement | 0.056 | 0.069 |
| 3 recherche caches + q* + lectures lancées | 0.045 | 0.061 |
| 4 calcul des experts déjà en VRAM | 0.032 | 0.049 |
| 5 copies lancées + branche CPU | 0.063 | 0.132 |
| 6 attente disque/copies + calcul des manquants | 1.822 | 1.657 |
| 7 prefetch | 0.281 | 0.353 |

### Phases, lo / split4 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.402 | 0.120 |
| 2 décisions HOBBIT + regroupement | 0.061 | 0.088 |
| 3 recherche caches + q* + lectures lancées | 0.048 | 0.074 |
| 4 calcul des experts déjà en VRAM | 0.054 | 0.079 |
| 5 copies lancées + branche CPU | 0.097 | 0.159 |
| 6 attente disque/copies + calcul des manquants | 0.646 | 0.590 |
| 7 prefetch | 0.315 | 0.404 |

## torch.profiler (décodage seul, variante split4)

Lancements de noyaux par token : 1 (+ 97 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaMemcpyAsync | 666 | 80.7 |
| cudaGraphLaunch | 388 | 60.1 |
| aten::copy_ | 666 | 8.8 |
| aten::empty | 470 | 5.5 |
| Activity Buffer Request | 5 | 3.5 |
| aten::_to_copy | 200 | 2.0 |
| aten::view | 212 | 1.9 |
| aten::record_stream | 340 | 1.5 |
| cudaStreamIsCapturing | 1660 | 1.4 |
| cudaEventQuery | 989 | 1.0 |
| aten::empty_strided | 200 | 1.0 |
| aten::to | 200 | 0.9 |
| cudaDeviceSynchronize | 2 | 0.9 |
| cudaEventRecordWithFlags | 535 | 0.8 |
| cudaStreamSynchronize | 200 | 0.7 |
| cudaEventRecord | 339 | 0.7 |
| cudaStreamWaitEvent | 463 | 0.7 |
| aten::reshape | 204 | 0.6 |
| Buffer Flush | 1 | 0.5 |
| aten::index_select | 4 | 0.4 |
