# Profil qwen3-next-80b-fp8 — 2026-10-02_1850

Machine : NVIDIA GeForce RTX 2060, k = 10

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 17.291 |
| sync_us | 23.535 |
| floor_loop_gpu_ms | 2.133 |
| floor_loop_wall_ms | 2.076 |
| floor_bmm_gpu_ms | 0.347 |
| floor_bmm_wall_ms | 0.327 |
| vram_bandwidth_gbs | 181.139 |
| dequant_int4_expert_gpu_ms | 0.389 |
| dequant_int4_expert_wall_ms | 0.384 |
| grouped_int4_v1_gpu_ms | 1.478 |
| grouped_int4_v1_gbs | 11.311 |
| grouped_int4_v2_gpu_ms | 0.378 |
| grouped_int4_v2_gbs | 44.169 |
| grouped_int4_v3_gpu_ms | 0.377 |
| grouped_int4_v3_gbs | 44.319 |
| grouped_int4_v4_gpu_ms | 0.384 |
| grouped_int4_v4_gbs | 43.467 |
| grouped_int4_v3_wall_ms | 0.387 |
| grouped_int4_auto_version | 3.000 |
| grouped_int4_graph_wall_ms | 0.117 |
| grouped_int4_v2_kernel_ms | 0.118 |
| grouped_int4_v2_kernel_gbs | 141.188 |
| grouped_int4_v3_kernel_ms | 0.112 |
| grouped_int4_v3_kernel_gbs | 149.219 |
| grouped_int4_v4_kernel_ms | 0.147 |
| grouped_int4_v4_kernel_gbs | 113.939 |
| grouped_int8_kernel_ms | 0.166 |
| grouped_int8_kernel_gbs | 195.849 |
| grouped_fp16_kernel_ms | 0.292 |
| grouped_fp16_kernel_gbs | 215.314 |
| grouped_int4_floor_ms | 0.092 |
| grouped_fp16_gpu_ms | 0.368 |
| grouped_fp16_gbs | 170.979 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token | Mo RAM→VRAM/token | prefetch précision | A5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lo / defaut | async | 286.8 | 3.49 | 262.4 | 24.4 | 5.47 | 84% | 60.8 | 226 | 84% | 0% |
| lo / defaut | sync | 294.5 | 3.40 | 270.8 | 23.7 | 5.64 | 84% | 57.1 | 226 | 84% | 0% |

- lo / defaut : CUDA graphs : 49 segments, 48 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 0 refaits

### Phases, lo / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 1.355 | 0.094 |
| 2 décisions HOBBIT + regroupement | 0.059 | 0.069 |
| 3 recherche caches + q* + lectures lancées | 0.028 | 0.044 |
| 4 calcul des experts déjà en VRAM | 0.031 | 0.048 |
| 5 copies lancées + branche CPU | 0.059 | 0.130 |
| 6 attente disque/copies + calcul des manquants | 2.916 | 2.987 |
| 7 prefetch | 1.005 | 1.041 |

## torch.profiler (décodage seul, variante defaut)

Lancements de noyaux par token : 1 (+ 97 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaMemcpyAsync | 670 | 245.1 |
| cudaGraphLaunch | 388 | 53.6 |
| aten::copy_ | 670 | 8.1 |
| aten::empty | 474 | 5.1 |
| Activity Buffer Request | 5 | 3.4 |
| aten::_to_copy | 200 | 1.9 |
| cudaDeviceSynchronize | 2 | 1.4 |
| aten::record_stream | 341 | 1.4 |
| cudaStreamIsCapturing | 1670 | 1.3 |
| cudaStreamSynchronize | 200 | 1.2 |
| aten::view | 212 | 1.1 |
| cudaEventQuery | 993 | 1.0 |
| aten::to | 200 | 0.9 |
| cudaEventRecordWithFlags | 537 | 0.8 |
| aten::empty_strided | 200 | 0.8 |
| cudaEventRecord | 340 | 0.8 |
| cudaStreamWaitEvent | 467 | 0.7 |
| Buffer Flush | 1 | 0.6 |
| aten::index_select | 4 | 0.5 |
| aten::embedding | 4 | 0.5 |
