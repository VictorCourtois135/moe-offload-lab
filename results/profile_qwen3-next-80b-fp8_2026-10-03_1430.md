# Profil qwen3-next-80b-fp8 — 2026-10-03_1430

Machine : NVIDIA GeForce RTX 2060, k = 10

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 16.745 |
| sync_us | 22.802 |
| floor_loop_gpu_ms | 2.097 |
| floor_loop_wall_ms | 1.985 |
| floor_bmm_gpu_ms | 0.410 |
| floor_bmm_wall_ms | 0.384 |
| vram_bandwidth_gbs | 153.384 |
| dequant_int4_expert_gpu_ms | 0.425 |
| dequant_int4_expert_wall_ms | 0.384 |
| grouped_int4_v1_gpu_ms | 1.158 |
| grouped_int4_v1_gbs | 14.435 |
| grouped_int4_v2_gpu_ms | 0.380 |
| grouped_int4_v2_gbs | 43.934 |
| grouped_int4_v3_gpu_ms | 0.401 |
| grouped_int4_v3_gbs | 41.710 |
| grouped_int4_v4_gpu_ms | 0.375 |
| grouped_int4_v4_gbs | 44.512 |
| grouped_int4_v3_wall_ms | 0.378 |
| grouped_int4_auto_version | 3.000 |
| grouped_int4_graph_wall_ms | 0.117 |
| grouped_int4_v2_kernel_ms | 0.125 |
| grouped_int4_v2_kernel_gbs | 133.808 |
| grouped_int4_v3_kernel_ms | 0.114 |
| grouped_int4_v3_kernel_gbs | 146.380 |
| grouped_int4_v4_kernel_ms | 0.150 |
| grouped_int4_v4_kernel_gbs | 111.204 |
| grouped_int8_kernel_ms | 0.184 |
| grouped_int8_kernel_gbs | 175.837 |
| grouped_fp16_kernel_ms | 0.262 |
| grouped_fp16_kernel_gbs | 240.150 |
| grouped_int4_floor_ms | 0.109 |
| grouped_fp16_gpu_ms | 0.425 |
| grouped_fp16_gbs | 147.902 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token | Mo RAM→VRAM/token | prefetch précision | A5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lo / defaut | async | 146.7 | 6.82 | 136.1 | 10.6 | 2.84 | 84% | 60.7 | 226 | 84% | 0% |
| lo / defaut | sync | 149.3 | 6.70 | 139.4 | 9.9 | 2.90 | 84% | 56.8 | 225 | 84% | 0% |

- lo / defaut : CUDA graphs : 49 segments, 48 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens ; token entier (A5) : 0 tokens, 0 refaits

### Phases, lo / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.561 | 0.094 |
| 2 décisions HOBBIT + regroupement | 0.055 | 0.068 |
| 3 recherche caches + q* + lectures lancées | 0.046 | 0.060 |
| 4 calcul des experts déjà en VRAM | 0.032 | 0.050 |
| 5 copies lancées + branche CPU | 0.063 | 0.132 |
| 6 attente disque/copies + calcul des manquants | 1.783 | 1.654 |
| 7 prefetch | 0.280 | 0.350 |

## torch.profiler (décodage seul, variante defaut)

Lancements de noyaux par token : 1 (+ 97 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaMemcpyAsync | 669 | 89.3 |
| cudaGraphLaunch | 388 | 61.8 |
| aten::copy_ | 669 | 8.7 |
| aten::empty | 473 | 5.5 |
| Activity Buffer Request | 5 | 3.1 |
| aten::_to_copy | 200 | 2.4 |
| aten::record_stream | 338 | 1.4 |
| cudaStreamIsCapturing | 1664 | 1.3 |
| cudaEventRecordWithFlags | 534 | 1.2 |
| aten::view | 212 | 1.2 |
| cudaEventQuery | 991 | 1.1 |
| cudaDeviceSynchronize | 2 | 0.9 |
| aten::empty_strided | 200 | 0.8 |
| cudaStreamSynchronize | 200 | 0.8 |
| aten::to | 200 | 0.8 |
| cudaEventRecord | 338 | 0.7 |
| cudaStreamWaitEvent | 465 | 0.7 |
| aten::reshape | 204 | 0.4 |
| Buffer Flush | 1 | 0.4 |
| aten::index_select | 4 | 0.3 |
