# Profil olmoe-hf — 2026-10-01_1137

Machine : NVIDIA GeForce RTX 2060, k = 8

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 20.295 |
| sync_us | 27.818 |
| floor_loop_gpu_ms | 1.826 |
| floor_loop_wall_ms | 1.688 |
| floor_bmm_gpu_ms | 0.356 |
| floor_bmm_wall_ms | 0.364 |
| vram_bandwidth_gbs | 282.507 |
| dequant_int4_expert_gpu_ms | 0.563 |
| dequant_int4_expert_wall_ms | 0.568 |
| grouped_int4_v1_gpu_ms | 0.527 |
| grouped_int4_v1_gbs | 50.743 |
| grouped_int4_v2_gpu_ms | 0.372 |
| grouped_int4_v2_gbs | 71.916 |
| grouped_int4_v3_gpu_ms | 0.382 |
| grouped_int4_v3_gbs | 69.957 |
| grouped_int4_v4_gpu_ms | 0.381 |
| grouped_int4_v4_gbs | 70.128 |
| grouped_int4_v3_wall_ms | 0.387 |
| grouped_int4_auto_version | 2.000 |
| grouped_int4_graph_wall_ms | 0.154 |
| grouped_int4_floor_ms | 0.095 |
| grouped_fp16_gpu_ms | 0.415 |
| grouped_fp16_gbs | 242.638 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token |
|---|---|---|---|---|---|---|---|---|
| hobbit@0.8 / defaut | async | 95.8 | 10.44 | 92.9 | 2.9 | 5.81 | 78% | 11.0 |
| hobbit@0.8 / defaut | sync | 110.7 | 9.03 | 108.0 | 2.8 | 6.75 | 78% | 10.6 |

- hobbit@0.8 / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens

### Phases, hobbit@0.8 / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.787 | 0.216 |
| 2 décisions HOBBIT + regroupement | 0.056 | 0.070 |
| 3 recherche caches + q* + lectures lancées | 0.033 | 0.046 |
| 4 calcul des experts déjà en VRAM | 0.030 | 0.054 |
| 5 copies lancées + branche CPU | 0.126 | 1.308 |
| 6 attente disque/copies + calcul des manquants | 2.409 | 2.238 |
| 7 prefetch | 2.359 | 2.491 |

## torch.profiler (décodage seul, variante defaut)

Lancements de noyaux par token : 171 (+ 33 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaMemcpyAsync | 688 | 199.5 |
| cudaGraphLaunch | 264 | 28.0 |
| cudaLaunchKernel | 1368 | 22.9 |
| aten::copy_ | 1304 | 14.6 |
| aten::empty_strided | 864 | 5.8 |
| aten::mm | 128 | 5.3 |
| aten::_to_copy | 864 | 5.1 |
| aten::cat | 248 | 4.9 |
| aten::empty | 440 | 4.9 |
| aten::view | 776 | 3.6 |
| aten::topk | 120 | 3.3 |
| Activity Buffer Request | 5 | 3.3 |
| aten::_softmax | 120 | 2.1 |
| aten::to | 864 | 1.8 |
| aten::reshape | 760 | 1.7 |
| cudaEventQuery | 1349 | 1.3 |
| aten::record_stream | 324 | 1.3 |
| cudaStreamIsCapturing | 1597 | 1.3 |
| cudaStreamSynchronize | 248 | 0.9 |
| cudaEventRecordWithFlags | 479 | 0.8 |
