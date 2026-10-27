# Profil olmoe-hf — 2026-10-01_1116

Machine : NVIDIA GeForce RTX 2060, k = 8

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 17.777 |
| sync_us | 22.852 |
| floor_loop_gpu_ms | 1.740 |
| floor_loop_wall_ms | 1.694 |
| floor_bmm_gpu_ms | 0.355 |
| floor_bmm_wall_ms | 0.359 |
| vram_bandwidth_gbs | 283.523 |
| dequant_int4_expert_gpu_ms | 0.562 |
| dequant_int4_expert_wall_ms | 0.569 |
| grouped_int4_v1_gpu_ms | 0.518 |
| grouped_int4_v1_gbs | 51.612 |
| grouped_int4_v2_gpu_ms | 0.385 |
| grouped_int4_v2_gbs | 69.450 |
| grouped_int4_v3_gpu_ms | 0.392 |
| grouped_int4_v3_gbs | 68.150 |
| grouped_int4_v3_wall_ms | 0.383 |
| grouped_int4_graph_wall_ms | 0.162 |
| grouped_int4_floor_ms | 0.094 |
| grouped_fp16_gpu_ms | 0.379 |
| grouped_fp16_gbs | 265.436 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token |
|---|---|---|---|---|---|---|---|---|
| lo / cg_v2 | async | 21.4 | 46.72 | 19.7 | 1.7 | 1.23 | 100% | 0.0 |
| lo / cg_v2 | sync | 25.4 | 39.37 | 23.6 | 1.8 | 1.48 | 100% | 0.0 |
| lo / cg_v3 | async | 21.3 | 47.00 | 19.6 | 1.6 | 1.23 | 100% | 0.0 |
| lo / cg_v3 | sync | 25.3 | 39.47 | 23.6 | 1.7 | 1.47 | 100% | 0.0 |
| lo / defaut | async | 11.2 | 89.63 | 9.8 | 1.4 | 0.61 | 100% | 0.0 |
| lo / defaut | sync | 14.8 | 67.72 | 13.2 | 1.6 | 0.83 | 100% | 0.0 |
| hobbit@0.8 / cg_v2 | async | 156.7 | 6.38 | 150.2 | 6.5 | 9.39 | 78% | 11.0 |
| hobbit@0.8 / cg_v2 | sync | 250.1 | 4.00 | 247.0 | 3.1 | 15.44 | 78% | 10.8 |
| hobbit@0.8 / cg_v3 | async | 126.4 | 7.91 | 123.4 | 2.9 | 7.71 | 78% | 11.0 |
| hobbit@0.8 / cg_v3 | sync | 155.8 | 6.42 | 152.7 | 3.1 | 9.54 | 78% | 10.8 |
| hobbit@0.8 / defaut | async | 116.3 | 8.60 | 113.1 | 3.2 | 7.07 | 78% | 11.0 |
| hobbit@0.8 / defaut | sync | 129.9 | 7.70 | 126.9 | 3.0 | 7.93 | 78% | 10.8 |

- lo / cg_v2 : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens
- lo / cg_v3 : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens
- lo / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens
- hobbit@0.8 / cg_v2 : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens
- hobbit@0.8 / cg_v3 : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens
- hobbit@0.8 / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens

### Phases, lo / cg_v2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.389 | 0.193 |
| 2 décisions HOBBIT + regroupement | 0.067 | 0.085 |
| 3 recherche caches + q* + lectures lancées | 0.018 | 0.029 |
| 4 calcul des experts déjà en VRAM | 0.007 | 0.016 |
| 5 copies lancées + branche CPU | 0.001 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.749 | 0.780 |
| 7 prefetch | 0.002 | 0.012 |

### Phases, lo / cg_v3 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.390 | 0.192 |
| 2 décisions HOBBIT + regroupement | 0.066 | 0.085 |
| 3 recherche caches + q* + lectures lancées | 0.017 | 0.029 |
| 4 calcul des experts déjà en VRAM | 0.007 | 0.016 |
| 5 copies lancées + branche CPU | 0.001 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.742 | 0.778 |
| 7 prefetch | 0.002 | 0.012 |

### Phases, lo / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.478 | 0.177 |
| 2 décisions HOBBIT + regroupement | 0.035 | 0.049 |
| 3 recherche caches + q* + lectures lancées | 0.015 | 0.026 |
| 4 calcul des experts déjà en VRAM | 0.005 | 0.015 |
| 5 copies lancées + branche CPU | 0.000 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.072 | 0.178 |
| 7 prefetch | 0.001 | 0.010 |

### Phases, hobbit@0.8 / cg_v2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.499 | 0.248 |
| 2 décisions HOBBIT + regroupement | 0.104 | 0.121 |
| 3 recherche caches + q* + lectures lancées | 0.043 | 0.054 |
| 4 calcul des experts déjà en VRAM | 0.043 | 0.056 |
| 5 copies lancées + branche CPU | 1.309 | 1.596 |
| 6 attente disque/copies + calcul des manquants | 3.701 | 9.815 |
| 7 prefetch | 3.682 | 3.164 |

### Phases, hobbit@0.8 / cg_v3 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.477 | 0.250 |
| 2 décisions HOBBIT + regroupement | 0.094 | 0.115 |
| 3 recherche caches + q* + lectures lancées | 0.040 | 0.052 |
| 4 calcul des experts déjà en VRAM | 0.038 | 0.055 |
| 5 copies lancées + branche CPU | 1.014 | 1.618 |
| 6 attente disque/copies + calcul des manquants | 2.934 | 3.980 |
| 7 prefetch | 3.109 | 3.156 |

### Phases, hobbit@0.8 / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.473 | 0.245 |
| 2 décisions HOBBIT + regroupement | 0.065 | 0.073 |
| 3 recherche caches + q* + lectures lancées | 0.035 | 0.049 |
| 4 calcul des experts déjà en VRAM | 0.038 | 0.053 |
| 5 copies lancées + branche CPU | 1.057 | 1.589 |
| 6 attente disque/copies + calcul des manquants | 2.295 | 2.509 |
| 7 prefetch | 3.100 | 3.089 |

## torch.profiler (décodage seul, variante defaut)

Lancements de noyaux par token : 51 (+ 33 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaMemcpyAsync | 136 | 265.9 |
| aten::embedding | 8 | 95.4 |
| cudaGraphLaunch | 264 | 57.7 |
| cudaLaunchKernel | 408 | 22.1 |
| aten::linear | 8 | 12.3 |
| aten::_to_copy | 384 | 12.1 |
| aten::copy_ | 392 | 4.2 |
| aten::empty_strided | 384 | 2.3 |
| aten::cat | 128 | 2.2 |
| Activity Buffer Request | 1 | 1.8 |
| aten::index_select | 8 | 1.7 |
| aten::view | 416 | 1.7 |
| aten::to | 384 | 1.3 |
| aten::reshape | 400 | 0.8 |
| cudaDeviceSynchronize | 2 | 0.6 |
| aten::resize_ | 8 | 0.4 |
| aten::mm | 8 | 0.4 |
| cudaStreamSynchronize | 128 | 0.3 |
| aten::argmax | 8 | 0.2 |
| cudaStreamIsCapturing | 128 | 0.1 |
