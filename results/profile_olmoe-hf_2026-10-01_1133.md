# Profil olmoe-hf — 2026-10-01_1133

Machine : NVIDIA GeForce RTX 2060, k = 8

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 19.226 |
| sync_us | 20.694 |
| floor_loop_gpu_ms | 1.781 |
| floor_loop_wall_ms | 1.742 |
| floor_bmm_gpu_ms | 0.348 |
| floor_bmm_wall_ms | 0.367 |
| vram_bandwidth_gbs | 288.877 |
| dequant_int4_expert_gpu_ms | 0.563 |
| dequant_int4_expert_wall_ms | 0.565 |
| grouped_int4_v1_gpu_ms | 0.528 |
| grouped_int4_v1_gbs | 50.664 |
| grouped_int4_v2_gpu_ms | 0.376 |
| grouped_int4_v2_gbs | 71.142 |
| grouped_int4_v3_gpu_ms | 0.372 |
| grouped_int4_v3_gbs | 71.925 |
| grouped_int4_v4_gpu_ms | 0.386 |
| grouped_int4_v4_gbs | 69.346 |
| grouped_int4_v3_wall_ms | 0.370 |
| grouped_int4_auto_version | 2.000 |
| grouped_int4_graph_wall_ms | 0.153 |
| grouped_int4_floor_ms | 0.093 |
| grouped_fp16_gpu_ms | 0.385 |
| grouped_fp16_gbs | 261.413 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token |
|---|---|---|---|---|---|---|---|---|
| lo / g2 | async | 11.3 | 88.23 | 9.9 | 1.4 | 0.62 | 100% | 0.0 |
| lo / g2 | sync | 15.0 | 66.77 | 13.4 | 1.6 | 0.84 | 100% | 0.0 |
| lo / g4 | async | 12.1 | 82.89 | 10.6 | 1.4 | 0.66 | 100% | 0.0 |
| lo / g4 | sync | 15.6 | 63.99 | 14.1 | 1.6 | 0.88 | 100% | 0.0 |
| lo / defaut | async | 11.2 | 89.11 | 9.8 | 1.4 | 0.62 | 100% | 0.0 |
| lo / defaut | sync | 15.1 | 66.39 | 13.5 | 1.6 | 0.84 | 100% | 0.0 |
| hobbit@0.8 / g2 | async | 126.0 | 7.94 | 122.1 | 3.9 | 7.63 | 78% | 11.1 |
| hobbit@0.8 / g2 | sync | 246.7 | 4.05 | 241.4 | 5.3 | 15.09 | 78% | 10.7 |
| hobbit@0.8 / g4 | async | 396.9 | 2.52 | 393.9 | 3.0 | 24.62 | 78% | 11.3 |
| hobbit@0.8 / g4 | sync | 126.3 | 7.92 | 123.3 | 3.0 | 7.71 | 78% | 10.9 |
| hobbit@0.8 / defaut | async | 117.8 | 8.49 | 114.0 | 3.8 | 7.12 | 78% | 11.2 |
| hobbit@0.8 / defaut | sync | 125.4 | 7.97 | 122.4 | 3.0 | 7.65 | 78% | 10.7 |

- lo / g2 : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens
- lo / g4 : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens
- lo / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens
- hobbit@0.8 / g2 : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens
- hobbit@0.8 / g4 : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens
- hobbit@0.8 / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, cache statique 2048 tokens

### Phases, lo / g2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.485 | 0.183 |
| 2 décisions HOBBIT + regroupement | 0.036 | 0.049 |
| 3 recherche caches + q* + lectures lancées | 0.014 | 0.025 |
| 4 calcul des experts déjà en VRAM | 0.005 | 0.015 |
| 5 copies lancées + branche CPU | 0.000 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.076 | 0.185 |
| 7 prefetch | 0.001 | 0.011 |

### Phases, lo / g4 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.534 | 0.178 |
| 2 décisions HOBBIT + regroupement | 0.034 | 0.048 |
| 3 recherche caches + q* + lectures lancées | 0.015 | 0.025 |
| 4 calcul des experts déjà en VRAM | 0.005 | 0.015 |
| 5 copies lancées + branche CPU | 0.000 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.072 | 0.231 |
| 7 prefetch | 0.001 | 0.010 |

### Phases, lo / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.486 | 0.184 |
| 2 décisions HOBBIT + regroupement | 0.035 | 0.049 |
| 3 recherche caches + q* + lectures lancées | 0.015 | 0.026 |
| 4 calcul des experts déjà en VRAM | 0.005 | 0.015 |
| 5 copies lancées + branche CPU | 0.000 | 0.008 |
| 6 attente disque/copies + calcul des manquants | 0.072 | 0.188 |
| 7 prefetch | 0.001 | 0.011 |

### Phases, hobbit@0.8 / g2 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.504 | 0.234 |
| 2 décisions HOBBIT + regroupement | 0.062 | 0.075 |
| 3 recherche caches + q* + lectures lancées | 0.035 | 0.051 |
| 4 calcul des experts déjà en VRAM | 0.037 | 0.055 |
| 5 copies lancées + branche CPU | 1.016 | 1.605 |
| 6 attente disque/copies + calcul des manquants | 2.983 | 9.411 |
| 7 prefetch | 2.986 | 3.271 |

### Phases, hobbit@0.8 / g4 (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.696 | 0.223 |
| 2 décisions HOBBIT + regroupement | 0.060 | 0.071 |
| 3 recherche caches + q* + lectures lancées | 0.034 | 0.048 |
| 4 calcul des experts déjà en VRAM | 0.036 | 0.052 |
| 5 copies lancées + branche CPU | 1.023 | 1.514 |
| 6 attente disque/copies + calcul des manquants | 19.610 | 2.518 |
| 7 prefetch | 3.153 | 2.964 |

### Phases, hobbit@0.8 / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.489 | 0.228 |
| 2 décisions HOBBIT + regroupement | 0.060 | 0.072 |
| 3 recherche caches + q* + lectures lancées | 0.035 | 0.049 |
| 4 calcul des experts déjà en VRAM | 0.037 | 0.053 |
| 5 copies lancées + branche CPU | 1.016 | 1.522 |
| 6 attente disque/copies + calcul des manquants | 2.501 | 2.404 |
| 7 prefetch | 2.977 | 3.001 |

## torch.profiler (décodage seul, variante defaut)

Lancements de noyaux par token : 51 (+ 33 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaMemcpyAsync | 136 | 550.7 |
| cudaGraphLaunch | 264 | 234.4 |
| aten::_to_copy | 384 | 131.1 |
| aten::to | 384 | 105.4 |
| cudaLaunchKernel | 408 | 40.5 |
| aten::embedding | 8 | 23.4 |
| aten::copy_ | 392 | 4.6 |
| aten::empty_strided | 384 | 2.6 |
| aten::cat | 128 | 2.5 |
| aten::view | 416 | 1.9 |
| Activity Buffer Request | 1 | 1.7 |
| aten::index_select | 8 | 1.6 |
| aten::reshape | 400 | 0.9 |
| cudaDeviceSynchronize | 2 | 0.6 |
| cudaStreamSynchronize | 128 | 0.6 |
| aten::mm | 8 | 0.5 |
| aten::resize_ | 8 | 0.3 |
| aten::argmax | 8 | 0.2 |
| cudaStreamIsCapturing | 128 | 0.1 |
| aten::matmul | 8 | 0.1 |
