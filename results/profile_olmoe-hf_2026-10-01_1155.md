# Profil olmoe-hf — 2026-10-01_1155

Machine : NVIDIA GeForce RTX 2060, k = 8

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 17.285 |
| sync_us | 20.032 |
| floor_loop_gpu_ms | 1.629 |
| floor_loop_wall_ms | 1.700 |
| floor_bmm_gpu_ms | 0.354 |
| floor_bmm_wall_ms | 0.357 |
| vram_bandwidth_gbs | 284.339 |
| dequant_int4_expert_gpu_ms | 0.563 |
| dequant_int4_expert_wall_ms | 0.565 |
| grouped_int4_v1_gpu_ms | 0.521 |
| grouped_int4_v1_gbs | 51.285 |
| grouped_int4_v2_gpu_ms | 0.400 |
| grouped_int4_v2_gbs | 66.783 |
| grouped_int4_v3_gpu_ms | 0.397 |
| grouped_int4_v3_gbs | 67.360 |
| grouped_int4_v4_gpu_ms | 0.369 |
| grouped_int4_v4_gbs | 72.423 |
| grouped_int4_v3_wall_ms | 0.371 |
| grouped_int4_auto_version | 2.000 |
| grouped_int4_graph_wall_ms | 0.162 |
| grouped_int4_v2_kernel_ms | 0.160 |
| grouped_int4_v2_kernel_gbs | 166.999 |
| grouped_int4_v3_kernel_ms | 0.152 |
| grouped_int4_v3_kernel_gbs | 175.752 |
| grouped_int4_v4_kernel_ms | 0.220 |
| grouped_int4_v4_kernel_gbs | 121.432 |
| grouped_int4_floor_ms | 0.094 |
| grouped_fp16_gpu_ms | 0.382 |
| grouped_fp16_gbs | 263.789 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token | Mo RAM→VRAM/token | prefetch précision |
|---|---|---|---|---|---|---|---|---|---|---|
| lo / sans_paliers | async | 11.2 | 89.41 | 9.8 | 1.4 | 0.61 | 100% | 0.0 | 0 | - |
| lo / sans_paliers | sync | 15.2 | 65.97 | 13.5 | 1.7 | 0.84 | 100% | 0.0 | 0 | - |
| lo / defaut | async | 9.4 | 105.93 | 8.0 | 1.5 | 0.50 | 100% | 0.0 | 0 | - |
| lo / defaut | sync | 13.0 | 76.65 | 11.5 | 1.5 | 0.72 | 100% | 0.0 | 0 | - |
| hobbit@0.8 / sans_paliers | async | 119.2 | 8.39 | 116.0 | 3.3 | 7.25 | 78% | 11.0 | 677 | 84% |
| hobbit@0.8 / sans_paliers | sync | 126.4 | 7.91 | 123.4 | 3.0 | 7.71 | 78% | 10.7 | 680 | 84% |
| hobbit@0.8 / defaut | async | 111.6 | 8.96 | 108.6 | 3.0 | 6.79 | 78% | 11.0 | 677 | 84% |
| hobbit@0.8 / defaut | sync | 125.0 | 8.00 | 122.0 | 3.0 | 7.62 | 78% | 10.8 | 683 | 84% |

- lo / sans_paliers : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [2048] sur [2048], cache statique 2048 tokens
- lo / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens
- hobbit@0.8 / sans_paliers : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [2048] sur [2048], cache statique 2048 tokens
- hobbit@0.8 / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens

### Phases, lo / sans_paliers (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.478 | 0.185 |
| 2 décisions HOBBIT + regroupement | 0.036 | 0.050 |
| 3 recherche caches + q* + lectures lancées | 0.015 | 0.027 |
| 4 calcul des experts déjà en VRAM | 0.005 | 0.016 |
| 5 copies lancées + branche CPU | 0.001 | 0.009 |
| 6 attente disque/copies + calcul des manquants | 0.073 | 0.190 |
| 7 prefetch | 0.001 | 0.011 |

### Phases, lo / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.368 | 0.174 |
| 2 décisions HOBBIT + regroupement | 0.035 | 0.048 |
| 3 recherche caches + q* + lectures lancées | 0.015 | 0.026 |
| 4 calcul des experts déjà en VRAM | 0.006 | 0.015 |
| 5 copies lancées + branche CPU | 0.001 | 0.009 |
| 6 attente disque/copies + calcul des manquants | 0.072 | 0.185 |
| 7 prefetch | 0.001 | 0.011 |

### Phases, hobbit@0.8 / sans_paliers (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.486 | 0.211 |
| 2 décisions HOBBIT + regroupement | 0.062 | 0.071 |
| 3 recherche caches + q* + lectures lancées | 0.037 | 0.049 |
| 4 calcul des experts déjà en VRAM | 0.040 | 0.053 |
| 5 copies lancées + branche CPU | 1.063 | 1.530 |
| 6 attente disque/copies + calcul des manquants | 2.482 | 2.479 |
| 7 prefetch | 3.067 | 2.996 |

### Phases, hobbit@0.8 / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.389 | 0.220 |
| 2 décisions HOBBIT + regroupement | 0.060 | 0.074 |
| 3 recherche caches + q* + lectures lancées | 0.034 | 0.050 |
| 4 calcul des experts déjà en VRAM | 0.038 | 0.054 |
| 5 copies lancées + branche CPU | 1.007 | 1.560 |
| 6 attente disque/copies + calcul des manquants | 2.343 | 2.408 |
| 7 prefetch | 2.908 | 3.044 |

## torch.profiler (décodage seul, variante defaut)

Lancements de noyaux par token : 51 (+ 33 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaMemcpyAsync | 136 | 137.6 |
| aten::embedding | 8 | 43.7 |
| cudaGraphLaunch | 264 | 24.7 |
| cudaLaunchKernel | 408 | 9.0 |
| aten::copy_ | 392 | 4.0 |
| aten::empty_strided | 384 | 2.3 |
| aten::_to_copy | 384 | 2.3 |
| aten::cat | 128 | 2.2 |
| aten::view | 416 | 1.7 |
| aten::index_select | 8 | 1.6 |
| cudaDeviceSynchronize | 2 | 1.6 |
| Activity Buffer Request | 1 | 1.5 |
| aten::to | 384 | 1.2 |
| aten::reshape | 400 | 0.8 |
| aten::mm | 8 | 0.4 |
| cudaStreamSynchronize | 128 | 0.4 |
| aten::resize_ | 8 | 0.3 |
| aten::argmax | 8 | 0.2 |
| cudaStreamIsCapturing | 128 | 0.1 |
| aten::matmul | 8 | 0.1 |
