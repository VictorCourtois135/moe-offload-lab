# Profil olmoe-hf — 2026-10-02_0952

Machine : NVIDIA GeForce RTX 2060, k = 8

## Planchers physiques

| Mesure | Valeur |
|---|---|
| launch_us | 18.279 |
| sync_us | 23.138 |
| floor_loop_gpu_ms | 1.742 |
| floor_loop_wall_ms | 1.749 |
| floor_bmm_gpu_ms | 0.357 |
| floor_bmm_wall_ms | 0.360 |
| vram_bandwidth_gbs | 282.201 |
| dequant_int4_expert_gpu_ms | 0.572 |
| dequant_int4_expert_wall_ms | 0.567 |
| grouped_int4_v1_gpu_ms | 0.529 |
| grouped_int4_v1_gbs | 50.590 |
| grouped_int4_v2_gpu_ms | 0.388 |
| grouped_int4_v2_gbs | 68.963 |
| grouped_int4_v3_gpu_ms | 0.389 |
| grouped_int4_v3_gbs | 68.783 |
| grouped_int4_v4_gpu_ms | 0.393 |
| grouped_int4_v4_gbs | 67.971 |
| grouped_int4_v3_wall_ms | 0.396 |
| grouped_int4_auto_version | 3.000 |
| grouped_int4_graph_wall_ms | 0.153 |
| grouped_int4_v2_kernel_ms | 0.156 |
| grouped_int4_v2_kernel_gbs | 171.066 |
| grouped_int4_v3_kernel_ms | 0.151 |
| grouped_int4_v3_kernel_gbs | 176.683 |
| grouped_int4_v4_kernel_ms | 0.217 |
| grouped_int4_v4_kernel_gbs | 123.256 |
| grouped_int8_kernel_ms | 0.237 |
| grouped_int8_kernel_gbs | 218.743 |
| grouped_fp16_kernel_ms | 0.403 |
| grouped_fp16_kernel_gbs | 249.584 |
| grouped_int4_floor_ms | 0.095 |
| grouped_fp16_gpu_ms | 0.377 |
| grouped_fp16_gbs | 266.730 |

## Décodage

| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token | Mo RAM→VRAM/token | prefetch précision |
|---|---|---|---|---|---|---|---|---|---|---|
| lo / defaut | async | 8.4 | 118.86 | 6.8 | 1.6 | 0.42 | 100% | 0.0 | 0 | - |
| lo / defaut | sync | 10.9 | 92.00 | 8.9 | 2.0 | 0.56 | 100% | 0.0 | 0 | - |
| hobbit@0.8 / defaut | async | 21.4 | 46.66 | 19.4 | 2.0 | 1.21 | 94% | 0.0 | 116 | 84% |
| hobbit@0.8 / defaut | sync | 27.8 | 35.98 | 25.6 | 2.2 | 1.60 | 95% | 0.0 | 112 | 84% |

- lo / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens
- hobbit@0.8 / defaut : CUDA graphs : 17 segments, 16 couches MoE entre eux, paliers capturés [128] sur [128, 256, 512, 1024, 2048], cache statique 2048 tokens

### Phases, lo / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.301 | 0.067 |
| 2 décisions HOBBIT + regroupement | 0.032 | 0.050 |
| 3 recherche caches + q* + lectures lancées | 0.015 | 0.028 |
| 4 calcul des experts déjà en VRAM | 0.005 | 0.016 |
| 5 copies lancées + branche CPU | 0.000 | 0.009 |
| 6 attente disque/copies + calcul des manquants | 0.069 | 0.181 |
| 7 prefetch | 0.001 | 0.012 |

### Phases, hobbit@0.8 / defaut (ms par couche)

| Phase | async | sync |
|---|---|---|
| 1 routage lu sur CPU (synchro) | 0.297 | 0.077 |
| 2 décisions HOBBIT + regroupement | 0.051 | 0.065 |
| 3 recherche caches + q* + lectures lancées | 0.024 | 0.036 |
| 4 calcul des experts déjà en VRAM | 0.015 | 0.025 |
| 5 copies lancées + branche CPU | 0.128 | 0.263 |
| 6 attente disque/copies + calcul des manquants | 0.203 | 0.382 |
| 7 prefetch | 0.492 | 0.566 |

## torch.profiler (décodage seul, variante defaut)

Lancements de noyaux par token : 2 (+ 33 CUDA graphs rejoués)

| Opération | Appels | CPU propre (ms) |
|---|---|---|
| cudaGraphLaunch | 264 | 26.4 |
| cudaMemcpyAsync | 136 | 24.8 |
| Activity Buffer Request | 1 | 2.0 |
| cudaLaunchKernel | 16 | 1.8 |
| aten::copy_ | 136 | 1.7 |
| aten::_to_copy | 128 | 1.2 |
| aten::index_select | 8 | 0.9 |
| aten::view | 168 | 0.8 |
| aten::embedding | 8 | 0.7 |
| aten::to | 128 | 0.6 |
| cudaStreamSynchronize | 128 | 0.3 |
| aten::reshape | 152 | 0.3 |
| aten::resize_ | 8 | 0.3 |
| aten::empty_strided | 128 | 0.3 |
| aten::argmax | 8 | 0.2 |
| cuLaunchKernelEx | 8 | 0.2 |
| cudaDeviceSynchronize | 2 | 0.2 |
| aten::empty | 16 | 0.2 |
| cudaStreamIsCapturing | 128 | 0.1 |
| aten::resolve_conj | 128 | 0.1 |
