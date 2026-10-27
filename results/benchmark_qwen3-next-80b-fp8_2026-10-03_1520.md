# Benchmark qwen3-next-80b-fp8 — 2026-10-03_1520

Machine : NVIDIA GeForce RTX 2060, 15.8 Gio RAM, Windows-11-10.0.26200-SP0
Store : C:\moe-store\qwen3-next (int4) ; budgets VRAM 2.24 Gio, RAM 5.0 Gio ; B_PCIe 12.53 Go/s, B_Host 0.38 Go/s
Prompts : 8 x 64 tokens ; qualité sur 0 tokens ; répétitions : 1

| Réglage | Politique | tok/s | TTFT (s) | accord top-1 | écart perplexité | KL | VRAM hit | haute préc. demandée | haute préc. utilisée | CPU | disque/token | prefetch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lo | lo | 6.81 | 6.87 | - | - | - | 61% | 0% | 0% | 0% | 117.2 | 85% | référence |
