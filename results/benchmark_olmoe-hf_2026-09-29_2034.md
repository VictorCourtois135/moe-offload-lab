# Benchmark olmoe-hf — 2026-09-29_2034

Machine : NVIDIA GeForce RTX 2060, 15.8 Gio RAM, Windows-11-10.0.26200-SP0
Store : C:\moe-store\olmoe (fp16, int4) ; budgets VRAM 2.98 Gio, RAM 6.65 Gio ; B_PCIe 12.51 Go/s, B_Host 3.49 Go/s
Prompts : 8 x 64 tokens ; qualité sur 256 tokens ; répétitions : 2

| Réglage | Politique | tok/s | TTFT (s) | accord top-1 | écart perplexité | KL | VRAM hit | fp16 demandé | fp16 utilisé | CPU | disque/token | prefetch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi | hi | 3.68 | 2.26 | 99.6% | -0.15% | 0.00101 | 55% | 100% | 100% | 0% | 36.1 | 84% | référence |
| lo | lo | 6.22 | 0.91 | 88.3% | +10.37% | 0.0698 | 98% | 0% | 0% | 0% | 0.0 | 84% | rejeté |
| hobbit | hobbit(t1=0.6) | 4.10 | 3.43 | 95.3% | +2.71% | 0.0164 | 61% | 49% | 77% | 0% | 15.7 | 84% | rejeté |
| hobbit_edges | hobbit(t1=0.6) + couches protégées [0, 1, 15] | 4.26 | 1.41 | 94.1% | +2.75% | 0.0138 | 61% | 59% | 81% | 0% | 18.0 | 84% | rejeté |
