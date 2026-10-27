# Benchmark olmoe-hf — 2026-10-02_1219

Machine : NVIDIA GeForce RTX 2060, 15.8 Gio RAM, Windows-11-10.0.26200-SP0
Store : C:\moe-store\olmoe-i8 (int8, int4) ; budgets VRAM 3.38 Gio, RAM 5.0 Gio ; B_PCIe 12.58 Go/s, B_Host 1.23 Go/s
Prompts : 8 x 64 tokens ; qualité sur 256 tokens ; répétitions : 1

| Réglage | Politique | tok/s | TTFT (s) | accord top-1 | écart perplexité | KL | VRAM hit | haute préc. demandée | haute préc. utilisée | CPU | disque/token | prefetch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lo | lo | 114.56 | 0.79 | 93.4% | +3.44% | 0.0494 | 98% | 0% | 0% | 0% | 1.6 | 84% | référence |
| hobbit@0.8 | hobbit(t1=0.8) | 18.94 | 1.01 | 98.0% | -0.87% | 0.00605 | 82% | 73% | 99% | 0% | 2.6 | 84% | rejeté |
