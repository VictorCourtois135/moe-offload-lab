# Benchmark olmoe-hf — 2026-10-02_1001

Machine : NVIDIA GeForce RTX 2060, 15.8 Gio RAM, Windows-11-10.0.26200-SP0
Store : C:\moe-store\olmoe-i8 (int8, int4) ; budgets VRAM 2.98 Gio, RAM 5.0 Gio ; B_PCIe 12.58 Go/s, B_Host 1.23 Go/s
Prompts : 8 x 64 tokens ; qualité sur 0 tokens ; répétitions : 2

| Réglage | Politique | tok/s | TTFT (s) | accord top-1 | écart perplexité | KL | VRAM hit | haute préc. demandée | haute préc. utilisée | CPU | disque/token | prefetch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hobbit@0.8 | hobbit(t1=0.8) | 26.08 | 0.75 | - | - | - | 78% | 73% | 99% | 0% | 2.6 | 84% | référence |
