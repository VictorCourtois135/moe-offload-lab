# Benchmark olmoe-hf — 2026-10-03_1543

Machine : NVIDIA GeForce RTX 2060, 15.8 Gio RAM, Windows-11-10.0.26200-SP0
Store : C:\moe-store\olmoe-i8 (int8, int4) ; budgets VRAM 3.58 Gio, RAM 5.0 Gio ; B_PCIe 12.58 Go/s, B_Host 1.23 Go/s
Prompts : 8 x 64 tokens ; qualité sur 256 tokens ; répétitions : 1

| Réglage | Politique | tok/s | TTFT (s) | accord top-1 | écart perplexité | KL | VRAM hit | haute préc. demandée | haute préc. utilisée | CPU | disque/token | prefetch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lo | lo | 116.22 | 0.76 | 94.9% | +2.84% | 0.051 | 98% | 0% | 0% | 0% | 1.6 | 83% | référence |
| hobbit | hobbit(t1=0.6) | 29.43 | 0.76 | 94.5% | +1.53% | 0.0146 | 84% | 49% | 95% | 0% | 3.1 | 84% | rejeté |
