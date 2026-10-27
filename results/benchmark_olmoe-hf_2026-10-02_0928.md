# Benchmark olmoe-hf — 2026-10-02_0928

Machine : NVIDIA GeForce RTX 2060, 15.8 Gio RAM, Windows-11-10.0.26200-SP0
Store : C:\moe-store\olmoe-i8 (int8, int4) ; budgets VRAM 2.98 Gio, RAM 5.0 Gio ; B_PCIe 12.58 Go/s, B_Host 1.23 Go/s
Prompts : 8 x 64 tokens ; qualité sur 256 tokens ; répétitions : 1

| Réglage | Politique | tok/s | TTFT (s) | accord top-1 | écart perplexité | KL | VRAM hit | haute préc. demandée | haute préc. utilisée | CPU | disque/token | prefetch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi | hi | 23.82 | 0.75 | 98.0% | +0.38% | 0.00266 | 78% | 100% | 100% | 0% | 2.0 | 84% | référence |
| lo | lo | 86.87 | 0.83 | 89.5% | +10.55% | 0.07 | 97% | 0% | 0% | 0% | 1.6 | 84% | rejeté |
| hobbit@0.8 | hobbit(t1=0.8) | 26.23 | 0.81 | 98.0% | +0.53% | 0.00789 | 78% | 73% | 99% | 0% | 2.6 | 84% | rejeté |
