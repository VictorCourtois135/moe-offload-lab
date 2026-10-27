# Benchmark olmoe-hf — 2026-10-02_1115

Machine : NVIDIA GeForce RTX 2060, 15.8 Gio RAM, Windows-11-10.0.26200-SP0
Store : C:\moe-store\olmoe-i8 (int8, int4) ; budgets VRAM 3.38 Gio, RAM 5.0 Gio ; B_PCIe 12.58 Go/s, B_Host 1.23 Go/s
Prompts : 8 x 64 tokens ; qualité sur 256 tokens ; répétitions : 1

| Réglage | Politique | tok/s | TTFT (s) | accord top-1 | écart perplexité | KL | VRAM hit | haute préc. demandée | haute préc. utilisée | CPU | disque/token | prefetch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lo | lo | 112.71 | 0.74 | 93.4% | +4.97% | 0.05 | 98% | 0% | 0% | 0% | 1.6 | 83% | référence |
| hobbit@0.8 | hobbit(t1=0.8) | 18.80 | 1.00 | 96.9% | +1.10% | 0.00652 | 82% | 73% | 99% | 0% | 2.6 | 84% | rejeté |
