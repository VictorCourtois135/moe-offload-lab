# Benchmark olmoe-hf — 2026-10-02_1500

Machine : NVIDIA GeForce RTX 2060, 15.8 Gio RAM, Windows-11-10.0.26200-SP0
Store : C:\moe-store\olmoe-i8 (int8, int4) ; budgets VRAM 3.38 Gio, RAM 5.0 Gio ; B_PCIe 12.58 Go/s, B_Host 1.23 Go/s
Prompts : 8 x 64 tokens ; qualité sur 256 tokens ; répétitions : 1

| Réglage | Politique | tok/s | TTFT (s) | accord top-1 | écart perplexité | KL | VRAM hit | haute préc. demandée | haute préc. utilisée | CPU | disque/token | prefetch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lo | lo | 117.15 | 0.77 | 93.8% | +3.42% | 0.0495 | 98% | 0% | 0% | 0% | 1.6 | 83% | référence |
| hobbit@0.8 | hobbit(t1=0.8) | 29.95 | 0.76 | 98.4% | -0.61% | 0.00618 | 82% | 73% | 99% | 0% | 2.6 | 84% | rejeté |
