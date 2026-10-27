# Benchmark olmoe-hf — 2026-09-30_0923

Machine : NVIDIA GeForce RTX 2060, 15.8 Gio RAM, Windows-11-10.0.26200-SP0
Store : C:\moe-store\olmoe (fp16, int4) ; budgets VRAM 2.98 Gio, RAM 1.87 Gio ; B_PCIe 12.51 Go/s, B_Host 3.49 Go/s
Prompts : 8 x 64 tokens ; qualité sur 256 tokens ; répétitions : 2

| Réglage | Politique | tok/s | TTFT (s) | accord top-1 | écart perplexité | KL | VRAM hit | fp16 demandé | fp16 utilisé | CPU | disque/token | prefetch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi | hi | 3.05 | 2.33 | 99.6% | +0.38% | 0.00114 | 60% | 44% | 60% | 0% | 27.7 | 84% | référence |
| lo | lo | 5.64 | 0.90 | 88.3% | +10.37% | 0.0705 | 61% | 40% | 56% | 0% | 26.9 | 84% | rejeté |
| hobbit@0.2 | hobbit(t1=0.2) | 3.98 | 1.37 | 90.6% | +3.26% | 0.0357 | 57% | 44% | 62% | 0% | 29.5 | 84% | rejeté |
| hobbit@0.4 | hobbit(t1=0.4) | 3.81 | 1.41 | 93.0% | +4.68% | 0.0298 | 57% | 47% | 64% | 0% | 30.2 | 84% | rejeté |
| hobbit@0.6 | hobbit(t1=0.6) | 3.93 | 1.44 | 94.5% | +3.43% | 0.0191 | 57% | 49% | 64% | 0% | 30.1 | 84% | rejeté |
| hobbit@0.8 | hobbit(t1=0.8) | 3.90 | 1.65 | 97.3% | +1.47% | 0.00576 | 58% | 49% | 63% | 0% | 29.3 | 84% | rejeté |
