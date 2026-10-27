# Benchmark olmoe-hf — 2026-10-01_1232

Machine : NVIDIA GeForce RTX 2060, 15.8 Gio RAM, Windows-11-10.0.26200-SP0
Store : C:\moe-store\olmoe-i8 (int8, int4) ; budgets VRAM 2.98 Gio, RAM 5.0 Gio ; B_PCIe 12.58 Go/s, B_Host 1.23 Go/s
Prompts : 8 x 64 tokens ; qualité sur 256 tokens ; répétitions : 1

| Réglage | Politique | tok/s | TTFT (s) | accord top-1 | écart perplexité | KL | VRAM hit | fp16 demandé | fp16 utilisé | CPU | disque/token | prefetch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lo | lo | 74.14 | 0.81 | 89.8% | +9.89% | 0.0721 | 97% | 0% | 0% | 0% | 1.6 | 84% | référence |
| hobbit@0.5 | hobbit(t1=0.5) | 21.05 | 0.92 | 95.7% | +3.63% | 0.0205 | 78% | 39% | 90% | 0% | 3.3 | 84% | rejeté |
| hobbit@0.8 | hobbit(t1=0.8) | 22.63 | 0.81 | 96.5% | +0.59% | 0.00661 | 78% | 73% | 99% | 0% | 2.6 | 84% | rejeté |
