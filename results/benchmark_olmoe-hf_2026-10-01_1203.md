# Benchmark olmoe-hf — 2026-10-01_1203

Machine : NVIDIA GeForce RTX 2060, 15.8 Gio RAM, Windows-11-10.0.26200-SP0
Store : C:\moe-store\olmoe (fp16, int4) ; budgets VRAM 2.98 Gio, RAM 5.0 Gio ; B_PCIe 12.58 Go/s, B_Host 4.11 Go/s
Prompts : 8 x 64 tokens ; qualité sur 256 tokens ; répétitions : 1

| Réglage | Politique | tok/s | TTFT (s) | accord top-1 | écart perplexité | KL | VRAM hit | fp16 demandé | fp16 utilisé | CPU | disque/token | prefetch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lo | lo | 75.10 | 0.79 | 89.5% | +10.38% | 0.07 | 97% | 0% | 0% | 0% | 1.6 | 84% | référence |
| hobbit | hobbit(t1=0.8) | 5.55 | 2.22 | 98.0% | +1.06% | 0.0074 | 59% | 73% | 92% | 0% | 24.6 | 84% | rejeté |
| hobbit_miss | hobbit(t1=0.8, défaut de cache : t1=0.3) | 5.89 | 2.17 | 94.9% | +3.43% | 0.0176 | 59% | 73% | 82% | 0% | 25.8 | 84% | rejeté |
