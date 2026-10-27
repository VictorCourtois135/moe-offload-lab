# Benchmark olmoe-hf — 2026-09-30_0956

Machine : NVIDIA GeForce RTX 2060, 15.8 Gio RAM, Windows-11-10.0.26200-SP0
Store : C:\moe-store\olmoe (fp16, int4) ; budgets VRAM 2.98 Gio, RAM 5.0 Gio ; B_PCIe 12.51 Go/s, B_Host 3.49 Go/s
Prompts : 8 x 64 tokens ; qualité sur 256 tokens ; répétitions : 2

| Réglage | Politique | tok/s | TTFT (s) | accord top-1 | écart perplexité | KL | VRAM hit | fp16 demandé | fp16 utilisé | CPU | disque/token | prefetch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi | hi | 5.00 | 1.44 | 98.8% | +0.12% | 0.00194 | 58% | 100% | 100% | 0% | 24.7 | 84% | référence |
| lo | lo | 5.92 | 0.88 | 89.1% | +10.35% | 0.0698 | 97% | 0% | 0% | 0% | 1.6 | 84% | rejeté |
| hobbit@0.2 | hobbit(t1=0.2) | 4.73 | 1.23 | 88.7% | +10.73% | 0.0648 | 70% | 16% | 58% | 0% | 11.2 | 84% | rejeté |
| hobbit@0.6 | hobbit(t1=0.6) | 4.84 | 1.31 | 93.8% | +4.91% | 0.0305 | 59% | 49% | 83% | 0% | 22.3 | 84% | rejeté |
| hobbit@0.8 | hobbit(t1=0.8) | 4.98 | 1.26 | 97.3% | +0.83% | 0.0066 | 58% | 73% | 92% | 0% | 24.9 | 84% | rejeté |
| hobbit@0.9 | hobbit(t1=0.9) | 5.06 | 1.24 | 99.2% | -0.17% | 0.00187 | 58% | 87% | 96% | 0% | 25.4 | 84% | rejeté |
