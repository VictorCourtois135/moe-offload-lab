# Benchmark olmoe-hf — 2026-10-02_0925

Machine : NVIDIA GeForce RTX 2060, 15.8 Gio RAM, Windows-11-10.0.26200-SP0
Store : C:\moe-store\olmoe (fp16, int4) ; budgets VRAM 2.98 Gio, RAM 5.0 Gio ; B_PCIe 12.58 Go/s, B_Host 4.11 Go/s
Prompts : 1 x 64 tokens ; qualité sur 256 tokens ; répétitions : 1

| Réglage | Politique | tok/s | TTFT (s) | accord top-1 | écart perplexité | KL | VRAM hit | haute préc. demandée | haute préc. utilisée | CPU | disque/token | prefetch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lo | lo | 24.91 | 14.68 | 89.1% | +10.47% | 0.0696 | 90% | 0% | 0% | 0% | 10.5 | 84% | référence |
