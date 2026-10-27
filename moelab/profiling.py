"""
Outils de mesure pour l'optimisation (doc 18).

Deux questions :
  1. Où part le temps d'une couche MoE ? -> PhaseTimer, branché dans MoERuntime.run_layer
     (mode "async" : temps CPU de chaque phase, ce que coûte le Python et le lancement des
     noyaux ; mode "sync" : torch.cuda.synchronize() à chaque frontière, le temps GPU compris).
  2. Quel est le plancher physique ? -> microbenchmarks : coût d'un lancement de noyau, d'une
     synchronisation GPU -> CPU, et temps GPU minimal pour calculer 8 experts.
"""

from __future__ import annotations

import time
from collections import defaultdict

import torch
import torch.nn.functional as F


class PhaseTimer:
    """Chronomètre par phase. `mark(nom)` attribue le temps écoulé depuis la marque précédente à `nom`."""

    def __init__(self, sync: bool = False):
        self.sync = sync and torch.cuda.is_available()
        self.totals: dict[str, float] = defaultdict(float)
        self.calls = 0
        self._last = 0.0

    def start(self) -> None:
        if self.sync:
            torch.cuda.synchronize()
        self.calls += 1
        self._last = time.perf_counter()

    def mark(self, name: str) -> None:
        if self.sync:
            torch.cuda.synchronize()
        t = time.perf_counter()
        self.totals[name] += t - self._last
        self._last = t

    def reset(self) -> None:
        self.totals.clear()
        self.calls = 0

    def per_call_ms(self) -> dict[str, float]:
        n = max(self.calls, 1)
        return {k: 1000 * v / n for k, v in self.totals.items()}


def _cuda_time(fn, repeat: int) -> float:
    """Temps GPU moyen d'un appel (ms), mesuré par événements CUDA après 3 appels d'échauffement."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(repeat):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / repeat


def _wall_time(fn, repeat: int, sync: bool = True) -> float:
    """Temps mur moyen d'un appel (ms)."""
    for _ in range(3):
        fn()
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    if sync:
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / repeat


@torch.no_grad()
def microbenchmarks(hidden: int, inter: int, k: int, dtype=torch.float16, repeat: int = 200) -> dict[str, float]:
    """
    Planchers physiques sur le GPU courant, pour un expert (hidden x inter) et k experts par token.
    Tous les temps en ms.
    """
    dev = "cuda"
    res: dict[str, float] = {}
    tiny = torch.zeros(16, device=dev)

    # 1. Lancement d'un noyau minuscule : le coût fixe de chaque opération PyTorch sur GPU.
    res["launch_us"] = 1000 * _wall_time(lambda: tiny.add_(1), repeat * 5)

    # 2. Aller-retour GPU -> CPU d'un petit tenseur (ce que fait le routage lu en Python).
    res["sync_us"] = 1000 * _wall_time(lambda: tiny.to("cpu"), repeat, sync=False)

    # 3. Plancher de calcul : k experts fp16 pour 1 token, poids déjà en VRAM.
    x = torch.randn(1, hidden, device=dev, dtype=dtype)
    gu = torch.randn(k, 2 * inter, hidden, device=dev, dtype=dtype) * 0.02
    dn = torch.randn(k, hidden, inter, device=dev, dtype=dtype) * 0.02
    w = torch.rand(k, device=dev, dtype=dtype)

    def loop():  # ce que fait le moteur aujourd'hui : un expert après l'autre
        out = torch.zeros_like(x)
        for i in range(k):
            g, u = F.linear(x, gu[i]).chunk(2, dim=-1)
            out += w[i] * F.linear(F.silu(g) * u, dn[i])
        return out

    def batched():  # les k experts en deux produits groupés (bmm)
        h = torch.bmm(gu, x.expand(k, -1).unsqueeze(-1)).squeeze(-1)  # (k, 2I)
        g, u = h.chunk(2, dim=-1)
        y = torch.bmm(dn, (F.silu(g) * u).unsqueeze(-1)).squeeze(-1)  # (k, H)
        return (w.unsqueeze(-1) * y).sum(0, keepdim=True)

    res["floor_loop_gpu_ms"] = _cuda_time(loop, repeat)
    res["floor_loop_wall_ms"] = _wall_time(loop, repeat)
    res["floor_bmm_gpu_ms"] = _cuda_time(batched, repeat)
    res["floor_bmm_wall_ms"] = _wall_time(batched, repeat)
    weight_bytes = (gu.numel() + dn.numel()) * gu.element_size()
    res["vram_bandwidth_gbs"] = weight_bytes / (res["floor_bmm_gpu_ms"] / 1000) / 1e9

    # 4. Déquantification int4 d'un expert (ce que refait materialize() à chaque usage).
    from .quant import dequantize, quantize

    q1, s1 = quantize(gu[0].float(), 4, 64)
    q2, s2 = quantize(dn[0].float(), 4, 64)
    q1, s1, q2, s2 = q1.to(dev), s1.to(dev), q2.to(dev), s2.to(dev)
    res["dequant_int4_expert_gpu_ms"] = _cuda_time(lambda: (dequantize(q1, s1, 4, dtype), dequantize(q2, s2, 4, dtype)), repeat)
    res["dequant_int4_expert_wall_ms"] = _wall_time(lambda: (dequantize(q1, s1, 4, dtype), dequantize(q2, s2, 4, dtype)), repeat)

    # 5. Noyaux groupés (A4+B1) : k experts int4 pour 1 token, v1 (30/09) contre v2 (01/10).
    #    Le plancher est la lecture des octets int4 à la bande passante de la VRAM.
    try:
        from .kernels import HAVE_TRITON, grouped_ffn_float, grouped_ffn_int4
        from .store import encode_record, record_layout, views

        if HAVE_TRITON:
            lay = record_layout("int4", hidden, inter)
            recs = [torch.from_numpy(encode_record(lay, gu[i].float().cpu(), dn[i].float().cpu())).to(dev) for i in range(k)]
            ents = [(views(lay, b), 1.0 / k) for b in recs]
            out = torch.zeros(1, hidden, device=dev, dtype=dtype)
            int4_bytes = sum(b.numel() for b in recs)
            for ver in (1, 2, 3, 4):
                ms = _cuda_time(lambda: grouped_ffn_int4(x, ents, lay["group_sizes"], out, version=ver), repeat)
                res[f"grouped_int4_v{ver}_gpu_ms"] = ms
                res[f"grouped_int4_v{ver}_gbs"] = int4_bytes / (ms / 1000) / 1e9
            # temps mur d'un appel (CPU compris) : lancement Python normal, puis graph (A6 bis)
            res["grouped_int4_v3_wall_ms"] = _wall_time(lambda: grouped_ffn_int4(x, ents, lay["group_sizes"], out, version=3), repeat)
            from .kernels import GroupedGraph

            from .kernels import pick_int4_version

            res["grouped_int4_auto_version"] = float(pick_int4_version(lay, dev, k))
            gg = GroupedGraph({"int4": lay}, x, out, k, version=int(res["grouped_int4_auto_version"]))
            items = {"int4": [(b, 1.0 / k) for b in recs]}
            res["grouped_int4_graph_wall_ms"] = _wall_time(lambda: gg.run(items), repeat)
            # temps GPU des noyaux seuls (rejoués en graph : sans le coût de lancement Python)
            from .kernels import grouped_graph_gpu_ms

            for ver in (2, 3, 4):
                ms = grouped_graph_gpu_ms(lay, x, out, k, ver, items)
                res[f"grouped_int4_v{ver}_kernel_ms"] = ms
                res[f"grouped_int4_v{ver}_kernel_gbs"] = int4_bytes / (ms / 1000) / 1e9
            # int8 (haute précision d'un store int8 + int4) et fp16, mêmes conditions
            lay8 = record_layout("int8", hidden, inter)
            recs8 = [torch.from_numpy(encode_record(lay8, gu[i].float().cpu(), dn[i].float().cpu())).to(dev) for i in range(k)]
            ms = grouped_graph_gpu_ms(lay8, x, out, k, 2, {"int8": [(b, 1.0 / k) for b in recs8]})
            res["grouped_int8_kernel_ms"] = ms
            res["grouped_int8_kernel_gbs"] = sum(b.numel() for b in recs8) / (ms / 1000) / 1e9
            lay16 = record_layout("fp16", hidden, inter)
            recs16 = [torch.from_numpy(encode_record(lay16, gu[i].float().cpu(), dn[i].float().cpu())).to(dev) for i in range(k)]
            ms = grouped_graph_gpu_ms(lay16, x, out, k, 2, {"fp16": [(b, 1.0 / k) for b in recs16]})
            res["grouped_fp16_kernel_ms"] = ms
            res["grouped_fp16_kernel_gbs"] = sum(b.numel() for b in recs16) / (ms / 1000) / 1e9
            res["grouped_int4_floor_ms"] = int4_bytes / (res["vram_bandwidth_gbs"] * 1e9) * 1000
            fents = [({"gate_up": gu[i], "down": dn[i]}, 1.0 / k) for i in range(k)]
            ms = _cuda_time(lambda: grouped_ffn_float(x, fents, out), repeat)
            res["grouped_fp16_gpu_ms"] = ms
            res["grouped_fp16_gbs"] = weight_bytes / (ms / 1000) / 1e9
    except Exception as e:  # noqa: BLE001 — mesure facultative
        res["grouped_error"] = f"{e.__class__.__name__}: {e}"[:200]
    return res
