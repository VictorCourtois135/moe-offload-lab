"""
A5 (02/10) : le token entier en un graph, décisions sur le GPU (fullgraph.py), et S3
(éviction least-stale, prefetch selon le score). Exécuté sur CPU : émulateur de graphs
+ noyaux groupés par l'interprète de Triton, comme test_kernels.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from moelab.cache import TierCache

_A5_SCRIPT = r"""
import sys, torch
sys.path.insert(0, {root!r})
from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
from moelab.generate import EagerDecoder, pick
from moelab.graphs import GraphDecoder

ids = torch.randint(2, 128, (1, 6), generator=torch.Generator().manual_seed(5))
AT_EVICT = []

def fill_vram(rt, precs):
    for l in range(len(rt.store.layers)):
        for e in range(rt.store.num_experts):
            for p in precs:
                rt.fast.insert((l, e, p), rt.store.read(l, e, p).clone())

def run(dec, rt, n=20, evict_at=None):
    lg = [dec.prefill(ids)]
    nxt = pick(lg[-1])
    for i in range(n):
        if evict_at is not None and i == evict_at:  # force un défaut de cache pendant un token A5
            for key in [k for k in rt.fast.entries if k[0] == 0]:
                rt.fast.evict(key)
            AT_EVICT.append(rt.stats.full_tokens)
        lg.append(dec.step(nxt))
        nxt = pick(lg[-1])
    return torch.cat(lg)

cases = [("lo", PrecisionPolicy("lo"), None), ("hobbit", PrecisionPolicy("hobbit", t1=0.5, upgrade=False), None),
         ("hobbit_upgrade", PrecisionPolicy("hobbit", t1=0.5), None), ("evict", PrecisionPolicy("lo"), 6)]
for name, pol, evict_at in cases:
    outs, stats = [], []
    for full in (False, True):
        cfg = RuntimeConfig(policy=pol, fast_device="cpu", fast_budget_bytes=1 << 26, host_budget_bytes=1 << 24,
                            grouped_experts=True, moe_graphs=True, full_graph=full, prefetch_adaptive=False)
        model, rt = load_offloaded_model({model!r}, {store!r}, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
        precs = rt.store.precisions if name.startswith("hobbit") else [rt.store.lo]
        fill_vram(rt, precs if name != "hobbit_upgrade" else [rt.store.hi])
        with torch.no_grad():
            dec = GraphDecoder(model, max_len=32, backend="emulate")
            outs.append(run(dec, rt, evict_at=evict_at))
        assert rt.triton_error is None, rt.triton_error
        assert dec.full_failed is None, dec.full_failed
        stats.append(rt.stats)
        rt.close()
    d = (outs[0] - outs[1]).abs().max().item()
    assert torch.allclose(outs[0], outs[1], atol=1e-4, rtol=1e-4), (name, d)
    assert stats[1].full_tokens > 0, (name, stats[1])
    assert stats[0].full_tokens == 0
    if evict_at is not None:
        assert stats[1].full_fallbacks > 0, stats[1]
        # après le défaut, le chemin A5 doit reprendre (le compteur de défauts est remis à zéro à chaque token)
        assert stats[1].full_tokens > AT_EVICT[-1], (AT_EVICT, stats[1])
    # mêmes compteurs de cache que le chemin normal (hits, haute précision utilisée)
    assert stats[0].requests == stats[1].requests and stats[0].used_hi == stats[1].used_hi, (name, stats)
    print(name, "A5", stats[1].full_tokens, "refaits", stats[1].full_fallbacks, "écart", d)
print("OK")
"""


def _run(code):
    res = subprocess.run([sys.executable, "-c", code], env={**os.environ, "TRITON_INTERPRET": "1"},
                         capture_output=True, text=True, timeout=900)
    assert res.returncode == 0 and "OK" in res.stdout, (res.stdout[-2000:], res.stderr[-4000:])
    return res.stdout


def test_full_graph_matches_normal_path(olmoe_dir, tmp_path):
    """Token entier en un graph = chemin normal (lo, hobbit, upgrade), y compris un défaut de cache refait."""
    pytest.importorskip("triton")
    from moelab.store import build_store

    store = build_store(olmoe_dir, tmp_path / "s8", ("int8", "int4"), log=lambda *a: None)
    root = str(Path(__file__).resolve().parents[1])
    print(_run(_A5_SCRIPT.format(root=root, model=str(olmoe_dir), store=str(store))))


def test_full_graph_qwen3next_restores_recurrent_state(qnext_dir, tmp_path):
    """Qwen3-Next (attention linéaire) : un token A5 raté restaure les états récurrents avant d'être refait."""
    pytest.importorskip("triton")
    from moelab.store import build_store

    store = build_store(qnext_dir, tmp_path / "q8", ("int8", "int4"), log=lambda *a: None)
    root = str(Path(__file__).resolve().parents[1])
    code = _A5_SCRIPT.format(root=root, model=str(qnext_dir), store=str(store))
    code = code.replace('cases = [("lo"', 'cases = [("evict", PrecisionPolicy("lo"), 6), ("lo"')
    print(_run(code))


# -- S3 ----------------------------------------------------------------------
def test_stale_eviction_protects_current_pass():
    rank = {"a": 0}.get
    for policy, expect_victim in (("lrfu", (0, 1, "a")), ("stale", (0, 0, "a"))):
        c = TierCache("t", "cpu", 3 * 64, rank, eviction=policy)
        blob = lambda: torch.zeros(64, dtype=torch.uint8)  # noqa: E731
        old = c.insert((0, 0, "a"), blob())
        for _ in range(5):  # ancien mais très utilisé : gros score LRFU
            c.touch(old)
        c.tick()
        c.insert((0, 1, "a"), blob())  # passe courante (vient d'être préchargé)
        c.insert((0, 2, "a"), blob())
        c.touch(c.entries[(0, 2, "a")])
        c.insert((0, 3, "a"), blob())  # plein : il faut évincer
        gone = {(0, 0, "a"), (0, 1, "a"), (0, 2, "a")} - set(c.entries)
        assert gone == {expect_victim}, (policy, gone)


_S3_SCRIPT = r"""
import sys, torch
sys.path.insert(0, {root!r})
from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
from moelab.generate import EagerDecoder, pick
from moelab.graphs import GraphDecoder

ids = torch.randint(2, 128, (1, 6), generator=torch.Generator().manual_seed(8))
def run(dec, n=8):
    lg = [dec.prefill(ids)]
    nxt = pick(lg[-1])
    for _ in range(n):
        lg.append(dec.step(nxt))
        nxt = pick(lg[-1])
    return torch.cat(lg)

res = {{}}
for mass in (None, 0.2, 0.95):
    for graph in (False, True):
        cfg = RuntimeConfig(policy=PrecisionPolicy("hobbit", t1=0.5, upgrade=False), fast_device="cpu",
                            fast_budget_bytes=8 * 64 * 1024, host_budget_bytes=1 << 22, grouped_experts=True,
                            prefetch_adaptive=False, prefetch_mass=mass, cache_eviction="stale")
        model, rt = load_offloaded_model({model!r}, {store!r}, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
        with torch.no_grad():
            out = run(GraphDecoder(model, max_len=32, backend="emulate") if graph else EagerDecoder(model))
        res[(mass, graph)] = (out, rt.stats.pred_total)
        rt.close()
base = res[(None, False)][0]
for key, (out, pt) in res.items():
    assert torch.allclose(out, base, atol=1e-4, rtol=1e-4), key  # le prefetch ne change jamais le résultat
    assert pt > 0, key
# masse faible : moins d'experts prédits ; masse forte : plus que les k du routeur
assert res[(0.2, True)][1] < res[(None, True)][1] < res[(0.95, True)][1], {{k: v[1] for k, v in res.items()}}
print("OK")
"""


def test_score_based_prefetch_and_stale_eviction(olmoe_dir, olmoe_store):
    pytest.importorskip("triton")
    root = str(Path(__file__).resolve().parents[1])
    _run(_S3_SCRIPT.format(root=root, model=str(olmoe_dir), store=str(olmoe_store)))
