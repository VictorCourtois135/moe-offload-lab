"""
Noyaux groupés v1 et v2 (int4 et flottants) contre le calcul de référence, exécutés
par l'interprète de Triton sur CPU : chemins masqués (dimensions qui ne tombent pas
juste, groupes qui ne sont pas des puissances de 2) et chemin sans masques (EVEN).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = r"""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, {root!r})
from moelab.store import record_layout, encode_record, views, materialize
from moelab.kernels import grouped_ffn_int4, grouped_ffn_float, grouped_ffn_int8, _even

def ref(x, recs, lay, ws):
    out = torch.zeros(1, x.shape[1])
    for b, w in zip(recs, ws):
        gu, dn = materialize(lay, b, torch.float32)
        g, u = F.linear(x, gu).chunk(2, -1)
        out += w * F.linear(F.silu(g) * u, dn)
    return out

assert _even((512, 64), (256, 256))
for H, I in ((64, 32), (96, 48), (512, 256), (256, 128)):
    for prec in ("int4", "int8", "fp16", "fp32"):
        torch.manual_seed(H + I)
        lay = record_layout(prec, H, I)
        recs = [torch.from_numpy(encode_record(lay, torch.randn(2 * I, H) * 0.1, torch.randn(H, I) * 0.1)) for _ in range(3)]
        ws = [0.5, 0.3, 0.2]
        x = torch.randn(1, H)
        r = ref(x, recs, lay, ws)
        ents = [(views(lay, b), w) for b, w in zip(recs, ws)]
        for ver in ((1, 2, 3, 4) if prec == "int4" else (2,)):
            out = torch.full((1, H), 0.5)  # le noyau AJOUTE à out
            if prec == "int4":
                grouped_ffn_int4(x, ents, lay["group_sizes"], out, version=ver)
            elif prec == "int8":
                grouped_ffn_int8(x, ents, lay, out)
            else:
                grouped_ffn_float(x, ents, out)
            err = ((out - 0.5 - r).abs().max() / r.abs().max()).item()
            assert err < 1e-4, (H, I, prec, ver, err)
print("OK")
"""


def test_grouped_kernels_v1_v2_interpreted():
    pytest.importorskip("triton")
    root = str(Path(__file__).resolve().parents[1])
    res = subprocess.run([sys.executable, "-c", _SCRIPT.format(root=root)], env={**os.environ, "TRITON_INTERPRET": "1"},
                         capture_output=True, text=True, timeout=900)
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-3000:]


_GG_SCRIPT = r"""
import sys, torch
sys.path.insert(0, {root!r})
from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
from moelab.generate import EagerDecoder, pick
from moelab.graphs import GraphDecoder

ids = torch.randint(2, 128, (1, 6), generator=torch.Generator().manual_seed(11))

def run(dec, n=6):
    lg = [dec.prefill(ids)]
    nxt = pick(lg[-1])
    for _ in range(n):
        lg.append(dec.step(nxt))
        nxt = pick(lg[-1])
    return torch.cat(lg)

for pol in (PrecisionPolicy("hi"), PrecisionPolicy("lo"), PrecisionPolicy("hobbit", t1=0.5, upgrade=False)):
    outs = []
    for graph in (False, True):
        # VRAM minuscule + q* : défauts de cache, transferts ET branche CPU, table complétée par des poids 0
        cfg = RuntimeConfig(policy=pol, fast_device="cpu", fast_budget_bytes=6 * 64 * 1024, host_budget_bytes=1 << 22,
                            grouped_experts=True, b_pcie=1.0, b_host=3.0, moe_graphs=True)
        model, rt = load_offloaded_model({model!r}, {store!r}, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
        with torch.no_grad():
            outs.append(run(GraphDecoder(model, max_len=32, backend="emulate") if graph else EagerDecoder(model)))
        assert rt.triton_error is None, rt.triton_error
        runs = sum(g.runs for g in rt._ggraphs.values() if g is not None)
        assert (runs > 0) == graph, (pol.mode, graph, runs)
        if graph:
            assert rt.stats.cpu_execs > 0 and rt.stats.fills > 0, rt.stats
        rt.close()
    assert torch.allclose(outs[0], outs[1], atol=1e-4, rtol=1e-4), (pol.mode, (outs[0] - outs[1]).abs().max())
print("OK")
"""


def test_grouped_graph_path_interpreted(olmoe_dir, olmoe_store):
    """A6 bis : calcul groupé par table fixe (poids 0 pour les lignes vides) = chemin normal, hi / lo / hobbit, avec q*."""
    pytest.importorskip("triton")
    root = str(Path(__file__).resolve().parents[1])
    code = _GG_SCRIPT.format(root=root, model=str(olmoe_dir), store=str(olmoe_store))
    res = subprocess.run([sys.executable, "-c", code], env={**os.environ, "TRITON_INTERPRET": "1"},
                         capture_output=True, text=True, timeout=900)
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-3000:]


def test_int8_int4_store_graph_path_interpreted(olmoe_dir, tmp_path):
    """Store int8 + int4 (int8 = haute précision) : calcul groupé en graph = chemin normal, en hobbit."""
    pytest.importorskip("triton")
    from moelab.store import build_store

    store = build_store(olmoe_dir, tmp_path / "store8", ("int8", "int4"), log=lambda *a: None)
    root = str(Path(__file__).resolve().parents[1])
    code = _GG_SCRIPT.format(root=root, model=str(olmoe_dir), store=str(store))
    res = subprocess.run([sys.executable, "-c", code], env={**os.environ, "TRITON_INTERPRET": "1"},
                         capture_output=True, text=True, timeout=900)
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-3000:]
