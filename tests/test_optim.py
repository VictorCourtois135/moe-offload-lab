"""Optimisations du doc 18 : chacune doit laisser les résultats identiques."""

from __future__ import annotations

import json

import pytest
import torch

from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
from moelab.store import ExpertStore, build_hot_store


def _aligned(n: int) -> torch.Tensor:
    raw = torch.empty(n + 4096, dtype=torch.uint8)
    off = (-raw.data_ptr()) % 4096
    return raw[off : off + n]


def test_direct_io_reads_same_bytes(olmoe_store):
    """D1 : lecture sans cache (ou repli propre si le système de fichiers ne la permet pas)."""
    a, b = ExpertStore(olmoe_store), ExpertStore(olmoe_store)
    b.direct = True
    for prec in a.precisions:
        rb = a.record_bytes(prec)
        for layer, expert in ((0, 0), (2, 7), (1, 3)):
            ref = a.read(layer, expert, prec)
            got = b.read(layer, expert, prec, out=_aligned(rb))
            assert torch.equal(ref, got)
            unaligned = torch.empty(rb + 1, dtype=torch.uint8)[1:]  # adresse non alignée : lecture ordinaire
            assert torch.equal(ref, b.read(layer, expert, prec, out=unaligned))
    assert b.direct_reads + b.direct_fallbacks > 0
    a.close()
    b.close()


def test_hot_store_serves_same_bytes(olmoe_store, tmp_path):
    """D4 : les experts du store chaud sont lus là-bas, et ce sont les mêmes octets."""
    store = ExpertStore(olmoe_store)
    usage = {"store_id": store.store_id, "sessions": 1,
             "counts": {"0:1": 50, "1:2": 40, "2:3": 30, "0:4": 1},
             "hi_counts": {"0:1": 50, "1:2": 0, "2:3": 30}}
    upath = tmp_path / "usage.json"
    upath.write_text(json.dumps(usage), encoding="utf-8")
    budget = store.record_bytes(store.hi) * 2 + store.record_bytes(store.lo)
    info = build_hot_store(olmoe_store, tmp_path / "hot", budget, upath, log=lambda *a: None)
    assert info["experts"] == 3  # 0:1 et 2:3 en haute précision, 1:2 en basse ; 0:4 ne tient plus
    hot = ExpertStore(olmoe_store)
    assert hot.attach_hot(tmp_path / "hot") == 3
    for (layer, expert, prec) in ((0, 1, store.hi), (1, 2, store.lo), (2, 3, store.hi), (0, 4, store.lo)):
        assert torch.equal(store.read(layer, expert, prec), hot.read(layer, expert, prec))
    assert hot.hot_reads == 3
    store.close()
    hot.close()


@pytest.mark.parametrize("opts", [
    {"prefetch_depth": 2},
    {"direct_io": True, "sort_reads": True},
    {"fast_decode": True, "prefetch_adaptive": True, "prefetch_depth": 2, "direct_io": True},
], ids=["c2", "d1+d3", "tout"])
def test_optimisations_keep_generation_identical(olmoe_dir, olmoe_store, opts):
    from transformers import AutoModelForCausalLM

    ids = torch.randint(2, 128, (1, 8), generator=torch.Generator().manual_seed(4))
    ref = AutoModelForCausalLM.from_pretrained(olmoe_dir, dtype=torch.float32).eval()
    cfg = RuntimeConfig(policy=PrecisionPolicy("hi"), fast_device="cpu", fast_budget_bytes=6 * 64 * 1024,
                        host_budget_bytes=5 * 64 * 1024, b_pcie=1.0, b_host=2.0, prefetch=True, **opts)
    model, rt = load_offloaded_model(olmoe_dir, olmoe_store, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
    with torch.no_grad():
        a = ref.generate(ids, max_new_tokens=12, do_sample=False)
        b = model.generate(ids, max_new_tokens=12, do_sample=False)
    assert torch.equal(a, b)
    assert rt.stats.disk_reads > 0
    rt.close()


def test_disk_parallel_measure_runs(olmoe_store):
    from moelab.hw import measure_disk_parallel

    store = ExpertStore(olmoe_store)
    assert measure_disk_parallel(store, store.lo, 2, n=8) > 0
    store.close()


def test_unpacked_int4_dequantizes_identically(olmoe_store):
    """B2 : le record int4 déballé donne exactement les mêmes poids."""
    from moelab.store import materialize, unpack_int4_record, unpacked_layout

    store = ExpertStore(olmoe_store)
    l4 = store.layouts["int4"]
    l8 = unpacked_layout(l4)
    assert l8 is not None and l8["record_bytes"] > l4["record_bytes"]
    blob = store.read(1, 5, "int4")
    a = materialize(l4, blob, torch.float32)
    b = materialize(l8, unpack_int4_record(blob, l4, l8), torch.float32)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    store.close()


def test_unpacked_int4_same_logits(olmoe_dir, olmoe_store):
    ids = torch.randint(2, 128, (2, 16), generator=torch.Generator().manual_seed(6))
    outs = []
    for unpacked in (False, True):
        cfg = RuntimeConfig(policy=PrecisionPolicy("lo"), fast_device="cpu", fast_budget_bytes=1 << 22,
                            host_budget_bytes=1 << 22, fast_int4_unpacked=unpacked)
        model, rt = load_offloaded_model(olmoe_dir, olmoe_store, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
        with torch.no_grad():
            model(ids)  # remplit le niveau rapide
            outs.append(model(ids).logits)
        assert rt.stats.fast_hits > 0
        rt.close()
    assert torch.equal(outs[0], outs[1])


def test_levels_one_matches_levels():
    g = torch.Generator().manual_seed(1)
    pols = [PrecisionPolicy("hobbit", t1=0.3, t_mid=0.7), PrecisionPolicy("hobbit", t1=0.5, t_mid=0.6, t2=0.9),
            PrecisionPolicy("hobbit", t1=0.6, t1_layers={1: 0.2, 3: 0.9})]
    for pol in pols:
        for layer in range(4):
            for _ in range(40):
                w = torch.rand(1, 8, generator=g) ** 2
                lvl, skip = pol.levels(layer, w)
                lvl1, skip1, _ = pol.levels_one(layer, w[0].tolist())
                assert lvl[0].tolist() == lvl1 and skip[0].tolist() == skip1
    pol = PrecisionPolicy("hobbit", t1=0.6, t1_layers={1: 0.0})
    assert pol.levels_one(1, [0.5, 0.3, 0.2])[0] == [0, 2, 2]  # seuil de la couche 1 : seul le premier en fp16
    assert pol.levels_one(0, [0.5, 0.3, 0.2])[0] == [0, 0, 2]


def test_level_to_precision():
    from moelab.policy import level_to_precision

    assert [level_to_precision(v, ["fp16", "int4"]) for v in (0, 1, 2)] == ["fp16", "int4", "int4"]
    assert [level_to_precision(v, ["fp16", "int8", "int4"]) for v in (0, 1, 2)] == ["fp16", "int8", "int4"]


def test_three_precision_store_uses_middle_level(olmoe_dir, tmp_path):
    """E2 : store fp32 / int8 / int4, le niveau intermédiaire va bien chercher l'int8."""
    from moelab import build_store

    store_dir = build_store(olmoe_dir, tmp_path / "s3", ("fp32", "int8", "int4"), log=lambda *a: None)
    pol = PrecisionPolicy("hobbit", t1=0.2, t_mid=0.7)
    cfg = RuntimeConfig(policy=pol, host_budget_bytes=1 << 24, fast_decode=True)
    model, rt = load_offloaded_model(olmoe_dir, store_dir, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
    seen = set()
    orig = rt._host_nowait

    def spy(layer, expert, need):
        seen.add(need)
        return orig(layer, expert, need)

    rt._host_nowait = spy
    with torch.no_grad():
        model.generate(torch.randint(2, 128, (1, 6)), max_new_tokens=6, do_sample=False)
    assert seen == {"fp32", "int8", "int4"}
    rt.close()


def test_miss_downgrade_e3(olmoe_dir, olmoe_store):
    pol = PrecisionPolicy("hobbit", t1=0.9, miss_t1=0.0)
    cfg = RuntimeConfig(policy=pol, fast_device="cpu", fast_budget_bytes=4 * 64 * 1024, host_budget_bytes=1 << 22,
                        fast_decode=True)
    model, rt = load_offloaded_model(olmoe_dir, olmoe_store, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
    with torch.no_grad():
        model.generate(torch.randint(2, 128, (1, 6)), max_new_tokens=8, do_sample=False)
    assert rt.stats.miss_downgrades > 0
    rt.close()


_TRITON_SCRIPT = r"""
import sys, torch
sys.path.insert(0, {root!r})
from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
from moelab.kernels import gemv_int4
from moelab.quant import dequantize, quantize

torch.manual_seed(0)
for O, I in ((64, 128), (96, 192)):
    w = torch.randn(O, I); q, s = quantize(w, 4, 64); x = torch.randn(I)
    ref = dequantize(q, s, 4, torch.float32) @ x
    assert torch.allclose(gemv_int4(q, s, x, 64), ref, atol=1e-4, rtol=1e-4)

ids = torch.randint(2, 128, (1, 6), generator=torch.Generator().manual_seed(7))
outs = []
for tri in (False, True):
    cfg = RuntimeConfig(policy=PrecisionPolicy("lo"), fast_device="cpu", fast_budget_bytes=1 << 22,
                        host_budget_bytes=1 << 22, fast_decode=True, triton_int4=tri)
    model, rt = load_offloaded_model({model!r}, {store!r}, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        logits = []
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        past = out.past_key_values
        for _ in range(3):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
            logits.append(out.logits)
    assert rt._triton == tri and rt.triton_error is None, rt.triton_error
    outs.append(torch.cat(logits))
    rt.close()
assert torch.allclose(outs[0], outs[1], atol=1e-4, rtol=1e-4), (outs[0] - outs[1]).abs().max()
print("OK")
"""


def test_triton_int4_kernel_interpreted(olmoe_dir, olmoe_store):
    """B1 : le noyau Triton, exécuté sur CPU par l'interprète de Triton, donne les mêmes résultats."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip("triton")
    root = str(Path(__file__).resolve().parents[1])
    code = _TRITON_SCRIPT.format(root=root, model=str(olmoe_dir), store=str(olmoe_store))
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-3000:]


_GROUPED_SCRIPT = r"""
import sys, torch
sys.path.insert(0, {root!r})
from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model

ids = torch.randint(2, 128, (1, 6), generator=torch.Generator().manual_seed(8))
# upgrade=False : sinon un fp16 arrivé en cache plus ou moins tôt (lectures en arrière-plan) change la précision
for pol in (PrecisionPolicy("hi"), PrecisionPolicy("lo"), PrecisionPolicy("hobbit", t1=0.5, upgrade=False)):
    outs = []
    for grouped in (False, True):
        cfg = RuntimeConfig(policy=pol, fast_device="cpu", fast_budget_bytes=6 * 64 * 1024, host_budget_bytes=1 << 22,
                            fast_decode=True, grouped_experts=grouped, b_pcie=1.0, b_host=0.5)
        model, rt = load_offloaded_model({model!r}, {store!r}, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
            logits = []
            for _ in range(4):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
                logits.append(out.logits)
        assert rt._grouped == grouped and rt.triton_error is None, rt.triton_error
        assert rt.stats.fills > 0 and rt.stats.fast_hits > 0
        outs.append(torch.cat(logits))
        rt.close()
    assert torch.allclose(outs[0], outs[1], atol=1e-4, rtol=1e-4), (pol.mode, (outs[0] - outs[1]).abs().max())
print("OK")
"""


def test_grouped_experts_interpreted(olmoe_dir, olmoe_store):
    """A4+B1 : tous les experts d'une couche en 2 noyaux groupés, mêmes logits (hi, lo, hobbit mélangé)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip("triton")
    root = str(Path(__file__).resolve().parents[1])
    code = _GROUPED_SCRIPT.format(root=root, model=str(olmoe_dir), store=str(olmoe_store))
    res = subprocess.run([sys.executable, "-c", code], env={**os.environ, "TRITON_INTERPRET": "1"},
                         capture_output=True, text=True, timeout=900)
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-3000:]
