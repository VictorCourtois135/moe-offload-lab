"""
A6 : génération avec la partie dense en CUDA graphs, vérifiée sur CPU grâce à
l'émulateur de graphs (moelab/graphs.py) : la passe rejouée sans Python doit donner
exactement la même génération que la passe normale.
"""

from __future__ import annotations

import pytest
import torch

from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
from moelab.generate import EagerDecoder, generate, pick
from moelab.graphs import GraphCaptureError, GraphDecoder, _EmuGraph, _EmuPool

MB = 2**20


def _load(model_dir, store_dir, **cfg):
    cfg.setdefault("host_budget_bytes", 64 * MB)
    cfg.setdefault("fast_budget_bytes", 0)
    return load_offloaded_model(model_dir, store_dir, RuntimeConfig(**cfg), dense="cpu", dtype=torch.float32,
                                log=lambda *a: None)


def _ids(seed=0, T=8, vocab=128):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, vocab, (1, T), generator=g)


def _run(dec, ids, n):
    logits = [dec.prefill(ids)]
    nxt = pick(logits[-1])
    toks = [int(nxt)]
    for _ in range(n - 1):
        logits.append(dec.step(nxt))
        nxt = pick(logits[-1])
        toks.append(int(nxt))
    return toks, torch.cat(logits)


@pytest.mark.parametrize("opts", [{"fast_decode": True}, {"fast_decode": False}], ids=["fast_decode", "general"])
def test_replayed_graphs_match_eager(olmoe_dir, olmoe_store, opts):
    ids = _ids()
    model, rt = _load(olmoe_dir, olmoe_store, policy=PrecisionPolicy("hi"), fast_device="cpu",
                      fast_budget_bytes=6 * 64 * 1024, b_pcie=1.0, b_host=2.0, prefetch=True, **opts)
    ref_toks, ref_logits = _run(EagerDecoder(model), ids, 10)
    dec = GraphDecoder(model, max_len=64, backend="emulate")
    toks, logits = _run(dec, ids, 10)
    assert dec.failed is None, dec.failed
    assert dec.active and len(dec.segs) == len(rt.store.layers) + 1
    assert toks == ref_toks
    assert torch.allclose(logits, ref_logits, atol=1e-5, rtol=1e-4), (logits - ref_logits).abs().max()
    # 2e requête : même graphs, cache statique remis à zéro
    toks2, logits2 = _run(dec, ids, 6)
    assert toks2 == ref_toks[:6] and dec.captures == 1
    assert torch.allclose(logits2, ref_logits[:6], atol=1e-5, rtol=1e-4)
    rt.close()


def test_generate_with_graphs_equals_transformers(olmoe_dir, olmoe_store):
    from transformers import AutoModelForCausalLM

    ids = _ids(3)
    ref = AutoModelForCausalLM.from_pretrained(olmoe_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        a = ref.generate(ids, max_new_tokens=12, do_sample=False)[0, ids.shape[1]:].tolist()
    model, rt = _load(olmoe_dir, olmoe_store, policy=PrecisionPolicy("hi"))
    dec = GraphDecoder(model, max_len=32, backend="emulate")
    b = generate(model, ids, 12, decoder=dec, eos_token_id=-1)
    assert dec.active
    assert b == a[: len(b)] and len(b) == 12
    assert rt.stats.tokens == ids.shape[1] + 11
    rt.close()


def test_static_cache_limit(olmoe_dir, olmoe_store):
    model, rt = _load(olmoe_dir, olmoe_store, policy=PrecisionPolicy("lo"))
    dec = GraphDecoder(model, max_len=10, backend="emulate")
    out = generate(model, _ids(T=8), 50, decoder=dec, eos_token_id=-1)
    assert len(out) == 2  # 10 - 8 tokens de place
    with pytest.raises(ValueError):
        dec.prefill(_ids(T=12))
    rt.close()


def test_emulator_rejects_value_reads():
    """Une lecture de valeur pendant la capture (ce qui casse un vrai CUDA graph) est détectée."""
    g = _EmuGraph(_EmuPool())
    x = torch.ones(3)
    g.begin()
    try:
        with pytest.raises(GraphCaptureError):
            float(x.sum())
    finally:
        g.end()


def test_emulator_replays_and_restores_state():
    """La capture ne doit rien changer (comme sur GPU) ; le replay refait les opérations sur les mêmes tenseurs."""
    pool = _EmuPool()
    state = torch.zeros(4)
    pos = torch.tensor(0)
    inp = torch.tensor([2.0])
    g = _EmuGraph(pool)
    g.begin()
    y = inp * 3
    state.index_copy_(0, pos.view(1), y)
    pos.add_(1)
    g.end()
    assert state.tolist() == [0, 0, 0, 0] and int(pos) == 0  # capture sans effet
    g.replay()
    inp.fill_(5.0)
    g.replay()
    assert state.tolist() == [6, 15, 0, 0] and int(pos) == 2


def test_failed_capture_falls_back(olmoe_dir, olmoe_store, monkeypatch):
    model, rt = _load(olmoe_dir, olmoe_store, policy=PrecisionPolicy("hi"))
    ref_toks, _ = _run(EagerDecoder(model), _ids(5), 6)
    dec = GraphDecoder(model, max_len=32, backend="emulate")

    def boom(self):
        raise GraphCaptureError("test")

    monkeypatch.setattr(_EmuGraph, "begin", boom)
    toks, _ = _run(dec, _ids(5), 6)
    assert dec.failed and not dec.active and rt.capture_hook is None
    assert toks == ref_toks
    rt.close()


def test_graphs_qwen3next(qnext_dir, tmp_path):
    """Même mécanisme sur Qwen3-Next (attention linéaire Gated DeltaNet + attention complète, expert partagé)."""
    from moelab.store import build_store

    store = build_store(qnext_dir, tmp_path / "store", ("fp32", "int4"), log=lambda *a: None)
    ids = _ids(4)
    model, rt = _load(qnext_dir, store, policy=PrecisionPolicy("hi"))
    ref_toks, ref_logits = _run(EagerDecoder(model), ids, 8)
    dec = GraphDecoder(model, max_len=32, backend="emulate")
    toks, logits = _run(dec, ids, 8)
    assert dec.failed is None and dec.active, dec.failed
    assert toks == ref_toks
    assert torch.allclose(logits, ref_logits, atol=1e-5, rtol=1e-4)
    rt.close()


@pytest.mark.parametrize("arch", ["olmoe", "qwen3next"])
def test_length_buckets(olmoe_dir, olmoe_store, qnext_dir, tmp_path, arch):
    """Paliers de longueur : la génération traverse 8 -> 16 -> 32 positions, mêmes logits que la passe normale."""
    from moelab.store import build_store

    if arch == "olmoe":
        mdir, store = olmoe_dir, olmoe_store
    else:
        mdir, store = qnext_dir, build_store(qnext_dir, tmp_path / "store", ("fp32", "int4"), log=lambda *a: None)
    ids = _ids(6, T=6)
    model, rt = _load(mdir, store, policy=PrecisionPolicy("hi"))
    ref_toks, ref_logits = _run(EagerDecoder(model), ids, 22)
    dec = GraphDecoder(model, max_len=64, backend="emulate", min_bucket=8)
    toks, logits = _run(dec, ids, 22)
    assert dec.failed is None, dec.failed
    assert sorted(dec.sets) == [8, 16, 32] and dec.buckets == [8, 16, 32, 64]
    assert toks == ref_toks
    assert torch.allclose(logits, ref_logits, atol=1e-5, rtol=1e-4), (logits - ref_logits).abs().max()
    rt.close()


def test_decode_forced_quality_matches_forward(olmoe_dir, olmoe_store):
    """Qualité « token par token » (mode decode du benchmark) = passe complète, en haute précision."""
    from moelab.bench import decode_forced_logits, teacher_forced_logits

    model, rt = _load(olmoe_dir, olmoe_store, policy=PrecisionPolicy("hi"))
    chunks = [_ids(7, T=12), _ids(8, T=9)]
    a = teacher_forced_logits(model, chunks)
    b = decode_forced_logits(model, chunks)
    for x, y in zip(a, b):
        assert x.shape == y.shape and torch.allclose(x, y, atol=1e-4, rtol=1e-4), (x - y).abs().max()
    rt.close()


@pytest.mark.parametrize("pack", [True, False])
def test_packed_routing_prefetch(olmoe_dir, olmoe_store, pack):
    """A9 : en graph, la prédiction du prefetch vient du graph (aucun appel au routeur en Python), mêmes prédictions."""
    ids = _ids(9)
    model, rt = _load(olmoe_dir, olmoe_store, policy=PrecisionPolicy("hi"), fast_device="cpu",
                      fast_budget_bytes=6 * 64 * 1024, b_pcie=1.0, b_host=2.0, prefetch=True, prefetch_adaptive=False)
    ref_toks, ref_logits = _run(EagerDecoder(model), ids, 8)
    ref_pred = (rt.stats.pred_total, rt.stats.pred_hits)
    dec = GraphDecoder(model, max_len=64, backend="emulate")
    dec.pack_routing = pack
    _run(dec, ids, 3)  # prefill + échauffement + capture
    calls = []
    hooks = [g.register_forward_hook(lambda *a: calls.append(1)) for g in rt.gates.values()]
    rt.reset_stats()
    toks, logits = _run(dec, ids, 8)
    for h in hooks:
        h.remove()
    assert toks == ref_toks and torch.allclose(logits, ref_logits, atol=1e-5, rtol=1e-4)
    assert rt.stats.pred_total > 0
    # le routeur n'est appelé que par le prefill (une fois par couche) ; en génération, seulement sans A9
    n_layers = len(rt.store.layers)
    assert (len(calls) == n_layers) if pack else (len(calls) > n_layers), len(calls)
    del ref_pred
    rt.close()


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graphs"])
def test_disk_ahead_prefetch_same_tokens_and_reads_ahead(qnext_dir, tmp_path, graph):
    """prefetch_disk_ahead (Qwen3-Next, 02/10) : lectures disque lancées pour L+2, même génération."""
    from moelab.store import build_store

    store = build_store(qnext_dir, tmp_path / "q", ("int4",), log=lambda *a: None)
    ids = _ids(seed=4)
    outs, stats = [], []
    for ahead in (0, 1):
        model, rt = _load(qnext_dir, store, policy=PrecisionPolicy("lo"), fast_device="cpu",
                          fast_budget_bytes=4 * 64 * 1024, host_budget_bytes=6 * 64 * 1024, prefetch_adaptive=False,
                          prefetch_disk_ahead=ahead)
        dec = GraphDecoder(model, max_len=64, backend="emulate") if graph else EagerDecoder(model)
        toks, logits = _run(dec, ids, 8)
        if graph:
            assert dec.failed is None, dec.failed
        outs.append((toks, logits))
        stats.append(rt.stats)
        rt.close()
    assert outs[0][0] == outs[1][0]
    assert torch.allclose(outs[0][1], outs[1][1], atol=1e-5, rtol=1e-4)
    assert stats[0].disk_ahead == 0 and stats[1].disk_ahead > 0, (stats[0].disk_ahead, stats[1].disk_ahead)
