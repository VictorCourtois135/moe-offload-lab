"""
Le test le plus important : en haute précision, le modèle déchargé doit donner les
MÊMES sorties que le modèle transformers d'origine, quel que soit l'endroit où
vivent les experts (principe de Colibrì : le placement ne change que la vitesse).
"""

from __future__ import annotations

import pytest
import torch
from conftest import reference_logits, to_fused_checkpoint

from moelab import PrecisionPolicy, RuntimeConfig, build_store, load_offloaded_model

MB = 2**20


def _load(model_dir, store_dir, **cfg):
    cfg.setdefault("host_budget_bytes", 64 * MB)
    cfg.setdefault("fast_budget_bytes", 0)
    return load_offloaded_model(model_dir, store_dir, RuntimeConfig(**cfg), dense="cpu", dtype=torch.float32,
                                log=lambda *a: None)


def _ids(seed=0, T=24, vocab=128):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, vocab, (2, T), generator=g)


def test_hi_matches_reference_ram_only(olmoe_dir, olmoe_store):
    ids = _ids()
    ref = reference_logits(olmoe_dir, ids)
    model, rt = _load(olmoe_dir, olmoe_store, policy=PrecisionPolicy("hi"))
    with torch.no_grad():
        out = model(ids).logits
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4), (out - ref).abs().max()
    assert rt.stats.used_hi == rt.stats.requests > 0
    rt.close()


def test_placement_does_not_change_output(olmoe_dir, olmoe_store):
    """Niveau rapide minuscule (forte éviction) + q* qui envoie une partie des experts sur la branche CPU."""
    ids = _ids(1)
    ref = reference_logits(olmoe_dir, ids)
    model, rt = _load(olmoe_dir, olmoe_store, policy=PrecisionPolicy("hi"), fast_device="cpu",
                      fast_budget_bytes=3 * 64 * 1024, host_budget_bytes=4 * 64 * 1024,
                      b_pcie=1.0, b_host=3.0, prefetch=True)
    with torch.no_grad():
        out = model(ids).logits
        out2 = model(ids).logits
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
    assert torch.allclose(out2, ref, atol=1e-5, rtol=1e-4)
    s = rt.stats
    assert s.cpu_execs > 0 and s.fills > 0, s
    assert rt.fast.stats.evictions > 0
    rt.close()


@pytest.mark.parametrize("opts", [{}, {"fast_decode": True}, {"fast_decode": True, "prefetch_adaptive": True}],
                         ids=["base", "fast_decode", "fast_decode+adaptive"])
def test_greedy_generation_identical(olmoe_dir, olmoe_store, opts):
    """Chaque optimisation du doc 18 doit laisser la génération identique à transformers."""
    from transformers import AutoModelForCausalLM

    ids = _ids(2, T=8)[:1]
    ref = AutoModelForCausalLM.from_pretrained(olmoe_dir, dtype=torch.float32).eval()
    model, rt = _load(olmoe_dir, olmoe_store, policy=PrecisionPolicy("hi"), fast_device="cpu",
                      fast_budget_bytes=6 * 64 * 1024, b_pcie=1.0, b_host=2.0, prefetch=True, **opts)
    with torch.no_grad():
        a = ref.generate(ids, max_new_tokens=12, do_sample=False)
        b = model.generate(ids, max_new_tokens=12, do_sample=False)
    assert torch.equal(a, b)
    s = rt.stats.as_dict()
    assert s["prefetch_recall"] is not None and s["tokens"] == 8 + 11
    assert s["cpu_execs"] > 0 and s["fills"] > 0  # branche CPU et transferts exercés en génération
    rt.close()


def test_fused_checkpoint_gives_same_store(olmoe_dir, olmoe_store, tmp_path):
    fused = to_fused_checkpoint(olmoe_dir, tmp_path / "fused")
    store2 = build_store(fused, tmp_path / "store", ("fp32", "int4"), log=lambda *a: None)
    for layer in range(3):
        for prec in ("fp32", "int4"):
            a = (olmoe_store / f"layer{layer:03d}.{prec}.bin").read_bytes()
            b = (store2 / f"layer{layer:03d}.{prec}.bin").read_bytes()
            assert a == b


def test_lo_and_hobbit_are_close_but_not_identical(olmoe_dir, olmoe_store):
    ids = _ids(3)
    ref = reference_logits(olmoe_dir, ids)
    res = {}
    for mode in ("lo", "hobbit"):
        model, rt = _load(olmoe_dir, olmoe_store, policy=PrecisionPolicy(mode, t1=0.6))
        with torch.no_grad():
            out = model(ids).logits
        p_ref, p_out = ref.double().log_softmax(-1), out.double().log_softmax(-1)
        kl = (p_ref.exp() * (p_ref - p_out)).sum(-1).mean()
        res[mode] = (kl.item(), rt.stats.as_dict()["hi_share"])
        rt.close()
    assert 0 < res["hobbit"][0] < res["lo"][0] < 0.05, res
    assert res["lo"][1] == 0.0 and 0 < res["hobbit"][1] < 1, res
