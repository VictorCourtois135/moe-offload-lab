"""
Prefill (03/10) : P1 reprise du cache en conversation (exacte), P2 à P4 (mêmes logits),
P5 saut des experts absents et peu demandés (compté, et rien quand le seuil vaut 0).
"""

from __future__ import annotations

import pytest
import torch

from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
from moelab.generate import generate, pick
from moelab.graphs import GraphDecoder

MB = 2**20


def _load(model_dir, store_dir, **cfg):
    cfg.setdefault("host_budget_bytes", 64 * MB)
    cfg.setdefault("fast_budget_bytes", 0)
    return load_offloaded_model(model_dir, store_dir, RuntimeConfig(**cfg), dense="cpu", dtype=torch.float32,
                                log=lambda *a: None)


def _ids(seed=0, T=8, vocab=128):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, vocab, (1, T), generator=g)


@pytest.fixture(params=["olmoe", "qwen3next"])
def arch_model(request, olmoe_dir, olmoe_store, qnext_dir, tmp_path):
    from moelab.store import build_store

    if request.param == "olmoe":
        return request.param, olmoe_dir, olmoe_store
    return request.param, qnext_dir, build_store(qnext_dir, tmp_path / "store", ("fp32", "int4"), log=lambda *a: None)


def _conversation(dec, turns, reuse):
    """Comme chat.py : chaque message = tout l'historique + le nouveau ; logits du prefill + 4 tokens."""
    hist, out = torch.zeros(1, 0, dtype=torch.long), []
    for new in turns:
        ids = torch.cat([hist, new], 1)
        lg = [dec.prefill(ids, reuse=reuse)]
        out.append((dec.reused, ids.shape[1]))
        toks = []
        nxt = pick(lg[-1])
        for _ in range(4):
            toks.append(int(nxt))
            lg.append(dec.step(nxt, int(nxt)))
            nxt = pick(lg[-1])
        hist = torch.cat([ids, torch.tensor([toks])], 1)
        out.append(torch.cat(lg))
    return out


def test_p1_reuse_is_exact(arch_model):
    """P1 : reprendre la conversation du cache donne les mêmes logits que tout recalculer."""
    arch, mdir, store = arch_model
    turns = [_ids(1, T=7), _ids(2, T=5), _ids(3, T=6)]
    model, rt = _load(mdir, store, policy=PrecisionPolicy("hi"))
    ref = _conversation(GraphDecoder(model, max_len=64, backend="emulate"), turns, reuse=False)
    got = _conversation(GraphDecoder(model, max_len=64, backend="emulate"), turns, reuse=True)
    for i in range(0, len(ref), 2):
        assert ref[i][0] == 0
        assert torch.allclose(got[i + 1], ref[i + 1], atol=1e-4, rtol=1e-4), (arch, i, (got[i + 1] - ref[i + 1]).abs().max())
    # messages 2 et 3 : tout l'historique (prompts et tokens générés, tous calculés ici) est repris
    assert got[0][0] == 0
    assert got[2][0] == 7 + 4 and got[4][0] == 7 + 4 + 5 + 4, (got[2], got[4])
    rt.close()


def test_p1_divergent_history(arch_model):
    """Historique modifié au milieu : attention seule = coupe au 1er token différent ;
    attention linéaire = retour à la fin du prompt précédent (ou tout recalculer si le prompt a changé)."""
    arch, mdir, store = arch_model
    model, rt = _load(mdir, store, policy=PrecisionPolicy("hi"))
    dec = GraphDecoder(model, max_len=64, backend="emulate")
    a = _ids(5, T=8)
    dec.prefill(a, reuse=True)
    nxt = pick(dec.step(torch.tensor([[7]]), 7))
    dec.step(nxt, int(nxt))
    # l'historique « réécrit » la réponse (re-tokenisation différente), puis un nouveau message
    b = torch.cat([a, torch.tensor([[9, 9, 9]]), _ids(6, T=4)], 1)
    lg = dec.prefill(b, reuse=True)
    assert dec.reused == 8, (arch, dec.reused)  # olmoe : 1er écart à la position 8 ; qwen3next : fin du prompt a
    ref = GraphDecoder(model, max_len=64, backend="emulate").prefill(b)
    assert torch.allclose(lg, ref, atol=1e-4, rtol=1e-4)
    # le prompt lui-même change au début : tout est recalculé
    c = torch.cat([torch.tensor([[3]]), b[:, 1:]], 1)
    lg = dec.prefill(c, reuse=True)
    assert dec.reused == 0
    assert torch.allclose(lg, GraphDecoder(model, max_len=64, backend="emulate").prefill(c), atol=1e-4, rtol=1e-4)
    # suivi perdu (step sans valeur, comme benchmark et profil) : plus de reprise
    dec.step(torch.tensor([[4]]))
    dec.prefill(torch.cat([c, torch.tensor([[4, 5]])], 1), reuse=True)
    assert dec.reused == 0
    rt.close()


def test_p1_generate_timings(olmoe_dir, olmoe_store):
    model, rt = _load(olmoe_dir, olmoe_store, policy=PrecisionPolicy("hi"))
    dec = GraphDecoder(model, max_len=64, backend="emulate")
    ids = _ids(8, T=6)
    tm: dict = {}
    new = generate(model, ids, 5, decoder=dec, eos_token_id=-1, reuse=True, timings=tm)
    assert tm["prompt_tokens"] == 6 and tm["reused"] == 0 and tm["ttft_s"] > 0
    ids2 = torch.cat([ids, torch.tensor([new]), _ids(9, T=3)], 1)
    generate(model, ids2, 3, decoder=dec, eos_token_id=-1, reuse=True, timings=tm)
    assert tm["reused"] == 6 + 4  # le 5e token généré n'a pas encore été calculé
    rt.close()


def _prefill_logits(mdir, store, **opts):
    model, rt = _load(mdir, store, policy=PrecisionPolicy("lo"), fast_device="cpu", fast_budget_bytes=4 * 64 * 1024,
                      host_budget_bytes=6 * 64 * 1024, prefetch_adaptive=False, **opts)
    outs = []
    with torch.no_grad():
        for s in range(3):
            outs.append(model(_ids(20 + s, T=12)).logits)
    st = rt.stats
    rt.close()
    return torch.cat(outs), st


@pytest.mark.parametrize("opts", [{"prefill_order": "completed"}, {"prefill_prefetch": True},
                                  {"prefill_cache": "ram"},
                                  {"prefill_order": "completed", "prefill_prefetch": True, "prefill_cache": "ram"},
                                  {"prefill_skip": 0.0}],
                         ids=["p2", "p3", "p4", "p234", "skip0"])
def test_prefill_options_same_logits(arch_model, opts):
    """P2 à P4 changent l'ordre et la place, pas le calcul ; P5 avec un seuil de 0 ne saute rien."""
    arch, mdir, store = arch_model
    ref, _ = _prefill_logits(mdir, store)
    got, st = _prefill_logits(mdir, store, **opts)
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), (arch, opts, (got - ref).abs().max())
    if opts.get("prefill_prefetch"):
        assert st.disk_ahead > 0
    assert st.prefill_skipped == 0


def test_prefill_skip_skips_only_absent_low_weight(arch_model):
    """P5 : avec un seuil de 1, tout expert à lire sur le disque est sauté ; les résidents sont calculés."""
    arch, mdir, store = arch_model
    ref, st0 = _prefill_logits(mdir, store)
    got, st = _prefill_logits(mdir, store, prefill_skip=1.0)
    assert st.prefill_skipped > 0 and st.disk_reads < st0.disk_reads
    assert not torch.allclose(got, ref, atol=1e-4)
    assert torch.isfinite(got).all()


def _ref_grouped_int4(x_row, entries, groups, out, version=2):
    """Noyau groupé int4 de référence (PyTorch) : out += somme des w * expert(x)."""
    from moelab.quant import dequantize
    from moelab.runtime import expert_ffn

    for v, w in entries:
        gu = dequantize(v["gate_up.q"], v["gate_up.s"], 4, x_row.dtype)
        dn = dequantize(v["down.q"], v["down.s"], 4, x_row.dtype)
        out.add_(w * expert_ffn(x_row.reshape(1, -1), gu, dn, torch.nn.SiLU()).view_as(out))


def _ref_grouped_float(x_row, entries, out):
    from moelab.runtime import expert_ffn

    for v, w in entries:
        out.add_(w * expert_ffn(x_row.reshape(1, -1), v["gate_up"].to(x_row.dtype), v["down"].to(x_row.dtype),
                                torch.nn.SiLU()).view_as(out))


@pytest.mark.parametrize("policy", ["lo", "hi"])
def test_p6_grouped_per_token_same_logits(olmoe_dir, olmoe_store, monkeypatch, policy):
    """P6 : prefill token par token avec les noyaux groupés (ici leur équivalent PyTorch) = mêmes logits."""
    import moelab.runtime as runtime

    monkeypatch.setattr(runtime, "grouped_ffn_int4", _ref_grouped_int4)
    monkeypatch.setattr(runtime, "grouped_ffn_float", _ref_grouped_float)
    outs, calls = [], []
    for grouped in (0, 64):
        model, rt = _load(olmoe_dir, olmoe_store, policy=PrecisionPolicy(policy), fast_device="cpu",
                          fast_budget_bytes=4 * 64 * 1024, host_budget_bytes=6 * 64 * 1024, prefetch_adaptive=False,
                          prefill_grouped=grouped, grouped_kernels=2, prefill_split=False)
        rt._grouped = True
        seen = []
        orig = rt._run_grouped_tokens
        monkeypatch.setattr(rt, "_run_grouped_tokens", lambda *a: (seen.append(1), orig(*a)))
        with torch.no_grad():
            outs.append(torch.cat([model(_ids(30 + s, T=10)).logits for s in range(2)]))
        calls.append(len(seen))
        assert rt._grouped, rt.triton_error
        rt.close()
    assert calls[0] == 0 and calls[1] > 0
    assert torch.allclose(outs[0], outs[1], atol=1e-4, rtol=1e-4), (outs[0] - outs[1]).abs().max()


def test_p9_prefix_cache_memory_and_disk(arch_model, tmp_path):
    """P9 : le début fixe (système + outils) calculé une fois, rechargé de la mémoire ou du disque
    (décodeur neuf = nouveau lancement) ; mêmes logits qu'en recalculant tout."""
    arch, mdir, store = arch_model
    model, rt = _load(mdir, store, policy=PrecisionPolicy("hi"))
    pre, a, b = _ids(40, T=9), _ids(41, T=4), _ids(42, T=5)

    def fresh(full):
        return GraphDecoder(model, max_len=64, backend="emulate").prefill(full)

    d1 = GraphDecoder(model, max_len=64, backend="emulate")
    d1.prefix_dir, d1.prefix_key = str(tmp_path / "pc"), "k"
    ids_a = torch.cat([pre, a], 1)
    lg = d1.prefill(ids_a, reuse=True, prefix_len=9)
    assert not d1.prefix_hit and len(list((tmp_path / "pc").glob("prefix_*.pt"))) == 1
    assert torch.allclose(lg, fresh(ids_a), atol=1e-4, rtol=1e-4)
    d1.step(torch.tensor([[5]]), 5)
    # nouvelle conversation (/reset) : même système, autre question -> début fixe repris de la mémoire
    ids_b = torch.cat([pre, b], 1)
    lg = d1.prefill(ids_b, reuse=True, prefix_len=9)
    assert d1.reused == 9 and (d1.prefix_hit or arch == "olmoe"), (arch, d1.reused, d1.prefix_hit)
    assert torch.allclose(lg, fresh(ids_b), atol=1e-4, rtol=1e-4)
    # nouveau lancement : décodeur neuf, cache vide, début fixe relu sur le disque
    d2 = GraphDecoder(model, max_len=64, backend="emulate")
    d2.prefix_dir, d2.prefix_key = str(tmp_path / "pc"), "k"
    lg = d2.prefill(ids_b, reuse=True, prefix_len=9)
    assert d2.prefix_hit and d2.reused == 9
    assert torch.allclose(lg, fresh(ids_b), atol=1e-4, rtol=1e-4)
    # la suite (génération) reste exacte
    ref = GraphDecoder(model, max_len=64, backend="emulate")
    ref.prefill(ids_b)
    assert torch.allclose(d2.step(torch.tensor([[7]]), 7), ref.step(torch.tensor([[7]])), atol=1e-4, rtol=1e-4)
    # autre contexte (clé différente) : rien n'est repris
    d3 = GraphDecoder(model, max_len=64, backend="emulate")
    d3.prefix_dir, d3.prefix_key = str(tmp_path / "pc"), "autre"
    d3.prefill(ids_b, reuse=True, prefix_len=9)
    assert not d3.prefix_hit and d3.reused == 0
    rt.close()


def test_export_import_state_around_other_use(arch_model):
    """L'autopilot mesure avec le décodeur entre deux tours de réflexion : la conversation du
    cerveau est exportée puis remise, et la suite reste exacte et reprise du cache."""
    arch, mdir, store = arch_model
    model, rt = _load(mdir, store, policy=PrecisionPolicy("hi"))
    dec = GraphDecoder(model, max_len=64, backend="emulate")
    a = _ids(50, T=8)
    dec.prefill(a, reuse=True)
    dec.step(torch.tensor([[6]]), 6)
    saved = dec.export_state()
    assert saved is not None and len(saved["tokens"]) == 9
    dec.prefill(_ids(51, T=11))          # autre usage (mesure) : cache écrasé, suivi perdu
    dec.step(torch.tensor([[3]]))
    assert dec.import_state(saved) and dec.tokens == saved["tokens"]
    nxt = torch.cat([a, torch.tensor([[6, 9]]), _ids(52, T=3)], 1)
    lg = dec.prefill(nxt, reuse=True)
    assert dec.reused == 9
    ref = GraphDecoder(model, max_len=64, backend="emulate").prefill(nxt)
    assert torch.allclose(lg, ref, atol=1e-4, rtol=1e-4), (arch, (lg - ref).abs().max())
    dec.step(torch.tensor([[4]]))         # suivi perdu : rien à exporter
    assert dec.export_state() is None
    rt.close()
