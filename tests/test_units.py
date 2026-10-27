"""Tests unitaires de chaque brique : quantification, store, cache, politiques."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from moelab.cache import TierCache, UsageLog
from moelab.policy import PrecisionPolicy, qstar
from moelab.quant import bits_per_weight, dequantize, pick_group_size, quantize
from moelab.store import ExpertStore, encode_record, materialize, record_layout

ROOT = Path(__file__).resolve().parents[1]


# -- quantification ----------------------------------------------------------
@pytest.mark.parametrize("bits,tol", [(8, 0.008), (4, 0.12)])
def test_quant_roundtrip(bits, tol):
    # Poids gaussiens, groupe de 64 : max|w| ~ 2.4 sigma, pas = 2.4/qmax sigma,
    # erreur ~ pas/sqrt(12) -> ~0.10 relatif en int4, ~0.0055 en int8.
    torch.manual_seed(0)
    w = torch.randn(96, 256)
    q, s = quantize(w, bits, 64)
    assert s.shape == (96, 4) and s.dtype == torch.float16
    assert q.shape == ((96, 256) if bits == 8 else (96, 128))
    back = dequantize(q, s, bits, torch.float32)
    rel = (back - w).norm() / w.norm()
    assert rel < tol, rel


def test_int4_packing_order():
    w = torch.zeros(1, 64)
    w[0, 0], w[0, 1] = 7.0, -7.0  # colonne paire -> quartet bas, impaire -> quartet haut
    q, s = quantize(w, 4, 64)
    assert q[0, 0].item() == (7 + 8) | ((-7 + 8) << 4)
    assert torch.equal(dequantize(q, s, 4, torch.float32)[0, :2], torch.tensor([7.0, -7.0]))


def test_zero_group_and_group_size():
    q, s = quantize(torch.zeros(4, 64), 4, 64)
    assert torch.equal(dequantize(q, s, 4, torch.float32), torch.zeros(4, 64))
    assert pick_group_size(1024) == 64 and pick_group_size(32) == 32 and pick_group_size(768) == 64
    assert bits_per_weight(4, 64) == 4.25


# -- records du store ------------------------------------------------------
@pytest.mark.parametrize("prec", ["fp32", "fp16", "bf16", "int8", "int4"])
def test_record_roundtrip(prec):
    torch.manual_seed(1)
    H, I = 128, 64
    gu, dn = torch.randn(2 * I, H), torch.randn(H, I)
    lay = record_layout(prec, H, I)
    assert lay["record_bytes"] % 4096 == 0
    blob = torch.from_numpy(encode_record(lay, gu, dn))
    g2, d2 = materialize(lay, blob, torch.float32)
    tol = {"fp32": 0, "fp16": 1e-3, "bf16": 1e-2, "int8": 1e-2, "int4": 0.12}[prec]
    assert (g2 - gu).norm() / gu.norm() <= tol
    assert (d2 - dn).norm() / dn.norm() <= tol


def test_store_read(olmoe_store):
    st = ExpertStore(olmoe_store)
    assert st.precisions == ["fp32", "int4"] and st.hi == "fp32" and st.lo == "int4"
    assert st.rank("fp32") < st.rank("int4")
    a = st.read(1, 5, "int4")
    buf = torch.empty(st.record_bytes("int4"), dtype=torch.uint8)
    b = st.read(1, 5, "int4", out=buf)
    assert b is buf and torch.equal(a, b)
    assert st.bytes_read == 2 * st.record_bytes("int4")


def test_incomplete_store_refused(tmp_path):
    (tmp_path / "manifest.json").write_text('{"complete": false}')
    with pytest.raises(RuntimeError):
        ExpertStore(tmp_path)


# -- cache ---------------------------------------------------------------------
RANK = {"fp16": 0, "int4": 1}.__getitem__


def blob(n):
    return torch.zeros(n, dtype=torch.uint8)


def test_lrfu_evicts_coldest():
    c = TierCache("t", "cpu", 300, RANK, half_life=4)
    for e in range(3):
        c.insert((0, e, "fp16"), blob(100))
    for _ in range(3):  # l'expert 0 devient chaud
        c.tick()
        c.touch(c.entries[(0, 0, "fp16")])
    c.tick()
    c.insert((0, 9, "fp16"), blob(100))
    assert (0, 0, "fp16") in c.entries and (0, 9, "fp16") in c.entries
    assert len(c.entries) == 3 and c.stats.evictions == 1


def test_lrfu_recency_beats_old_frequency():
    c = TierCache("t", "cpu", 200, RANK, half_life=1)
    c.insert((0, 0, "fp16"), blob(100))
    for _ in range(3):
        c.touch(c.entries[(0, 0, "fp16")])
    c.clock += 20  # l'expert 0 n'a plus servi depuis longtemps
    c.insert((0, 1, "fp16"), blob(100))
    c.insert((0, 2, "fp16"), blob(100))
    assert (0, 0, "fp16") not in c.entries


def test_precision_substitution():
    c = TierCache("t", "cpu", 1000, RANK)
    c.insert((2, 7, "fp16"), blob(10))
    assert c.lookup(2, 7, "int4").key[2] == "fp16"  # un fp16 sert une demande int4
    c2 = TierCache("t", "cpu", 1000, RANK)
    c2.insert((2, 7, "int4"), blob(10))
    assert c2.lookup(2, 7, "fp16") is None  # mais pas l'inverse
    assert c2.lookup(2, 7, "int4", exact=True) is not None


def test_pins_and_protect():
    c = TierCache("t", "cpu", 300, RANK, pin_fraction=0.5)
    c.insert((0, 0, "fp16"), blob(100), pinned=True)
    c.insert((0, 1, "fp16"), blob(100))
    c.insert((0, 2, "fp16"), blob(100))
    assert c.insert((0, 3, "fp16"), blob(100), protect=frozenset({(0, 1, "fp16"), (0, 2, "fp16")})) is None
    assert c.insert((0, 3, "fp16"), blob(100)) is not None
    assert (0, 0, "fp16") in c.entries  # l'épinglé a survécu
    assert c.insert((0, 4, "fp16"), blob(400)) is None


def test_repin_hysteresis_and_max_swaps():
    c = TierCache("t", "cpu", 10_000, RANK, pin_fraction=0.5, half_life=1000)
    for e in range(5):
        c.insert((0, e, "fp16"), blob(1000), pinned=True)
    for e in range(5, 10):
        c.insert((0, e, "fp16"), blob(1000))
    for e in range(5, 10):
        for _ in range(3):
            c.touch(c.entries[(0, e, "fp16")])
    swaps = c.repin(hysteresis=0.25, max_swaps=4)
    assert len(swaps) == 4
    assert sum(ent.pinned for ent in c.entries.values()) == 5
    # Candidat seulement 10% plus chaud : pas d'échange (hystérésis)
    d = TierCache("t", "cpu", 2000, RANK, pin_fraction=0.5, half_life=1000)
    d.insert((0, 0, "fp16"), blob(1000), pinned=True)
    d.insert((0, 1, "fp16"), blob(1000))
    d.entries[(0, 0, "fp16")].f = 10.0
    d.entries[(0, 1, "fp16")].f = 11.0
    assert d.repin() == []


def test_usage_log_roundtrip(tmp_path):
    p = tmp_path / "usage.json"
    u = UsageLog(p, "abc")
    for _ in range(3):
        u.record(1, 4, hi=True)
    u.record(0, 2, hi=False)
    u.save()
    v = UsageLog(p, "abc")
    assert v.top(1) == [(1, 4)] and v.prefers_hi(1, 4) and not v.prefers_hi(0, 2) and v.sessions == 1
    w = UsageLog(p, "autre")
    assert not w.counts and w.ignored_reason


# -- politiques -------------------------------------------------------------------
def test_hobbit_on_real_olmoe_example():
    """Les scores mesurés dans docs/06 : 4 experts en haute précision, 3 en basse, 1 sauté."""
    w = torch.tensor([[0.1345, 0.1063, 0.0921, 0.0855, 0.0829, 0.0705, 0.0682, 0.0214]])
    pol = PrecisionPolicy("hobbit", t1=0.6, t2=0.9)
    s = pol.unimportance(w)[0]
    assert torch.allclose(s, torch.tensor([0.0, 0.2034, 0.3641, 0.5034, 0.6327, 0.7580, 0.8646, 0.9677]), atol=2e-4)
    need_hi, skip = pol.decide(0, w)
    assert need_hi[0].tolist() == [True] * 4 + [False] * 4
    assert skip[0].tolist() == [False] * 7 + [True]


def test_hobbit_order_independent_and_modes():
    w = torch.tensor([[0.05, 0.6, 0.35]])
    need_hi, _ = PrecisionPolicy("hobbit", t1=0.6).decide(3, w)
    assert need_hi[0].tolist() == [False, True, True]
    assert PrecisionPolicy("hi").decide(0, w)[0].all()
    assert not PrecisionPolicy("lo").decide(0, w)[0].any()
    assert PrecisionPolicy("hobbit", protect_layers={3}).decide(3, w)[0].all()
    pol = PrecisionPolicy("hobbit")
    assert pol.expert_is_hi(torch.tensor([True, False])) and not pol.expert_is_hi(torch.tensor([True, False, False]))


def test_qstar_matches_freetoken_figure():
    assert qstar(4, 1.0, 4.0) == 1        # figure 2 du papier : B_P:B_H = 1:4, m=4 -> 1
    assert qstar(10, 1.0, 4.0) == 2       # round(2.5) = 2 (arrondi bancaire de Python)
    assert qstar(3, 5.0, 4.0) == 3        # B_H <= B_P : tout transférer
    assert qstar(8, 1.0, 100.0) == 1      # au moins un transfert
    assert qstar(0, 1.0, 2.0) == 0


def test_store_close_releases_files(olmoe_store):
    st = ExpertStore(olmoe_store)
    st.read(0, 0, "int4")
    assert st._all_handles and not st._all_handles[0].closed
    fh = st._all_handles[0]
    st.close()
    assert fh.closed and not st._all_handles
    st.read(0, 1, "int4")  # rouvre à la demande


def test_split_wrappers_keep_generate_working(olmoe_dir, olmoe_store):
    """Même montage que le placement split, mais sur CPU : generate() doit rester exact."""
    from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
    from moelab.loader import CpuEmbedding, CpuLinear
    from transformers import AutoModelForCausalLM

    cfg = RuntimeConfig(policy=PrecisionPolicy("hi"), fast_budget_bytes=0, host_budget_bytes=1 << 24)
    model, rt = load_offloaded_model(olmoe_dir, olmoe_store, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    model.set_input_embeddings(CpuEmbedding(emb.weight.data, emb.padding_idx, out_device="cpu"))
    model.set_output_embeddings(CpuLinear(head.weight.data))
    ids = torch.randint(2, 128, (1, 6), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=8, do_sample=False)
    ref = AutoModelForCausalLM.from_pretrained(olmoe_dir, dtype=torch.float32).eval().generate(
        ids, max_new_tokens=8, do_sample=False)
    assert torch.equal(gen, ref)
    rt.close()


def test_generate_uses_kv_cache_even_if_config_disables_it(tmp_path):
    """OLMoE a "use_cache": false dans config.json : generate() doit quand même utiliser le cache KV."""
    from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
    from moelab.store import build_store
    from moelab.testing import make_model_dir, tiny_olmoe_config

    d = make_model_dir(tmp_path / "m", tiny_olmoe_config(use_cache=False))
    build_store(d, tmp_path / "s", ["fp16", "int4"], log=lambda *a: None)
    model, rt = load_offloaded_model(d, tmp_path / "s", RuntimeConfig(policy=PrecisionPolicy("hi")),
                                     dense="cpu", dtype=torch.float32, log=lambda *a: None)
    seen = []
    orig = rt.run_layer

    def spy(layer, x, *a):
        if layer == 0:
            seen.append(x.shape[0])
        return orig(layer, x, *a)

    rt.run_layer = spy
    with torch.no_grad():
        model.generate(torch.randint(2, 128, (1, 7)), max_new_tokens=5, do_sample=False)
    assert seen == [7, 1, 1, 1, 1], seen  # prefill, puis un seul token par pas
    rt.close()


def test_host_pool_exact_slots_cap_and_quarantine():
    from moelab.hostpool import HostPool

    pool = HostPool({"fp16": 3000, "int4": 800}, cap_bytes=64 * 1024, pin=False, log=lambda *a: None)
    a = pool.acquire("fp16")
    assert a.numel() == 3000 and pool.is_pooled(a)
    assert pool.allocated & (pool.allocated - 1) == 0  # plaques de taille puissance de 2 : pas de perte à l'arrondi
    pool.release(a)
    pool.release(a)  # double libération ignorée
    b = pool.acquire("fp16")
    assert b.data_ptr() != a.data_ptr()  # en quarantaine jusqu'à la couche suivante
    pool.advance()
    got = {pool.acquire("fp16").data_ptr() for _ in range(len(pool.free["fp16"]) + 1)}
    assert a.data_ptr() in got
    blobs = [pool.acquire("int4") for _ in range(200)]  # bien plus que le plafond
    assert pool.allocated <= pool.cap and pool.fallbacks > 0
    assert all(x.numel() == 800 for x in blobs)


def test_host_pool_recycles_free_slabs_across_precisions():
    """Un réglage lo remplit le pool d'int4 ; ensuite hobbit doit pouvoir y loger son int8 (mesures du 02/10)."""
    from moelab.hostpool import HostPool

    pool = HostPool({"int8": 6000, "int4": 3000}, cap_bytes=4 * 65536, pin=False, log=lambda *a: None)
    lo = []
    while True:  # tout le plafond en int4, comme après un benchmark `lo`
        b = pool.acquire("int4")
        if not pool.is_pooled(b):
            break
        lo.append(b)
    assert pool.allocated == pool.cap and pool.free["int8"] == []
    for b in lo:
        pool.release(b)
    pool.advance()
    hi = [pool.acquire("int8") for _ in range(20)]
    assert all(pool.is_pooled(x) for x in hi) and pool.reclaims > 0 and pool.fallbacks == 1
    assert pool.allocated == pool.cap  # aucune nouvelle allocation : les plaques ont été redécoupées
    assert len({x.data_ptr() for x in hi}) == 20


def test_runtime_host_memory_stays_bounded(olmoe_dir, olmoe_store):
    """Beaucoup d'évictions en RAM : le pool ne grossit pas au-delà du plafond et ne perd aucun slot."""
    from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model

    REC = 64 * 32 * 3 * 4
    cfg = RuntimeConfig(policy=PrecisionPolicy("hobbit", t1=0.3), host_budget_bytes=3 * REC)
    model, rt = load_offloaded_model(olmoe_dir, olmoe_store, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
    ids = torch.randint(2, 128, (1, 8), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        model.generate(ids, max_new_tokens=12, do_sample=False)
    assert rt.host.stats.evictions > 0
    assert rt.pool.allocated <= rt.pool.cap
    for f in rt.pending.values():
        f.result()
    held = sum(1 for e in rt.host.entries.values() if rt.pool.is_pooled(e.blob))
    assert len(rt.pool.in_use) == held + len(rt.pending)  # tout slot prêté est en cache, en lecture (prefetch), ou rendu
    rt.reset_caches()
    assert len(rt.pool.in_use) == 0
    rt.close()


def test_benchmark_configs_accept_per_config_t1():
    import importlib.util
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts" / "benchmark.py")
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)
    c = bm.make_configs(["hi", "hobbit", "hobbit@0.3", "hobbit_edges@0.8"], 0.6, 16)
    assert c["hi"].mode == "hi" and c["hobbit"].t1 == 0.6 and c["hobbit@0.3"].t1 == 0.3
    assert c["hobbit_edges@0.8"].protect_layers == {0, 1, 15}
    with pytest.raises(SystemExit):
        bm.make_configs(["hobbit@abc"], 0.6, 16)


def test_warm_start_resets_stats_even_without_preload(olmoe_dir, olmoe_store):
    """Bug du 30/09 : sans rien à précharger, warm_start ne remettait pas les compteurs à zéro."""
    from moelab import RuntimeConfig, load_offloaded_model

    cfg = RuntimeConfig(policy=PrecisionPolicy("hi"), host_budget_bytes=1 << 24)
    model, rt = load_offloaded_model(olmoe_dir, olmoe_store, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
    with torch.no_grad():
        model(torch.randint(2, 128, (1, 8)))
    assert rt.stats.requests > 0
    rt.usage = UsageLog(None, rt.store.store_id)  # historique vierge, comme le benchmark : rien à précharger
    rt.warm_start(preload_lo=False, log=lambda *a: None)
    assert rt.stats.requests == 0 and rt.stats.tokens == 0
    rt.close()


def test_policy_upgrade_flag():
    assert PrecisionPolicy("hobbit").allow_upgrade
    assert not PrecisionPolicy("hobbit", upgrade=False).allow_upgrade
    assert not PrecisionPolicy("lo").allow_upgrade


def test_phase_timer_records_every_phase(olmoe_dir, olmoe_store):
    from moelab import RuntimeConfig, load_offloaded_model
    from moelab.profiling import PhaseTimer

    cfg = RuntimeConfig(policy=PrecisionPolicy("hobbit"), host_budget_bytes=1 << 24)
    model, rt = load_offloaded_model(olmoe_dir, olmoe_store, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
    rt.prof = PhaseTimer()
    with torch.no_grad():
        model(torch.randint(2, 128, (1, 8)))
    assert rt.prof.calls == len(rt.store.layers)
    assert {n[0] for n in rt.prof.totals} >= {"1", "2", "3", "4", "5", "6"}
    rt.close()


def test_decide_one_matches_decide():
    g = torch.Generator().manual_seed(0)
    for pol in (PrecisionPolicy("hobbit", t1=0.6), PrecisionPolicy("hobbit", t1=0.3, t2=0.9),
                PrecisionPolicy("hi"), PrecisionPolicy("lo"), PrecisionPolicy("hobbit", protect_layers={2})):
        for layer in range(4):
            for _ in range(50):
                w = torch.rand(1, 8, generator=g) ** 3
                need, skip = pol.decide(layer, w)
                need1, skip1 = pol.decide_one(layer, w[0].tolist())
                assert need[0].tolist() == need1 and skip[0].tolist() == skip1


def test_heap_eviction_matches_sorted_lrfu():
    """Éviction en tas (03/10) : mêmes victimes (mêmes scores) que le tri complet des entrées."""
    import random

    from moelab.cache import TierCache

    rnd = random.Random(0)
    c = TierCache("t", "cpu", 40 * 64, {"a": 0}.get, half_life=8.0, pin_fraction=0.25)
    n = 0
    for step in range(3000):
        r = rnd.random()
        if r < 0.5:
            key = (rnd.randrange(4), rnd.randrange(60), "a")
            if key in c.entries:
                c.touch(c.entries[key])
                continue
            protect = frozenset(rnd.sample(list(c.entries), min(3, len(c.entries)))) if c.entries else frozenset()
            need = 64
            expected = None
            if c.used + need > c.budget:
                cands = sorted((e for e in c.entries.values() if not e.pinned and e.key not in protect), key=c.score)
                expected = sorted(round(c.score(e), 9) for e in cands[:1])
            before = {k: c.score(e) for k, e in c.entries.items()}
            c.insert(key, torch.zeros(64, dtype=torch.uint8), protect=protect, pinned=rnd.random() < 0.05)
            gone = [k for k in before if k not in c.entries]
            if expected is not None:
                assert sorted(round(before[k], 9) for k in gone) == expected, step
                assert not set(gone) & protect
                n += 1
        elif r < 0.8:
            c.tick()
        elif r < 0.9 and c.entries:
            c.unpin(rnd.choice(list(c.entries)))
        elif c.entries:
            c.pin(rnd.choice(list(c.entries)))
    assert n > 500
    assert c.used <= c.budget


@pytest.mark.parametrize("parts", [2, 3, 8])
def test_split_reads_same_bytes(olmoe_store, parts):
    """Un record lu en plusieurs morceaux simultanés (03/10) = le même record lu d'un coup."""
    st = ExpertStore(olmoe_store)
    for prec in st.precisions:
        ref = st.read(1, 3, prec).clone()
        st.read_parts = parts
        got = st.read(1, 3, prec)
        st.read_parts = 1
        assert torch.equal(ref, got), (prec, parts)
    st.close()


def test_auto_kv_reserve():
    """03/10 : longueur max et réserve VRAM d'après le cache KV du modèle."""
    from types import SimpleNamespace

    from moelab.plan import ACTIVATIONS_RESERVE, auto_graph_max_len, auto_vram_reserve, kv_bytes_per_token

    olmoe = SimpleNamespace(num_hidden_layers=16, num_attention_heads=16, num_key_value_heads=16, hidden_size=2048,
                            max_position_embeddings=4096)
    qnext = SimpleNamespace(num_hidden_layers=48, num_attention_heads=16, num_key_value_heads=2, head_dim=256,
                            hidden_size=2048, full_attention_interval=4, max_position_embeddings=262144)
    qnext_types = SimpleNamespace(**vars(qnext), layer_types=["linear_attention"] * 3 + ["full_attention"])
    qnext_types.layer_types = qnext_types.layer_types * 12
    assert kv_bytes_per_token(olmoe) == 128 * 1024
    assert kv_bytes_per_token(qnext) == kv_bytes_per_token(qnext_types) == 24 * 1024
    assert auto_graph_max_len(olmoe) == 2048 and auto_graph_max_len(qnext) == 8192
    assert auto_vram_reserve(qnext, 8192) == ACTIVATIONS_RESERVE + 192 * 2**20
    assert abs(auto_vram_reserve(olmoe, 2048) / 2**30 - 0.80) < 0.01
