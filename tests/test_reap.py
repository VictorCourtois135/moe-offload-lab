"""REAP (doc 22, 08/10) : scores des experts, reprise, données figées, banc d'évaluation."""

from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F

from moelab import PrecisionPolicy, RuntimeConfig, build_store, load_offloaded_model
from moelab.prune_data import build_calibration, build_eval, load_or_build, pack, spread_chunks
from moelab.prune_eval import evaluate, load_tool_tasks, perplexity, score_task, tool_tasks
from moelab.reap import ReapCollector, calibrate, prune_mask, run_batches, summary
from moelab.store import materialize
from moelab.testing import make_demo_model

MB = 2**20
QUIET = dict(log=lambda *a, **k: None)


@pytest.fixture(scope="module")
def qstore(qnext_dir, tmp_path_factory):
    return build_store(qnext_dir, tmp_path_factory.mktemp("reap") / "s", ("fp32", "int4"), **QUIET)


def _load(model_dir, store, fast: bool):
    cfg = RuntimeConfig(policy=PrecisionPolicy("hi"), fast_device="cpu" if fast else None,
                        fast_budget_bytes=6 * 2 * 64 * 32 * 4 if fast else 0, host_budget_bytes=64 * MB,
                        b_pcie=1.0, b_host=2.0)
    return load_offloaded_model(model_dir, store, cfg, dense="cpu", dtype=torch.float32, **QUIET)


def _ids(n=6, t=24, seed=0):
    return torch.randint(2, 128, (n, t), generator=torch.Generator().manual_seed(seed))


def _reference(model, rt, ids):
    """g * ||f|| recalculé à part : entrées et routage de chaque couche, poids lus dans le store."""
    seen = []
    rt.calib_hook = lambda l, x, idx, w: seen.append((l, x.clone(), idx.clone(), w.clone()))
    with torch.no_grad():
        model.get_decoder()(input_ids=ids, use_cache=False)
    rt.calib_hook = None
    L, E = len(rt.store.layers), rt.store.num_experts
    sg, cnt = torch.zeros(L, E, dtype=torch.float64), torch.zeros(L, E, dtype=torch.int64)
    lay = rt.store.layouts["fp32"]
    for l, x, idx, w in seen:
        for e in idx.unique().tolist():
            rows, slot = (idx == e).nonzero(as_tuple=True)
            gu, dn = materialize(lay, rt.store.read(l, e, "fp32"), torch.float32)
            g, u = F.linear(x[rows], gu).chunk(2, dim=-1)
            y = F.linear(F.silu(g) * u, dn)
            sg[l, e] += (w[rows, slot].float() * y.norm(dim=-1)).sum().double()
            cnt[l, e] += rows.numel()
    return sg, cnt


@pytest.mark.parametrize("fast", [True, False])
def test_reap_scores_match_reference(qnext_dir, qstore, tmp_path, fast):
    """Crochet du runtime (chemin GPU simulé, puis branche CPU) = calcul direct, à 1e-5 près."""
    model, rt = _load(qnext_dir, qstore, fast)
    ids = _ids()
    sg, cnt = _reference(model, rt, ids)
    col = calibrate(model, ids, tmp_path / "s.pt", batch=2, **QUIET)
    assert col.samples_done == 6 and col.tokens == ids.numel()
    assert torch.equal(col.count, cnt)
    assert torch.allclose(col.totals["sum_gnorm"], sg, rtol=1e-5, atol=1e-7)
    S = col.scores()
    assert (S[cnt == 0] == 0).all() and (S[cnt > 0] > 0).all()
    assert torch.allclose(col.frequency().sum(1), torch.ones(len(rt.store.layers), dtype=torch.float64))
    assert rt.expert_hook is None and rt.cfg.use_qstar  # réglages remis comme avant
    if fast:
        assert rt.stats.fills > 0
    else:
        assert rt.stats.cpu_execs > 0
    rt.close()


def test_reap_resume_after_interruption(qnext_dir, qstore, tmp_path, monkeypatch):
    model, rt = _load(qnext_dir, qstore, True)
    ids = _ids(8)
    full = calibrate(model, ids, tmp_path / "full.pt", batch=2, **QUIET)

    import moelab.reap as reap

    real, calls = reap.hidden_states, {"n": 0}

    def boom(m, chunk):
        calls["n"] += 1
        if calls["n"] == 3:  # 3e lot : interrompu au milieu (Ctrl+C)
            real(m, chunk[:1])
            raise KeyboardInterrupt
        return real(m, chunk)

    monkeypatch.setattr(reap, "hidden_states", boom)
    path = tmp_path / "part.pt"
    with pytest.raises(KeyboardInterrupt):
        calibrate(model, ids, path, batch=2, save_every=1, **QUIET)
    part = ReapCollector.load(path)
    assert part.samples_done == 4  # les 2 lots finis, rien du lot interrompu
    monkeypatch.setattr(reap, "hidden_states", real)
    msgs = []
    done = calibrate(model, ids, path, batch=2, log=msgs.append)
    assert any("reprise : 4/8" in m for m in msgs)
    assert torch.equal(done.count, full.count)
    assert torch.allclose(done.totals["sum_gnorm"], full.totals["sum_gnorm"], rtol=1e-9)
    # d'autres échantillons : l'ancien fichier est ignoré
    msgs = []
    other = calibrate(model, _ids(4, seed=1), path, batch=2, log=msgs.append)
    assert other.samples_done == 4 and any("ignoré" in m for m in msgs)
    rt.close()


def test_run_batches_halves_on_oom():
    seen = []

    def fn(chunk):
        if chunk.shape[0] > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        seen.append(chunk[:, 0].tolist())

    final = run_batches(torch.arange(7).view(7, 1), 8, fn, **QUIET)
    assert final == 2 and sum(seen, []) == list(range(7))
    with pytest.raises(RuntimeError, match="autre"):
        run_batches(torch.arange(3).view(3, 1), 2, lambda c: (_ for _ in ()).throw(RuntimeError("autre")), **QUIET)


def test_prune_mask_and_summary():
    S = torch.tensor([[0.1, 0.5, 0.0, 0.9, 0.2, 0.3, 0.4, 0.8], [1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.float64)
    m = prune_mask(S, 0.25, top_k=2)
    assert m.sum(1).tolist() == [2, 2]
    assert m[0].nonzero().flatten().tolist() == [0, 2] and m[1].nonzero().flatten().tolist() == [0, 1]
    assert prune_mask(S, 0.9, top_k=3).sum(1).tolist() == [5, 5]  # toujours au moins top_k experts
    assert prune_mask(S, 0.5, top_k=2, layers=[1])[0].sum() == 0
    col = ReapCollector(2, 8)
    col.count = torch.tensor([[1, 5, 0, 9, 2, 3, 4, 8], [1] * 8])
    col.totals["sum_gnorm"] = S * col.count
    s = summary(col, ratios=(0.25,), top_k=2, record_bytes=2**20)
    assert s["jamais_choisis"] == 1
    assert s["par_couche"][0]["choix_perdus_25"] == pytest.approx(1 / 32)
    assert s["taux"]["0.25"]["store_gio"] == pytest.approx(round(2 * 6 / 1024, 2))
    json.dumps(s)


class _Tok:
    """Tokenizer factice : un token par caractère (codes 2..)."""
    eos_token_id = 1
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        class R:
            pass

        r = R()
        r.input_ids = [2 + (ord(c) % 100) for c in text]
        return r


def test_pack_and_calibration_mix():
    tok = _Tok()
    assert pack(["abc", "de"], tok, 2, seq_len=3) == [[2 + 97, 2 + 98, 2 + 99], [1, 2 + 0, 2 + 1]]
    assert len(spread_chunks(list(range(100)), 3, 10)) == 3
    assert spread_chunks(list(range(100)), 3, 10)[1][0] == 30

    def broken():
        raise ConnectionError("pas de réseau")

    srcs = {"local": lambda: [], "code": broken, "wiki_en": lambda: ["x" * 10_000]}
    d = build_calibration(tok, 10, seq_len=16, seed=0, sources=srcs, mix={"local": 0.2, "code": 0.3, "wiki_en": 0.5},
                          **QUIET)
    assert d["ids"].shape == (10, 16) and d["counts"] == {"local": 0, "code": 0, "wiki_en": 10}
    again = build_calibration(tok, 10, seq_len=16, seed=0, sources=srcs, mix={"local": 0.2, "code": 0.3, "wiki_en": 0.5},
                              **QUIET)
    assert torch.equal(d["ids"], again["ids"])  # même graine, mêmes échantillons
    with pytest.raises(RuntimeError, match="pas assez"):
        build_calibration(tok, 10, seq_len=16, sources={"wiki_en": lambda: ["x" * 20]}, mix={"wiki_en": 1}, **QUIET)
    ev = build_eval(tok, 3, 2, seq_len=8, sources={"wikitext": lambda: ["a" * 100], "fr": lambda: ["é" * 30]})
    assert ev["wikitext"].shape == (3, 8) and ev["fr"].shape == (2, 8)


def test_load_or_build_is_frozen(tmp_path):
    n = {"calls": 0}

    def build():
        n["calls"] += 1
        return {"ids": torch.arange(4)}

    a = load_or_build(tmp_path / "d.pt", build, **QUIET)
    b = load_or_build(tmp_path / "d.pt", build, **QUIET)
    assert n["calls"] == 1 and torch.equal(a["ids"], b["ids"])


def test_score_task():
    t = {"outil": "calculer", "args": {"expression": [["3.35", "3,35"], "6.8"]}}
    assert score_task(t, "", [{"name": "calculer", "arguments": {"expression": "(6.8 - 3,35) / 3,35"}}])[0]
    ok, why = score_task(t, "", [{"name": "calculer", "arguments": {"expression": "6.8/2"}}])
    assert not ok and "manque" in why
    assert "mauvais outil" in score_task(t, "", [{"name": "lire_page", "arguments": {}}])[1]
    assert "aucun" in score_task(t, "réponse", [])[1]
    any_tool = {"outil": ["lister_dossier", "chercher_fichiers"], "args": {"*": ["test"]}}
    assert score_task(any_tool, "", [{"name": "chercher_fichiers", "arguments": {"dossier": "D:\\x\\Test", "motif": "*"}}])[0]
    direct = {"outil": None, "contient": ["cat", ["sofa", "couch"]]}
    assert score_task(direct, "The cat sleeps on the  Couch.", [])[0]
    assert not score_task(direct, "The cat sleeps.", [])[0]
    assert not score_task(direct, "", [{"name": "calculer", "arguments": {}}])[0]


def test_perplexity_matches_reference(qnext_dir, qstore):
    from transformers import AutoModelForCausalLM

    model, rt = _load(qnext_dir, qstore, True)
    ids = _ids(5, 20, seed=3)
    p = perplexity(model, ids, batch=2, chunk=7, **QUIET)
    ref = AutoModelForCausalLM.from_pretrained(qnext_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        lp = ref(ids).logits.log_softmax(-1)
    nll = -lp[:, :-1].gather(-1, ids[:, 1:, None]).mean().item()
    assert p["ppl"] == pytest.approx(math.exp(nll), rel=1e-4)
    assert torch.equal(p["top1"], lp[:, :-1].argmax(-1).to(torch.int32))
    rt.close()


def test_evaluate_end_to_end(tmp_path):
    """Modèle miniature avec tokenizer : les trois mesures, le JSON, puis l'accord avec la référence."""
    mdir = make_demo_model(tmp_path / "m", arch="qwen3next")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mdir)
    store = build_store(mdir, tmp_path / "s", ("fp32", "int4"), **QUIET)
    model, rt = _load(mdir, store, True)
    rt.cfg.cuda_graphs = False
    tasks = load_tool_tasks()
    assert len(tasks["taches"]) == 20 and len({t["id"] for t in tasks["taches"]}) == 20
    r = tool_tasks(model, tok, tasks, max_new_tokens=4, **QUIET)
    assert r["total"] == 20 and len(r["details"]) == 20
    ids = {"wikitext": _ids(3, 16), "fr": _ids(2, 16, seed=5)}
    res = evaluate(model, tok, ids, "baseline", tmp_path / "out", batch=2, parts=("vitesse", "perplexite"), **QUIET)
    saved = json.loads((tmp_path / "out" / "baseline.json").read_text(encoding="utf-8"))
    assert saved["perplexite"]["wikitext"]["extraits"] == 3 and saved["vitesse"]["invites"] == 3
    top = torch.load(tmp_path / "out" / "baseline_top1.pt", weights_only=True)
    again = evaluate(model, tok, ids, "essai", tmp_path / "out", batch=3, ref_top1=top, parts=("perplexite",), **QUIET)
    assert again["perplexite"]["fr"]["accord_top1_ref"] == 1.0
    assert again["perplexite"]["wikitext"]["ppl"] == pytest.approx(res["perplexite"]["wikitext"]["ppl"], rel=1e-6)
    rt.close()


def test_prune_mask_exact_and_never_loaded(qnext_dir, qstore):
    """Étape 3 : masque nul = modèle inchangé ; masque = même résultat que le modèle de référence avec le même
    masque ; aucun expert masqué n'est jamais choisi ni chargé ; le masque se retire."""
    import types

    from transformers import AutoModelForCausalLM

    from moelab.reap import _masked_forward, apply_prune_mask, parse_layers

    model, rt = _load(qnext_dir, qstore, True)
    ids = _ids(1, 20, seed=7)
    with torch.no_grad():
        base = model(ids).logits
    L, E = len(rt.store.layers), rt.store.num_experts
    assert apply_prune_mask(model, torch.zeros(L, E, dtype=torch.bool), **QUIET) == 0
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, base, atol=1e-6)

    mask = prune_mask(torch.rand(L, E, generator=torch.Generator().manual_seed(1)).double(), 0.5, top_k=4)
    assert apply_prune_mask(model, mask, **QUIET) == int(mask.sum())
    chosen = []
    rt.calib_hook = lambda l, x, idx, w: chosen.append((l, idx.clone()))
    rt.reset_stats()
    with torch.no_grad():
        out = model(ids).logits
    rt.calib_hook = None
    assert all(not mask[l][idx.flatten()].any() for l, idx in chosen)            # jamais choisis
    in_cache = {(k[0], k[1]) for k in rt.host.entries} | {(k[0], k[1]) for k in rt.fast.entries}
    assert not (in_cache & rt.pruned)                                              # jamais chargés
    ref = AutoModelForCausalLM.from_pretrained(qnext_dir, dtype=torch.float32).eval()
    for li, layer in enumerate(ref.model.layers):
        g = layer.mlp.gate
        g.moelab_bias = torch.where(mask[li], float("-inf"), 0.0).float()
        g.forward = types.MethodType(_masked_forward, g)
    with torch.no_grad():
        assert torch.allclose(out, ref(ids).logits, atol=1e-4, rtol=1e-3)       # exact, au calcul près
        assert not torch.allclose(out, base, atol=1e-3)

    apply_prune_mask(model, None, **QUIET)
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, base, atol=1e-6) and not rt.pruned
    assert parse_layers("10-12,15", 48) == [10, 11, 12, 15] and parse_layers(None, 48) is None
    with pytest.raises(ValueError):
        parse_layers("40-50", 48)
    with pytest.raises(ValueError):
        apply_prune_mask(model, torch.ones(L, E, dtype=torch.bool), **QUIET)    # moins de top_k experts
    rt.close()
