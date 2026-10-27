"""
Autopilot (05/10) : la boucle garde un gain seulement s'il dépasse le bruit et se
confirme, refuse ce qui dégrade la qualité, remet la meilleure configuration après chaque
essai, respecte son budget, écrit un profil que chat.py / agent.py réappliquent.
Le « modèle » est ici un script d'appels d'outils ; la mesure, un faux banc réglable.
"""

from __future__ import annotations

import json

import pytest
import torch

from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
from moelab.autopilot import Autopilot, Bench, apply_profile, check_change, search_space

MB = 2**20


def _load(olmoe_dir, store):
    return load_offloaded_model(olmoe_dir, store, RuntimeConfig(policy=PrecisionPolicy("lo"), host_budget_bytes=64 * MB,
                                                                fast_budget_bytes=0), dense="cpu", dtype=torch.float32,
                                log=lambda *a: None)


class FakeBench:
    """Score d'après la config du runtime : read_parts=4 gagne 20 %, prefill_skip abîme la qualité,
    cache_eviction=stale gagne 10 % à la 1re mesure seulement (bruit)."""

    def __init__(self, rt):
        self.rt, self.calls, self.stale_runs = rt, [], 0

    def run(self):
        c = self.rt.cfg
        self.calls.append((c.read_parts, c.prefill_skip, c.cache_eviction))
        s, kl = 10.0, 0.0
        if c.read_parts == 4:
            s *= 0.8
        if c.prefill_skip:
            s, kl = s * 0.5, 0.02
        if c.cache_eviction == "stale":
            self.stale_runs += 1
            s *= 0.9 if self.stale_runs == 1 else 1.0
        ttft = 0.7 if c.prefill_grouped == 64 else 1.0  # 1er token -30 %, réponse type -3,75 % seulement : gardé
        s -= 1.0 - ttft
        return {"premier_token_s": ttft, "tok_s": 20.0, "reponse_type_s": round(s, 2), "hit_vram_pourcent": 50.0,
                "disque_par_token": 3.0, "kl": kl, "top1_pourcent": 100.0}


def _call(name, **args):
    return '<tool_call>' + json.dumps({"name": name, "arguments": args}) + '</tool_call>'


def test_autopilot_loop(olmoe_dir, olmoe_store, tmp_path):
    model, rt = _load(olmoe_dir, olmoe_store)
    rt.cfg.read_parts = 1
    rt.cfg.prefill_grouped = 0
    bench = FakeBench(rt)
    pilot = Autopilot(model, None, bench, max_trials=6, log=lambda *a: None)
    script = [
        _call("machine") + _call("espace"),
        _call("mesurer", changements={"read_parts": 4}),
        _call("mesurer", changements={"prefill_skip": 0.1}),
        _call("mesurer", changements={"cache_eviction": "stale"}),
        _call("mesurer", changements={"nimporte": 1}) + _call("mesurer", changements={"read_parts": 3}),
        _call("mesurer", changements={"read_parts": 4}) + _call("mesurer", changements={"prefill_skip": 0.1}),
        _call("mesurer", changements={"prefill_grouped": 64}),
        _call("journal"),
        _call("terminer", rapport="read_parts=4 aide ; le reste non."),
    ]
    seen, box = [], {}

    def gen(messages, tools):
        box["m"] = messages
        seen.append(1)
        return script[len(seen) - 1]

    s = pilot.run(generate_fn=gen)
    verdicts = [t.verdict for t in pilot.trials]
    assert verdicts[0] == "départ"
    assert verdicts[1].startswith("GARDÉ")                    # read_parts 4 : -20 %, confirmé
    assert verdicts[2].startswith("qualité dégradée")         # prefill_skip : KL 0,02
    assert "bruit" in verdicts[3]                             # stale : gain non confirmé
    assert "1er token" in verdicts[4] and verdicts[4].startswith("GARDÉ")  # 1er token -30 %, réponse type -3,75 %
    assert len(pilot.trials) == 5                             # clé inconnue, valeur interdite, doublon : refusés sans mesure
    assert s["meilleur"]["read_parts"] == 4 and s["meilleur"]["prefill_grouped"] == 64
    assert s["meilleur_reponse_type_s"] == 7.7 and s["gain_pourcent"] == 23.0
    assert rt.cfg.read_parts == 4 and rt.store.read_parts == 4 and rt.cfg.prefill_skip is None  # meilleure config remise
    assert rt.cfg.cache_eviction == "lrfu"
    tool_out = [m["content"] for m in box["m"] if m["role"] == "tool"]
    assert any("n'est pas dans l'espace" in x for x in tool_out) and any("non permis" in x for x in tool_out)
    assert any("déjà mesurée" in x for x in tool_out) and any("déjà celles de la meilleure" in x for x in tool_out)
    fin = s["verification_finale"]
    assert fin["depart"]["reponse_type_s"] == 10.0 and fin["meilleur"]["reponse_type_s"] == 7.7 and fin["gain_pourcent"] == 23.0
    assert len(seen) == len(script)  # terminer arrête la boucle
    assert s["rapport"].startswith("read_parts=4")
    ppath, rpath = pilot.save(tmp_path / "res", "olmoe-mini")
    assert "GARDÉ" in rpath.read_text(encoding="utf-8")
    prof = json.loads(ppath.read_text(encoding="utf-8"))
    assert prof["changes_par_rapport_au_depart"] == {"read_parts": 4, "prefill_grouped": 64}
    rt.close()
    # nouveau lancement : le profil est réappliqué (sauf les clés données en ligne de commande)
    model2, rt2 = _load(olmoe_dir, olmoe_store)
    rt2.cfg.read_parts, rt2.cfg.prefill_grouped = 1, 0
    assert apply_profile(model2, olmoe_store, skip=("read_parts",), log=lambda *a: None) == {"prefill_grouped": 64}
    assert apply_profile(model2, olmoe_store, log=lambda *a: None) == {"read_parts": 4} and rt2.cfg.read_parts == 4
    rt2.close()
    ppath.unlink()


def test_budget_and_errors(olmoe_dir, olmoe_store):
    model, rt = _load(olmoe_dir, olmoe_store)
    pilot = Autopilot(model, None, FakeBench(rt), max_trials=1, log=lambda *a: None)
    pilot.baseline()
    assert pilot.experiment({}) == {"erreur": "aucun changement demandé"}
    r = pilot.experiment({"prefetch_depth": 2})
    assert r["essai"] == 1 and "pas mieux" in r["verdict"] and r["ecart_tok_s_pourcent"] == 0.0
    assert "épuisé" in pilot.experiment({"prefetch_depth": 1})["erreur"]
    rt.close()


def test_failure_restores_best(olmoe_dir, olmoe_store):
    """Une mesure qui plante (ex. mémoire GPU) : verdict d'échec, meilleure config remise."""
    model, rt = _load(olmoe_dir, olmoe_store)

    class Boom(FakeBench):
        def run(self):
            if self.rt.cfg.prefetch_depth == 2:
                raise RuntimeError("CUDA out of memory")
            return super().run()

    pilot = Autopilot(model, None, Boom(rt), log=lambda *a: None)
    pilot.baseline()
    r = pilot.experiment({"prefetch_depth": 2})
    assert r["verdict"].startswith("échec") and "out of memory" in r["verdict"]
    assert rt.cfg.prefetch_depth == 1
    rt.close()


def test_check_change_and_space(olmoe_dir, olmoe_store):
    model, rt = _load(olmoe_dir, olmoe_store)
    sp = search_space(rt)
    assert "vram_gb" not in sp and "policy" in sp  # pas de GPU ; store à deux précisions
    assert check_change(sp, "prefill_skip", "none") is None
    assert check_change(sp, "ram_gb", "2.5") == 2.5
    for bad in (("ram_gb", 1000), ("read_parts", 3), ("x", 1)):
        with pytest.raises(ValueError):
            check_change(sp, *bad)
    rt.close()


def test_real_bench_on_tiny_model(olmoe_dir, olmoe_store, tmp_path):
    from moelab.testing import make_tiny_tokenizer

    tok = make_tiny_tokenizer(tmp_path / "tok")
    model, rt = _load(olmoe_dir, olmoe_store)
    b = Bench(model, tok, ["abc", "def gh"], gen=5, warm="salut")
    r1 = b.run()
    assert r1["kl"] == 0.0 and r1["tok_s"] > 0 and r1["reponse_type_s"] > 0
    r2 = b.run()  # même config : mêmes probabilités
    assert r2["kl"] < 1e-6 and r2["top1_pourcent"] == 100.0
    rt.close()


def test_nudges_when_model_forgets_to_call(olmoe_dir, olmoe_store):
    model, rt = _load(olmoe_dir, olmoe_store)
    pilot = Autopilot(model, None, FakeBench(rt), log=lambda *a: None)
    outs = iter(["Mesurons.", "", _call("terminer", rapport="ok")])
    s = pilot.run(generate_fn=lambda m, t: next(outs))
    assert s["relances"] == 2 and s["rapport"] == "ok"
    pilot2 = Autopilot(model, None, FakeBench(rt), log=lambda *a: None)
    s2 = pilot2.run(generate_fn=lambda m, t: "rien")
    assert s2["relances"] == 5 and "sans appeler terminer" in s2["rapport"]
    rt.close()


def test_drift_is_detected_and_reference_follows(olmoe_dir, olmoe_store):
    """Les conditions changent (disque qui chauffe...) : la meilleure config est remesurée à côté
    du candidat, et la comparaison se fait contre cette nouvelle mesure."""
    model, rt = _load(olmoe_dir, olmoe_store)

    class Drifting(FakeBench):
        slow = 1.0

        def run(self):
            r = super().run()
            r["reponse_type_s"] = round(r["reponse_type_s"] * self.slow, 2)
            r["tok_s"] = round(r["tok_s"] / self.slow, 2)
            return r

    b = Drifting(rt)
    pilot = Autopilot(model, None, b, log=lambda *a: None)
    pilot.baseline()
    b.slow = 1.5  # tout ralentit de 50 % : le candidat, neutre, a l'air bien pire
    r = pilot.experiment({"prefetch_depth": 2})
    assert r["derive_pourcent"] == -50.0 and "pas mieux" in r["verdict"]
    assert abs(r["ecart_reponse_type_pourcent"]) < 1  # comparé à la remesure : neutre
    assert pilot.best_res["reponse_type_s"] == 15.0
    rt.close()
