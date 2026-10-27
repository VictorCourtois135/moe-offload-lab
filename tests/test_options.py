"""Catalogue des réglages (moelab/options.py) : cohérent avec le code, applicable à chaud."""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model, options

MB = 2**20


def test_catalog_matches_code():
    cfg_fields = {f.name: f for f in dataclasses.fields(RuntimeConfig)}
    pol_fields = {f.name: f for f in dataclasses.fields(PrecisionPolicy)}
    defaults_cfg, defaults_pol = RuntimeConfig(), PrecisionPolicy()
    keys = set()
    for o in options.catalog():
        assert o.key not in keys
        keys.add(o.key)
        assert o.group in options.GROUPS and o.apply in ("live", "caches", "reload", "store")
        kind, _, name = o.target.partition(".")
        if kind == "cfg":
            assert name in cfg_fields, o.key
            if o.default is not None:
                assert getattr(defaults_cfg, name) == o.default, (o.key, getattr(defaults_cfg, name), o.default)
        elif kind == "policy":
            assert name in pol_fields, o.key
            if o.key not in ("protect", "t1_layers"):
                assert getattr(defaults_pol, name) == o.default, o.key
        else:
            assert kind in ("load", "store"), o.key
    for name, preset in options.PRESETS.items():
        assert set(preset) <= keys, name
    data = json.loads(options.describe_json())
    assert len(data["options"]) == len(options.catalog()) and data["presets"]


def test_parse():
    assert options.parse("cuda_graphs", "off") is False
    assert options.parse("t1", "0.5") == 0.5
    assert options.parse("grouped_kernels", "3") == 3
    with pytest.raises(ValueError):
        options.parse("t1", "1.5")
    with pytest.raises(ValueError):
        options.parse("grouped_kernels", "9")


def test_apply_live(olmoe_dir, olmoe_store):
    model, rt = load_offloaded_model(olmoe_dir, olmoe_store, RuntimeConfig(host_budget_bytes=64 * MB, fast_budget_bytes=0),
                                     dense="cpu", dtype=torch.float32, log=lambda *a: None)
    options.apply(model, "policy", "lo")
    options.apply(model, "t1", "0.3")
    assert rt.policy.mode == "lo" and rt.policy.t1 == 0.3
    options.apply(model, "protect", "edges")
    assert rt.policy.protect_layers == frozenset({0, 1, len(rt.store.layers) - 1})
    options.apply(model, "direct_io", "off")
    assert rt.cfg.direct_io is False and rt.store.direct is False
    options.apply(model, "grouped_kernels", 2)
    assert rt.cfg.grouped_kernels == 2
    with pytest.raises(ValueError):
        options.apply(model, "dense", "gpu")       # demande un rechargement
    options.apply(model, "ram_gb", 0.05)           # budgets à chaud (05/10)
    assert rt.host.budget == int(0.05 * 2**30) and options.current(model)["ram_gb"] == 0.05
    with pytest.raises(ValueError):
        options.apply(model, "policy", "medium")   # choix inconnu
    with pytest.raises(KeyError):
        options.apply(model, "nope", 1)
    options.apply_preset(model, "qualité")
    assert rt.policy.mode == "hobbit" and rt.policy.t1 == 0.8
    cur = options.current(model)
    assert cur["policy"] == "hobbit" and cur["t1"] == 0.8 and cur["direct_io"] is False
    # le réglage agit vraiment : en lo (sans couches protégées), aucun expert en haute précision
    options.apply(model, "protect", "")
    options.apply(model, "policy", "lo")
    rt.reset_stats()
    with torch.no_grad():
        model(torch.randint(2, 128, (1, 6)))
    assert rt.stats.used_hi == 0 and rt.stats.requests > 0
    assert "Précision" in options.table(model) and "graph_max_len" in options.table(model, advanced=True)
    rt.close()
