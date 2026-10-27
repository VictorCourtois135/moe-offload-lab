"""
Qwen3-MoE (famille de Qwen3-30B-A3B, 05/10) : moelab le fait tourner sans adaptateur.
Mêmes logits que transformers, CUDA graphs (émulés) identiques à la passe normale.
"""

from __future__ import annotations

import torch

from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
from moelab.generate import EagerDecoder, pick
from moelab.graphs import GraphDecoder
from moelab.store import build_store
from moelab.testing import make_model_dir, reference_logits, tiny_qwen3moe_config


def test_qwen3moe_runs_exact(tmp_path):
    md = make_model_dir(tmp_path / "m", tiny_qwen3moe_config())
    st = build_store(md, tmp_path / "s", ("fp32", "int4"), log=lambda *a: None)
    ids = torch.randint(2, 128, (1, 9), generator=torch.Generator().manual_seed(0))
    ref = reference_logits(md, ids)
    model, rt = load_offloaded_model(md, st, RuntimeConfig(policy=PrecisionPolicy("hi"), host_budget_bytes=2**26,
                                                           fast_budget_bytes=0), dense="cpu", dtype=torch.float32,
                                     log=lambda *a: None)
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, ref, atol=1e-4, rtol=1e-4)

    def run(dec):
        lg = [dec.prefill(ids)]
        for _ in range(6):
            lg.append(dec.step(pick(lg[-1])))
        return torch.cat(lg)

    a, b = run(EagerDecoder(model)), run(GraphDecoder(model, max_len=32, backend="emulate"))
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
    rt.close()
