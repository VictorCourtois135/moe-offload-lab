"""Palier final : la même mécanique sur l'architecture Qwen3-Next (DeltaNet + attention, expert partagé, FP8)."""

from __future__ import annotations

import torch
from conftest import reference_logits, to_fp8_checkpoint

from moelab import PrecisionPolicy, RuntimeConfig, build_store, load_offloaded_model
from moelab.checkpoint import Checkpoint
from moelab.loader import CpuEmbedding, CpuLinear, discover_experts

MB = 2**20


def test_discover(qnext_dir):
    info = discover_experts(qnext_dir)
    assert info["model_type"] == "qwen3_next"
    assert info["num_experts"] == 16 and info["hidden"] == 64 and info["inter"] == 32
    assert len(info["layers"]) == 4 and info["layers"][0] == "model.layers.0.mlp.experts"


def test_qwen3next_hi_matches_reference(qnext_dir, tmp_path):
    store = build_store(qnext_dir, tmp_path / "s", ("fp32", "int4"), log=lambda *a: None)
    ids = torch.randint(2, 128, (1, 20), generator=torch.Generator().manual_seed(0))
    ref = reference_logits(qnext_dir, ids)
    model, rt = load_offloaded_model(qnext_dir, store, RuntimeConfig(policy=PrecisionPolicy("hi"), fast_device="cpu",
                                     fast_budget_bytes=5 * 2 * 64 * 32 * 4, host_budget_bytes=64 * MB,
                                     b_pcie=1.0, b_host=2.0), dense="cpu", dtype=torch.float32, log=lambda *a: None)
    with torch.no_grad():
        out = model(ids).logits
        gen = model.generate(ids[:, :6], max_new_tokens=6, do_sample=False)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3), (out - ref).abs().max()
    assert rt.stats.cpu_execs > 0 and rt.stats.fills > 0
    assert gen.shape == (1, 12)
    rt.close()


def test_fp8_checkpoint_dequantized(qnext_dir, tmp_path):
    fp8 = to_fp8_checkpoint(qnext_dir, tmp_path / "fp8")
    ck, ck8 = Checkpoint(qnext_dir), Checkpoint(fp8)
    k = "model.layers.0.mlp.experts.3.down_proj.weight"
    assert ck8.get_raw(k).dtype == torch.float8_e4m3fn
    w, w8 = ck.get(k), ck8.get(k)
    assert w8.dtype == torch.float32
    assert (w - w8).abs().max() <= w.abs().max() * 0.07  # e4m3 : 3 bits de mantisse

    store = build_store(fp8, tmp_path / "s8", ("fp16", "int4"), log=lambda *a: None)
    model, rt = load_offloaded_model(fp8, store, RuntimeConfig(policy=PrecisionPolicy("hi"), host_budget_bytes=64 * MB),
                                     dense="cpu", dtype=torch.float32, log=lambda *a: None)
    ids = torch.randint(2, 128, (1, 16), generator=torch.Generator().manual_seed(1))
    ref = reference_logits(qnext_dir, ids)
    with torch.no_grad():
        out = model(ids).logits
    agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
    assert agree >= 0.8, agree
    rt.close()


def test_cpu_wrappers_match_modules():
    torch.manual_seed(0)
    emb = torch.nn.Embedding(50, 8, padding_idx=0)
    lin = torch.nn.Linear(8, 50, bias=False)
    ids = torch.randint(0, 50, (2, 5))
    ce, cl = CpuEmbedding(emb.weight.data, 0), CpuLinear(lin.weight.data, chunk=7)
    # aucun paramètre : sinon transformers croirait le modèle "sur CPU" en placement split
    assert list(ce.parameters()) == [] and list(cl.parameters()) == []
    assert torch.equal(ce(ids), emb(ids))
    assert CpuEmbedding(emb.weight.data, 0, out_device="meta")(ids).device.type == "meta"
    x = torch.randn(2, 5, 8)
    assert torch.allclose(cl(x), lin(x), atol=1e-6)
