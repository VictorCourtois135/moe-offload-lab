"""Chemin GPU (streams de copie, mémoire épinglée, placement split). Sautés sans CUDA."""

from __future__ import annotations

import pytest
import torch

from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
from moelab.testing import reference_logits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA indisponible")
REC = 64 * 32 * 3 * 4  # un record fp32 du modèle miniature


@pytest.mark.parametrize("dense", ["gpu", "split"])
def test_gpu_tiers_match_reference(olmoe_dir, olmoe_store, dense):
    ids = torch.randint(2, 128, (2, 24), generator=torch.Generator().manual_seed(5))
    ref = reference_logits(olmoe_dir, ids)
    cfg = RuntimeConfig(policy=PrecisionPolicy("hi"), fast_device="cuda", fast_budget_bytes=6 * REC,
                        host_budget_bytes=8 * REC, b_pcie=1.0, b_host=3.0, prefetch=True)
    model, rt = load_offloaded_model(olmoe_dir, olmoe_store, cfg, dense=dense, dtype=torch.float32, log=lambda *a: None)
    assert model.device.type == "cuda"  # c'est ce que generate() utilise pour placer ses tenseurs
    with torch.no_grad():
        out = model(ids.cuda()).logits.cpu()
        gen = model.generate(ids[:1, :6].cuda(), max_new_tokens=10, do_sample=False).cpu()
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3), (out - ref).abs().max()
    from transformers import AutoModelForCausalLM

    ref_gen = AutoModelForCausalLM.from_pretrained(olmoe_dir, dtype=torch.float32).eval().generate(
        ids[:1, :6], max_new_tokens=10, do_sample=False)
    assert torch.equal(gen, ref_gen)
    s = rt.stats
    assert s.fills > 0 and s.cpu_execs > 0 and s.bytes_h2d > 0
    rt.close()
