"""
Q1 / Q2 (02/10) : choix de l'échelle int4 ("mse"), GPTQ, et réécriture en place de
l'int4 d'un store.
"""

from __future__ import annotations

import json
import shutil

import torch

from moelab.quant import choose_scales, dequantize, gptq_quantize, quantize
from moelab.store import ExpertStore, build_store, materialize, record_layout


def _err(w, q, s):
    return ((dequantize(q, s, 4, torch.float32) - w) ** 2).mean().item()


def test_mse_scale_never_worse_and_better_on_average():
    torch.manual_seed(0)
    w = torch.distributions.StudentT(4).sample((256, 512))
    e_abs = _err(w, *quantize(w, 4, 64, "absmax"))
    e_mse = _err(w, *quantize(w, 4, 64, "mse"))
    assert e_mse < 0.8 * e_abs, (e_abs, e_mse)
    # groupe par groupe : jamais pire (le candidat absmax fait partie de la recherche)
    g = w.reshape(256, 8, 64)
    sa, sm = choose_scales(g, 4, "absmax").float(), choose_scales(g, 4, "mse").float()
    ea = ((torch.round(g / sa).clamp(-8, 7) * sa - g) ** 2).sum(-1)
    em = ((torch.round(g / sm).clamp(-8, 7) * sm - g) ** 2).sum(-1)
    assert bool((em <= ea + 1e-6).all())


def test_mse_scale_uses_minus_eight_and_zero_groups():
    w = torch.zeros(2, 64)
    w[0, 0] = 8.0  # le plus grand poids tombe exactement sur -8 avec une échelle négative
    w[0, 1:5] = torch.tensor([1.0, 2.0, 3.0, -4.0])
    q, s = quantize(w, 4, 64, "mse")
    back = dequantize(q, s, 4, torch.float32)
    assert torch.allclose(back[0, :5], w[0, :5]) and s[0, 0].item() == -1.0
    assert torch.equal(back[1], torch.zeros(64))  # groupe nul : échelle non nulle, aucun NaN


def test_gptq_reduces_output_error():
    torch.manual_seed(1)
    O, I = 64, 256
    w = torch.randn(O, I) * 0.05
    X = torch.randn(2048, I) @ (torch.eye(I) + 0.4 * torch.randn(I, I) / I ** 0.5)
    X[:, :8] *= 6
    H = X.T @ X

    def out_err(q, s):
        return ((X @ (dequantize(q, s, 4, torch.float32) - w).T) ** 2).mean().item()

    e_mse = out_err(*quantize(w, 4, 64, "mse"))
    e_gptq = out_err(*gptq_quantize(w, H, 4, 64))
    assert e_gptq < 0.85 * e_mse, (e_mse, e_gptq)
    q, s = gptq_quantize(w, H, 4, 64)
    assert q.shape == (O, I // 2) and q.dtype == torch.uint8 and s.shape == (O, I // 64) and s.dtype == torch.float16


def test_layout_scale_and_store_id(olmoe_dir, tmp_path):
    assert record_layout("int4", 64, 32)["scale"] == "mse"
    assert record_layout("int8", 64, 32)["scale"] == "absmax"
    assert "scale" not in record_layout("fp16", 64, 32)
    a = build_store(olmoe_dir, tmp_path / "a", ("fp32", "int4"), int4_scale="absmax", log=lambda *x: None)
    b = build_store(olmoe_dir, tmp_path / "b", ("fp32", "int4"), log=lambda *x: None)
    ma = json.loads((a / "manifest.json").read_text())
    mb = json.loads((b / "manifest.json").read_text())
    assert ma["store_id"] != mb["store_id"]


def test_requantize_store_mse_in_place(olmoe_dir, tmp_path):
    from moelab.checkpoint import Checkpoint
    from moelab.requant import requantize_store

    src = build_store(olmoe_dir, tmp_path / "s", ("fp32", "int4"), int4_scale="absmax", log=lambda *x: None)
    (src / "usage.json").write_text(json.dumps({"store_id": json.loads((src / "manifest.json").read_text())["store_id"],
                                                "counts": {"0:1": 3}}))
    hi_before = (src / "layer000.fp32.bin").read_bytes()
    stats = requantize_store(src, olmoe_dir, "mse", device="cpu", log=lambda *x: None)
    st = ExpertStore(src)
    assert st.layouts["int4"]["scale"] == "mse" and st.store_id == stats["store_id"]
    assert (src / "layer000.fp32.bin").read_bytes() == hi_before  # la haute précision n'a pas bougé
    assert json.loads((src / "usage.json").read_text())["store_id"] == st.store_id
    ck = Checkpoint(olmoe_dir)
    gu, dn = ck.read_expert(st.layers[0], 2, st.hidden, st.inter)
    got_gu, _ = materialize(st.layouts["int4"], st.read(0, 2, "int4"), torch.float32)
    lay = st.layouts["int4"]
    q, s = quantize(gu, 4, lay["group_sizes"]["gate_up"], "mse")
    assert torch.equal(got_gu, dequantize(q, s, 4, torch.float32))
    st.close()
    ck.close()


def test_gptq_pipeline_improves_lo_quality(olmoe_dir, tmp_path):
    """Calibration (entrées enregistrées) -> GPTQ par expert -> logits en `lo` plus proches de `hi`."""
    from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
    from moelab.requant import collect_calibration, requantize_store

    base = build_store(olmoe_dir, tmp_path / "base", ("fp32", "int4"), int4_scale="absmax", log=lambda *x: None)
    gptq = tmp_path / "gptq"
    shutil.copytree(base, gptq)
    g = torch.Generator().manual_seed(3)
    calib = [torch.randint(2, 128, (1, 96), generator=g) for _ in range(12)]
    test = torch.randint(2, 128, (1, 64), generator=torch.Generator().manual_seed(99))

    def logits(store, mode):
        cfg = RuntimeConfig(policy=PrecisionPolicy(mode), fast_device=None, host_budget_bytes=1 << 24)
        model, rt = load_offloaded_model(olmoe_dir, store, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
        with torch.no_grad():
            out = model(input_ids=test).logits
        rt.close()
        return out

    cfg = RuntimeConfig(policy=PrecisionPolicy("hi"), fast_device=None, host_budget_bytes=1 << 24)
    model, rt = load_offloaded_model(olmoe_dir, gptq, cfg, dense="cpu", dtype=torch.float32, log=lambda *a: None)
    meta = collect_calibration(model, calib, tmp_path / "calib", log=lambda *a: None)
    rt.close()
    assert meta["tokens"] == 12 * 96 and (tmp_path / "calib" / "x_000.npy").exists()
    stats = requantize_store(gptq, olmoe_dir, "gptq", calib_dir=tmp_path / "calib", min_tokens=8, device="cpu",
                             log=lambda *a: None)
    assert stats["gptq"] > 0
    ref = logits(base, "hi")
    e_abs = (logits(base, "lo") - ref).pow(2).mean().item()
    e_gptq = (logits(gptq, "lo") - ref).pow(2).mean().item()
    assert e_gptq < e_abs, (e_abs, e_gptq)
