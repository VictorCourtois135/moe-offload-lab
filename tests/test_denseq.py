"""D5 : partie dense en int8 (QLinear)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
from moelab.denseq import QLinear, quantizable
from moelab.generate import EagerDecoder, pick
from moelab.graphs import GraphDecoder

MB = 2**20


def test_qlinear_close_to_linear():
    torch.manual_seed(0)
    lin = nn.Linear(256, 96, bias=True)
    ql = QLinear.from_linear(lin, "cpu")
    x = torch.randn(3, 5, 256)
    ref, out = lin(x), ql(x)
    assert out.shape == ref.shape
    assert ((out - ref).abs().max() / ref.abs().max()) < 1e-2


def _load(model_dir, store_dir, **kw):
    return load_offloaded_model(model_dir, store_dir, RuntimeConfig(policy=PrecisionPolicy("hi"), host_budget_bytes=64 * MB,
                                                                    fast_budget_bytes=0), dense="cpu",
                                dtype=torch.float32, log=lambda *a: None, **kw)


def test_dense_int8_model(olmoe_dir, olmoe_store):
    ids = torch.randint(2, 128, (1, 8), generator=torch.Generator().manual_seed(1))
    m0, rt0 = _load(olmoe_dir, olmoe_store)
    n_lin = len(quantizable(m0))
    with torch.no_grad():
        ref = m0(ids).logits
    rt0.close()
    m1, rt1 = _load(olmoe_dir, olmoe_store, dense_int8=True)
    assert not quantizable(m1) and sum(isinstance(m, QLinear) for m in m1.modules()) == n_lin > 0
    with torch.no_grad():
        out = m1(ids).logits
    assert (out - ref).abs().max() < 0.05 * ref.abs().max()
    assert (out.argmax(-1) == ref.argmax(-1)).float().mean() > 0.8
    # génération rejouée (émulateur de graphs) = génération normale, avec la partie dense en int8
    def run(dec, n=6):
        lg = [dec.prefill(ids)]
        nxt = pick(lg[-1])
        for _ in range(n):
            lg.append(dec.step(nxt))
            nxt = pick(lg[-1])
        return torch.cat(lg)
    a = run(EagerDecoder(m1))
    b = run(GraphDecoder(m1, max_len=32, backend="emulate"))
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)
    rt1.close()


_SCRIPT = r"""
import sys, torch
sys.path.insert(0, {root!r})
from moelab.denseq import QLinear
for O, K in ((96, 256), (128, 512), (50, 192)):
    torch.manual_seed(O)
    lin = torch.nn.Linear(K, O, bias=False)
    ql = QLinear.from_linear(lin, "cpu")
    assert ql.kernel or K == 192
    x = torch.randn(K)
    ref = torch.nn.functional.linear(x, torch.ops.aten.mul(ql.q.float().reshape(O, -1, ql.group_size),
                                    ql.s.float()[:, :, None]).reshape(O, K))
    out = ql(x[None, None])[0, 0]
    assert ((out - ref).abs().max() / ref.abs().max()) < 1e-4, (O, K)
print("OK")
"""


def test_gemv_int8_interpreted():
    pytest.importorskip("triton")
    root = str(Path(__file__).resolve().parents[1])
    res = subprocess.run([sys.executable, "-c", _SCRIPT.format(root=root)], env={**os.environ, "TRITON_INTERPRET": "1"},
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-3000:]
