"""
Store autonome (05/10) : la partie dense copiée dans le store ; le modèle se charge sans
le checkpoint d'origine (supprimé dans le test), avec les mêmes logits. FP8 compris.
"""

from __future__ import annotations

import shutil

import torch

from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
from moelab.standalone import dense_bytes, is_standalone, make_standalone, resolve_model_dir, standalone_dir
from moelab.store import build_store
from moelab.testing import make_model_dir, make_tiny_tokenizer, reference_logits, tiny_qwen3next_config


def _logits(model_dir, store, ids):
    model, rt = load_offloaded_model(model_dir, store, RuntimeConfig(policy=PrecisionPolicy("hi"), host_budget_bytes=2**26,
                                                                     fast_budget_bytes=0), dense="cpu",
                                     dtype=torch.float32, log=lambda *a: None)
    with torch.no_grad():
        out = model(ids).logits
    rt.close()
    return out


def test_standalone_store_runs_without_checkpoint(tmp_path):
    from conftest import to_fp8_checkpoint

    md = make_model_dir(tmp_path / "m", tiny_qwen3next_config())
    fp8 = to_fp8_checkpoint(md, tmp_path / "fp8")
    make_tiny_tokenizer(fp8)
    st = build_store(fp8, tmp_path / "s", ("fp32", "int4"), log=lambda *a: None)
    ids = torch.randint(2, 128, (1, 7), generator=torch.Generator().manual_seed(1))
    ref = _logits(fp8, st, ids)
    assert not is_standalone(st)
    n = dense_bytes(fp8)
    out = make_standalone(fp8, st, log=lambda *a: None)
    assert is_standalone(st) and out == standalone_dir(st)
    assert (out / "tokenizer.json").exists() and (out / "config.json").exists()
    assert sum(f.stat().st_size for f in out.glob("dense-*.safetensors")) >= n
    shutil.rmtree(fp8)  # le checkpoint d'origine n'existe plus
    md2 = resolve_model_dir(None, st)
    assert torch.allclose(_logits(md2, st, ids), ref, atol=1e-5, rtol=1e-4)
    from transformers import AutoTokenizer

    assert AutoTokenizer.from_pretrained(md2) is not None
    _ = reference_logits  # (référence transformers déjà vérifiée par test_qwen3next)
