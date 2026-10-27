"""
Modèles miniatures aléatoires pour les tests (quelques centaines de Ko) : même code
transformers que les vrais OLMoE / Qwen3-Next, mêmes formats de checkpoint sur
disque, mais des dimensions minuscules. Aucun téléchargement nécessaire.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers.utils import logging as hf_logging  # noqa: E402

from moelab.testing import make_model_dir, reference_logits, tiny_olmoe_config, tiny_qwen3next_config  # noqa: E402,F401

hf_logging.set_verbosity_error()


def to_fused_checkpoint(src: Path, dst: Path) -> Path:
    """Réécrit un checkpoint par expert au format fusionné (E, 2I, H) / (E, H, I)."""
    dst.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for f in src.glob("*.safetensors"):
        tensors.update(load_file(str(f)))
    fused: dict[str, dict] = {}
    out = {}
    for k, v in tensors.items():
        if ".mlp.experts." in k:
            prefix, rest = k.split(".mlp.experts.")
            e, proj, _ = rest.split(".")
            fused.setdefault(prefix, {}).setdefault(proj, {})[int(e)] = v
        else:
            out[k] = v
    for prefix, projs in fused.items():
        n = len(projs["gate_proj"])
        out[f"{prefix}.mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat([projs["gate_proj"][e], projs["up_proj"][e]], 0) for e in range(n)])
        out[f"{prefix}.mlp.experts.down_proj"] = torch.stack([projs["down_proj"][e] for e in range(n)])
    save_file(out, str(dst / "model.safetensors"))
    for f in src.glob("*.json"):
        (dst / f.name).write_text(f.read_text())
    return dst


def to_fp8_checkpoint(src: Path, dst: Path, block=(16, 16)) -> Path:
    """Réécrit les poids 2D en FP8 par blocs, comme les checkpoints Qwen *-FP8 (avec des blocs réduits)."""
    dst.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for f in src.glob("*.safetensors"):
        tensors.update(load_file(str(f)))
    out = {}
    for k, v in tensors.items():
        if v.dim() == 2 and k.endswith("proj.weight") and v.shape[0] % block[0] == 0 and v.shape[1] % block[1] == 0:
            o, i = v.shape
            blocks = v.reshape(o // block[0], block[0], i // block[1], block[1])
            scale = blocks.abs().amax(dim=(1, 3)) / 448.0
            q = (blocks / scale[:, None, :, None]).reshape(o, i).to(torch.float8_e4m3fn)
            out[k] = q
            out[k + "_scale_inv"] = scale.float()
        else:
            out[k] = v
    save_file(out, str(dst / "model.safetensors"))
    for f in src.glob("*.json"):
        (dst / f.name).write_text(f.read_text())
    cfg = json.loads((dst / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "fp8", "weight_block_size": list(block)}
    (dst / "config.json").write_text(json.dumps(cfg))
    return dst


@pytest.fixture(scope="session")
def olmoe_dir(tmp_path_factory):
    return make_model_dir(tmp_path_factory.mktemp("olmoe"), tiny_olmoe_config())


@pytest.fixture(scope="session")
def qnext_dir(tmp_path_factory):
    return make_model_dir(tmp_path_factory.mktemp("qnext"), tiny_qwen3next_config(), seed=1)


@pytest.fixture(scope="session")
def olmoe_store(olmoe_dir, tmp_path_factory):
    from moelab.store import build_store

    return build_store(olmoe_dir, tmp_path_factory.mktemp("store_olmoe"), ("fp32", "int4"), log=lambda *a: None)

