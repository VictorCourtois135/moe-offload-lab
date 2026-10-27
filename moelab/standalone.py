"""
Store autonome (05/10) : la partie dense du modèle copiée dans le store.

Jusqu'ici, moelab relisait la partie dense (attention, embeddings, tête, experts partagés)
dans le checkpoint téléchargé à chaque lancement : il fallait garder ce checkpoint, alors
que les experts, 95 % de sa taille, n'y servaient plus. Qwen3-Next : 82 Go gardés pour
~2,5 Go utiles.

`make_standalone(model_dir, store_dir)` copie dans <store>/model :
  - les tenseurs qui ne sont pas des experts, tels quels (FP8 et leurs `_scale_inv` compris :
    la lecture fait la même déquantification), en fichiers de 1 Go au plus ;
  - config.json, generation_config.json et les fichiers du tokenizer.
Ensuite <store>/model est un dossier de modèle complet (sans ses experts, qui sont dans le
store) : `--model` devient facultatif, et le checkpoint téléchargé peut être supprimé.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

SHARD_BYTES = 1 << 30
SMALL_FILE = 64 << 20  # fichiers annexes copiés : tokenizer, configs (pas les poids)


def is_expert_key(key: str) -> bool:
    return ".experts." in key


def standalone_dir(store_dir) -> Path:
    return Path(store_dir) / "model"


def is_standalone(store_dir) -> bool:
    d = standalone_dir(store_dir)
    return (d / "config.json").exists() and (d / "model.safetensors.index.json").exists()


def resolve_model_dir(model_dir, store_dir) -> Path:
    """Le dossier du modèle : celui donné, sinon la partie dense du store autonome."""
    if model_dir:
        return Path(model_dir)
    if is_standalone(store_dir):
        return standalone_dir(store_dir)
    raise SystemExit(f"--model manquant, et le store {store_dir} n'est pas autonome "
                     "(scripts\\make_standalone.py --model ... --store ... pour qu'il le devienne)")


def dense_bytes(model_dir) -> int:
    from .checkpoint import Checkpoint

    ckpt = Checkpoint(model_dir)
    total = 0
    for k in ckpt.keys():
        if not is_expert_key(k):
            h = ckpt._handle(ckpt.key_to_file[k]).header[k]
            total += h["data_offsets"][1] - h["data_offsets"][0]
    ckpt.close()
    return total


def make_standalone(model_dir, store_dir, log=print) -> Path:
    from safetensors.torch import save_file

    from .checkpoint import Checkpoint

    src, out = Path(model_dir), standalone_dir(store_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(src)
    keys = sorted(k for k in ckpt.keys() if not is_expert_key(k))
    weight_map, shard, size, n, total = {}, {}, 0, 0, 0

    def flush():
        nonlocal shard, size, n
        if not shard:
            return
        n += 1
        name = f"dense-{n:05d}.safetensors"
        save_file(shard, str(out / name), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = name
        log(f"  {name} : {len(shard)} tenseurs, {size / 2**20:.0f} Mio")
        shard, size = {}, 0

    for k in keys:
        t = ckpt.get_raw(k).contiguous()
        b = t.numel() * t.element_size()
        if shard and size + b > SHARD_BYTES:
            flush()
        shard[k] = t
        size += b
        total += b
    flush()
    ckpt.close()
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total, "moelab_standalone": True}, "weight_map": weight_map}, indent=1),
        encoding="utf-8")
    copied = []
    for f in src.iterdir():
        if (f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json"
                and f.stat().st_size <= SMALL_FILE):
            shutil.copy2(f, out / f.name)
            copied.append(f.name)
    man_path = Path(store_dir) / "manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["standalone"] = "model"
        man_path.write_text(json.dumps(man, indent=1), encoding="utf-8")
    log(f"  store autonome : partie dense {total / 2**30:.2f} Gio + {len(copied)} fichiers ({', '.join(sorted(copied)[:6])}"
        + (" ..." if len(copied) > 6 else "") + f") dans {out}")
    return out
