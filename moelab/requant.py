"""
Q1 / Q2 (02/10) : requantifier la basse précision (int4) d'un store existant, en place.

Le format des records ne change pas (mêmes entiers, une échelle float16 par groupe de
64) : seuls les nombres écrits changent. Les noyaux, le runtime, les caches : rien ne
bouge. Seuls les fichiers layerXXX.int4.bin sont réécrits, une couche à la fois
(il suffit d'une couche de place libre en plus sur le disque).

  - "mse"  : meilleure échelle par groupe (voir quant.py), sans calibration ;
  - "gptq" : GPTQ expert par expert, à partir des vraies entrées de chaque expert
             (voir `collect_calibration`). Chaque expert a sa Hessienne
             H = somme des (w_t x_t)(w_t x_t)ᵀ sur les tokens qui l'ont choisi,
             pondérés par le poids du routeur w_t : la sortie de l'expert est
             multipliée par w_t, son erreur aussi. Un expert vu par trop peu de
             tokens garde la méthode "mse".

Les poids d'origine sont relus dans le checkpoint Hugging Face (pas dans le store :
on ne requantifie pas des poids déjà arrondis).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .quant import gptq_quantize
from .store import ExpertStore, encode_record, json_ident

# Débuts de textes variés pour l'auto-calibration (le modèle écrit lui-même son texte de
# calibration, cf. Williams et al. 2025, « Self-calibration for language model
# quantization and pruning ») ; aucun ne vient de bench/ (le texte d'évaluation de la qualité).
SEEDS = [
    "Explique comment fonctionne un moteur à combustion interne.",
    "Write a Python function that parses a CSV file and computes column statistics.",
    "Raconte l'histoire d'une petite ville au bord de la mer au XIXe siècle.",
    "What are the main causes of inflation, and how do central banks respond?",
    "Résous pas à pas : un train part à 14 h à 80 km/h, un autre à 15 h à 110 km/h...",
    "Describe the life cycle of a star, from nebula to remnant.",
    "Donne une recette détaillée de lasagnes végétariennes.",
    "Summarize the key ideas of object-oriented programming with examples in Java.",
    "Quels sont les avantages et inconvénients du télétravail pour une entreprise ?",
    "Write a short dialogue between a doctor and a patient about sleep problems.",
    "Explain the difference between TCP and UDP, with use cases.",
    "Écris une lettre de motivation pour un poste de développeur junior.",
    "How does photosynthesis work at the molecular level?",
    "Liste les étapes pour créer une API REST avec FastAPI et une base PostgreSQL.",
    "Translate into English and explain the grammar: « Il aurait fallu qu'on le sût plus tôt. »",
    "Prove that the square root of 2 is irrational.",
]


# ----------------------------------------------------------------------------
# Calibration : texte, puis entrées de chaque couche MoE
# ----------------------------------------------------------------------------
@torch.no_grad()
def self_calibration_ids(model, tok, n_tokens: int, seq_len: int = 512, temperature: float = 0.8, top_p: float = 0.95,
                         log=print) -> list[torch.Tensor]:
    """Le modèle écrit son propre texte de calibration, à partir de débuts variés."""
    from .bench import chat_ids
    from .generate import generate

    dev = model.moelab_device
    out, total, i = [], 0, 0
    torch.manual_seed(0)
    t0 = time.time()
    while total < n_tokens:
        ids = chat_ids(tok, SEEDS[i % len(SEEDS)], dev)
        new = generate(model, ids, seq_len - ids.shape[1], temperature=temperature, top_p=top_p, eos_token_id=[])
        seq = torch.cat([ids[0].cpu(), torch.tensor(new)])[:seq_len]
        out.append(seq.unsqueeze(0))
        total += seq.numel()
        i += 1
        log(f"  calibration : {total}/{n_tokens} tokens générés ({time.time() - t0:.0f} s)")
    return out


def file_calibration_ids(tok, paths: list[str], n_tokens: int, seq_len: int = 512) -> list[torch.Tensor]:
    """Texte de calibration lu dans des fichiers (.txt, .md, .jsonl avec un champ "text")."""
    texts = []
    for p in paths:
        raw = Path(p).read_text(encoding="utf-8", errors="ignore")
        if p.endswith(".jsonl"):
            texts += [json.loads(l).get("text", "") for l in raw.splitlines() if l.strip()]
        else:
            texts.append(raw)
    ids = tok("\n\n".join(texts), return_tensors="pt").input_ids[0][:n_tokens]
    return [ids[i:i + seq_len].unsqueeze(0) for i in range(0, len(ids), seq_len) if len(ids[i:i + seq_len]) > 16]


@torch.no_grad()
def collect_calibration(model, chunks: list[torch.Tensor], work_dir: str | Path, log=print) -> dict:
    """
    Fait passer le texte de calibration (passe normale, prefill) et enregistre, pour
    chaque couche MoE, ses entrées x et le routage (experts choisis, poids) sur le disque
    (tableaux numpy mappés : la RAM ne dépend pas du nombre de tokens).
    """
    rt = model.moelab_runtime
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    n = sum(c.shape[1] for c in chunks)
    L, H = len(rt.store.layers), rt.store.hidden
    k = None
    arrays: dict = {}
    pos = {"i": 0}

    def hook(layer, x, idx, w):
        nonlocal k
        if k is None:
            k = idx.shape[-1]
            for l in range(L):
                arrays[l] = (
                    np.lib.format.open_memmap(work / f"x_{l:03d}.npy", "w+", np.float16, (n, H)),
                    np.lib.format.open_memmap(work / f"idx_{l:03d}.npy", "w+", np.int32, (n, k)),
                    np.lib.format.open_memmap(work / f"w_{l:03d}.npy", "w+", np.float32, (n, k)),
                )
        xa, ia, wa = arrays[layer]
        T = x.shape[0]
        s = pos["i"]
        xa[s:s + T] = x.detach().to("cpu", torch.float16).numpy()
        ia[s:s + T] = idx.detach().to("cpu", torch.int32).numpy()
        wa[s:s + T] = w.detach().to("cpu", torch.float32).numpy()

    rt.calib_hook = hook
    t0 = time.time()
    try:
        for j, c in enumerate(chunks):
            model(input_ids=c.to(model.moelab_device), use_cache=False)
            pos["i"] += c.shape[1]
            if j % 8 == 0 or j == len(chunks) - 1:
                log(f"  entrées des couches MoE : {pos['i']}/{n} tokens ({time.time() - t0:.0f} s)")
    finally:
        rt.calib_hook = None
    for a in arrays.values():
        for m in a:
            m.flush()
    meta = {"tokens": n, "k": k, "layers": L, "hidden": H, "store_id": rt.store.store_id}
    (work / "calib.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


# ----------------------------------------------------------------------------
# Réécriture de l'int4 du store
# ----------------------------------------------------------------------------
def _expert_hessians(X, idx, w, e, gate_up, down, min_tokens):
    """(H_gate_up, H_down) de l'expert e, pondérées par le poids du routeur ; None si trop peu de tokens."""
    rows, slot = (idx == e).nonzero(as_tuple=True)
    if rows.numel() < min_tokens:
        return None
    we = w[rows, slot].unsqueeze(1)
    xe = X[rows]
    xw = xe * we
    h_gu = xw.T @ xw
    g, u = (xe @ gate_up.T).chunk(2, dim=-1)
    aw = (F.silu(g) * u) * we
    h_dn = aw.T @ aw
    return h_gu, h_dn


def requantize_store(store_dir, model_dir, method: str = "mse", calib_dir=None, min_tokens: int = 32,
                     device: str | None = None, log=print) -> dict:
    """Réécrit les records int4 du store avec `method` ("mse" ou "gptq"). Reprenable."""
    from .checkpoint import Checkpoint

    if method not in ("mse", "gptq", "absmax"):
        raise ValueError(f"méthode inconnue : {method}")
    store_dir = Path(store_dir)
    store = ExpertStore(store_dir)
    if "int4" not in store.precisions:
        raise ValueError(f"{store_dir} n'a pas de précision int4")
    manifest = store.manifest
    old_id = store.store_id
    store.close()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lay = dict(next(l for l in manifest["layouts"] if l["precision"] == "int4"))
    lay["scale"] = method
    L, E, H, I = len(manifest["layers"]), manifest["num_experts"], manifest["hidden"], manifest["inter"]
    gs = lay["group_sizes"]

    calib = None
    if method == "gptq":
        if calib_dir is None:
            raise ValueError("gptq : il faut un dossier de calibration (collect_calibration)")
        calib = Path(calib_dir)
        meta = json.loads((calib / "calib.json").read_text(encoding="utf-8"))
        if meta["layers"] != L or meta["hidden"] != H:
            raise ValueError("la calibration ne correspond pas à ce store")

    prog_path = store_dir / "progress_requant.json"
    done: set[int] = set()
    if prog_path.exists():
        prog = json.loads(prog_path.read_text(encoding="utf-8"))
        if prog.get("method") == method:
            done = set(prog["done"])
            if done:
                log(f"  reprise : {len(done)} couche(s) déjà réécrites")
    ckpt = Checkpoint(model_dir)
    stats = {"gptq": 0, "fallback": 0}
    t_start = time.time()
    todo = [l for l in range(L) if l not in done]
    for n_done, li in enumerate(todo):
        prefix = manifest["layers"][li]
        if calib is not None:
            X = torch.from_numpy(np.array(np.load(calib / f"x_{li:03d}.npy", mmap_mode="r"))).to(dev, torch.float32)
            idx = torch.from_numpy(np.load(calib / f"idx_{li:03d}.npy")).to(dev)
            w = torch.from_numpy(np.load(calib / f"w_{li:03d}.npy")).to(dev)
        path = store_dir / f"layer{li:03d}.int4.bin"
        tmp = path.with_suffix(".bin.new")
        with open(tmp, "wb") as f:
            f.truncate(lay["record_bytes"] * E)
            for e in range(E):
                gate_up, down = ckpt.read_expert(prefix, e, H, I)
                gate_up, down = gate_up.to(dev), down.to(dev)
                quantized = None
                if calib is not None:
                    hs = _expert_hessians(X, idx, w, e, gate_up, down, min_tokens)
                    if hs is None:
                        stats["fallback"] += 1
                    else:
                        quantized = {"gate_up": gptq_quantize(gate_up, hs[0], 4, gs["gate_up"]),
                                     "down": gptq_quantize(down, hs[1], 4, gs["down"])}
                        stats["gptq"] += 1
                rec_lay = lay if quantized is not None or method != "gptq" else {**lay, "scale": "mse"}
                f.seek(e * lay["record_bytes"])
                f.write(encode_record(rec_lay, gate_up, down, quantized=quantized).tobytes())
        os.replace(tmp, path)
        done.add(li)
        prog_path.write_text(json.dumps({"method": method, "done": sorted(done)}), encoding="utf-8")
        el = time.time() - t_start
        log(f"  couche {li + 1}/{L} réécrite ({method}) — {el:.0f} s, reste ~{el / (n_done + 1) * (len(todo) - n_done - 1):.0f} s")
    ckpt.close()

    # manifest : nouvelle méthode d'échelle, donc nouvel identifiant de store
    manifest["layouts"] = [lay if l["precision"] == "int4" else l for l in manifest["layouts"]]
    info = {"model_type": manifest["model_type"], "layers": manifest["layers"]}
    import hashlib

    manifest["store_id"] = hashlib.sha1(json_ident(info, E, H, I, manifest["layouts"]).encode()).hexdigest()[:12]
    (store_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    prog_path.unlink(missing_ok=True)
    # l'historique d'usage reste valable (mêmes experts) : on le rattache au nouvel identifiant
    usage = store_dir / "usage.json"
    if usage.exists():
        data = json.loads(usage.read_text(encoding="utf-8"))
        if data.get("store_id") == old_id:
            data["store_id"] = manifest["store_id"]
            usage.write_text(json.dumps(data), encoding="utf-8")
    stats["store_id"] = manifest["store_id"]
    stats["seconds"] = time.time() - t_start
    return stats
