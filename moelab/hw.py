"""
Mesure des bandes passantes de la machine, pour la politique q* de FreeToken :

  B_PCIe : octets/s copiés de la RAM (épinglée) vers la VRAM
  B_Host : octets de poids d'experts traités par seconde quand on calcule un
           expert sur CPU (lecture en RAM + déquantification + produit matrice-vecteur)
  disque : octets/s en lisant des experts au hasard dans le store

FreeToken insiste : ces valeurs "ne se lisent pas sur une fiche technique", il
faut les mesurer sur la machine qui fait tourner le modèle.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .runtime import expert_ffn
from .store import ExpertStore, materialize

BANDWIDTH_FILE = "bandwidth.json"


def measure_pcie(n_bytes: int = 256 * 2**20, repeats: int = 5) -> float | None:
    if not torch.cuda.is_available():
        return None
    src = torch.empty(n_bytes, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return n_bytes / best


def measure_host_compute(store: ExpertStore, prec: str, n: int = 16, dtype=torch.float32, seed: int = 0) -> float:
    """Débit du calcul d'un expert sur CPU pour UN token (le cas de la génération)."""
    rng = random.Random(seed)
    picks = [(rng.randrange(len(store.layers)), rng.randrange(store.num_experts)) for _ in range(n)]
    blobs = [store.read(l, e, prec) for l, e in picks]  # en RAM avant de chronométrer
    x = torch.randn(1, store.hidden, dtype=dtype)
    expert_ffn(x, *materialize(store.layouts[prec], blobs[0], dtype), F.silu)
    t0 = time.perf_counter()
    for b in blobs:
        expert_ffn(x, *materialize(store.layouts[prec], b, dtype), F.silu)
    return store.record_bytes(prec) * n / (time.perf_counter() - t0)


def measure_disk(store: ExpertStore, prec: str, n: int = 32, seed: int = 1) -> float:
    """Lectures aléatoires d'experts. Attention : le cache du système peut embellir le résultat."""
    rng = random.Random(seed)
    buf = torch.empty(store.record_bytes(prec), dtype=torch.uint8)
    t0 = time.perf_counter()
    for _ in range(n):
        store.read(rng.randrange(len(store.layers)), rng.randrange(store.num_experts), prec, out=buf)
    return store.record_bytes(prec) * n / (time.perf_counter() - t0)


def measure_disk_parallel(store: ExpertStore, prec: str, workers: int, n: int = 64, seed: int = 2) -> float:
    """
    D2 : débit en lisant n experts au hasard avec `workers` lectures en vol. Un NVMe ne donne
    son plein débit qu'avec plusieurs lectures simultanées ; un HDD, lui, préfère 1 ou 2.
    Lectures sans cache du système quand c'est possible (sinon le résultat mesure la RAM).
    """
    from concurrent.futures import ThreadPoolExecutor

    rng = random.Random(seed + workers)
    picks = [(rng.randrange(len(store.layers)), rng.randrange(store.num_experts)) for _ in range(n)]
    rb = store.record_bytes(prec)
    bufs = []
    for _ in range(workers):
        raw = torch.empty(rb + 4096, dtype=torch.uint8)
        off = (-raw.data_ptr()) % 4096
        bufs.append(raw[off : off + rb])
    was = store.direct
    store.direct = True
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            t0 = time.perf_counter()
            list(ex.map(lambda i: store.read(*picks[i], prec, out=bufs[i % workers]), range(n)))
            dt = time.perf_counter() - t0
    finally:
        store.direct = was
    return rb * n / dt


def measure_read_split(store: ExpertStore, prec: str, parts: int, n: int = 32, seed: int = 3) -> float:
    """
    Débit d'experts lus UN PAR UN (le cas d'un expert manquant qu'on attend), chacun découpé
    en `parts` lectures simultanées. Lecture sans cache du système quand c'est possible.
    """
    rng = random.Random(seed + parts)
    rb = store.record_bytes(prec)
    raw = torch.empty(rb + 4096, dtype=torch.uint8)
    off = (-raw.data_ptr()) % 4096
    buf = raw[off:off + rb]
    was, was_parts = store.direct, store.read_parts
    store.direct, store.read_parts = True, parts
    try:
        t0 = time.perf_counter()
        for _ in range(n):
            store.read(rng.randrange(len(store.layers)), rng.randrange(store.num_experts), prec, out=buf)
        dt = time.perf_counter() - t0
    finally:
        store.direct, store.read_parts = was, was_parts
    return rb * n / dt


def measure_all(store: ExpertStore, log=print) -> dict:
    res = {"b_pcie": measure_pcie()}
    for p in store.precisions:
        res[f"b_host_{p}"] = measure_host_compute(store, p)
        res[f"disk_{p}"] = measure_disk(store, p)
    res["b_host"] = res[f"b_host_{store.hi}"]
    for k, v in res.items():
        log(f"  {k:<14} {'-' if v is None else f'{v / 1e9:.2f} Go/s'}")
    # D2 : combien de lectures en parallèle pour ce disque ?
    per_workers = {w: measure_disk_parallel(store, store.lo, w) for w in (1, 2, 4, 8, 16)}
    best = max(per_workers, key=per_workers.get)
    res["disk_parallel"] = {str(w): v for w, v in per_workers.items()}
    res["io_workers"] = best
    # Un expert attendu seul : en combien de morceaux simultanés le lire ?
    # Le choix se fait sur la courbe des lectures en parallèle (64 lectures, stable) : si le disque
    # gagne nettement à avoir plusieurs lectures en vol, un expert attendu seul est lu en autant
    # de morceaux. La mesure directe ci-dessous (32 experts un par un) est trop bruitée pour
    # décider (03/10 : 1 -> 0,70, 2 -> 1,05, 4 -> 0,41 Go/s, alors que le profil de Qwen3-Next
    # donne 6,8 tok/s en 1 morceau et 10,9 tok/s en 4) ; elle reste affichée pour information.
    per_parts = {k: measure_read_split(store, store.lo, k) for k in (1, 2, 4)}
    best_parts = min(best, 4) if per_workers[best] >= 1.2 * per_workers[1] else 1  # au-delà de 4 : rien mesuré de mieux
    res["read_split"] = {str(k): v for k, v in per_parts.items()}
    res["read_parts"] = best_parts
    log("  un expert lu en k morceaux simultanés (indicatif) : "
        + ", ".join(f"{k} -> {v / 1e9:.2f} Go/s" for k, v in per_parts.items())
        + f" ; retenu : {best_parts} (d'après les lectures en parallèle)")
    res["direct_io_supported"] = store._direct_error is None
    log("  lectures en parallèle (sans cache) : " + ", ".join(f"{w} -> {v / 1e9:.2f} Go/s" for w, v in per_workers.items())
        + f" ; retenu : {best}")
    if store._direct_error:
        log(f"  lecture directe impossible ici ({store._direct_error}) : mesures avec le cache du système")
    return res


def save_bandwidth(store_dir, res: dict) -> Path:
    path = Path(store_dir) / BANDWIDTH_FILE
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return path


def load_bandwidth(store_dir) -> dict | None:
    path = Path(store_dir) / BANDWIDTH_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
