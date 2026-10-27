"""
Auto-dimensionnement (idée de Colibrì `coli plan`) : combien de VRAM et de RAM
donner aux caches d'experts, d'après la mémoire réellement libre sur la machine.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import psutil
import torch

from .store import PRECISIONS, ExpertStore, record_layout


def ram_available() -> int:
    return psutil.virtual_memory().available


def vram_free() -> int:
    return torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0


# Réserve VRAM automatique (mesures du 03/10, Qwen3-Next) : les activations d'un token
# et du prefill tiennent dans ~0,55 Gio ; le reste de la réserve, c'est le cache KV
# statique des CUDA graphs, dont la taille dépend du modèle. Réserve fixe de 1 Gio
# avant : Qwen3-Next (cache KV de 48 Mio à 2 048 tokens) perdait ~0,4 Gio de cache
# d'experts — 0,6 Gio de réserve : 5,93 -> 6,81 tok/s.
ACTIVATIONS_RESERVE = int(0.55 * 2**30)
KV_TARGET_BYTES = 256 * 2**20          # cache KV visé pour choisir la longueur max automatique
GRAPH_LENS = (8192, 4096, 2048)


def kv_bytes_per_token(config, dtype_bytes: int = 2) -> int:
    """Octets de cache KV par token : seules les couches à attention complète en ont un
    (Qwen3-Next : 12 sur 48, les autres ont un état de taille fixe)."""
    n = int(config.num_hidden_layers)
    types = getattr(config, "layer_types", None)
    if types:
        n_full = sum(1 for t in types if t in ("full_attention", "attention", "sliding_attention"))
    elif getattr(config, "full_attention_interval", None):
        n_full = n // int(config.full_attention_interval)
    else:
        n_full = n
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", None) or heads)
    head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // heads)
    return n_full * 2 * kv_heads * head_dim * dtype_bytes


def auto_graph_max_len(config) -> int:
    """La plus grande longueur (8 192, 4 096 ou 2 048) dont le cache KV tient dans
    KV_TARGET_BYTES. OLMoE : 2 048 (256 Mio) ; Qwen3-Next : 8 192 (192 Mio)."""
    per_tok = kv_bytes_per_token(config)
    cap = int(getattr(config, "max_position_embeddings", 0) or 1 << 30)
    for n in GRAPH_LENS:
        if n * per_tok <= KV_TARGET_BYTES and n <= cap:
            return n
    return min(GRAPH_LENS[-1], cap)


def auto_vram_reserve(config, max_len: int) -> int:
    """Réserve VRAM (octets) : activations + cache KV statique de max_len tokens."""
    return ACTIVATIONS_RESERVE + max_len * kv_bytes_per_token(config)


# Avec un GPU, le cache RAM est en mémoire épinglée : Windows ne peut ni la déplacer
# ni l'écrire dans le fichier d'échange. Mesuré sur la machine de référence (16 Go) :
# au-delà de ~7 Gio épinglés, Windows refuse d'en donner plus et le reste du système
# se met à paginer (TTFT de 30 à 50 s par moments). D'où un plafond en part de la RAM totale.
PINNED_MAX_FRACTION = 0.35


def auto_budgets(cfg, main_device: torch.device, vram_reserve_gb: float = 1.0, ram_reserve_gb: float = 3.0) -> None:
    """
    Remplit cfg.fast_budget_bytes / cfg.host_budget_bytes s'ils valent None.
    VRAM : ce qui est libre après la partie dense, moins une réserve pour les
    activations et le cache KV. RAM : ce qui est disponible, moins une réserve pour
    Windows et les autres programmes, et avec un GPU au plus PINNED_MAX_FRACTION de
    la RAM totale (mémoire épinglée).
    """
    if cfg.host_budget_bytes is None:
        budget = ram_available() - int(ram_reserve_gb * 2**30)
        if main_device.type == "cuda":
            budget = min(budget, int(PINNED_MAX_FRACTION * psutil.virtual_memory().total))
        cfg.host_budget_bytes = max(256 * 2**20, budget)
    if cfg.fast_budget_bytes is None:
        if main_device.type == "cuda":
            cfg.fast_budget_bytes = max(0, vram_free() - int(vram_reserve_gb * 2**30))
        else:
            cfg.fast_budget_bytes = 0


def estimate_store(info: dict, precisions, group_size: int = 64) -> dict[str, int]:
    """Taille de chaque fichier de précision d'un store, avant de le construire."""
    n = info["num_experts"] * len(info["layers"])
    return {p: record_layout(p, info["hidden"], info["inter"], group_size)["record_bytes"] * n for p in precisions}


def disk_free(path: str | Path) -> int:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def plan_report(store: ExpertStore, fast_budget: int, host_budget: int) -> list[str]:
    """Ce que les budgets permettent de garder en VRAM / RAM, en nombre d'experts."""
    lines = []
    n_total = store.num_experts * len(store.layers)
    for name, budget in (("VRAM", fast_budget), ("RAM", host_budget)):
        if budget <= 0:
            lines.append(f"  {name} : pas de cache d'experts")
            continue
        parts = []
        for p in store.precisions:
            n = min(n_total, budget // store.record_bytes(p))
            parts.append(f"{n} experts {p} ({100 * n / n_total:.0f}%)")
        lines.append(f"  {name} : {budget / 2**30:.2f} Gio -> jusqu'à " + " ou ".join(parts))
    lo_total = store.total_bytes(store.lo)
    share = lo_total / max(host_budget, 1)
    if preload_advised(lo_total, host_budget):
        verdict = f"{share:.0%} du budget RAM -> préchargé automatiquement au lancement"
    elif share <= 0.95:
        verdict = (f"tient en RAM mais en prendrait {share:.0%} (peu de place pour la haute précision) : "
                   "pas préchargé par défaut, --preload-lo yes pour forcer")
    else:
        verdict = "ne tient pas en RAM, une partie viendra du disque"
    lines.append(f"  pool {store.lo} complet : {lo_total / 2**30:.2f} Gio -> {verdict}")
    return lines


def preload_advised(lo_total: int, host_budget: int) -> bool:
    """Règle de --preload-lo auto : le pool basse précision ne doit pas prendre plus de 60% du budget RAM."""
    return lo_total <= 0.6 * host_budget


def bits_label(prec: str) -> str:
    spec = PRECISIONS[prec]
    return spec["dtype"] if spec["kind"] == "float" else f"int{spec['bits']}"
