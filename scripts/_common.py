"""Arguments et chargement communs aux scripts (à lancer depuis la racine du repo : python scripts\\xxx.py ...)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model  # noqa: E402
from moelab.hw import load_bandwidth  # noqa: E402
from moelab.plan import preload_advised  # noqa: E402
from moelab.store import ExpertStore  # noqa: E402

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
GIB = 2**30


def add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("modèle et mémoire")
    g.add_argument("--model", default=None,
                   help="dossier du checkpoint Hugging Face (facultatif si le store est autonome : make_standalone.py)")
    g.add_argument("--store", required=True, help="dossier du store d'experts (créé par build_store.py)")
    g.add_argument("--dense", default="auto", choices=["auto", "gpu", "cpu", "split"],
                   help="où mettre la partie dense (défaut : auto)")
    g.add_argument("--dtype", default="float16", choices=list(DTYPES), help="type de calcul (défaut : float16)")
    g.add_argument("--vram-gb", type=float, default=None, help="VRAM pour les experts (défaut : auto)")
    g.add_argument("--ram-gb", type=float, default=None, help="RAM pour les experts (défaut : auto)")
    g.add_argument("--vram-reserve-gb", type=float, default=None,
                   help="VRAM laissée aux activations et au cache KV (défaut : auto, 0,55 Gio + cache KV à --graph-max-len)")
    g.add_argument("--ram-reserve-gb", type=float, default=3.0, help="RAM laissée à Windows et aux autres programmes")
    g.add_argument("--preload-lo", default="auto", choices=["auto", "yes", "no"],
                   help="précharger tout le pool basse précision en RAM (auto : si ça prend <= 60%% du budget RAM)")
    g.add_argument("--dense-int8", action="store_true", default=True,
                   help="D5 : partie dense (attention, tête de sortie) en int8 — actif par défaut depuis le 02/10")
    g.add_argument("--no-dense-int8", dest="dense_int8", action="store_false",
                   help="D5 désactivé : partie dense en fp16 (référence de qualité, ou comparaison)")
    g.add_argument("--no-prefetch", action="store_true", help="désactiver le prefetch de la couche suivante")
    g.add_argument("--no-qstar", action="store_true", help="désactiver q* (tous les experts manquants vont en VRAM)")
    g.add_argument("--usage", default=None, help="historique d'usage (défaut : <store>/usage.json)")
    o = p.add_argument_group("optimisations (doc 18)")
    o.add_argument("--no-fast-decode", action="store_true", help="désactiver A2 (décisions de génération allégées)")
    o.add_argument("--no-prefetch-adaptive", action="store_true", help="prefetch même quand tout est résident")
    o.add_argument("--prefetch-depth", type=int, default=1, help="C2 : couches préchargées à l'avance (1 ou 2)")
    o.add_argument("--prefetch-disk-ahead", type=int, default=0,
                   help="lectures disque -> RAM lancées pour les couches L+2.. (0 = non ; utile quand le store ne tient pas en RAM)")
    o.add_argument("--no-direct-io", action="store_true", help="D1 désactivé : lectures disque via le cache du système")
    o.add_argument("--no-sort-reads", action="store_true", help="D3 : ne pas trier les lectures dans l'ordre du fichier")
    o.add_argument("--io-workers", type=int, default=None, help="D2 : lectures en parallèle (défaut : mesuré, sinon 4)")
    o.add_argument("--read-parts", type=int, default=None,
                   help="morceaux simultanés par expert lu sur disque (défaut : mesuré par measure_bandwidth.py, sinon 1)")
    o.add_argument("--int4-unpacked", action="store_true", help="B2 : int4 rangé déballé (int8) dans la VRAM")
    o.add_argument("--no-triton", action="store_true", help="désactiver les noyaux Triton (B1)")
    o.add_argument("--no-grouped", action="store_true", help="A4+B1 désactivé : un expert à la fois (sans noyaux groupés)")
    o.add_argument("--no-cuda-graphs", action="store_true", help="A6 désactivé : partie dense sans CUDA graphs (cache KV dynamique)")
    o.add_argument("--graph-max-len", type=int, default=None,
                   help="A6 : taille du cache KV statique, prompt + réponse (défaut : auto, OLMoE 2048, Qwen3-Next 8192)")
    o.add_argument("--hot-store", default=None, help="D4 : store chaud sur un disque rapide (build_hot_store.py)")
    o.add_argument("--no-full-graph", action="store_true",
                   help="A5 désactivé : pas de token entier en un graph (actif par défaut, décisions sur le GPU)")
    o.add_argument("--cache-eviction", default="lrfu", choices=["lrfu", "stale"], help="S3 : éviction du cache VRAM")
    o.add_argument("--prefetch-mass", type=float, default=None,
                   help="S3 : prefetch des experts prédits jusqu'à cette masse de probabilité (ex. 0.9)")


def add_policy_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("précision (HOBBIT)")
    g.add_argument("--policy", default="hobbit", choices=["hi", "lo", "hobbit"],
                   help="hi = tout en haute précision, lo = tout en basse, hobbit = selon l'importance (défaut)")
    g.add_argument("--t1", type=float, default=0.6, help="seuil HOBBIT haute/basse précision (défaut 0.6)")
    g.add_argument("--t2", type=float, default=None, help="seuil de saut d'expert (défaut : jamais)")
    g.add_argument("--protect", default="", help="couches toujours en haute précision : '0,1,15' ou 'edges'")
    g.add_argument("--hi-vote", type=float, default=0.5, help="part des tokens qui doivent demander la haute précision")
    g.add_argument("--t1-layers", default=None, help="E1 : seuils t1 par couche ('auto' = <store>/t1_layers.json, ou un chemin)")
    g.add_argument("--t-mid", type=float, default=None, help="E2 : seuil du niveau intermédiaire (store à 3 précisions)")
    g.add_argument("--miss-t1", type=float, default=None, help="E3 : seuil t1 appliqué aux experts absents de la VRAM")


def parse_protect(spec: str, n_layers: int) -> frozenset[int]:
    if not spec:
        return frozenset()
    if spec == "edges":
        return frozenset({0, 1, n_layers - 1})
    return frozenset(int(x) for x in spec.split(","))


def load_t1_layers(spec: str | None, store_dir) -> dict[int, float] | None:
    if not spec:
        return None
    import json

    path = Path(store_dir) / "t1_layers.json" if spec == "auto" else Path(spec)
    if not path.exists():
        raise SystemExit(f"{path} introuvable : lance d'abord scripts\\measure_routing.py")
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8"))["t1_layers"].items()}


def policy_from_args(args, n_layers: int) -> PrecisionPolicy:
    return PrecisionPolicy(args.policy, t1=args.t1, t2=args.t2, protect_layers=parse_protect(args.protect, n_layers),
                           hi_vote=args.hi_vote, t_mid=getattr(args, "t_mid", None),
                           t1_layers=load_t1_layers(getattr(args, "t1_layers", None), args.store),
                           miss_t1=getattr(args, "miss_t1", None))


def load(args, log=print):
    """Charge modèle + runtime + tokenizer d'après les arguments, avec préchargement."""
    from transformers import AutoTokenizer

    from moelab.standalone import resolve_model_dir

    args.model = str(resolve_model_dir(args.model, args.store))

    store = ExpertStore(args.store)
    log(store.summary())
    bw = load_bandwidth(args.store) or {}
    cfg = RuntimeConfig(
        policy=policy_from_args(args, len(store.layers)) if hasattr(args, "policy") else PrecisionPolicy("hi"),
        fast_budget_bytes=None if args.vram_gb is None else int(args.vram_gb * GIB),
        host_budget_bytes=None if args.ram_gb is None else int(args.ram_gb * GIB),
        use_qstar=not args.no_qstar,
        prefetch=not args.no_prefetch,
        usage_path=args.usage or str(Path(args.store) / "usage.json"),
        fast_decode=not args.no_fast_decode,
        prefetch_adaptive=not args.no_prefetch_adaptive,
        prefetch_depth=args.prefetch_depth,
        prefetch_disk_ahead=getattr(args, "prefetch_disk_ahead", 0),
        direct_io=not args.no_direct_io,
        sort_reads=not args.no_sort_reads,
        io_workers=args.io_workers or int(bw.get("io_workers", 4)),
        read_parts=args.read_parts or int(bw.get("read_parts", 1)),
        hot_store=args.hot_store,
        fast_int4_unpacked=args.int4_unpacked,
        triton_int4=not args.no_triton,
        grouped_experts=not args.no_grouped,
        cuda_graphs=not args.no_cuda_graphs,
        graph_max_len=args.graph_max_len,
        full_graph=not getattr(args, "no_full_graph", False),
        cache_eviction=getattr(args, "cache_eviction", "lrfu"),
        prefetch_mass=getattr(args, "prefetch_mass", None),
    )
    if bw.get("b_pcie") and bw.get("b_host"):
        cfg.b_pcie, cfg.b_host = bw["b_pcie"], bw["b_host"]
    else:
        log("  bandwidth.json absent : q* utilise des valeurs par défaut (lance scripts\\measure_bandwidth.py)")
    model, rt = load_offloaded_model(args.model, args.store, cfg, dense=args.dense, dtype=DTYPES[args.dtype],
                                     vram_reserve_gb=args.vram_reserve_gb, ram_reserve_gb=args.ram_reserve_gb, log=log,
                                     dense_int8=getattr(args, "dense_int8", False))
    lo_total = store.total_bytes(store.lo)
    preload = {"yes": True, "no": False}.get(args.preload_lo, preload_advised(lo_total, rt.cfg.host_budget_bytes))
    log(f"  budgets : VRAM {(rt.cfg.fast_budget_bytes or 0) / GIB:.2f} Gio, RAM {rt.cfg.host_budget_bytes / GIB:.2f} Gio ; "
        f"q* avec B_PCIe={rt.cfg.b_pcie / 1e9:.1f} Go/s, B_Host={rt.cfg.b_host / 1e9:.1f} Go/s")
    rt.warm_start(preload_lo=preload, log=log)
    tok = AutoTokenizer.from_pretrained(args.model)
    return model, rt, tok


def model_tag(args) -> str:
    """Nom court du modèle pour les fichiers de résultats (store autonome : le nom du store)."""
    p = Path(args.model)
    return Path(args.store).name if p.name == "model" and p.parent.resolve() == Path(args.store).resolve() else p.name


def stats_line(rt, n_tokens: int, seconds: float) -> str:
    s = rt.stats.as_dict()
    parts = [f"{n_tokens / seconds:.2f} tok/s" if seconds > 0 else "-"]
    if rt.fast is not None:
        parts.append(f"VRAM hit {s['fast_hit_rate']:.0%}")
    parts += [f"haute précision demandée {s['need_hi_share']:.0%} / utilisée {s['hi_share']:.0%}", f"CPU {s['cpu_share']:.0%}", f"disque {s['disk_per_token']:.1f}/token"]
    if s["skip_rate"]:
        parts.append(f"sautés {s['skip_rate']:.1%}")
    if s["prefetch_recall"] is not None:
        parts.append(f"prefetch rappel {s['prefetch_recall']:.0%}")
    return " | ".join(parts)


def reap_dir(args) -> Path:
    """
    Dossier des résultats REAP (13/10) : <store>\\reap, À CÔTÉ DU STORE et pas dans le repo, pour qu'une
    nouvelle version du repo (zip copié par-dessus ou dossier remplacé) ne les efface jamais. Les fichiers d'un
    ancien results\\reap du repo y sont copiés au premier lancement.
    """
    import shutil

    if getattr(args, "out_dir", None):
        out = Path(args.out_dir)
    else:
        out = Path(args.store) / "reap"
        old = ROOT / "results" / "reap"
        if old.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            moved = [f.name for f in old.iterdir() if f.is_file() and not (out / f.name).exists()]
            for name in moved:
                shutil.copy2(old / name, out / name)
            if moved:
                print(f"Résultats REAP repris de {old} vers {out} : {', '.join(sorted(moved))}")
    out.mkdir(parents=True, exist_ok=True)
    return out
