"""
Phase 0 de l'optimisation (doc 18) : où part le temps d'un token ?

    python scripts\\profile_moe.py --model D:\\models\\olmoe-hf --store C:\\moe-store\\olmoe

Mesure, dans l'ordre :
  1. les planchers physiques du GPU : coût d'un lancement de noyau, d'une synchro GPU -> CPU,
     temps minimal pour calculer 8 experts (un par un, ou groupés), déquantification int4 ;
  2. le décodage réel, pour chaque politique et chaque variante d'optimisation : ms par token,
     part des couches MoE et de la partie dense, détail de chaque phase de run_layer, en temps
     CPU ("async") puis GPU compris ("sync") ;
  3. torch.profiler sur quelques tokens de DÉCODAGE (le prefill n'est plus compté) : les
     opérations les plus coûteuses et le nombre de lancements de noyaux par token (trace
     Chrome exportée dans results/).

Résultat : results\\profile_<modèle>_<date>.md (+ .json).
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import _common
import torch

from moelab.bench import chat_ids, environment, load_prompts
from moelab.cache import UsageLog
from moelab.generate import make_decoder
from moelab.kernels import triton_usable
from moelab.policy import PrecisionPolicy
from moelab.profiling import PhaseTimer, microbenchmarks


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def decode(model, ids, n_tokens: int, ctx=None):
    """
    Prefill puis 1 token d'échauffement (non chronométrés : avec les CUDA graphs, c'est là
    qu'a lieu la capture), puis n_tokens gloutons chronométrés. Retourne ms par token.
    `ctx` (ex. torch.profiler) n'entoure que les tokens chronométrés.
    """
    rt = model.moelab_runtime
    dec = make_decoder(model, log=print)
    nxt = dec.prefill(ids).argmax(-1, keepdim=True).view(1, 1)
    nxt = dec.step(nxt).argmax(-1, keepdim=True).view(1, 1)
    sync()
    rt.reset_stats()
    if rt.prof:
        rt.prof.reset()
    with ctx or nullcontext():
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            nxt = dec.step(nxt).argmax(-1, keepdim=True).view(1, 1)
        sync()
        dt = time.perf_counter() - t0
    if rt.cfg.cuda_graphs and not getattr(dec, "active", False):
        print(f"   attention : {dec.describe()}")
    return 1000 * dt / n_tokens


# Variantes d'optimisation comparées (doc 18) : chaque nom active des options de RuntimeConfig.
_V30 = {"fast_decode": True, "prefetch_adaptive": True, "triton_int4": True}  # défaut du 30/09
_V0110 = {**_V30, "grouped_experts": True, "direct_io": True}                  # défaut du 01/10 matin
_DEF = {**_V0110, "cuda_graphs": True, "full_graph": True}                     # défaut actuel (A6, A6 bis, v3, A5)
VARIANTS = {
    "base": {},
    "a2": {"fast_decode": True},
    "a2+adapt": {"fast_decode": True, "prefetch_adaptive": True},
    "b2": {"fast_int4_unpacked": True},
    "c2": {"prefetch_depth": 2},
    "d1": {"direct_io": True},
    "b1": {"fast_decode": True, "triton_int4": True},
    "v30": _V30,
    "grp": {**_V30, "grouped_experts": True},
    "grp+d1": _V0110,
    "grp+d1+c2": {**_V0110, "prefetch_depth": 2},
    "v0110": _V0110,
    "cg_v1": {**_DEF, "full_graph": False, "grouped_kernels": 1, "moe_graphs": False},  # CUDA graphs, noyaux groupés v1
    "cg_v2": {**_DEF, "full_graph": False, "grouped_kernels": 2, "moe_graphs": False},  # mesuré le 01/10 : 48,4 tok/s en lo
    "cg_v3": {**_DEF, "full_graph": False, "grouped_kernels": 3, "moe_graphs": False},  # v3 sans graph du calcul groupé
    "g2": {**_DEF, "full_graph": False, "grouped_kernels": 2},                           # graph du calcul groupé, noyaux v2 (01/10 : 89,6 tok/s)
    "g4": {**_DEF, "full_graph": False, "grouped_kernels": 4},                           # ... noyaux v4
    "defaut": _DEF,                                                 # noyaux choisis par mesure, paliers de longueur
    "sans_paliers": {**_DEF, "full_graph": False, "graph_buckets": False},              # attention sur les 2 048 positions du cache
    "sans_a9": {**_DEF, "full_graph": False, "graph_pack_routing": False},              # prédiction du prefetch lancée en Python
    # 02/10 (recherche bibliographique) : à mesurer avant d'activer par défaut
    "sans_a5": {**_DEF, "full_graph": False},                       # 02/10 : lo 117,5 tok/s (132,1 avec A5)
    "a5": _DEF,
    "stale": {**_DEF, "cache_eviction": "stale"},                   # éviction least-stale (SpecMD)
    "mass": {**_DEF, "prefetch_mass": 0.9},                         # prefetch selon le score (SpecMD)
    "s3": {**_DEF, "cache_eviction": "stale", "prefetch_mass": 0.9},
    "ahead1": {**_DEF, "prefetch_disk_ahead": 1},                   # lectures disque -> RAM pour L+2 (Qwen3-Next)
    "ahead2": {**_DEF, "prefetch_disk_ahead": 2},                   # ... pour L+2 et L+3
    "split1": {**_DEF, "read_parts": 1},                            # un expert = une lecture disque
    "split4": {**_DEF, "read_parts": 4},                            # un expert = 4 lectures simultanées
    "tout": {**_DEF, "cache_eviction": "stale", "prefetch_mass": 0.9},
}
OFF = {"fast_decode": False, "prefetch_adaptive": False, "fast_int4_unpacked": False, "prefetch_depth": 1,
       "direct_io": False, "triton_int4": False, "grouped_experts": False, "cuda_graphs": False, "grouped_kernels": 0,
       "graph_buckets": True, "graph_pack_routing": True,
       "moe_graphs": True, "full_graph": False, "cache_eviction": "lrfu", "prefetch_mass": None,
       "prefetch_disk_ahead": 0, "read_parts": None}


def apply_variant(rt, var: str) -> None:
    if var not in VARIANTS:
        raise SystemExit(f"variante inconnue : {var} (choix : {', '.join(VARIANTS)})")
    if not hasattr(rt, "_measured_read_parts"):  # valeur mesurée (bandwidth.json), gardée si la variante ne la fixe pas
        rt._measured_read_parts = rt.cfg.read_parts
    measured_parts = rt._measured_read_parts
    for key, flag in {**OFF, **VARIANTS[var]}.items():
        setattr(rt.cfg, key, measured_parts if (key == "read_parts" and flag is None) else flag)
    rt.store.direct = rt.cfg.direct_io
    rt.store.read_parts = max(1, int(rt.cfg.read_parts))
    if rt.fast is not None:
        rt.fast.eviction = rt.cfg.cache_eviction
    rt.set_unpacked(rt.cfg.fast_int4_unpacked)
    rt._ggraphs.clear()
    rt._auto_int4 = None
    rt._triton = bool(rt.cfg.triton_int4 and rt.fast is not None and triton_usable(rt.fast.device))
    rt._grouped = bool(rt.cfg.grouped_experts and rt.fast is not None and triton_usable(rt.fast.device))
    if (rt.cfg.triton_int4 or rt.cfg.grouped_experts) and not (rt._triton or rt._grouped):
        print(f"   (variante {var} : Triton indisponible, chemin PyTorch)")


def make_policy(spec: str) -> PrecisionPolicy:
    base, _, t = spec.partition("@")
    if base not in ("hi", "lo", "hobbit"):
        raise SystemExit(f"politique inconnue : {spec}")
    return PrecisionPolicy(base, t1=float(t) if t else 0.6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common.add_model_args(ap)
    ap.add_argument("--policies", default="lo,hobbit@0.8",
                    help="politiques mesurées (défaut : lo,hobbit@0.8 ; hi est très lent et sensible à la pagination)")
    ap.add_argument("--variants", default="sans_a5,defaut", help=f"optimisations comparées ({', '.join(VARIANTS)})")
    ap.add_argument("--tokens", type=int, default=48, help="tokens de décodage chronométrés par mesure")
    ap.add_argument("--trace-tokens", type=int, default=8, help="tokens passés sous torch.profiler")
    ap.add_argument("--trace-variant", default=None, help="variante tracée par torch.profiler (défaut : la dernière)")
    ap.add_argument("--trace-policy", default=None, help="politique tracée par torch.profiler (défaut : la première)")
    ap.add_argument("--out", default=str(_common.ROOT / "results"))
    args = ap.parse_args()
    args.policy, args.t1, args.t2, args.protect, args.hi_vote = "hi", 0.6, None, "", 0.5

    model, rt, tok = _common.load(args)
    cfg = model.config
    k = getattr(cfg, "num_experts_per_tok", 8)
    ids = chat_ids(tok, load_prompts()[0], model.moelab_device)
    report: dict = {"environment": environment(rt), "k": k}
    variants = args.variants.split(",")
    for var in variants + ([args.trace_variant] if args.trace_variant else []):
        if var not in VARIANTS:
            raise SystemExit(f"variante inconnue : {var} (choix : {', '.join(VARIANTS)})")

    # 1. Planchers physiques
    if torch.cuda.is_available():
        print("1. Planchers physiques du GPU ...")
        mb = microbenchmarks(rt.store.hidden, rt.store.inter, k)
        report["micro"] = mb
        for key, v in mb.items():
            print(f"   {key:32s} {v:9.3f}" if isinstance(v, float) else f"   {key:32s} {v}")
    else:
        print("1. Pas de GPU : planchers non mesurés")

    # 2. Décodage réel, par politique et par variante d'optimisation
    report["decode"] = {}
    n_layers = len(rt.store.layers)
    for spec in args.policies.split(","):
        pol = make_policy(spec)
        for var in variants:
            apply_variant(rt, var)
            rt.policy = rt.cfg.policy = pol
            rt.reset_caches()
            rt.usage = UsageLog(None, rt.store.store_id)
            rt.warm_start(preload_lo=rt.preload_lo, log=lambda *a: None)
            print(f"2. Décodage, politique {pol.describe()}, variante {var} ...")
            decode(model, ids, args.tokens)  # échauffement : remplit les caches
            entry = {}
            for mode in ("async", "sync"):
                rt.prof = PhaseTimer(sync=(mode == "sync"))
                ms_tok = decode(model, ids, args.tokens)
                st = rt.stats.as_dict()
                phases = rt.prof.per_call_ms()
                moe_ms_tok = 1000 * st["t_moe"] / args.tokens
                entry[mode] = {"ms_per_token": ms_tok, "moe_ms_per_token": moe_ms_tok,
                               "dense_ms_per_token": ms_tok - moe_ms_tok,
                               "moe_ms_per_layer": moe_ms_tok / n_layers, "phases_ms_per_layer": phases,
                               "fast_hit_rate": st["fast_hit_rate"], "disk_per_token": st["disk_per_token"],
                               "fills_per_token": st["fills"] / args.tokens,
                               "prefetch_per_token": st["prefetch_issued"] / args.tokens,
                               "h2d_mb_per_token": st["bytes_h2d"] / args.tokens / 2**20,
                               "prefetch_precision": st["prefetch_precision"], "prefetch_recall": st["prefetch_recall"],
                               "full_share": st["full_tokens"] / args.tokens, "full_fallbacks": st["full_fallbacks"]}
                print(f"   [{mode}] {ms_tok:.1f} ms/token ({1000 / ms_tok:.2f} tok/s) : couches MoE {moe_ms_tok:.1f} ms "
                      f"({moe_ms_tok / n_layers:.2f} ms/couche), reste (dense) {ms_tok - moe_ms_tok:.1f} ms, "
                      f"hit VRAM {st['fast_hit_rate']:.0%}, disque {st['disk_per_token']:.1f}/token")
                e = entry[mode]
                pp = (f", prefetch précision {e['prefetch_precision']:.0%} rappel {e['prefetch_recall']:.0%}"
                      if e["prefetch_precision"] is not None else "")
                print(f"      transferts : {e['fills_per_token']:.1f} à la demande + {e['prefetch_per_token']:.1f} prefetch "
                      f"par token, {e['h2d_mb_per_token']:.0f} Mo RAM -> VRAM par token{pp}"
                      + (f", {st['disk_ahead'] / args.tokens:.1f} lectures disque à l'avance par token" if st.get("disk_ahead") else ""))
                if rt.cfg.full_graph:
                    print(f"      A5 : {e['full_share']:.0%} des tokens en un seul graph, {e['full_fallbacks']} refaits "
                          "(les couches MoE de ces tokens ne sont pas chronométrées à part)")
                for name, v in sorted(phases.items()):
                    print(f"      {name:48s} {v:7.3f} ms/couche")
            rt.prof = None
            if rt.triton_error:
                print(f"   Triton a échoué, repli PyTorch : {rt.triton_error}")
                entry["triton_error"] = rt.triton_error
                rt.triton_error = None
            dec = getattr(model, "moelab_graph_decoder", None)
            if rt.cfg.cuda_graphs and dec is not None:
                entry["cuda_graphs"] = dec.describe()
            report["decode"][f"{spec} / {var}"] = entry

    # 3. torch.profiler, sur le décodage seulement
    tvar = args.trace_variant or variants[-1]
    tpol = args.trace_policy or args.policies.split(",")[0]
    print(f"3. torch.profiler (politique {tpol}, variante {tvar}) ...")
    apply_variant(rt, tvar)
    rt.policy = rt.cfg.policy = make_policy(tpol)
    # mêmes caches que pour les mesures : sinon la VRAM contient encore les fp16 de la politique
    # précédente, et la trace de `lo` montre des copies qui n'existent pas en régime normal (01/10)
    rt.reset_caches()
    rt.usage = UsageLog(None, rt.store.store_id)
    rt.warm_start(preload_lo=rt.preload_lo, log=lambda *a: None)
    decode(model, ids, args.tokens)
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    decode(model, ids, 4)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    tag = _common.model_tag(args)
    p = profile(activities=acts)
    decode(model, ids, args.trace_tokens, ctx=p)
    trace = out_dir / f"trace_{tag}_{stamp}.json"
    p.export_chrome_trace(str(trace))
    ka = p.key_averages()
    launch_names = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cudaGraphLaunch")
    launches = sum(e.count for e in ka if e.key in launch_names[:3])
    graph_launches = sum(e.count for e in ka if e.key == "cudaGraphLaunch")
    report["trace_variant"] = tvar
    report["launches_per_token"] = launches / args.trace_tokens
    report["graph_launches_per_token"] = graph_launches / args.trace_tokens
    by_cpu = sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)[:20]
    report["top_cpu"] = [{"op": e.key, "calls": e.count, "self_cpu_ms": e.self_cpu_time_total / 1000} for e in by_cpu]
    print(f"   {report['launches_per_token']:.0f} lancements de noyaux par token"
          + (f" + {report['graph_launches_per_token']:.0f} graphs rejoués" if graph_launches else ""))
    print(ka.table(sort_by="self_cpu_time_total", row_limit=15))

    # Rapport
    (out_dir / f"profile_{tag}_{stamp}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    md = [f"# Profil {tag} — {stamp}", "", f"Machine : {report['environment'].get('gpu')}, k = {k}", ""]
    if "micro" in report:
        md += ["## Planchers physiques", "", "| Mesure | Valeur |", "|---|---|"]
        md += [f"| {key} | {v:.3f} |" if isinstance(v, float) else f"| {key} | {v} |" for key, v in report["micro"].items()]
        md.append("")
    md += ["## Décodage", "",
           "| Politique / variante | Mode | ms/token | tok/s | MoE ms/token | dense ms/token | ms par couche MoE | hit VRAM | disque/token | Mo RAM→VRAM/token | prefetch précision | A5 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for spec, entry in report["decode"].items():
        for mode in ("async", "sync"):
            e = entry[mode]
            md.append(f"| {spec} | {mode} | {e['ms_per_token']:.1f} | {1000 / e['ms_per_token']:.2f} | "
                      f"{e['moe_ms_per_token']:.1f} | {e['dense_ms_per_token']:.1f} | {e['moe_ms_per_layer']:.2f} | "
                      f"{e['fast_hit_rate']:.0%} | {e['disk_per_token']:.1f} | {e['h2d_mb_per_token']:.0f} | "
                      f"{'-' if e['prefetch_precision'] is None else format(e['prefetch_precision'], '.0%')} | "
                      f"{format(e.get('full_share', 0), '.0%')} |")
    md.append("")
    for spec, entry in report["decode"].items():
        if "cuda_graphs" in entry:
            md += [f"- {spec} : {entry['cuda_graphs']}"]
    md.append("")
    for spec, entry in report["decode"].items():
        md += [f"### Phases, {spec} (ms par couche)", "", "| Phase | async | sync |", "|---|---|---|"]
        names = sorted(set(entry["async"]["phases_ms_per_layer"]) | set(entry["sync"]["phases_ms_per_layer"]))
        md += [f"| {n} | {entry['async']['phases_ms_per_layer'].get(n, 0):.3f} | {entry['sync']['phases_ms_per_layer'].get(n, 0):.3f} |" for n in names]
        md.append("")
    md += [f"## torch.profiler (décodage seul, variante {tvar})", "",
           f"Lancements de noyaux par token : {report['launches_per_token']:.0f}"
           + (f" (+ {report['graph_launches_per_token']:.0f} CUDA graphs rejoués)" if graph_launches else ""), "",
           "| Opération | Appels | CPU propre (ms) |", "|---|---|---|"]
    md += [f"| {e['op']} | {e['calls']} | {e['self_cpu_ms']:.1f} |" for e in report["top_cpu"]]
    (out_dir / f"profile_{tag}_{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nRésultats : {out_dir / f'profile_{tag}_{stamp}.md'} (+ .json, trace Chrome : {trace.name})")
    rt.close()


if __name__ == "__main__":
    main()
