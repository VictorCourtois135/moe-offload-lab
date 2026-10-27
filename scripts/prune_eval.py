"""
Élagage virtuel REAP (doc 22, étape 3) : le routeur ne choisit plus les experts au score le plus bas, puis la
même mesure que eval_ppl.py. Rien n'est écrit dans le store.

    # d'abord sur les couches 10 à 20 seulement (valider le mécanisme)
    python scripts\\prune_eval.py --store C:\\moe-store\\qwen3-next --ram-gb 5 --prune-ratio 0.25 --layers 10-20

    # puis toutes les couches, aux trois taux
    python scripts\\prune_eval.py --store C:\\moe-store\\qwen3-next --ram-gb 5 --prune-ratio 0.25
    python scripts\\prune_eval.py --store C:\\moe-store\\qwen3-next --ram-gb 5 --prune-ratio 0.40
    python scripts\\prune_eval.py --store C:\\moe-store\\qwen3-next --ram-gb 5 --prune-ratio 0.50

Il faut <store>\\reap\\reap_stats.pt (calibrate_reap.py) et <store>\\reap\\baseline.json (eval_ppl.py).
Sortie : <store>\\reap\\prune_25.json (ou prune_25_l10-20.json), comparable à baseline.json, avec
l'accord top-1 avec la référence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import _common
from moelab.prune_eval import evaluate, short_report
from moelab.reap import ReapCollector, apply_prune_mask, parse_layers, prune_mask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common.add_model_args(ap)
    _common.add_policy_args(ap)
    g = ap.add_argument_group("élagage")
    g.add_argument("--prune-ratio", type=float, required=True, help="part des experts retirés par couche (ex. 0.25)")
    g.add_argument("--layers", default=None, help="couches élaguées : « 10-20 », « 0,5,7 » (défaut : toutes)")
    g.add_argument("--stats", default=None, help="accumulateurs REAP (défaut : <out-dir>\\reap_stats.pt)")
    g.add_argument("--score", default="reap", choices=["reap", "ean", "frequence"],
                   help="score utilisé (défaut reap ; ean et frequence pour comparer)")
    g.add_argument("--name", default=None, help="nom de la mesure (défaut : prune_<taux>[_l<couches>])")
    g.add_argument("--out-dir", default=None, help="dossier des résultats (défaut : <store>\\reap)")
    g.add_argument("--parts", nargs="+", default=["vitesse", "outils", "perplexite"],
                   choices=["vitesse", "outils", "perplexite"], help="mesures à faire (défaut : les trois)")
    g.add_argument("--batch", type=int, default=8, help="extraits par passe de perplexité")
    args = ap.parse_args()

    out = _common.reap_dir(args)
    stats_path = Path(args.stats) if args.stats else out / "reap_stats.pt"
    if not stats_path.exists():
        raise SystemExit(f"{stats_path} introuvable : lance d'abord scripts\\calibrate_reap.py")
    col = ReapCollector.load(stats_path)
    model, rt, tok = _common.load(args)
    if col.store_id != rt.store.store_id:
        raise SystemExit(f"{stats_path.name} vient d'un autre store ({col.store_id} ≠ {rt.store.store_id})")
    scores = {"reap": col.scores, "ean": col.ean, "frequence": col.frequency}[args.score]()
    layers = parse_layers(args.layers, len(rt.store.layers))
    top_k = int(model.config.num_experts_per_tok)
    mask = prune_mask(scores, args.prune_ratio, top_k, layers)
    n = apply_prune_mask(model, mask)
    L, E = mask.shape
    per_layer = int(mask.sum(1).max())
    full = sum(rt.store.total_bytes(p) for p in rt.store.precisions) / 2**30
    est = full * (1 - n / (L * E))
    print(f"Élagage virtuel ({args.score}) : {n} experts masqués ({per_layer} par couche"
          f"{', couches ' + args.layers if layers else ''}) ; store élagué estimé {est:.2f} Gio au lieu de {full:.2f}")

    name = args.name or f"prune_{int(round(args.prune_ratio * 100))}" + (f"_l{args.layers}" if layers else "") \
        + ("" if args.score == "reap" else f"_{args.score}")
    ids = {}
    if "perplexite" in args.parts:
        if not (out / "eval_ids.pt").exists():
            raise SystemExit("eval_ids.pt introuvable : lance d'abord scripts\\eval_ppl.py (la référence)")
        ids = torch.load(out / "eval_ids.pt", map_location="cpu", weights_only=False)
    ref_path = out / "baseline_top1.pt"
    ref = torch.load(ref_path, map_location="cpu", weights_only=True) if ref_path.exists() else None
    extra = {"elagage": {"taux": args.prune_ratio, "score": args.score, "couches": args.layers or "toutes",
                         "experts_masques": n, "par_couche": per_layer, "store_gio_estime": round(est, 2)}}
    res = evaluate(model, tok, ids, name, out, batch=args.batch, ref_top1=ref, parts=tuple(args.parts), extra=extra)
    print()
    print(short_report(res))
    base = out / "baseline.json"
    if base.exists():
        b = json.loads(base.read_text(encoding="utf-8"))
        print("\nComparé à la référence :")
        for k, p in (res.get("perplexite") or {}).items():
            if k in (b.get("perplexite") or {}):
                r = b["perplexite"][k]["ppl"]
                print(f"  perplexité {k:8s} : {r:.3f} -> {p['ppl']:.3f} ({100 * (p['ppl'] / r - 1):+.1f} %)")
        if res.get("outils") and b.get("outils"):
            print(f"  outils              : {b['outils']['reussies']}/20 -> {res['outils']['reussies']}/20")
        if res.get("vitesse") and b.get("vitesse"):
            v, w = b["vitesse"], res["vitesse"]
            print(f"  vitesse             : {v['tok_s']:.1f} -> {w['tok_s']:.1f} tok/s ; hit VRAM "
                  f"{100 * v['hit_vram']:.0f} -> {100 * w['hit_vram']:.0f} % ; lectures disque/token "
                  f"{v['lectures_disque_par_token']:.0f} -> {w['lectures_disque_par_token']:.0f}")
    print(f"\nRésultats : {out / (name + '.json')}")
    rt.close()


if __name__ == "__main__":
    main()
