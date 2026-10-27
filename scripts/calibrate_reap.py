"""
Calibration REAP (doc 22, 08/10) : score de chaque expert routé, pour l'élagage.

    # 1. essai de bout en bout sur 10 échantillons (quelques minutes)
    python scripts\\calibrate_reap.py --store C:\\moe-store\\qwen3-next --ram-gb 5 --test

    # 2. la vraie calibration (256 échantillons de 512 tokens) ; reprend toute seule si on l'arrête
    python scripts\\calibrate_reap.py --store C:\\moe-store\\qwen3-next --ram-gb 5

Le jeu de calibration est construit une fois (téléchargements : pip install datasets) puis
figé dans <store>\\reap\\calib_ids.pt. Il mélange texte général, français, code, maths,
conversations, et tes conversations et runs d'agents de l'app (app_data).
Sorties : <store>\\reap\\reap_stats.pt (accumulateurs, sauvegardés tous les --save-every lots)
et <store>\\reap\\reap_summary.json (ce que coûterait chaque taux d'élagage).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import _common
from moelab.prune_data import build_calibration, load_or_build
from moelab.reap import calibrate, write_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common.add_model_args(ap)
    g = ap.add_argument_group("calibration")
    g.add_argument("--samples", type=int, default=256, help="échantillons de 512 tokens (défaut 256)")
    g.add_argument("--batch", type=int, default=8, help="échantillons par passe (réduit tout seul si la VRAM manque)")
    g.add_argument("--save-every", type=int, default=4, help="sauvegarde tous les N lots (défaut 4)")
    g.add_argument("--seed", type=int, default=0, help="graine du jeu de calibration (défaut 0)")
    g.add_argument("--data", default=str(_common.ROOT / "app_data"),
                   help="données de l'app (conversations, runs d'agents) ajoutées à la calibration")
    g.add_argument("--out-dir", default=None, help="dossier des résultats (défaut : <store>\\reap)")
    g.add_argument("--test", action="store_true", help="essai sur 10 échantillons (fichiers *_test)")
    g.add_argument("--rebuild-data", action="store_true", help="reconstruire le jeu de calibration")
    g.add_argument("--no-resume", action="store_true", help="repartir de zéro même si reap_stats.pt existe")
    args = ap.parse_args()

    out = _common.reap_dir(args)
    suffix = "_test" if args.test else ""
    n = 10 if args.test else args.samples
    args.policy, args.t1, args.t2, args.protect, args.hi_vote = "hi", 0.6, None, "", 0.5
    model, rt, tok = _common.load(args)
    rt.policy.mode = "hi"  # la meilleure précision du store (Qwen3-Next : il n'y a que l'int4)

    data = load_or_build(out / "calib_ids.pt",
                         lambda: build_calibration(tok, args.samples, seed=args.seed, data_dir=args.data),
                         rebuild=args.rebuild_data)
    ids = data["ids"][:n].to(torch.int64)
    print(f"Calibration : {ids.shape[0]} échantillons de {ids.shape[1]} tokens "
          f"({', '.join(f'{k} {v}' for k, v in data['counts'].items())})")
    ev = out / "eval_ids.pt"
    if ev.exists():  # la calibration ne doit jamais contenir un extrait de l'évaluation
        e = torch.load(ev, map_location="cpu", weights_only=False)
        rows = {tuple(r.tolist()) for k in ("wikitext", "fr") for r in e[k]}
        same = sum(tuple(r.tolist()) in rows for r in data["ids"])
        if same:
            raise SystemExit(f"{same} échantillon(s) de calibration identiques à l'évaluation : --rebuild-data")

    t0 = time.time()
    col = calibrate(model, ids, out / f"reap_stats{suffix}.pt", batch=args.batch, save_every=args.save_every,
                    resume=not args.no_resume)
    k = getattr(model.config, "num_experts_per_tok", None)
    s = write_summary(col, out / f"reap_summary{suffix}.json", top_k=k,
                      record_bytes=rt.store.record_bytes(rt.store.lo), layer_names=rt.store.layers)
    print(f"\nTerminé en {(time.time() - t0) / 60:.1f} min : {col.samples_done} échantillons, {col.tokens} tokens")
    print(f"Experts jamais choisis : {s['jamais_choisis']} sur {s['couches'] * s['experts']}")
    for r, t in s["taux"].items():
        size = f", store ~{t['store_gio']} Gio (au lieu de {s['store_gio_complet']})" if "store_gio" in t else ""
        print(f"  élagage {float(r):.0%} : {t['experts_retires_par_couche']} experts retirés par couche, "
              f"{100 * t['choix_perdus_moyen']:.1f} % des choix du routeur à reporter (pire couche "
              f"{100 * t['choix_perdus_max']:.1f} %){size}")
    print(f"\nRésultats : {out / f'reap_stats{suffix}.pt'} et {out / f'reap_summary{suffix}.json'}")
    rt.close()


if __name__ == "__main__":
    main()
