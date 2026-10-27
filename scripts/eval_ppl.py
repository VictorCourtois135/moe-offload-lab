"""
Banc d'évaluation de l'élagage (doc 22, 08/10) : vitesse, appels d'outils, perplexité.

    # la référence, modèle complet (une fois) : ~20-30 min sur la RTX 2060 avec Qwen3-Next
    python scripts\\eval_ppl.py --store C:\\moe-store\\qwen3-next --ram-gb 5

    # seulement la perplexité, ou seulement les outils
    python scripts\\eval_ppl.py --store C:\\moe-store\\qwen3-next --ram-gb 5 --parts perplexite
    python scripts\\eval_ppl.py --store C:\\moe-store\\qwen3-next --ram-gb 5 --parts outils --name essai

Le jeu d'évaluation (WikiText-2 test, Wikipédia FR) est téléchargé une fois puis figé dans
<store>\\reap\\eval_ids.pt : toutes les mesures utilisent les mêmes extraits. Sortie :
<store>\\reap\\<nom>.json (+ <nom>_top1.pt, pour l'accord top-1 des mesures suivantes).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _common
from moelab.prune_data import build_eval, load_or_build
from moelab.prune_eval import evaluate, short_report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common.add_model_args(ap)
    _common.add_policy_args(ap)
    g = ap.add_argument_group("évaluation")
    g.add_argument("--name", default="baseline", help="nom de la mesure (défaut : baseline = la référence)")
    g.add_argument("--out-dir", default=None, help="dossier des résultats (défaut : <store>\\reap)")
    g.add_argument("--parts", nargs="+", default=["vitesse", "outils", "perplexite"],
                   choices=["vitesse", "outils", "perplexite"], help="mesures à faire (défaut : les trois)")
    g.add_argument("--batch", type=int, default=8, help="extraits par passe (réduit tout seul si la VRAM manque)")
    g.add_argument("--n-en", type=int, default=50, help="extraits WikiText-2 (défaut 50)")
    g.add_argument("--n-fr", type=int, default=20, help="extraits Wikipédia FR (défaut 20)")
    g.add_argument("--rebuild-data", action="store_true", help="reconstruire le jeu d'évaluation (déconseillé)")
    g.add_argument("--ref", default=None, help="top-1 de la référence (défaut : <out-dir>\\baseline_top1.pt)")
    args = ap.parse_args()

    out = _common.reap_dir(args)
    model, rt, tok = _common.load(args)
    ids = None
    if "perplexite" in args.parts:
        ids = load_or_build(out / "eval_ids.pt", lambda: build_eval(tok, args.n_en, args.n_fr),
                            rebuild=args.rebuild_data)
        print(f"Jeu d'évaluation : {tuple(ids['wikitext'].shape)} WikiText-2, {tuple(ids['fr'].shape)} français")
    ref = None
    ref_path = Path(args.ref) if args.ref else out / "baseline_top1.pt"
    if args.name != "baseline" and ref_path.exists():
        import torch

        ref = torch.load(ref_path, map_location="cpu", weights_only=True)
    res = evaluate(model, tok, ids or {}, args.name, out, batch=args.batch, ref_top1=ref, parts=tuple(args.parts))
    print()
    print(short_report(res))
    print(f"\nRésultats : {out / (args.name + '.json')}")
    rt.close()


if __name__ == "__main__":
    main()
