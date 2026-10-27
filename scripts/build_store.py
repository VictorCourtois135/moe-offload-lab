"""
Étape 8 : convertit les experts d'un checkpoint Hugging Face en store (un record
contigu par expert, une précision par fichier). Un expert à la fois en mémoire :
marche pour OLMoE comme pour Qwen3-Next-80B. Reprenable si interrompu.

    python scripts\\build_store.py --model D:\\models\\olmoe-hf --out C:\\moe-store\\olmoe
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import _common  # noqa: F401

from moelab.loader import discover_experts
from moelab.plan import disk_free, estimate_store
from moelab.store import PRECISIONS, build_store

GIB = 2**30


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="dossier du checkpoint Hugging Face")
    ap.add_argument("--out", required=True, help="dossier du store à créer")
    ap.add_argument("--hi", default="fp16", choices=[p for p in PRECISIONS if p != "int4"],
                    help="précision haute (défaut fp16)")
    ap.add_argument("--lo", default="int4", choices=list(PRECISIONS), help="précision basse (défaut int4)")
    ap.add_argument("--only", choices=["hi", "lo"], default=None,
                    help="ne construire qu'une précision (ex. --only lo pour un store int4 seul)")
    ap.add_argument("--precisions", default=None,
                    help="liste explicite, ex. fp16,int8,int4 (trois paliers pour --t-mid, optimisation E2)")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--int4-scale", default="mse", choices=["mse", "absmax"],
                    help="Q1 : échelle int4 choisie par recherche (mse, défaut) ou max|w|/7 (absmax, stores d'avant le 02/10)")
    ap.add_argument("--no-resume", action="store_true", help="repartir de zéro même si une conversion est en cours")
    ap.add_argument("--no-standalone", action="store_true",
                    help="ne pas copier la partie dense dans le store (sinon : store autonome, le checkpoint peut être supprimé)")
    args = ap.parse_args()

    precs = {"hi": (args.hi,), "lo": (args.lo,)}.get(args.only, (args.hi, args.lo))
    if args.precisions:
        order = list(PRECISIONS)  # fp32, fp16, bf16, int8, int4 : la plus précise d'abord
        precs = tuple(sorted(dict.fromkeys(p.strip() for p in args.precisions.split(",")), key=order.index))
    print(f"Analyse de {args.model} ...")
    info = discover_experts(args.model)
    print(f"  {info['model_type']} : {len(info['layers'])} couches MoE x {info['num_experts']} experts "
          f"(H={info['hidden']}, I={info['inter']})")
    sizes = estimate_store(info, precs, args.group_size)
    need = sum(sizes.values())
    if not args.no_standalone:
        from moelab.standalone import dense_bytes

        need += dense_bytes(args.model)
    free = disk_free(args.out)
    # Reprise : les fichiers déjà écrits occupent déjà leur place (chaque fichier de couche est
    # créé à sa taille finale), il ne reste à trouver que la place des fichiers absents.
    out = Path(args.out)
    present = 0
    if not args.no_resume and out.exists():
        present = sum(f.stat().st_size for p in precs for f in out.glob(f"layer*.{p}.bin"))
    remaining = max(0, need - present)
    print(f"  taille du store : {need / GIB:.1f} Gio ({', '.join(f'{p} {s / GIB:.1f}' for p, s in sizes.items())}), "
          f"libre sur la destination : {free / GIB:.1f} Gio"
          + (f", déjà écrit : {present / GIB:.1f} Gio (reste {remaining / GIB:.1f} Gio)" if present else ""))
    if remaining > free * 0.98:
        raise SystemExit(f"Pas assez de place : il manque {(remaining - free * 0.98) / GIB:.1f} Gio sur la destination "
                         "(libère de la place, choisis une autre destination, ou --only lo pour la seule basse précision).")

    t0 = time.time()
    build_store(args.model, args.out, precs, args.group_size, resume=not args.no_resume, int4_scale=args.int4_scale)
    if not args.no_standalone:
        from moelab.standalone import make_standalone

        print("Copie de la partie dense dans le store (store autonome) ...")
        make_standalone(args.model, args.out)
    print(f"Store prêt en {(time.time() - t0) / 60:.1f} min : {args.out}"
          + ("" if args.no_standalone else
             f"\nStore autonome : --model devient facultatif, et {args.model} peut être supprimé."))


if __name__ == "__main__":
    main()
