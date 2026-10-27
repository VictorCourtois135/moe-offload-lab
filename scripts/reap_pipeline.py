"""
Toute la chaîne REAP en une commande (doc 22), reprenable : ce qui est déjà fait est sauté.

    python scripts\\reap_pipeline.py --store C:\\moe-store\\qwen3-next --ram-gb 5

Dans l'ordre :
  1. référence (eval_ppl.py)                         -> <store>\\reap\\baseline.json      (~25 min)
  2. calibration REAP, 256 échantillons             -> <store>\\reap\\reap_stats.pt      (~35 min)
  3. élagage virtuel 25 %, couches 10 à 20 seulement -> prune_25_l10-20.json            (~25 min)
  4. élagage virtuel 25 %, 40 %, 50 %, toutes couches -> prune_25.json, prune_40.json…  (~25 min chacun)
  5. tableau comparatif                              -> <store>\\reap\\comparaison.md

Les résultats sont à côté du store, pas dans le repo : copier une nouvelle version du repo ne les efface pas.
Arrêt avec Ctrl+C à tout moment ; relancer la même commande reprend à l'étape en cours (la calibration
reprend même au milieu). Chaque étape tourne dans son propre processus : la mémoire GPU repart de zéro.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(HERE / script), *args]
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"{script} s'est arrêté (code {r.returncode}). Relance la même commande pour reprendre.")
    print(f"=== {script} terminé en {(time.time() - t0) / 60:.1f} min", flush=True)


def calibration_done(out: Path, samples: int) -> bool:
    summ = out / "reap_summary.json"
    if not summ.exists():
        return False
    try:
        return json.loads(summ.read_text(encoding="utf-8")).get("echantillons", 0) >= samples
    except ValueError:
        return False


def table(out: Path) -> str:
    base = json.loads((out / "baseline.json").read_text(encoding="utf-8"))
    rows = [("référence", base)]
    for f in sorted(out.glob("prune_*.json")):
        rows.append((f.stem, json.loads(f.read_text(encoding="utf-8"))))
    lines = ["| essai | store (Gio) | PPL WikiText | PPL FR | accord top-1 | outils | tok/s | hit VRAM | lectures disque/token |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows:
        p = r.get("perplexite") or {}
        v = r.get("vitesse") or {}
        o = r.get("outils") or {}
        acc = (p.get("wikitext") or {}).get("accord_top1_ref")
        gio = (r.get("elagage") or {}).get("store_gio_estime") or (r.get("store") or {}).get("gio")
        cells = [name, f"{gio:.1f}" if gio else "-",
                 f"{p['wikitext']['ppl']:.3f}" if "wikitext" in p else "-",
                 f"{p['fr']['ppl']:.3f}" if "fr" in p else "-",
                 f"{100 * acc:.1f} %" if acc is not None else "-",
                 f"{o['reussies']}/{o['total']}" if o else "-",
                 f"{v['tok_s']:.1f}" if v.get("tok_s") else "-",
                 f"{100 * v['hit_vram']:.0f} %" if "hit_vram" in v else "-",
                 f"{v['lectures_disque_par_token']:.0f}" if "lectures_disque_par_token" in v else "-"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="dossier du store (ex. C:\\moe-store\\qwen3-next)")
    ap.add_argument("--ram-gb", default="5", help="RAM pour les experts (défaut 5)")
    ap.add_argument("--ratios", nargs="+", default=["0.25", "0.40", "0.50"], help="taux d'élagage (défaut 0.25 0.40 0.50)")
    ap.add_argument("--samples", type=int, default=256, help="échantillons de calibration (défaut 256)")
    ap.add_argument("--skip-layer-test", action="store_true", help="sauter l'essai sur les couches 10 à 20")
    ap.add_argument("--stop-after", default=None, choices=["reference", "calibration", "essai"],
                    help="s'arrêter après cette étape")
    args, extra = ap.parse_known_args()  # le reste (--vram-gb, --dense…) est passé tel quel aux scripts

    sys.path.insert(0, str(HERE))
    import _common

    out = _common.reap_dir(argparse.Namespace(store=args.store, out_dir=None))
    common = ["--store", args.store, "--ram-gb", str(args.ram_gb), *extra]
    t0 = time.time()

    if (out / "baseline.json").exists():
        print("1. référence : déjà faite (baseline.json)")
    else:
        run("eval_ppl.py", common)
    if args.stop_after == "reference":
        return

    if calibration_done(out, args.samples):
        print("2. calibration : déjà faite (reap_summary.json)")
    else:
        run("calibrate_reap.py", [*common, "--samples", str(args.samples)])
    if args.stop_after == "calibration":
        return

    if not args.skip_layer_test:
        if (out / "prune_25_l10-20.json").exists():
            print("3. essai couches 10-20 : déjà fait")
        else:
            run("prune_eval.py", [*common, "--prune-ratio", "0.25", "--layers", "10-20"])
    if args.stop_after == "essai":
        return

    for r in args.ratios:
        name = f"prune_{int(round(float(r) * 100))}.json"
        if (out / name).exists():
            print(f"4. élagage {r} : déjà fait ({name})")
        else:
            run("prune_eval.py", [*common, "--prune-ratio", r])

    md = table(out)
    (out / "comparaison.md").write_text(md + "\n", encoding="utf-8")
    print(f"\nTout est fait ({(time.time() - t0) / 60:.0f} min pour cette séance).\n")
    print(md)
    print(f"\nTableau : {out / 'comparaison.md'}")


if __name__ == "__main__":
    main()
