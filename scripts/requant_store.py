"""
Q1 / Q2 (doc 18, 02/10) : requantifie EN PLACE la basse précision (int4) d'un store,
sans toucher au reste (format, haute précision, historique d'usage).

    # Q1 : meilleure échelle par groupe, sans calibration (~5 min sur OLMoE avec le GPU)
    python scripts\\requant_store.py --model D:\\models\\olmoe-hf --store C:\\moe-store\\olmoe-i8 --method mse

    # Q2 : GPTQ, calibré sur du texte écrit par le modèle lui-même (ou --calib-file)
    python scripts\\requant_store.py --model D:\\models\\olmoe-hf --store C:\\moe-store\\olmoe-i8 --method gptq --work-dir D:\\moe-calib

Il faut une couche de place libre en plus (~33 Mio pour OLMoE). Interrompu, il reprend
là où il s'était arrêté (même commande). Un store chaud (--hot-store) construit avant
doit être reconstruit (build_hot_store.py) : ses copies ont les anciens nombres.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import _common
from moelab.requant import collect_calibration, file_calibration_ids, requantize_store, self_calibration_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common.add_model_args(ap)
    ap.add_argument("--method", default="mse", choices=["mse", "gptq", "absmax"],
                    help="mse = meilleure échelle (Q1), gptq = calibré (Q2), absmax = revenir à l'ancien format")
    ap.add_argument("--calib-tokens", type=int, default=32768, help="gptq : tokens de calibration (défaut 32768)")
    ap.add_argument("--calib-file", nargs="*", default=None,
                    help="gptq : fichiers texte de calibration (sinon, le modèle écrit lui-même son texte)")
    ap.add_argument("--work-dir", default=None,
                    help="gptq : dossier des entrées enregistrées (~2 Gio pour OLMoE ; défaut : <store>\\calib)")
    ap.add_argument("--keep-calib", action="store_true", help="gptq : garder le dossier de calibration")
    ap.add_argument("--min-tokens", type=int, default=32, help="gptq : en dessous, l'expert garde la méthode mse")
    args = ap.parse_args()

    t0 = time.time()
    calib_dir = None
    if args.method == "gptq":
        calib_dir = Path(args.work_dir or Path(args.store) / "calib")
        args.policy, args.t1, args.t2, args.protect, args.hi_vote = "hi", 0.6, None, "", 0.5
        model, rt, tok = _common.load(args)
        rt.policy.mode = "hi"  # entrées calculées avec la haute précision du store
        if args.calib_file:
            chunks = file_calibration_ids(tok, args.calib_file, args.calib_tokens)
            print(f"Calibration : {sum(c.shape[1] for c in chunks)} tokens lus dans {len(args.calib_file)} fichier(s)")
        else:
            print(f"Calibration : le modèle écrit {args.calib_tokens} tokens de texte varié ...")
            chunks = self_calibration_ids(model, tok, args.calib_tokens)
        print("Enregistrement des entrées de chaque couche MoE ...")
        collect_calibration(model, chunks, calib_dir)
        rt.close()
        del model, rt
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Réécriture de l'int4 de {args.store} ({args.method}) ...")
    stats = requantize_store(args.store, args.model, args.method, calib_dir=calib_dir, min_tokens=args.min_tokens)
    if args.method == "gptq":
        print(f"  {stats['gptq']} experts GPTQ, {stats['fallback']} en mse (trop peu de tokens de calibration)")
        if not args.keep_calib and calib_dir is not None:
            shutil.rmtree(calib_dir, ignore_errors=True)
    print(f"Terminé en {(time.time() - t0) / 60:.1f} min. Nouvel identifiant du store : {stats['store_id']}")
    print("Mesure la qualité avec benchmark.py --ref results\\ref_fp16.pt (même commande qu'avant).")


if __name__ == "__main__":
    main()
