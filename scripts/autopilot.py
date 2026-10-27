"""
moelab autopilot (doc 19) : le modèle local optimise son propre moteur, sur ce PC.

    python scripts\\autopilot.py --model D:\\models\\qwen3-next-80b-fp8 --store C:\\moe-store\\qwen3-next --ram-gb 5

Le modèle choisit les expériences (outils machine, espace, mesurer, journal, terminer) ;
le code mesure, refuse ce qui dégrade la qualité, ne garde un gain qu'au-delà du bruit et
après une 2e mesure. À la fin :
    <store>\\autopilot_profile.json            la meilleure configuration (chat.py et agent.py
                                               l'appliquent au démarrage ; --no-profile pour s'en passer)
    results\\autopilot_<modèle>_<date>.md       le rapport du modèle et le journal des essais

L'épreuve (repartir des réglages du 02/10, sans montrer au modèle les effets déjà mesurés ;
aucun profil écrit) :
    python scripts\\autopilot.py ... --start 02-10 --blind

Durée : ~1 min par essai sur Qwen3-Next (2 si le gain est confirmé), plus la réflexion du
modèle. 12 essais : 30 à 45 min.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import _common
from transformers import TextStreamer

from moelab import options
from moelab.autopilot import Autopilot, Bench, start_0210
from moelab.bench import load_prompts
from moelab.generate import warmup


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common.add_model_args(ap)
    ap.add_argument("--policy", default="lo", choices=["hi", "lo", "hobbit"], help="politique de départ (défaut : lo)")
    ap.add_argument("--t1", type=float, default=0.6)
    ap.add_argument("--start", default="actuel", choices=["actuel", "02-10"],
                    help="point de départ : réglages actuels, ou ceux du 02/10 (épreuve)")
    ap.add_argument("--blind", action="store_true", help="ne pas montrer au modèle l'effet déjà mesuré des réglages")
    ap.add_argument("--trials", type=int, default=12, help="essais au plus (défaut 12)")
    ap.add_argument("--minutes", type=float, default=60, help="durée au plus (défaut 60 min)")
    ap.add_argument("--prompts", type=int, default=3, help="prompts par mesure (défaut 3)")
    ap.add_argument("--gen", type=int, default=24, help="tokens générés par prompt et par mesure (défaut 24)")
    ap.add_argument("--noise", type=float, default=0.05, help="gain minimal cru (défaut 0.05 = 5 %%)")
    ap.add_argument("--kl-max", type=float, default=0.005, help="KL au plus contre la référence (défaut 0.005)")
    ap.add_argument("--out", default=str(_common.ROOT / "results"))
    args = ap.parse_args()
    args.t2, args.protect, args.hi_vote = None, "", 0.5

    print("Chargement ...")
    model, rt, tok = _common.load(args)
    warmup(model, tok, log=print)
    if args.start == "02-10":
        for k, v in start_0210(rt).items():
            print("  départ 02/10 : " + options.apply(model, k, v))
    texts = load_prompts()
    bench = Bench(model, tok, texts[1:1 + args.prompts], gen=args.gen, warm=texts[0])
    pilot = Autopilot(model, tok, bench, max_trials=args.trials, minutes=args.minutes, noise=args.noise,
                      kl_max=args.kl_max, blind=args.blind)
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def on_step(step):
        for c, r in zip(step.calls, step.results):
            shown = r if len(r) < 300 else r[:300] + " ..."
            print(f"\n  [outil] {c.get('name')}({c.get('arguments', {})}) -> {shown}")

    from moelab.agent import moelab_generate_fn

    gen = moelab_generate_fn(model, tok, max_new_tokens=400, streamer=streamer)
    t0 = time.time()
    s = pilot.run(generate_fn=gen, on_step=on_step)
    # épreuve (--start 02-10) : un test, pas de profil écrit (il partirait de réglages volontairement mauvais)
    ppath, rpath = pilot.save(args.out, _common.model_tag(args), write_profile=args.start == "actuel")
    print(f"\nRéponse type : {s['depart_reponse_type_s']} -> {s['meilleur_reponse_type_s']} s ({s['gain_pourcent']:+.1f} %), "
          f"{s['essais']} essais en {(time.time() - t0) / 60:.0f} min ; gardés : {s['gardes'] or 'aucun'}")
    print(f"Profil : {ppath or 'pas écrit (épreuve, ou aucun gain gardé)'}\nRapport : {rpath}")
    rt.close()


if __name__ == "__main__":
    main()
