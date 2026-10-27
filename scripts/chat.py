"""
Discuter avec le modèle déchargé, avec les statistiques du système après chaque réponse.

    python scripts\\chat.py --model D:\\models\\olmoe-hf --store C:\\moe-store\\olmoe

Commandes pendant la discussion :
    /stats                 état des caches VRAM/RAM
    /options               les réglages, leur valeur actuelle et leur effet mesuré (/options all : tous)
    /set <réglage> <val>   changer un réglage à chaud, ex. /set t1 0.5, /set cuda_graphs off
    /preset <nom>          vitesse, équilibre, qualité, référence
    /policy hi|lo|hobbit   raccourci de /set policy
    /t1 0.5                raccourci de /set t1
    /reset                 oublier la conversation
    /quit
"""

from __future__ import annotations

import argparse
import re
import time

import _common
import torch
from transformers import TextStreamer

from moelab import options
from moelab.generate import generate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common.add_model_args(ap)
    _common.add_policy_args(ap)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7, help="0 = glouton")
    ap.add_argument("--system", default=None, help="message système optionnel")
    ap.add_argument("--no-reuse", action="store_true",
                    help="P1 désactivé : recalculer toute la conversation à chaque message (avant le 03/10)")
    ap.add_argument("--no-profile", action="store_true",
                    help="ne pas appliquer <store>/autopilot_profile.json (réglages trouvés par l'autopilot)")
    ap.add_argument("--no-warmup", action="store_true",
                    help="pas d'échauffement au chargement (compilation des noyaux au 1er message)")
    args = ap.parse_args()

    print("Chargement ...")
    t0 = time.time()
    model, rt, tok = _common.load(args)
    if not args.no_profile:  # réglages trouvés par l'autopilot (doc 19) ; ceux donnés en ligne de commande priment
        from moelab.autopilot import apply_profile

        apply_profile(model, args.store, skip=tuple(k for k in ("vram_gb", "ram_gb", "read_parts")
                                                    if getattr(args, k, None) is not None), log=print)
    if not args.no_warmup:
        from moelab.generate import warmup

        warmup(model, tok, log=print)
    print(f"Prêt en {time.time() - t0:.1f} s — politique {rt.policy.describe()} — {rt.describe()}")
    if rt.cfg.cuda_graphs and model.moelab_device.type == "cuda":
        print(f"CUDA graphs (A6) : capturés au 1er token de la 1re réponse, cache KV statique de {rt.cfg.graph_max_len} tokens")
    print("Tape ton message (/quit pour sortir).")
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    history = [{"role": "system", "content": args.system}] if args.system else []

    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/stats":
            print(rt.describe())
            continue
        if user == "/reset":
            history = history[:1] if args.system else []
            print("conversation effacée")
            continue
        if user.startswith("/") or re.search(r"\s/(set|preset|options|policy|t1|stats|reset|quit)\b", user):
            if not user.startswith("/"):
                print("les commandes se tapent seules, sur leur propre ligne (ex. : /set t1 0.5). Message non envoyé.")
                continue
            parts = user.split()
            cmd, rest = parts[0], parts[1:]
            try:
                if cmd == "/options":
                    print(options.table(model, advanced=bool(rest and rest[0] == "all")))
                elif cmd == "/set" and len(rest) == 2:
                    print(options.apply(model, rest[0], rest[1]))
                elif cmd == "/preset" and len(rest) == 1:
                    for line in options.apply_preset(model, rest[0]):
                        print(line)
                elif cmd == "/policy" and len(rest) == 1:
                    print(options.apply(model, "policy", rest[0]))
                elif cmd == "/t1" and len(rest) == 1:
                    print(options.apply(model, "t1", rest[0]))
                else:
                    print("commande inconnue. Commandes : /stats, /options, /set <réglage> <valeur>, /preset <nom>, "
                          "/policy hi|lo|hobbit, /t1 0.5, /reset, /quit")
            except (KeyError, ValueError) as e:
                print(f"impossible : {e}")
            continue

        history.append({"role": "user", "content": user})
        ids = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt", return_dict=False)
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        ids = ids.to(model.moelab_device)
        rt.reset_stats()
        t0 = time.perf_counter()
        tm: dict = {}
        new = generate(model, ids, args.max_new_tokens, temperature=args.temperature, top_p=0.9,
                       streamer=streamer, log=print, reuse=rt.cfg.kv_reuse and not args.no_reuse, timings=tm)
        dt = time.perf_counter() - t0
        history.append({"role": "assistant", "content": tok.decode(new, skip_special_tokens=True)})
        gen_s = dt - tm.get("ttft_s", 0.0)
        print(f"\n[{_common.stats_line(rt, max(len(new) - 1, 0), gen_s)} | 1er token {tm.get('ttft_s', 0):.1f} s, "
              f"prompt {tm.get('prompt_tokens', 0)} tokens dont {tm.get('reused', 0)} repris du cache]")
        rt.end_of_request()

    rt.end_of_request()
    rt.close()
    print("Historique d'usage enregistré (les experts chauds seront préchargés au prochain lancement).")


if __name__ == "__main__":
    main()
