"""
Étape 13 : benchmark A/B des politiques de précision, sur le même modèle, le même
matériel et les mêmes budgets mémoire. C'est aussi l'expérience qui répond à
l'issue Colibrì #249 : la précision dynamique par expert (HOBBIT) fait-elle mieux
que la quantification statique uniforme ?

Réglages comparés (par défaut) :
    hi            tout en haute précision (fp16) — la référence, qualité exacte
    lo            tout en int4 — quantification statique (ce que fait Colibrì)
    hobbit        précision choisie par expert et par token (t1 = 0.6)
    hobbit_edges  hobbit + couches 0, 1 et dernière toujours en fp16 (pattern en U, doc 07)
    hobbit@0.3    un seuil t1 propre au réglage (balayage : --configs hi,lo,hobbit@0.3,hobbit@0.6)

Chaque réglage repart de caches vides + le même préchargement, puis :
  1. vitesse : génération gloutonne sur les prompts de bench/prompts.json
  2. qualité : teacher forcing sur bench/eval_text.txt, comparé à 'hi'

    python scripts\\benchmark.py --model D:\\models\\olmoe-hf --store C:\\moe-store\\olmoe
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime
from pathlib import Path

import _common
import torch

from moelab.bench import (accept, chat_ids, environment, eval_chunks, load_eval_text, load_prompts,
                          markdown_table, median, quality, teacher_forced_logits, timed_generation, decode_forced_logits)
from moelab.cache import UsageLog
from moelab.policy import PrecisionPolicy


def make_configs(names: list[str], t1: float, n_layers: int, t1_layers=None, miss_t1: float = 0.3) -> dict[str, PrecisionPolicy]:
    """
    Noms reconnus : hi, lo, hobbit, hobbit_edges, hobbit_skip, hobbit_cal (E1 : seuils par
    couche de <store>/t1_layers.json), hobbit_miss (E3 : --miss-t1 pour les experts absents
    de la VRAM). Un seuil propre à un
    réglage s'écrit avec @ : "hobbit@0.3" (balayage de t1 dans un seul benchmark).
    """
    def build(base: str, t: float) -> PrecisionPolicy | None:
        return {
            "hi": lambda: PrecisionPolicy("hi"),
            "lo": lambda: PrecisionPolicy("lo"),
            "hobbit": lambda: PrecisionPolicy("hobbit", t1=t),
            "hobbit_edges": lambda: PrecisionPolicy("hobbit", t1=t, protect_layers={0, 1, n_layers - 1}),
            "hobbit_skip": lambda: PrecisionPolicy("hobbit", t1=t, t2=max(0.9, t)),
            "hobbit_cal": lambda: PrecisionPolicy("hobbit", t1=t, t1_layers=t1_layers),
            "hobbit_miss": lambda: PrecisionPolicy("hobbit", t1=t, miss_t1=miss_t1),
        }.get(base, lambda: None)()

    out = {}
    for n in names:
        base, _, t = n.partition("@")
        try:
            pol = build(base, float(t) if t else t1)
        except ValueError:
            pol = None
        if pol is None:
            raise SystemExit(f"réglage inconnu : {n} (choix : hi, lo, hobbit, hobbit_edges, hobbit_skip, ex. hobbit@0.3)")
        out[n] = pol
    return out


def run_once(model, rt, tok, policy, prompts, chunks, max_new, preload, qmode="decode") -> dict:
    rt.policy = rt.cfg.policy = policy
    rt.reset_caches()
    # Historique vierge et non sauvegardé : sinon l'ordre des réglages influencerait le préchargement.
    rt.usage = UsageLog(None, rt.store.store_id)
    rt.warm_start(preload_lo=preload, log=lambda *a: None)
    per_prompt = []
    for p in prompts:
        per_prompt.append(timed_generation(model, chat_ids(tok, p, model.moelab_device), max_new, tok.eos_token_id))
        rt.end_of_request()
    stats = rt.stats.as_dict()
    q_logits = None
    if chunks:
        # Qualité de la règle seule : caches vidés et pas de « fp16 gratuit » pris dans le
        # cache, sinon le résultat dépend de ce que les prompts précédents y ont laissé.
        rt.policy = rt.cfg.policy = dataclasses.replace(policy, upgrade=False)
        rt.reset_caches()
        if qmode == "decode":  # token par token, caches remplis comme au démarrage (E3 dépend de la VRAM)
            rt.warm_start(preload_lo=preload, log=lambda *a: None)
            q_logits = decode_forced_logits(model, chunks)
        else:
            q_logits = teacher_forced_logits(model, chunks)
        rt.policy = rt.cfg.policy = policy
    return {"per_prompt": per_prompt, "stats": stats, "logits": q_logits}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common.add_model_args(ap)
    ap.add_argument("--configs", default="hi,lo,hobbit,hobbit_edges")
    ap.add_argument("--t1", type=float, default=0.6)
    ap.add_argument("--miss-t1-bench", type=float, default=0.3, help="seuil E3 du réglage hobbit_miss")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--prompts", type=int, default=8, help="nombre de prompts utilisés (max 8)")
    ap.add_argument("--eval-tokens", type=int, default=256, help="tokens de texte pour la qualité (0 = pas de mesure)")
    ap.add_argument("--repeats", type=int, default=1, help="répétitions, en ordre alterné (ABCD puis DCBA)")
    ap.add_argument("--save-ref", default=None, help="enregistrer la référence de qualité (logits en haute précision) dans ce fichier")
    ap.add_argument("--ref", default=None, help="référence de qualité enregistrée par --save-ref (ex. celle du store fp16)")
    ap.add_argument("--quality-mode", default="decode", choices=["decode", "forward"],
                    help="qualité mesurée token par token comme en génération (défaut, seul mode où E3 agit) "
                         "ou en une passe sur tout le texte (forward, plus rapide)")
    ap.add_argument("--out", default=str(_common.ROOT / "results"))
    args = ap.parse_args()
    args.policy, args.t2, args.protect, args.hi_vote = "hi", None, "", 0.5

    model, rt, tok = _common.load(args)
    names_req = args.configs.split(",")
    t1_layers = _common.load_t1_layers("auto", args.store) if any(n.startswith("hobbit_cal") for n in names_req) else None
    configs = make_configs(names_req, args.t1, len(rt.store.layers), t1_layers, args.miss_t1_bench)
    prompts = load_prompts()[: args.prompts]
    chunks = eval_chunks(tok, load_eval_text(), args.eval_tokens) if args.eval_tokens else []
    preload = rt.preload_lo
    lo_gib = rt.store.total_bytes(rt.store.lo) / 2**30
    host_gib = rt.cfg.host_budget_bytes / 2**30
    if not preload:
        print(f"  Note : pool {rt.store.lo} ({lo_gib:.2f} Gio) non préchargé avec un budget RAM de {host_gib:.2f} Gio "
              "(préchargement si le pool prend au plus 60% du budget). Les vitesses ne se comparent qu'à des "
              "benchmarks lancés avec le même budget : fixe-le avec --ram-gb.")

    ref = None
    if chunks and args.ref:
        # référence calculée sur un autre store (ex. fp16), pour mesurer aussi la perte de la haute précision
        saved = torch.load(args.ref)
        if saved["eval_tokens"] != args.eval_tokens or len(saved["logits"]) != len(chunks):
            raise SystemExit(f"{args.ref} : référence faite avec --eval-tokens {saved['eval_tokens']}, relance avec la même valeur")
        ref = saved["logits"]
        print(f"Référence de qualité : {args.ref} ({saved['store']}, {saved['precision']})")
    elif chunks:
        print("Référence de qualité (tout en haute précision) ...")
        rt.policy = rt.cfg.policy = PrecisionPolicy("hi")
        ref = teacher_forced_logits(model, chunks)
        if args.save_ref:
            torch.save({"logits": ref, "eval_tokens": args.eval_tokens, "store": str(rt.store.path),
                        "precision": rt.store.hi}, args.save_ref)
            print(f"  référence enregistrée : {args.save_ref} (à passer à --ref pour un autre store)")

    order = []
    names = list(configs)
    for r in range(args.repeats):
        order += names if r % 2 == 0 else names[::-1]
    runs: dict[str, list] = {n: [] for n in names}
    for i, name in enumerate(order, 1):
        print(f"[{i}/{len(order)}] {name} : {configs[name].describe()} ...")
        res = run_once(model, rt, tok, configs[name], prompts, chunks, args.max_new_tokens, preload, args.quality_mode)
        s = median([p["decode_tok_s"] for p in res["per_prompt"]])
        print(f"    {s:.2f} tok/s en génération" if s else "    vitesse non mesurée")
        runs[name].append(res)

    results = {}
    for name, rs in runs.items():
        pp = [p for r in rs for p in r["per_prompt"]]
        results[name] = {
            "policy": configs[name].describe(),
            "speed": {"decode_tok_s": median([p["decode_tok_s"] for p in pp]), "ttft_s": median([p["ttft_s"] for p in pp])},
            "stats": rs[-1]["stats"],
            "quality": quality(ref, rs[-1]["logits"], chunks) if ref else None,
            "per_prompt": [{k: v for k, v in p.items() if k != "tokens"} for p in pp],
        }
    base = "hi" if "hi" in results else names[0]
    for name, r in results.items():
        if name != base:
            r["accepted"], r["reasons"] = accept(r, results[base])

    table = markdown_table(results, base)
    print()
    print(table)
    for name, r in results.items():
        if r.get("reasons"):
            print(f"  {name} rejeté : {'; '.join(r['reasons'])}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    tag = _common.model_tag(args)
    payload = {"date": stamp, "model": str(args.model), "args": vars(args), "environment": environment(rt), "results": results}
    (out / f"benchmark_{tag}_{stamp}.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    env = payload["environment"]
    md = [f"# Benchmark {tag} — {stamp}", "",
          f"Machine : {env.get('gpu') or 'CPU seul'}, {env['ram_total_gib']} Gio RAM, {env['platform']}",
          f"Store : {env['store']} ({', '.join(env['store_precisions'])}) ; budgets VRAM {env['fast_budget_gib']} Gio, "
          f"RAM {env['host_budget_gib']} Gio ; B_PCIe {env['b_pcie_gbs']} Go/s, B_Host {env['b_host_gbs']} Go/s",
          f"Prompts : {len(prompts)} x {args.max_new_tokens} tokens ; qualité sur {sum(c.shape[1] for c in chunks)} tokens ; "
          f"répétitions : {args.repeats}", "", table, ""]
    (out / f"benchmark_{tag}_{stamp}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nRésultats : {out / f'benchmark_{tag}_{stamp}.md'} (+ .json avec les données brutes)")
    rt.close()


if __name__ == "__main__":
    main()
