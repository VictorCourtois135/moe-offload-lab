"""
Le temps avant le premier token (TTFT) : où part-il, et que gagnent P1 à P5 (doc 18) ?

    python scripts\\profile_prefill.py --model D:\\models\\qwen3-next-80b-fp8 --store C:\\moe-store\\qwen3-next --ram-gb 5

Mesure, pour chaque variante :
  1. le prefill de plusieurs prompts (courts, et un long fait de plusieurs prompts mis bout à
     bout) : temps total, part des couches MoE et de la partie dense, détail de chaque phase
     d'une couche (GPU compris), experts distincts par couche, d'où ils viennent (VRAM, RAM,
     disque), débit disque obtenu comparé au débit mesuré par measure_bandwidth.py ;
  2. les tokens générés juste après (vitesse : un prefill qui vide le cache VRAM ralentit la
     suite), et leur qualité par rapport à la variante de référence (KL, accord top-1 ;
     seul P5 doit changer quelque chose) ;
  3. une conversation de 3 messages, avec et sans P1 (reprise du cache d'un message au suivant).

Variantes (05/10, défaut) : defaut (avec P6, activé), sans_p6, p6_256, p6_512, skip05, skip10.
--long-tokens 400 ajoute un prompt de 400 tokens (pour le seuil de P6).
2e série : p6 (noyaux groupés token par token), p7 (une lecture
par expert), p8 (bascule du GIL à 0,5 ms), p678, p678_skip05 / p678_skip10 (+ P5).
1re série : p2 (ordre d'arrivée), p3 (prefetch disque), p4 (hors du cache VRAM), p23, p234,
skip05 / skip10. Résultat : results\\prefill_<modèle>_<date>.md (+ .json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import _common
import torch
import torch.nn.functional as F

from moelab.bench import chat_ids, environment, load_prompts
from moelab.cache import UsageLog
from moelab.generate import generate, make_decoder
from moelab.hw import load_bandwidth
from moelab.profiling import PhaseTimer

BASE = {"prefill_order": "fills", "prefill_prefetch": False, "prefill_cache": "vram", "prefill_skip": None,
        "prefill_grouped": 192, "prefill_split": True, "gil_switch_ms": None}
_P678 = {"prefill_grouped": 64, "prefill_split": False, "gil_switch_ms": 0.5}
VARIANTS = {
    # 05/10 : P6 activé par défaut (defaut = avec P6)
    "sans_p6": {"prefill_grouped": 0},
    "p6_64": {"prefill_grouped": 64},
    "p6_256": {"prefill_grouped": 256},
    "p6_512": {"prefill_grouped": 512},
    "skip05": {"prefill_skip": 0.05},
    "skip10": {"prefill_skip": 0.10},
    # 03/10, 2e série : le fil principal ralenti pendant les lectures (P6 à P8)
    "p6": {"prefill_grouped": 64},
    "p7": {"prefill_split": False},
    "p8": {"gil_switch_ms": 0.5},
    "p678": _P678,
    "p678_skip05": {**_P678, "prefill_skip": 0.05},
    "p678_skip10": {**_P678, "prefill_skip": 0.10},
    # 1re série (16 h) : P2, P3, P4 rejetés
    "defaut": {},
    "p2": {"prefill_order": "completed"},
    "p3": {"prefill_prefetch": True},
    "p4": {"prefill_cache": "ram"},
    "p23": {"prefill_order": "completed", "prefill_prefetch": True},
    "p234": {"prefill_order": "completed", "prefill_prefetch": True, "prefill_cache": "ram"},
    "p23_skip05": {"prefill_order": "completed", "prefill_prefetch": True, "prefill_skip": 0.05},
}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def fresh(rt):
    """Mêmes conditions pour chaque variante : caches vidés puis préchargés comme au démarrage."""
    rt.reset_caches()
    rt.usage = UsageLog(None, rt.store.store_id)
    rt.warm_start(preload_lo=rt.preload_lo, log=lambda *a: None)


@torch.no_grad()
def measure_prompt(model, rt, ids, ref: dict | None, n_gen: int) -> dict:
    """Prefill chronométré (phases GPU comprises), puis n_gen tokens : gloutons (référence) ou imposés."""
    dec = make_decoder(model)
    rt.reset_stats()
    rt.prof = PhaseTimer(sync=True)
    sync()
    t0 = time.perf_counter()
    logits = dec.prefill(ids)
    sync()
    t_prefill = time.perf_counter() - t0
    phases = {k: 1000 * v for k, v in rt.prof.totals.items()}  # ms sur tout le prefill
    rt.prof = None
    st = rt.stats.as_dict()
    out = {"tokens": int(ids.shape[1]), "prefill_s": t_prefill, "moe_s": st["t_moe"], "dense_s": t_prefill - st["t_moe"],
           "io_wait_s": st["t_io_wait"], "phases_ms": phases, "layer_calls": st["layer_calls"],
           "experts_per_layer": st["requests"] / max(st["layer_calls"], 1),
           "vram_hits": st["fast_hits"], "ram_hits": st["host_hits"], "disk_reads": st["disk_reads"],
           "disk_gb": st["bytes_disk"] / 1e9, "skipped": st["prefill_skipped"], "requests": st["requests"],
           "disk_ahead": st["disk_ahead"], "cpu_execs": st["cpu_execs"]}
    out["disk_gbps"] = out["disk_gb"] / t_prefill if t_prefill > 0 else 0.0
    # Génération juste après : vitesse (le prefill a-t-il chassé les experts utiles ?) et qualité
    logps = [F.log_softmax(logits.float(), -1).cpu()]
    toks = ref["tokens"] if ref else []
    nxt = logits.argmax(-1, keepdim=True).view(1, 1) if not ref else torch.tensor([[toks[0]]], device=ids.device)
    gen = [int(nxt)]
    rt.reset_stats()
    sync()
    t0 = time.perf_counter()
    for i in range(n_gen - 1):
        lg = dec.step(nxt)
        logps.append(F.log_softmax(lg.float(), -1).cpu())
        nxt = lg.argmax(-1, keepdim=True).view(1, 1) if not ref else torch.tensor([[toks[i + 1]]], device=ids.device)
        gen.append(int(nxt))
    sync()
    dt = time.perf_counter() - t0
    out["gen_tok_s"] = (n_gen - 1) / dt if dt > 0 else 0.0
    out["gen_vram_hit"] = rt.stats.as_dict()["fast_hit_rate"]
    out["tokens_out"] = gen
    lp = torch.cat(logps)
    if ref is not None:
        rp = ref["logp"]
        out["kl"] = float((rp.exp() * (rp - lp)).sum(-1).mean())
        out["top1"] = float((rp.argmax(-1) == lp.argmax(-1)).float().mean())
    out["logp"] = lp
    return out


@torch.no_grad()
def conversation(model, rt, tok, prompts: list[str], reuse: bool, n_gen: int) -> list[dict]:
    """3 messages à la suite, comme dans chat.py : temps avant le premier token de chaque réponse."""
    fresh(rt)
    history, rows = [], []
    dev = model.moelab_device
    for p in prompts:
        history.append({"role": "user", "content": p})
        ids = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt", return_dict=False)
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        tm: dict = {}
        sync()
        new = generate(model, ids.to(dev), n_gen, temperature=0.0, reuse=reuse, timings=tm)
        history.append({"role": "assistant", "content": tok.decode(new, skip_special_tokens=True)})
        rows.append({"prompt_tokens": tm.get("prompt_tokens", 0), "reused": tm.get("reused", 0),
                     "ttft_s": tm.get("ttft_s", 0.0)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common.add_model_args(ap)
    ap.add_argument("--policy", default="lo", choices=["hi", "lo", "hobbit"], help="politique (défaut : lo)")
    ap.add_argument("--t1", type=float, default=0.6)
    ap.add_argument("--variants", default="defaut,sans_p6,skip05,skip10",
                    help=f"variantes mesurées ({', '.join(VARIANTS)}) ; la 1re sert de référence de qualité")
    ap.add_argument("--prompts", type=int, default=3, help="prompts courts mesurés (après 1 d'échauffement)")
    ap.add_argument("--gen", type=int, default=24, help="tokens générés après chaque prefill")
    ap.add_argument("--no-long", action="store_true", help="sans le prompt long (prompts mis bout à bout)")
    ap.add_argument("--long-tokens", type=int, default=0,
                    help="ajoute un prompt « tres_long » d'au moins ce nombre de tokens (ex. 400 : résultat d'outil d'un agent)")
    ap.add_argument("--no-conversation", action="store_true", help="sans la mesure de P1 en conversation")
    ap.add_argument("--out", default=str(_common.ROOT / "results"))
    args = ap.parse_args()
    args.t2, args.protect, args.hi_vote = None, "", 0.5

    variants = args.variants.split(",")
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"variante inconnue : {v} (choix : {', '.join(VARIANTS)})")
    model, rt, tok = _common.load(args)
    dev = model.moelab_device
    texts = load_prompts()
    warm = chat_ids(tok, texts[0], dev)
    tests = [(f"court {i}", chat_ids(tok, texts[i], dev)) for i in range(1, 1 + args.prompts)]
    if not args.no_long:
        long_text = "\n\n".join(texts[1 + args.prompts: 1 + args.prompts + 6] or texts[1:7])
        tests.append(("long", chat_ids(tok, long_text, dev)))
    if args.long_tokens:
        parts, i = [], 0
        while True:
            parts.append(texts[1 + i % (len(texts) - 1)])
            ids_l = chat_ids(tok, "\n\n".join(parts), dev)
            if ids_l.shape[1] >= args.long_tokens or i > 500:
                break
            i += 1
        tests.append(("tres_long", ids_l))
    bw = load_bandwidth(args.store) or {}
    disk_best = max((v for v in (bw.get("disk_parallel") or {}).values() if v), default=None)
    n_layers = len(rt.store.layers)
    report: dict = {"environment": environment(rt), "disk_parallel_best_gbps": disk_best and disk_best / 1e9,
                    "prompts": {name: int(ids.shape[1]) for name, ids in tests}, "variants": {}}
    print(f"Prompts : {', '.join(f'{n} = {t} tokens' for n, t in report['prompts'].items())}"
          + (f" ; disque mesuré : {disk_best / 1e9:.2f} Go/s au mieux" if disk_best else ""))

    refs: dict[str, dict] = {}
    for var in variants:
        for key, val in {**BASE, **VARIANTS[var]}.items():
            setattr(rt.cfg, key, val)
        sys.setswitchinterval((rt.cfg.gil_switch_ms or 5.0) / 1000)  # P8 (5 ms : la valeur de Python)
        fresh(rt)
        print(f"\nVariante {var} : {', '.join(f'{k}={v}' for k, v in VARIANTS[var].items()) or 'chemin par défaut'}")
        measure_prompt(model, rt, warm, None, 8)  # échauffement : capture des CUDA graphs, noyaux
        rows = {}
        for name, ids in tests:
            r = measure_prompt(model, rt, ids, refs.get(name), args.gen)
            if name not in refs:
                refs[name] = {"tokens": r["tokens_out"], "logp": r["logp"]}
            r.pop("logp")
            rows[name] = r
            q = f", KL {r['kl']:.4f}, top-1 {r['top1']:.0%}" if "kl" in r else ""
            print(f"  {name:8s} {r['tokens']:4d} tokens : prefill {r['prefill_s']:.2f} s (MoE {r['moe_s']:.2f}, "
                  f"dense {r['dense_s']:.2f}, attente disque {r['io_wait_s']:.2f}) ; {r['experts_per_layer']:.0f} experts "
                  f"par couche : VRAM {r['vram_hits']}, RAM {r['ram_hits']}, disque {r['disk_reads']} "
                  f"({r['disk_gb']:.2f} Go, {r['disk_gbps']:.2f} Go/s)"
                  + (f", sautés {r['skipped']}" if r["skipped"] else "")
                  + (f", lus à l'avance {r['disk_ahead']}" if r["disk_ahead"] else "")
                  + f" ; puis {r['gen_tok_s']:.2f} tok/s (hit VRAM {r['gen_vram_hit']:.0%}){q}")
        ph: dict[str, float] = {}
        for r in rows.values():
            for k, v in r["phases_ms"].items():
                ph[k] = ph.get(k, 0.0) + v / max(r["layer_calls"], 1)
        ph = {k: v / len(rows) for k, v in ph.items()}
        print("  phases d'une couche MoE au prefill (ms, moyenne des prompts, GPU compris) :")
        for k, v in sorted(ph.items()):
            print(f"    {k:48s} {v:8.2f}")
        report["variants"][var] = {"settings": VARIANTS[var], "prompts": rows, "phases_ms_per_layer": ph}

    for key, val in BASE.items():
        setattr(rt.cfg, key, val)
    sys.setswitchinterval(0.005)
    if not args.no_conversation:
        conv_prompts = texts[1:4]
        report["conversation"] = {}
        for reuse in (False, True):
            rows = conversation(model, rt, tok, conv_prompts, reuse, 32)
            report["conversation"]["avec P1" if reuse else "sans P1"] = rows
            print(f"\nConversation {'avec' if reuse else 'sans'} P1 : "
                  + " ; ".join(f"message {i + 1} : {r['prompt_tokens']} tokens dont {r['reused']} repris, "
                               f"1er token en {r['ttft_s']:.2f} s" for i, r in enumerate(rows)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    tag = _common.model_tag(args)
    for v in report["variants"].values():
        for r in v["prompts"].values():
            r.pop("tokens_out", None)
    (out_dir / f"prefill_{tag}_{stamp}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    md = [f"# Prefill {tag} — {stamp}", "", f"Machine : {report['environment'].get('gpu')} ; politique {rt.policy.describe()} ; "
          f"{n_layers} couches MoE" + (f" ; disque mesuré {disk_best / 1e9:.2f} Go/s au mieux" if disk_best else ""), "",
          "| Variante | Prompt | tokens | prefill (s) | MoE (s) | dense (s) | attente disque (s) | experts/couche | "
          "VRAM | RAM | disque | Go lus | Go/s | sautés | tok/s ensuite | KL | top-1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for var, v in report["variants"].items():
        for name, r in v["prompts"].items():
            md.append(f"| {var} | {name} | {r['tokens']} | {r['prefill_s']:.2f} | {r['moe_s']:.2f} | {r['dense_s']:.2f} | "
                      f"{r['io_wait_s']:.2f} | {r['experts_per_layer']:.0f} | {r['vram_hits']} | {r['ram_hits']} | "
                      f"{r['disk_reads']} | {r['disk_gb']:.2f} | {r['disk_gbps']:.2f} | {r['skipped']} | "
                      f"{r['gen_tok_s']:.2f} | {format(r['kl'], '.4f') if 'kl' in r else '-'} | "
                      f"{format(r['top1'], '.0%') if 'top1' in r else '-'} |")
    md.append("")
    for var, v in report["variants"].items():
        md += [f"### Phases d'une couche MoE au prefill, {var} (ms)", "", "| Phase | ms |", "|---|---|"]
        md += [f"| {k} | {x:.2f} |" for k, x in sorted(v["phases_ms_per_layer"].items())]
        md.append("")
    if "conversation" in report:
        md += ["## Conversation (P1)", "", "| | message | tokens du prompt | repris du cache | 1er token (s) |", "|---|---|---|---|---|"]
        for mode, rows in report["conversation"].items():
            md += [f"| {mode} | {i + 1} | {r['prompt_tokens']} | {r['reused']} | {r['ttft_s']:.2f} |" for i, r in enumerate(rows)]
        md.append("")
    (out_dir / f"prefill_{tag}_{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nRésultats : {out_dir / f'prefill_{tag}_{stamp}.md'} (+ .json)")
    rt.close()


if __name__ == "__main__":
    main()
