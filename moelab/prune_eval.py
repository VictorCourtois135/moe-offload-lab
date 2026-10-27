"""
Banc d'évaluation de l'élagage (08/10) : une mesure = un JSON comparable aux autres.

Trois mesures, dans cet ordre (la vitesse d'abord, caches dans l'état d'après le chargement) :

  1. vitesse    : génération gloutonne de quelques réponses (bench/prompts.json) : tokens/s,
                  temps avant le 1er token, taux de hit VRAM, lectures disque par token ;
  2. outils     : bench/tool_tasks.json, 20 questions en un tour, notées automatiquement :
                  le bon outil avec les bons arguments, ou une réponse directe qui contient
                  ce qu'il faut. C'est l'usage réel du modèle (chat avec outils, agents) ;
                  la perplexité seule voit mal ce que l'élagage abîme (génération, REAP §5) ;
  3. perplexité : WikiText-2 test et Wikipédia en français (prune_data.py), par lots.
                  Le token le plus probable à chaque position est gardé (<nom>_top1.pt) :
                  une mesure suivante donne son accord top-1 avec la référence.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

import torch

from .reap import bulk_mode, hidden_states, run_batches

BENCH = Path(__file__).resolve().parents[1] / "bench"


# ----------------------------------------------------------------------------
# Perplexité
# ----------------------------------------------------------------------------
@torch.no_grad()
def perplexity(model, ids: torch.Tensor, batch: int = 8, chunk: int = 128, log=print, name: str = "") -> dict:
    """NLL moyenne et perplexité sur ids (N x T), plus le top-1 à chaque position (N x T-1)."""
    head = model.get_output_embeddings()
    rt = model.moelab_runtime
    N, T = ids.shape
    top1 = torch.zeros(N, T - 1, dtype=torch.int32)
    nll = torch.zeros(N, dtype=torch.float64)
    t0 = time.time()
    pos = {"i": 0}

    def one(sub):
        h = hidden_states(model, sub)
        for r in range(sub.shape[0]):
            row = pos["i"] + r
            tgt = sub[r, 1:].to(torch.int64)
            for s in range(0, T - 1, chunk):
                e = min(s + chunk, T - 1)
                lg = head(h[r:r + 1, s:e]).float()[0]
                lp = torch.log_softmax(lg, dim=-1).cpu()
                nll[row] -= lp.gather(-1, tgt[s:e, None]).sum().double()
                top1[row, s:e] = lp.argmax(-1).to(torch.int32)
        pos["i"] += sub.shape[0]

    def after(j, b):
        log(f"  perplexité {name} : {j}/{N} extraits ({time.time() - t0:.0f} s)")

    with bulk_mode(rt):
        run_batches(ids, batch, one, log=log, on_batch=after)
    n_pred = N * (T - 1)
    mean = float(nll.sum() / n_pred)
    return {"ppl": math.exp(mean), "nll": mean, "extraits": N, "tokens_predits": n_pred,
            "secondes": round(time.time() - t0, 1), "top1": top1}


def agreement(top1: torch.Tensor, ref: torch.Tensor) -> float | None:
    if ref is None or ref.shape != top1.shape:
        return None
    return float((top1 == ref).float().mean())


# ----------------------------------------------------------------------------
# Appels d'outils
# ----------------------------------------------------------------------------
def load_tool_tasks(path=None) -> dict:
    return json.loads(Path(path or BENCH / "tool_tasks.json").read_text(encoding="utf-8"))


def _norm(s) -> str:
    return " ".join(str(s).lower().split())


def _has(text: str, needle) -> bool:
    """needle : une chaîne, ou une liste d'alternatives (une seule suffit)."""
    alts = needle if isinstance(needle, list) else [needle]
    return any(_norm(a) in text for a in alts)


def score_task(task: dict, text: str, calls: list[dict]) -> tuple[bool, str]:
    """(réussi, raison) pour une réponse du modèle."""
    want = task.get("outil")
    if want is None:
        if calls:
            return False, f"outil appelé alors qu'il fallait répondre : {calls[0].get('name')}"
        t = _norm(text)
        miss = [n for n in task.get("contient", []) if not _has(t, n)]
        return (not miss), ("" if not miss else f"manque {miss}")
    if not calls:
        return False, "aucun appel d'outil"
    c = calls[0]
    names = want if isinstance(want, list) else [want]
    if c.get("name") not in names:
        return False, f"mauvais outil : {c.get('name')} (attendu {' ou '.join(names)})"
    args = c.get("arguments") or {}
    if not isinstance(args, dict):
        return False, "arguments illisibles"
    for key, needles in (task.get("args") or {}).items():
        val = _norm(" ".join(str(v) for v in args.values())) if key == "*" else _norm(args.get(key, ""))
        if key != "*" and key not in args:
            return False, f"argument {key} absent"
        miss = [n for n in needles if not _has(val, n)]
        if miss:
            return False, f"{key} : manque {miss}"
    return True, ""


@torch.no_grad()
def tool_tasks(model, tok, tasks: dict | None = None, max_new_tokens: int = 192, log=print) -> dict:
    from .agent import Tool, parse_tool_calls, render
    from .generate import generate

    tasks = tasks or load_tool_tasks()
    tools = tasks["outils"]
    known = {f["name"]: Tool(f["name"], f.get("description", ""), f["parameters"], lambda **kw: None)
             for f in (t["function"] for t in tools)}
    specials = sorted({s for s in [tok.eos_token, tok.pad_token, "<|im_end|>", "<|endoftext|>",
                                    *getattr(tok, "all_special_tokens", [])] if s and "tool_call" not in s},
                      key=len, reverse=True)
    details, ok = [], 0
    t0 = time.time()
    for task in tasks["taches"]:
        msgs = [{"role": "system", "content": tasks["systeme"]}, {"role": "user", "content": task["question"]}]
        ids = render(tok, msgs, tools).to(model.moelab_device)
        new = generate(model, ids, max_new_tokens, temperature=0.0)
        text = tok.decode(new, skip_special_tokens=False)
        for s in specials:
            text = text.replace(s, "")
        rest, calls = parse_tool_calls(text, known)
        good, why = score_task(task, rest if task.get("outil") is None else text, calls)
        ok += good
        details.append({"id": task["id"], "ok": good, "raison": why, "reponse": text.strip()[:400]})
        log(f"  outils : {task['id']:18s} {'ok' if good else 'raté  ' + why}")
    n = len(tasks["taches"])
    return {"score": ok / n if n else 0.0, "reussies": ok, "total": n, "secondes": round(time.time() - t0, 1),
            "details": details}


# ----------------------------------------------------------------------------
# Vitesse
# ----------------------------------------------------------------------------
@torch.no_grad()
def speed(model, tok, prompts: list[str] | None = None, max_new_tokens: int = 64, log=print) -> dict:
    from .bench import chat_ids, load_prompts, timed_generation

    rt = model.moelab_runtime
    prompts = prompts or load_prompts()[:3]
    rt.reset_stats()
    runs = []
    for p in prompts:
        r = timed_generation(model, chat_ids(tok, p, model.moelab_device), max_new_tokens,
                             eos_token_id=getattr(model.generation_config, "eos_token_id", None))
        runs.append(r)
        tps = r["decode_tok_s"]
        log(f"  vitesse : {r['new_tokens']} tokens, 1er token {r['ttft_s']:.1f} s, "
            f"{'%.1f tok/s' % tps if tps else '-'}")
    st = rt.stats.as_dict()
    tps = [r["decode_tok_s"] for r in runs if r["decode_tok_s"]]
    return {"tok_s": statistics.median(tps) if tps else None, "ttft_s": statistics.median(r["ttft_s"] for r in runs),
            "hit_vram": st["fast_hit_rate"], "lectures_disque_par_token": st["disk_per_token"],
            "hit_ram": st["host_hits"] / max(st["requests"] - st["fast_hits"], 1), "invites": len(runs)}


# ----------------------------------------------------------------------------
# Une mesure complète
# ----------------------------------------------------------------------------
def evaluate(model, tok, eval_ids: dict, name: str, out_dir, batch: int = 8, ref_top1: dict | None = None,
             parts=("vitesse", "outils", "perplexite"), extra: dict | None = None, log=print) -> dict:
    """Mesure et écrit <out_dir>/<name>.json (+ <name>_top1.pt). ref_top1 : top-1 de la référence."""
    from .bench import environment
    from .generate import warmup

    rt = model.moelab_runtime
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    store = rt.store
    res = {"nom": name, "date": time.strftime("%Y-%m-%d %H:%M"),
           "store": {"chemin": str(store.path), "id": store.store_id, "couches": len(store.layers),
                     "experts": store.num_experts,
                     "gio": round(sum(store.total_bytes(p) for p in store.precisions) / 2**30, 2)},
           "moteur": rt.describe(), "politique": rt.policy.describe(), **(extra or {})}
    try:
        res["environnement"] = environment(rt)
    except Exception:
        pass
    warmup(model, tok)
    if "vitesse" in parts:
        log("Vitesse ...")
        res["vitesse"] = speed(model, tok, log=log)
    if "outils" in parts:
        log("Appels d'outils (20 questions) ...")
        res["outils"] = tool_tasks(model, tok, log=log)
    if "perplexite" in parts:
        log("Perplexité ...")
        res["perplexite"], tops = {}, {}
        for key in ("wikitext", "fr"):
            if key not in eval_ids:
                continue
            p = perplexity(model, eval_ids[key].to(torch.int64), batch=batch, log=log, name=key)
            tops[key] = p.pop("top1")
            if ref_top1 is not None:
                p["accord_top1_ref"] = agreement(tops[key], ref_top1.get(key))
            res["perplexite"][key] = p
        torch.save(tops, out_dir / f"{name}_top1.pt")
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return res


def short_report(res: dict) -> str:
    lines = [f"{res['nom']} ({res['minutes']} min)"]
    v = res.get("vitesse")
    if v:
        lines.append(f"  vitesse    : {v['tok_s'] or 0:.1f} tok/s, 1er token {v['ttft_s']:.1f} s, "
                     f"hit VRAM {100 * v['hit_vram']:.0f} %, {v['lectures_disque_par_token']:.1f} lectures disque/token")
    o = res.get("outils")
    if o:
        lines.append(f"  outils     : {o['reussies']}/{o['total']}")
    for k, p in (res.get("perplexite") or {}).items():
        acc = p.get("accord_top1_ref")
        lines.append(f"  perplexité {k:8s} : {p['ppl']:.3f}" + (f" (accord top-1 avec la référence {100 * acc:.1f} %)"
                                                            if acc is not None else ""))
    return "\n".join(lines)
