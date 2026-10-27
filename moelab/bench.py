"""
Mesures pour le benchmark A/B (étape 13) et la contribution à Colibrì (étape 16).

Qualité — on compare au modèle d'origine (tout en haute précision), en
"teacher forcing" sur un texte fixe : à chaque position, on donne le vrai début
du texte et on regarde la distribution du token suivant.
  - KL moyenne       : écart entre les distributions (0 = identique)
  - accord top-1     : % de positions où le token le plus probable est le même
  - perplexité       : qualité de prédiction du vrai texte ; on rapporte l'écart en %

Vitesse — génération gloutonne token par token avec cache KV :
  - TTFT             : temps jusqu'au premier token (prefill + 1er token)
  - tokens/s         : vitesse de génération après le premier token

Règle d'acceptation (adaptée de `coli tune`) : un réglage n'est retenu que si la
génération est au moins 3% plus rapide que la référence, le TTFT pas plus de 20%
plus lent, et la qualité reste dans les bornes (accord top-1 >= 99%, perplexité
+1% au plus). Colibrì exige aussi un taux de hit stable à 0,5 point près ; ici les
réglages comparés changent volontairement la précision, donc on rapporte les taux
de hit sans en faire un critère.
"""

from __future__ import annotations

import json
import math
import platform
import statistics
import time
from pathlib import Path

import psutil
import torch

BENCH_DIR = Path(__file__).resolve().parents[1] / "bench"


def load_eval_text() -> str:
    return (BENCH_DIR / "eval_text.txt").read_text(encoding="utf-8")


def load_prompts() -> list[str]:
    return json.loads((BENCH_DIR / "prompts.json").read_text(encoding="utf-8"))


def chat_ids(tokenizer, prompt: str, device) -> torch.Tensor:
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                            return_tensors="pt", return_dict=False)
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return ids.to(device)


def eval_chunks(tokenizer, text: str, n_tokens: int, chunk: int = 256) -> list[torch.Tensor]:
    ids = tokenizer(text, return_tensors="pt").input_ids[0][:n_tokens]
    return [ids[i : i + chunk].unsqueeze(0) for i in range(0, len(ids), chunk) if len(ids[i : i + chunk]) > 1]


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def teacher_forced_logits(model, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    dev = model.moelab_device
    out = []
    for c in chunks:
        logits = model(input_ids=c.to(dev), use_cache=False).logits[0].float().cpu()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("logits non finis (NaN/inf) : essaie --dtype bfloat16 ou float32")
        out.append(logits)
    return out


@torch.no_grad()
def decode_forced_logits(model, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Même chose que teacher_forced_logits, mais token par token avec le décodeur de la
    génération (cache KV, CUDA graphs) : chaque token passe par le chemin « 1 token »
    du runtime, celui où s'appliquent E3 (précision selon la présence en VRAM) et les
    noyaux de génération. C'est la qualité que l'on obtient vraiment en discutant.
    """
    from .generate import make_decoder

    dev = model.moelab_device
    out = []
    for c in chunks:
        ids = c.to(dev)
        dec = make_decoder(model)
        lg = [dec.prefill(ids[:, :1]).float()]
        for t in range(1, ids.shape[1]):
            lg.append(dec.step(ids[:, t: t + 1]).float())
        logits = torch.cat(lg).cpu()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("logits non finis (NaN/inf) : essaie --dtype bfloat16 ou float32")
        out.append(logits)
    return out


def quality(ref: list[torch.Tensor], test: list[torch.Tensor], chunks: list[torch.Tensor]) -> dict:
    kl_sum = agree = n = 0
    nll_ref = nll_test = 0.0
    for r, t, c in zip(ref, test, chunks):
        lr, lt = r.double().log_softmax(-1), t.double().log_softmax(-1)
        kl_sum += (lr.exp() * (lr - lt)).sum(-1).sum().item()
        agree += (r.argmax(-1) == t.argmax(-1)).sum().item()
        n += r.shape[0]
        target = c[0, 1:]
        nll_ref -= lr[:-1].gather(-1, target[:, None]).sum().item()
        nll_test -= lt[:-1].gather(-1, target[:, None]).sum().item()
    n_pred = sum(c.shape[1] - 1 for c in chunks)
    ppl_ref, ppl_test = math.exp(nll_ref / n_pred), math.exp(nll_test / n_pred)
    return {
        "kl": kl_sum / n,
        "top1_agreement": agree / n,
        "ppl_ref": ppl_ref,
        "ppl": ppl_test,
        "dppl_pct": 100 * (ppl_test / ppl_ref - 1),
        "positions": n,
    }


@torch.no_grad()
def timed_generation(model, ids: torch.Tensor, max_new_tokens: int, eos_token_id=None) -> dict:
    """
    Génération gloutonne chronométrée (prefill et décodage séparés), avec le même
    décodeur que chat.py : CUDA graphs si RuntimeConfig.cuda_graphs (A6), sinon passe normale.
    Le 1er token de décodage (échauffement + capture des graphs) est compté dans le TTFT.
    """
    from .generate import make_decoder

    dev = model.moelab_device
    eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]) - {None}
    dec = make_decoder(model)
    t0 = time.perf_counter()
    nxt = dec.prefill(ids).argmax(-1, keepdim=True).view(1, 1)
    tokens = [int(nxt)]
    if tokens[-1] not in eos and max_new_tokens > 1:
        nxt = dec.step(nxt).argmax(-1, keepdim=True).view(1, 1)
        tokens.append(int(nxt))
    _sync(dev)
    ttft = time.perf_counter() - t0
    t1 = time.perf_counter()
    for _ in range(max_new_tokens - len(tokens)):
        if tokens[-1] in eos:
            break
        if dec.max_len is not None and ids.shape[1] + len(tokens) >= dec.max_len:
            break
        nxt = dec.step(nxt).argmax(-1, keepdim=True).view(1, 1)
        tokens.append(int(nxt))
    _sync(dev)
    decode_s = time.perf_counter() - t1
    n_dec = len(tokens) - 2
    return {
        "prompt_tokens": ids.shape[1],
        "new_tokens": len(tokens),
        "ttft_s": ttft,
        "decode_tok_s": n_dec / decode_s if n_dec > 0 and decode_s > 0 else None,
        "tokens": tokens,
        "cuda_graphs": bool(getattr(dec, "active", False)),
    }


def accept(cand: dict, base: dict, min_speedup: float = 0.03, max_ttft_ratio: float = 1.2,
           min_top1: float = 0.99, max_dppl_pct: float = 1.0) -> tuple[bool, list[str]]:
    reasons = []
    s_c, s_b = cand["speed"]["decode_tok_s"], base["speed"]["decode_tok_s"]
    if not (s_c and s_b and s_c >= s_b * (1 + min_speedup)):
        reasons.append(f"vitesse {s_c:.2f} vs {s_b:.2f} tok/s (< +{100 * min_speedup:.0f}%)" if s_c and s_b else "vitesse non mesurée")
    if cand["speed"]["ttft_s"] > base["speed"]["ttft_s"] * max_ttft_ratio:
        reasons.append(f"TTFT {cand['speed']['ttft_s']:.2f} s vs {base['speed']['ttft_s']:.2f} s (> +{100 * (max_ttft_ratio - 1):.0f}%)")
    q = cand.get("quality")
    if q:
        if q["top1_agreement"] < min_top1:
            reasons.append(f"accord top-1 {100 * q['top1_agreement']:.1f}% < {100 * min_top1:.0f}%")
        if q["dppl_pct"] > max_dppl_pct:
            reasons.append(f"perplexité +{q['dppl_pct']:.2f}% > +{max_dppl_pct:.0f}%")
    return not reasons, reasons


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def environment(runtime=None) -> dict:
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "cpu_threads": psutil.cpu_count(),
        "ram_total_gib": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else None,
    }
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover
        pass
    if runtime is not None:
        c = runtime.cfg
        env.update({
            "store": str(runtime.store.path),
            "store_precisions": runtime.store.precisions,
            "fast_budget_gib": round((c.fast_budget_bytes or 0) / 2**30, 2),
            "host_budget_gib": round(c.host_budget_bytes / 2**30, 2),
            "b_pcie_gbs": round(c.b_pcie / 1e9, 2),
            "b_host_gbs": round(c.b_host / 1e9, 2),
            "prefetch": c.prefetch,
            "qstar": c.use_qstar,
        })
    return env


def markdown_table(results: dict, baseline: str) -> str:
    head = ("| Réglage | Politique | tok/s | TTFT (s) | accord top-1 | écart perplexité | KL | VRAM hit | haute préc. demandée | haute préc. utilisée | CPU | disque/token | prefetch | verdict |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = [head]
    for name, r in results.items():
        q, s, st = r.get("quality") or {}, r["speed"], r["stats"]
        fmt = lambda v, f: "-" if v is None else f.format(v)  # noqa: E731
        verdict = "référence" if name == baseline else ("retenu" if r.get("accepted") else "rejeté")
        rows.append("| " + " | ".join([
            name, r["policy"], fmt(s["decode_tok_s"], "{:.2f}"), fmt(s["ttft_s"], "{:.2f}"),
            fmt(q.get("top1_agreement"), "{:.1%}"), fmt(q.get("dppl_pct"), "{:+.2f}%"), fmt(q.get("kl"), "{:.3g}"),
            fmt(st.get("fast_hit_rate") if st.get("fast_hits") or st.get("fills") else None, "{:.0%}"),
            fmt(st.get("need_hi_share"), "{:.0%}"), fmt(st.get("hi_share"), "{:.0%}"), fmt(st.get("cpu_share"), "{:.0%}"), fmt(st.get("disk_per_token"), "{:.1f}"),
            fmt(st.get("prefetch_recall"), "{:.0%}"), verdict,
        ]) + " |")
    return "\n".join(rows)
