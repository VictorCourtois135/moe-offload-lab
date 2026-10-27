"""
S5 (doc 18, 02/10) : le décodage spéculatif vaudrait-il le coup ? On mesure AVANT de construire.

Le décodage spéculatif propose plusieurs tokens d'un coup (un « brouillon »), puis le
vrai modèle les vérifie en UNE passe : sans perte de qualité (chaque token est vérifié),
plus rapide si la vérification de K+1 tokens coûte moins que K+1 pas, et si le
brouillon est souvent juste. La variante sans entraînement (« prompt lookup ») cherche
dans le texte déjà écrit la dernière occurrence des n derniers tokens et propose la suite.

Ce script mesure les deux facteurs sur ta machine :
  1. le taux d'acceptation : génère les réponses des prompts du benchmark (glouton),
     puis rejoue le brouillon hors ligne sur ces réponses (exact en glouton) ;
  2. le coût de vérification : une passe de K+1 tokens (chemin prefill) contre un pas
     de génération normal (CUDA graphs).
Il en déduit le gain attendu. Si le gain est < 1, inutile de construire la fonction.

    python scripts\\measure_lookup.py --model D:\\models\\olmoe-hf --store C:\\moe-store\\olmoe-i8 --ram-gb 5 --policy lo
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import _common
import torch

from moelab.bench import chat_ids, load_prompts
from moelab.generate import generate, make_decoder, pick


def lookup_draft(seq: list[int], k: int, n_max: int = 3) -> list[int]:
    """Prompt lookup : la suite de la dernière occurrence des n derniers tokens (n = n_max ... 1)."""
    for n in range(min(n_max, len(seq) - 1), 0, -1):
        tail = seq[-n:]
        for i in range(len(seq) - n - 1, -1, -1):
            if seq[i:i + n] == tail:
                return seq[i + n:i + n + k]
    return []


def simulate(prompt: list[int], gen: list[int], k: int) -> tuple[int, int, int]:
    """(pas de vérification, tokens produits, tokens de brouillon acceptés) pour une réponse gloutonne."""
    pos, steps, accepted = 0, 0, 0
    while pos < len(gen):
        draft = lookup_draft(prompt + gen[:pos], k)
        a = 0
        while a < len(draft) and pos + a < len(gen) and draft[a] == gen[pos + a]:
            a += 1
        accepted += a
        pos += a + 1  # les tokens acceptés + celui que la vérification donne de toute façon
        steps += 1
    return steps, len(gen), accepted


@torch.no_grad()
def verify_cost(model, ids: torch.Tensor, k: int, repeat: int = 10) -> float:
    """ms d'une passe de k+1 tokens après le prompt (chemin prefill, cache dynamique remis à la même longueur)."""
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    n = ids.shape[1]
    extra = ids[:, -(k + 1):].clone()
    sync = torch.cuda.synchronize if ids.is_cuda else (lambda: None)
    for _ in range(2):
        model(input_ids=extra, past_key_values=past, use_cache=True)
        past.crop(-(k + 1))
    sync()
    t0 = time.perf_counter()
    for _ in range(repeat):
        model(input_ids=extra, past_key_values=past, use_cache=True)
        past.crop(-(k + 1))
    sync()
    return 1000 * (time.perf_counter() - t0) / repeat


@torch.no_grad()
def step_cost(model, ids: torch.Tensor, n: int = 32) -> float:
    dec = make_decoder(model)
    nxt = pick(dec.prefill(ids))
    for _ in range(4):
        nxt = pick(dec.step(nxt))
    sync = torch.cuda.synchronize if ids.is_cuda else (lambda: None)
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        nxt = pick(dec.step(nxt))
    sync()
    return 1000 * (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common.add_model_args(ap)
    _common.add_policy_args(ap)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--k", default="2,4,6", help="tailles de brouillon essayées")
    ap.add_argument("--out", default=str(_common.ROOT / "results"))
    args = ap.parse_args()

    model, rt, tok = _common.load(args)
    dev = model.moelab_device
    prompts = load_prompts()[: args.prompts]
    gens = []
    print(f"1. Réponses gloutonnes de {len(prompts)} prompts ...")
    for p in prompts:
        ids = chat_ids(tok, p, dev)
        gen = generate(model, ids, args.max_new_tokens)
        gens.append((ids[0].tolist(), gen))
    ks = [int(x) for x in args.k.split(",")]
    ids0 = chat_ids(tok, prompts[0], dev)
    print("2. Coût d'un pas de génération normal ...")
    t_step = step_cost(model, ids0)
    print(f"   {t_step:.2f} ms par token")
    report = {"step_ms": t_step, "policy": rt.policy.describe(), "k": {}}
    print("3. Brouillon « prompt lookup » et coût de vérification :")
    for k in ks:
        steps = toks = acc = 0
        for prompt, gen in gens:
            s, t, a = simulate(prompt, gen, k)
            steps, toks, acc = steps + s, toks + t, acc + a
        per_step = toks / steps
        t_ver = verify_cost(model, ids0, k)
        gain = per_step * t_step / t_ver
        report["k"][k] = {"tokens_per_step": per_step, "accepted_share": acc / toks, "verify_ms": t_ver,
                          "expected_speedup": gain}
        print(f"   K={k} : {per_step:.2f} tokens par vérification ({acc / toks:.0%} des tokens venus du brouillon), "
              f"vérification {t_ver:.1f} ms (= {t_ver / t_step:.1f} pas) -> gain attendu x{gain:.2f}")
    print("   (vérification mesurée par le chemin prefill, sans CUDA graphs : borne basse du gain ; une vérification "
          "en graph coûterait au mieux ~ (K+1) fois moins cher côté dense, pas côté experts)")
    best = max(report["k"].values(), key=lambda v: v["expected_speedup"])
    verdict = ("à construire" if best["expected_speedup"] > 1.15 else
               "pas rentable sur cette machine (gain < 15 %)")
    report["verdict"] = verdict
    print(f"Verdict : {verdict}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    path = out / f"lookup_{_common.model_tag(args)}_{stamp}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Résultats : {path}")
    rt.close()


if __name__ == "__main__":
    main()
