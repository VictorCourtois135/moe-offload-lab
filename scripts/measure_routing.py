"""
Étape 12 : trois mesures sur le routage réel du modèle, à comparer aux papiers.

1. Proxy d'importance (HOBBIT) : le poids du routeur prédit-il la vraie
   contribution d'un expert, ||w * E(x)|| ? (HOBBIT : Pearson 0,99 sur Mixtral)
2. Prefetch : le routeur de la couche L+1 appliqué à l'état de la couche L
   retrouve-t-il les experts de L+1 ? (Colibrì : 71,6% du top-8 sur GLM-5.2)
3. Localité temporelle : un token réutilise-t-il les experts du token précédent ?
   (Colibrì : 41,3%, c'est l'heuristique de référence qu'on doit battre)

    python scripts\\measure_routing.py --model D:\\models\\olmoe-hf --store C:\\moe-store\\olmoe
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import _common
import torch

from moelab.bench import environment, load_eval_text
from moelab.policy import PrecisionPolicy
from moelab.runtime import expert_ffn
from moelab.store import materialize


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-12)).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common.add_model_args(ap)
    ap.add_argument("--tokens", type=int, default=256, help="nombre de tokens du texte d'évaluation")
    ap.add_argument("--proxy-tokens", type=int, default=32, help="tokens utilisés pour la mesure du proxy (coûteuse)")
    ap.add_argument("--coverage", type=float, default=0.9,
                    help="E1 : part de la vraie contribution à garder en fp16 dans chaque couche (défaut 0.9)")
    ap.add_argument("--out", default=str(_common.ROOT / "results"))
    args = ap.parse_args()
    args.policy, args.t1, args.t2, args.protect, args.hi_vote = "hi", 0.6, None, "", 0.5

    model, rt, tok = _common.load(args)
    store = rt.store
    n_layers = len(store.layers)
    cap: dict[int, dict] = {}
    handles = []
    for L, g in rt.gates.items():
        def hook(m, inp, out, L=L):
            cap[L] = {"x": inp[0].detach().reshape(-1, inp[0].shape[-1]), "w": out[1].detach(), "idx": out[2].detach()}
        handles.append(g.register_forward_hook(hook))
    ids = tok(load_eval_text(), return_tensors="pt").input_ids[:, : args.tokens].to(model.moelab_device)
    print(f"Passe avant sur {ids.shape[1]} tokens ...")
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    k = cap[0]["idx"].shape[1]

    # 2 + 3 : prefetch et localité temporelle
    pref, temporal = [], []
    with torch.no_grad():
        for L in range(n_layers):
            act = cap[L]["idx"].cpu()
            if L + 1 < n_layers:
                _, _, pred = rt.gates[L + 1](cap[L]["x"])
                nxt = cap[L + 1]["idx"].cpu()
                pred = pred.cpu()
                pref.append(sum(len(set(pred[t].tolist()) & set(nxt[t].tolist())) for t in range(len(nxt))) / (k * len(nxt)))
            temporal.append(sum(len(set(act[t - 1].tolist()) & set(act[t].tolist())) for t in range(1, len(act))) / (k * (len(act) - 1)))

    # 1 : proxy d'importance
    g_all, c_all, top1 = [], [], []
    t1_layers, coverage_at_06 = {}, []
    grid = [round(0.02 * i, 2) for i in range(51)]
    n_sub = min(args.proxy_tokens, ids.shape[1])
    with torch.no_grad():
        for L in range(n_layers):
            x = cap[L]["x"][:n_sub].float().cpu()
            w = cap[L]["w"][:n_sub].float().cpu()
            idx = cap[L]["idx"][:n_sub].cpu()
            contrib = torch.zeros_like(w)
            for e in idx.unique().tolist():
                t, j = (idx == e).nonzero(as_tuple=True)
                gu, dn = materialize(store.layouts[store.hi], store.read(L, e, store.hi), torch.float32)
                y = expert_ffn(x[t], gu, dn, rt.act_fns[L])
                contrib[t, j] = (y * w[t, j, None]).norm(dim=-1)
            wn = w / w.sum(-1, keepdim=True)
            cn = contrib / contrib.sum(-1, keepdim=True)
            g_all.append(wn.flatten())
            c_all.append(cn.flatten())
            top1.append((wn.argmax(-1) == cn.argmax(-1)).float().mean().item())
            # E1 : le plus petit t1 qui garde en fp16 au moins `coverage` de la vraie contribution
            s = PrecisionPolicy().unimportance(w)
            cov = {t: (cn * (s <= t + 1e-6)).sum(-1).mean().item() for t in grid}
            t1_layers[L] = next((t for t in grid if cov[t] >= args.coverage), 1.0)
            coverage_at_06.append(cov[0.6])
    corr = pearson(torch.cat(g_all), torch.cat(c_all))

    res = {
        "date": date.today().isoformat(),
        "model": str(args.model),
        "tokens": ids.shape[1],
        "top_k": k,
        "proxy_pearson": corr,
        "proxy_top1_agreement": sum(top1) / len(top1),
        "prefetch_recall_mean": sum(pref) / len(pref),
        "prefetch_recall_per_layer": pref,
        "temporal_reuse_mean": sum(temporal) / len(temporal),
        "temporal_reuse_per_layer": temporal,
        "coverage_target": args.coverage,
        "t1_layers": t1_layers,
        "coverage_at_t1_0.6": coverage_at_06,
        "environment": environment(rt),
    }
    print()
    print(f"1. Proxy HOBBIT     : Pearson {corr:.3f} (HOBBIT sur Mixtral : 0,99), "
          f"le plus gros poids du routeur = la plus grosse contribution dans {100 * res['proxy_top1_agreement']:.0f}% des cas")
    print(f"2. Prefetch L -> L+1 : {100 * res['prefetch_recall_mean']:.1f}% du top-{k} retrouvé (Colibrì sur GLM-5.2 : 71,6%)")
    print(f"3. Localité         : {100 * res['temporal_reuse_mean']:.1f}% des experts du token précédent réutilisés "
          f"(Colibrì sur GLM-5.2 : 41,3%)")
    print("   Prefetch par couche : " + " ".join(f"{100 * r:.0f}" for r in pref))
    print(f"4. Seuils t1 calibrés (E1, {100 * args.coverage:.0f}% de la vraie contribution en fp16) : "
          + " ".join(f"{t1_layers[L]:.2f}" for L in sorted(t1_layers)))
    cal = Path(args.store) / "t1_layers.json"
    cal.write_text(json.dumps({"coverage": args.coverage, "t1_layers": t1_layers}, indent=1), encoding="utf-8")
    print(f"   enregistrés dans {cal} (utilisés avec --t1-layers auto)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"routing_{_common.model_tag(args)}_{res['date']}.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"Résultats : {path}")
    rt.close()


if __name__ == "__main__":
    main()
