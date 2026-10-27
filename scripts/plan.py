"""
Affiche le plan de placement sans lancer de génération : taille de la partie
dense, placement choisi, budgets VRAM/RAM, part des experts qui tiendra dans
chaque niveau. L'équivalent de `coli plan` de Colibrì.

    python scripts\\plan.py --model D:\\models\\olmoe-hf --store C:\\moe-store\\olmoe
"""

from __future__ import annotations

import argparse

import _common
import torch

from moelab.hw import load_bandwidth
from moelab.loader import build_skeleton, choose_dense_placement, iter_experts_modules, load_config
from moelab.plan import auto_graph_max_len, auto_vram_reserve, kv_bytes_per_token, plan_report, ram_available, vram_free
from moelab.policy import qstar
from moelab.store import ExpertStore

GIB = 2**30


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="facultatif si le store est autonome")
    ap.add_argument("--store", required=True)
    ap.add_argument("--dtype", default="float16", choices=list(_common.DTYPES))
    ap.add_argument("--vram-reserve-gb", type=float, default=None, help="défaut : auto (comme le chargement)")
    ap.add_argument("--graph-max-len", type=int, default=None, help="défaut : auto (comme le chargement)")
    ap.add_argument("--ram-reserve-gb", type=float, default=3.0)
    ap.add_argument("--no-dense-int8", dest="dense_int8", action="store_false",
                    help="estimer avec la partie dense en fp16 (D5, actif par défaut, la passe en int8)")
    args = ap.parse_args()
    from moelab.standalone import resolve_model_dir

    args.model = str(resolve_model_dir(args.model, args.store))

    store = ExpertStore(args.store)
    print(store.summary())
    config = load_config(args.model)
    model = build_skeleton(config, _common.DTYPES[args.dtype])
    max_len = args.graph_max_len or auto_graph_max_len(config)
    if args.vram_reserve_gb is None:
        args.vram_reserve_gb = auto_vram_reserve(config, max_len) / GIB
    print(f"Cache KV : {kv_bytes_per_token(config) // 1024} Ko/token, longueur max {max_len} tokens ; "
          f"réserve VRAM {args.vram_reserve_gb:.2f} Gio")
    expert_params = {id(p) for _, m in iter_experts_modules(model) for p in m.parameters()}
    dense = [p for p in model.parameters() if id(p) not in expert_params]
    dense_bytes = sum(p.numel() * p.element_size() for p in dense)
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    nbytes = lambda ps: sum(p.numel() * p.element_size() for p in ps)  # noqa: E731
    emb_bytes = nbytes(emb.parameters())
    head_bytes = nbytes(head.parameters()) if head is not None else 0
    eh_bytes = emb_bytes + head_bytes
    head_q = False
    print(f"Partie dense : {dense_bytes / GIB:.2f} Gio en fp16 (dont embeddings + tête {eh_bytes / GIB:.2f} Gio)")
    if args.dense_int8:  # D5 : mêmes règles que le chargement (loader.py)
        from moelab.denseq import quantizable

        q = quantizable(model)
        lin = sum(m.weight.numel() * m.weight.element_size() for _, m in q)
        head_q = any(m is head for _, m in q)
        dense_bytes -= int(0.47 * lin)
        if head_q:
            head_bytes -= int(0.47 * head_bytes)
            eh_bytes = emb_bytes + head_bytes
        print(f"  avec D5 (int8, défaut) : {dense_bytes / GIB:.2f} Gio"
              + (", tête de sortie gardée sur le GPU en placement split" if head_q else ""))
    placement = choose_dense_placement(dense_bytes, eh_bytes, int(args.vram_reserve_gb * GIB), emb_bytes, head_q)
    print(f"  -> placement '{placement}'")
    # en split, les embeddings restent sur CPU ; la tête aussi, sauf si D5 l'a passée en int8
    on_cpu_split = emb_bytes + (0 if head_q else head_bytes)
    on_gpu = {"gpu": dense_bytes, "split": dense_bytes - on_cpu_split}.get(placement, 0)
    fast = max(0, vram_free() - on_gpu - int(args.vram_reserve_gb * GIB)) if placement != "cpu" else 0
    on_cpu = {"cpu": dense_bytes, "split": on_cpu_split}.get(placement, 0)
    host = max(256 * 2**20, ram_available() - on_cpu - int(args.ram_reserve_gb * GIB))
    print("Budgets estimés pour les experts :")
    for line in plan_report(store, fast, host):
        print(line)
    bw = load_bandwidth(args.store)
    if bw and bw.get("b_pcie"):
        q = qstar(8, bw["b_pcie"], bw["b_host"])
        print(f"q* : pour 8 experts manquants, {q} transférés et {8 - q} calculés sur CPU")
    elif bw:
        print("q* : pas de GPU mesuré, tous les experts sont calculés sur CPU")
    else:
        print("q* : bandwidth.json absent, lance scripts\\measure_bandwidth.py")
    if not torch.cuda.is_available():
        print("Attention : CUDA indisponible, tout tournera sur CPU.")


if __name__ == "__main__":
    main()
