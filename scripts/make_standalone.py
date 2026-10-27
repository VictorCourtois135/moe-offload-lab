"""
Rend un store existant autonome (05/10) : la partie dense du modèle (attention, embeddings,
tête, experts partagés) et les fichiers du tokenizer sont copiés dans <store>\\model.
Ensuite --model devient facultatif, et le checkpoint téléchargé peut être supprimé.

    python scripts\\make_standalone.py --model D:\\models\\qwen3-next-80b-fp8 --store C:\\moe-store\\qwen3-next

Place nécessaire sur le disque du store : ~2,5 Go pour Qwen3-Next, ~1 Go pour OLMoE.
"""

from __future__ import annotations

import argparse

import _common  # noqa: F401

from moelab.plan import disk_free
from moelab.standalone import dense_bytes, make_standalone

GIB = 2**30


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="dossier du checkpoint Hugging Face")
    ap.add_argument("--store", required=True, help="store à rendre autonome")
    args = ap.parse_args()
    need, free = dense_bytes(args.model), disk_free(args.store)
    print(f"Partie dense : {need / GIB:.2f} Gio, libre sur le disque du store : {free / GIB:.1f} Gio")
    if need > free * 0.98:
        raise SystemExit("Pas assez de place sur le disque du store.")
    make_standalone(args.model, args.store)
    print(f"Fait. Le checkpoint {args.model} n'est plus nécessaire : vérifie avec chat.py sans --model, puis supprime-le.")


if __name__ == "__main__":
    main()
