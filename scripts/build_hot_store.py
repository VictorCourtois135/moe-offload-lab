"""
Optimisation D4 (doc 18) : copier sur un disque rapide les experts les plus utilisés.

Le store complet reste où il est (par exemple sur le HDD) ; les experts qui servent le
plus, d'après l'historique d'usage (usage.json, rempli en discutant avec chat.py), sont
copiés dans un « store chaud » sur le NVMe. Au chargement, --hot-store le branche :
ces experts sont lus sur le NVMe, les autres sur le HDD.

    python scripts\\build_hot_store.py --store D:\\moe-store\\qwen3-next --hot C:\\moe-hot\\qwen3-next --gb 20
"""

from __future__ import annotations

import argparse

import _common  # noqa: F401

from moelab.store import build_hot_store


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="store complet (celui de build_store.py)")
    ap.add_argument("--hot", required=True, help="dossier du store chaud, sur le disque rapide")
    ap.add_argument("--gb", type=float, required=True, help="place à y consacrer, en Gio")
    ap.add_argument("--usage", default=None, help="historique d'usage (défaut : <store>/usage.json)")
    args = ap.parse_args()
    build_hot_store(args.store, args.hot, int(args.gb * 2**30), args.usage)


if __name__ == "__main__":
    main()
