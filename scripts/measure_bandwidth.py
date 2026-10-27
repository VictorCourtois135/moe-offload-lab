"""
Étape 10 : mesure B_PCIe (RAM -> VRAM), B_Host (calcul d'un expert sur CPU) et la
vitesse de lecture du store, puis les enregistre dans <store>/bandwidth.json.
Tous les autres scripts les relisent pour la politique q* de FreeToken.

    python scripts\\measure_bandwidth.py --store C:\\moe-store\\olmoe
"""

from __future__ import annotations

import argparse

import _common  # noqa: F401

from moelab.hw import measure_all, save_bandwidth
from moelab.policy import qstar
from moelab.store import ExpertStore


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    args = ap.parse_args()
    store = ExpertStore(args.store)
    print(store.summary())
    res = measure_all(store)
    path = save_bandwidth(args.store, res)
    print(f"Enregistré dans {path}")
    if res["b_pcie"]:
        ratio = res["b_pcie"] / res["b_host"]
        print(f"B_PCIe / B_Host = {ratio:.2f} -> sur 8 experts manquants, q* en transfère "
              f"{qstar(8, res['b_pcie'], res['b_host'])} et en calcule {8 - qstar(8, res['b_pcie'], res['b_host'])} sur CPU")
    print("Note : relance ce script après un redémarrage pour une mesure disque 'à froid' (sans cache du système).")


if __name__ == "__main__":
    main()
