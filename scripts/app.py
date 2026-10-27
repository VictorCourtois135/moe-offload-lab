"""
L'app moelab (05/10) : chat, mise en route, autopilot, outils et MCP, moteur, tâches.

    pip install fastapi uvicorn huggingface_hub mcp pypdf
    python scripts\\app.py
    python scripts\\app.py --store C:\\moe-store\\qwen3-next --ram-gb 5     (charge ce modèle au démarrage)

Puis ouvre http://127.0.0.1:8000 (le navigateur s'ouvre tout seul). Tout reste sur ce PC :
le serveur n'écoute que 127.0.0.1. Données de l'app (conversations, bibliothèque,
réglages des outils) : app_data\\ à la racine du repo.
"""

from __future__ import annotations

import argparse
import threading
import webbrowser

import _common

from moelab.app.engine import Engine


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="adresse d'écoute (défaut : seulement ce PC)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data", default=str(_common.ROOT / "app_data"), help="dossier des données de l'app")
    ap.add_argument("--store", default=None, help="store à charger au démarrage (sinon : le dernier utilisé)")
    ap.add_argument("--model", default=None, help="checkpoint (facultatif si le store est autonome)")
    ap.add_argument("--ram-gb", type=float, default=None)
    ap.add_argument("--vram-gb", type=float, default=None)
    ap.add_argument("--policy", default="lo", choices=["lo", "hobbit", "hi"])
    ap.add_argument("--no-load", action="store_true", help="ne charger aucun modèle au démarrage")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit("Il manque des paquets : pip install fastapi uvicorn huggingface_hub mcp pypdf") from e
    from moelab.app.server import create_app

    engine = Engine(args.data)
    app = create_app(engine, results_dir=_common.ROOT / "results")
    lib = app.state.library
    start = None
    if args.store:
        start = {"store": args.store, "model_dir": args.model, "ram_gb": args.ram_gb, "vram_gb": args.vram_gb,
                 "policy": args.policy}
        try:
            lib.add(args.store, checkpoint=args.model)
        except Exception as e:  # store incomplet ou illisible : l'app démarre quand même
            print(f"(store non ajouté à la bibliothèque : {e})")
    elif not args.no_load:
        start = lib.data.get("dernier")
    if start and not args.no_load:
        def run(job, eng):
            info = eng.load(log=lambda m: (print(m), job.emit("log", texte=m)), **start)
            lib.data["dernier"] = start
            lib.save()
            return info

        engine.submit("load", f"Chargement de {start['store']}", run, needs_model=False)
    url = f"http://{args.host}:{args.port}"
    print(f"moelab : {url}  (Ctrl+C pour arrêter)")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    engine.close()


if __name__ == "__main__":
    main()
