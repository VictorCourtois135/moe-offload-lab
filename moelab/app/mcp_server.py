"""
L'app vue comme un serveur MCP (05/10), sur http://127.0.0.1:<port>/mcp (transport HTTP
« streamable ») : Claude Desktop, VS Code ou un autre agent peuvent utiliser le modèle local
et l'état du moteur. Désactivé tant que « exposer_mcp » n'est pas coché dans l'écran Outils.

Les demandes passent par la même file que le chat : un seul modèle, une demande à la fois.
"""

from __future__ import annotations

import time


def build_mcp(engine, toolbox, convs):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        return None
    mcp = FastMCP("moelab", instructions="Un modèle MoE qui tourne en local sur ce PC grâce à moelab.",
                  streamable_http_path="/", stateless_http=True, json_response=True)

    def check():
        if not toolbox.cfg.get("exposer_mcp"):
            raise RuntimeError("le serveur MCP de moelab est désactivé (écran Outils de l'app)")
        if engine.model is None:
            raise RuntimeError("aucun modèle chargé dans moelab")

    @mcp.tool()
    def demander(question: str, max_tokens: int = 800) -> str:
        """Pose une question au modèle local de moelab (rien ne sort du PC). Réponse en texte."""
        check()
        from .chat import chat_job

        c = convs.create(titre="[MCP] " + question[:50])
        job = engine.submit("chat", "MCP : " + question[:40], chat_job(convs, toolbox, c["id"], question, max_tokens, 0.3))
        while job.status in ("en attente", "en cours"):
            job.wait_events(len(job.events), timeout=2)
        if job.status != "terminé":
            raise RuntimeError(job.error or job.status)
        return job.result["texte"]

    @mcp.tool()
    def etat_moteur() -> dict:
        """État du moteur moelab : modèle chargé, VRAM, RAM, caches, dernière vitesse mesurée."""
        if not toolbox.cfg.get("exposer_mcp"):
            raise RuntimeError("le serveur MCP de moelab est désactivé")
        st = engine.status()
        st.pop("busy", None)
        return st

    @mcp.tool()
    def lancer_autopilot(essais: int = 12) -> str:
        """Lance l'autopilot de moelab (optimise le moteur pour ce PC, ~30 min). Renvoie l'identifiant du job."""
        check()
        from .jobs import autopilot_job

        job = engine.submit("autopilot", "Autopilot (MCP)", autopilot_job(trials=essais))
        return f"autopilot lancé : job {job.id} (suivi dans l'écran Autopilot de l'app)"

    _ = time
    return mcp
