"""
Conversations de l'app (05/10) : enregistrées dans <data>/conversations/<id>.json, réponses
générées par le moteur partagé, avec ou sans outils.

Les messages gardent le texte brut du modèle (appels d'outils compris) : d'un message au
suivant, le début du prompt est identique et P1 reprend l'historique du cache.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

from .engine import TextSink, read_json, write_json

SYSTEM_TOOLS = ("Tu es un assistant qui tourne en local sur le PC de l'utilisateur, grâce au moteur moelab. "
                "Utilise les outils quand ils aident (fichiers, web, état du moteur, calculs) au lieu de deviner. "
                "Chaque tour est lent sur ce PC : mets dans le même message les appels qui ne dépendent pas l'un de "
                "l'autre, cherche dans une longue page avec lire_page(chercher=…), et dès que tu as l'essentiel, "
                "fais ce qui est demandé (par exemple écrire le fichier) puis réponds. Un résultat partiel vaut mieux "
                "que rien : écris ce que tu as trouvé, avec « non trouvé » pour le reste. Ne recopie pas le contenu "
                "d'un fichier dans ta réponse : mets-le directement dans l'appel ecrire_fichier. Chaque date ou chiffre "
                "écrit doit venir d'une page lue. Les résultats déjà obtenus "
                "[r1], [r2]… se relisent avec relire, sans recharger la page. "
                "Réponds dans la langue de l'utilisateur.")
CHAT_STEPS = 16  # 13/10 : 8 tours ne suffisaient pas pour chercher, lire trois pages et écrire un fichier
CHAT_LAST = ("(moelab) Limite de tours atteinte : termine maintenant avec ce que tu as. Si un fichier est demandé, "
             "appelle ecrire_fichier avec ce que tu as trouvé (« non trouvé » pour ce qui manque) au lieu de recopier "
             "le contenu dans ta réponse, puis réponds en disant ce qui manque.")
_PATH = re.compile(r"(?<![\w:/.])(?:[A-Za-z]:[\\/]|/)[^\s\"'«»<>|*?]+\.[A-Za-z0-9]{1,6}")
_BLOCK = re.compile(r"```[\w-]*\n(.*?)```", re.S)


def unwritten_block(request: str, answer: str, written: set) -> tuple[str, str] | None:
    """
    13/10 : au dernier tour, le modèle a recopié le fichier demandé dans sa réponse au lieu d'appeler ecrire_fichier.
    (chemin, contenu) si la demande nomme UN fichier, qu'il n'a pas été écrit, et que la réponse contient UN bloc.
    """
    paths = list(dict.fromkeys(_PATH.findall(request or "")))
    blocks = _BLOCK.findall(answer or "")
    if len(paths) != 1 or len(blocks) != 1 or paths[0].lower() in written:
        return None
    return paths[0], blocks[0].strip() + "\n"
SYSTEM_PLAIN = ("Tu es un assistant qui tourne en local sur le PC de l'utilisateur, grâce au moteur moelab. "
                "Réponds dans la langue de l'utilisateur.")
_CALL = re.compile(r"<tool_call>.*?(</tool_call>|$)", re.DOTALL)


class Conversations:
    def __init__(self, data_dir):
        self.dir = Path(data_dir) / "conversations"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, cid: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", cid or ""):
            raise KeyError(cid)
        return self.dir / f"{cid}.json"

    def list(self) -> list[dict]:
        out = []
        for f in self.dir.glob("*.json"):
            c = read_json(f, None)
            if c:
                out.append({"id": c["id"], "titre": c["titre"], "maj": c["maj"], "messages": len(c["messages"]),
                            "outils": c.get("outils", False)})
        return sorted(out, key=lambda c: -c["maj"])

    def get(self, cid: str) -> dict:
        c = read_json(self._path(cid), None)
        if c is None:
            raise KeyError(cid)
        return c

    def create(self, titre: str = "Nouvelle conversation", outils: bool = False, systeme: str | None = None) -> dict:
        c = {"id": uuid.uuid4().hex[:12], "titre": titre, "cree": time.time(), "maj": time.time(), "outils": outils,
             "systeme": systeme, "messages": []}
        self.save(c)
        return c

    def save(self, c: dict) -> None:
        c["maj"] = time.time()
        write_json(self._path(c["id"]), c)

    def delete(self, cid: str) -> None:
        self._path(cid).unlink(missing_ok=True)


def visible_text(text: str) -> str:
    """Texte d'un message de l'assistant sans ses appels d'outils (affichés à part)."""
    return _CALL.sub("", text or "").strip()


def fit_context(messages: list[dict], tools, model, tok, rt, max_new_tokens: int, job, factor: float = 1.0) -> None:
    """Le prompt doit tenir dans le cache statique (graph_max_len) avec la place de la réponse."""
    from .. import agent as agent_mod

    d = getattr(model, "moelab_graph_decoder", None)
    max_len = getattr(d, "max_len", None) or getattr(rt.cfg, "graph_max_len", None)
    if not max_len:
        return
    b = max_len - min(max_new_tokens, 700) - 32
    count = lambda m: agent_mod.render(tok, m, tools).shape[1]  # noqa: E731
    from .agents import CLEAN_TARGET, CLEAN_TRIGGER

    n = agent_mod.clean_context(messages, count, int(b * CLEAN_TRIGGER * factor), int(b * CLEAN_TARGET * factor))
    if n:
        job.emit("log", texte=f"contexte nettoyé ({n} messages réduits ou retirés) pour tenir en mémoire")
    if agent_mod.compact_messages(messages, count, int(b * factor)):
        job.emit("log", texte="conversation raccourcie pour tenir dans la fenêtre")


def chat_job(convs: Conversations, toolbox, cid: str, message: str, max_new_tokens: int = 1024,
             temperature: float = 0.7):
    """Fonction de job : ajoute le message, génère la réponse (outils si la conversation les active)."""

    def run(job, engine):
        from ..agent import Agent, moelab_generate_fn

        c = convs.get(cid)
        if not c["messages"] and c["titre"] == "Nouvelle conversation":
            c["titre"] = message.strip().splitlines()[0][:60] or c["titre"]
        c["messages"].append({"role": "user", "content": message, "t": time.time()})
        convs.save(c)
        model, tok, rt = engine.model, engine.tok, engine.rt
        rt.reset_stats()
        sink = TextSink(tok, job)
        timings: list = []
        # 13/10 : un appel ecrire_fichier coupé à 1 024 tokens (JSON illisible)
        n_new = max(max_new_tokens, 1536) if c.get("outils") else max_new_tokens
        raw_gen = moelab_generate_fn(model, tok, n_new, temperature=temperature, streamer=sink,
                                     timings=timings)
        record: list = []      # messages ajoutés pendant ce tour, dans leur version complète (enregistrée)

        def collect(messages):
            # 13/10 : repérés par une marque, pas par id() : sous Windows, un message nettoyé libéré laissait son
            # adresse à un nouveau message, qui passait pour déjà vu et n'était pas enregistré
            for m in messages:
                if not m.get("_vu") and not m.get("_nettoye") and not m.get("_raccourci"):
                    m["_vu"] = True
                    record.append(m)

        def gen(messages, tools):
            """13/10 : huit pages lues dans le chat -> « CUDA out of memory » (le contexte dépassait le cache
            statique). Même garde que les agents : grand nettoyage à 80 %, filet de sécurité, un nouvel essai."""
            import torch

            collect(messages)
            fit_context(messages, tools, model, tok, rt, n_new, job)
            sink.reset()
            try:
                return raw_gen(messages, tools)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                d = getattr(model, "moelab_graph_decoder", None)
                if d is not None:
                    d.tokens = None
                job.emit("log", texte="mémoire GPU pleine : conversation raccourcie et nouvel essai")
                fit_context(messages, tools, model, tok, rt, n_new, job, factor=0.6)
                sink.reset()
                return raw_gen(messages, tools)
        from ..agent import DATE_RULE, today_fr

        system = c.get("systeme") or (SYSTEM_TOOLS if c.get("outils") else SYSTEM_PLAIN)
        system += "\n" + DATE_RULE.format(date=today_fr())  # 06/10 : sans la date, le modèle se croit en 2024
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {k: m[k] for k in ("role", "content", "name") if k in m} for m in c["messages"]]
        for m in msgs:
            m["_vu"] = True
        t0 = time.perf_counter()
        if c.get("outils"):
            from .tools import ARCHIVED_TOOLS, archived, rebuild_archive, relire_tool

            archive = rebuild_archive(c["messages"])
            written: set = set()

            from .factcheck import FactCheck

            bounced: set = set()

            def track(t):
                def fn(**kw):
                    # 13/10 : dates écrites qui n'étaient dans aucune page lue (2.14.0 « 2026-08-27 » au lieu du 2 sept.).
                    # Même contrôle que chez les agents : refusé une fois, avec la liste ; réécrit, il passe.
                    path = str(kw.get("chemin", "")).lower()
                    missing = FactCheck(message, system, *archive.values()).missing(str(kw.get("contenu", "")))
                    if missing and path not in bounced:
                        bounced.add(path)
                        return ("NON ÉCRIT (vérification, une seule fois) : ces dates ou chiffres ne sont dans aucune page "
                                f"ni aucun fichier lus : {', '.join(missing)}. Vérifie-les avec relire(ref, chercher=…) "
                                "ou écris « non trouvé », puis rappelle ecrire_fichier.")
                    out = t.fn(**kw)
                    if not str(out).startswith(("NON ÉCRIT", "refusé")):
                        written.add(str(kw.get("chemin", "")).lower())
                    return out
                return type(t)(t.name, t.description, t.parameters, fn)

            tools = [archived(t, archive) if t.name in ARCHIVED_TOOLS
                     else track(t) if t.name in ("ecrire_fichier", "ajouter_fichier") else t
                     for t in toolbox.for_job(model, job)]
            tools.append(relire_tool(archive))

            def on_step(step):
                for call, res in zip(step.calls, step.results):
                    job.emit("tool", nom=call.get("name"), arguments=call.get("arguments", {}), resultat=res[:4000])
                if step.calls:
                    job.emit("turn")

            final = Agent(tools, gen, max_steps=CHAT_STEPS, on_step=on_step, append_stop=False).run(msgs)
            if final.startswith("(arrêt"):  # dernier tour : conclure plutôt que s'arrêter au milieu
                msgs.append({"role": "user", "content": CHAT_LAST})
                job.emit("log", texte="limite de tours : dernier tour pour conclure")
                Agent(tools, gen, max_steps=3, on_step=on_step).run(msgs)
            last = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
            todo = unwritten_block(message, last.get("content") if last else "", written)
            write = next((t for t in tools if t.name == "ecrire_fichier"), None)
            if todo and write is not None and last is not None:
                # le fichier passe par les mêmes contrôles et la même demande d'accord qu'un appel du modèle
                try:
                    res = str(write.fn(chemin=todo[0], contenu=todo[1]))
                except Exception as e:  # noqa: BLE001
                    res = f"erreur : {e}"
                job.emit("tool", nom="ecrire_fichier", arguments={"chemin": todo[0], "contenu": todo[1][:200] + "…"},
                         resultat=res[:4000])
                ok = todo[0].lower() in written
                last["content"] += (f"\n\n(moelab : le bloc ci-dessus a été écrit dans {todo[0]}.)" if ok else
                                    f"\n\n(moelab : écriture de {todo[0]} non faite : {res[:200]})")
        else:
            msgs.append({"role": "assistant", "content": gen(msgs, [])})
        collect(msgs)
        dt = time.perf_counter() - t0
        from .tools import _REF

        for m in record:
            m = {k: v for k, v in m.items() if not k.startswith("_")} | {"t": time.time()}
            r = _REF.match(m.get("content") or "") if m.get("role") == "tool" else None
            if r and c.get("outils") and len(archive.get(r.group(1), "")) > len(m["content"]):
                m["archive"] = archive[r.group(1)]   # texte complet : relire marche encore au message suivant
            c["messages"].append(m)
        toks = sum(t["new_tokens"] for t in timings)
        ttft = timings[0]["ttft_s"] if timings else 0.0
        gen_s = max(dt - sum(t.get("ttft_s", 0) for t in timings), 1e-6)
        s = rt.stats.as_dict()
        stats = {"tokens": toks, "tours": len(timings), "premier_token_s": round(ttft, 2),
                 "tok_s": round(max(toks - len(timings), 0) / gen_s, 2), "total_s": round(dt, 1),
                 "repris_du_cache": timings[0].get("reused", 0) if timings else 0,
                 "prompt_tokens": timings[0].get("prompt_tokens", 0) if timings else 0,
                 "hit_vram": round(s["fast_hit_rate"], 3), "disque_par_token": round(s["disk_per_token"], 1)}
        c["messages"][-1]["stats"] = stats
        convs.save(c)
        rt.end_of_request()
        engine.last_answer = stats
        return {"stats": stats, "texte": visible_text(c["messages"][-1]["content"])}

    return run
