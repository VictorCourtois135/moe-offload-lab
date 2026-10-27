"""
Le serveur de l'app (05/10) : FastAPI, en local (127.0.0.1). Il garde le moteur chargé et
sert l'interface (moelab/app/static) et une API JSON ; le texte généré, les appels d'outils
et l'avancement des jobs arrivent en direct (Server-Sent Events).
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import options
from .agents import AgentLibrary, AgentSpec, list_runs, orchestrator_job
from .lessons import LessonBook
from .chat import Conversations, chat_job, visible_text
from .engine import Engine
from .jobs import autopilot_job, batch_job
from .onboarding import Library, catalog, install_job, machine, plan
from .tools import PERMS, ToolBox

STATIC = Path(__file__).parent / "static"


def create_app(engine: Engine, results_dir: Path | None = None) -> FastAPI:
    data = engine.data_dir
    engine.results_dir = results_dir
    convs, library, toolbox = Conversations(data), Library(data), ToolBox(data)
    from .mcp_server import build_mcp

    mcp = build_mcp(engine, toolbox, convs)
    mcp_app = mcp.streamable_http_app() if mcp is not None else None

    @contextlib.asynccontextmanager
    async def lifespan(app):
        toolbox.start_mcp()
        if mcp is not None:
            async with mcp.session_manager.run():
                yield
        else:
            yield

    app = FastAPI(title="moelab", lifespan=lifespan)
    app.state.engine, app.state.library, app.state.convs, app.state.toolbox = engine, library, convs, toolbox
    agent_lib, runs_dir = AgentLibrary(data), Path(data) / "agent_runs"
    lessons = LessonBook(data)
    app.state.agents, app.state.lessons = agent_lib, lessons
    if mcp_app is not None:
        app.mount("/mcp", mcp_app)

    def job_out(job):
        return job.summary()

    # ------------------------------------------------------------------ état, jobs
    @app.get("/api/status")
    def status():
        st = engine.status()
        st["bibliotheque"] = library.list()
        st["dernier"] = library.data.get("dernier")
        return st

    @app.get("/api/jobs")
    def jobs():
        return sorted((j.summary() for j in engine.jobs.values()), key=lambda j: -j["created"])

    @app.get("/api/jobs/{jid}")
    def job_detail(jid: str):
        job = engine.jobs.get(jid)
        if job is None:
            raise HTTPException(404, "job inconnu")
        return {**job.summary(), "events": job.events[-500:]}

    @app.get("/api/jobs/{jid}/events")
    def job_events(jid: str, after: int = 0):
        job = engine.jobs.get(jid)
        if job is None:
            raise HTTPException(404, "job inconnu")

        def stream():
            n = after
            idle = 0
            while True:
                evs = job.wait_events(n, timeout=10)
                if evs:
                    idle = 0
                    for e in evs:
                        yield f"id: {e['seq']}\ndata: {json.dumps(e, ensure_ascii=False, default=str)}\n\n"
                    n = evs[-1]["seq"] + 1
                    if any(e["type"] == "done" for e in evs):
                        return
                else:
                    idle += 1
                    yield ": ping\n\n"
                    if job.status not in ("en attente", "en cours") and idle > 1:
                        return

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/api/jobs/{jid}/cancel")
    def job_cancel(jid: str):
        return {"ok": engine.cancel(jid)}

    @app.post("/api/jobs/{jid}/approve")
    def job_approve(jid: str, body: dict = Body(...)):
        job = engine.jobs.get(jid)
        if job is None:
            raise HTTPException(404, "job inconnu")
        return {"ok": job.answer(body.get("id"), bool(body.get("accepte")), body.get("valeur"))}

    # ------------------------------------------------------------------ modèles
    @app.get("/api/models")
    def models():
        return {"modeles": library.list(), "dernier": library.data.get("dernier"), "charge": engine.model_info or None}

    @app.post("/api/models/load")
    def models_load(body: dict = Body(...)):
        store = body.get("store")
        if body.get("id"):
            store = library.get(body["id"])["store"]
        if not store:
            raise HTTPException(400, "store manquant")
        args = {"store": store, "model_dir": body.get("model_dir"), "ram_gb": body.get("ram_gb"),
                "vram_gb": body.get("vram_gb"), "policy": body.get("policy") or "lo"}

        def run(job, eng):
            info = eng.load(log=lambda m: job.emit("log", texte=m), **args)
            library.data["dernier"] = args
            library.save()
            return info

        return job_out(engine.submit("load", f"Chargement de {Path(store).name}", run, needs_model=False))

    @app.post("/api/models/unload")
    def models_unload():
        engine.in_worker("déchargement", lambda e: e.unload(), priority=0)
        return {"ok": True}

    @app.post("/api/models/add")
    def models_add(body: dict = Body(...)):
        try:
            return library.add(body["store"], body.get("nom"), body.get("checkpoint"))
        except (OSError, ValueError, KeyError) as e:
            raise HTTPException(400, f"store illisible : {e}") from e

    @app.post("/api/models/{mid}/remove")
    def models_remove(mid: str, body: dict = Body({})):
        m = library.get(mid)
        if engine.model_info.get("store") == m["store"] and body.get("supprimer_store"):
            raise HTTPException(409, "ce modèle est chargé : décharge-le d'abord")
        library.remove(mid, bool(body.get("supprimer_store")), bool(body.get("supprimer_checkpoint")))
        return {"ok": True}

    @app.post("/api/models/{mid}/standalone")
    def models_standalone(mid: str, body: dict = Body({})):
        m = library.get(mid)
        ckpt = body.get("checkpoint") or m.get("checkpoint")
        if not ckpt:
            raise HTTPException(400, "dossier du checkpoint manquant")

        def run(job, eng):
            from ..standalone import make_standalone

            make_standalone(ckpt, m["store"], log=lambda s: job.emit("log", texte=s))
            m["checkpoint"] = ckpt
            library.save()
            return {"ok": True}

        return job_out(engine.spawn("standalone", f"Store autonome : {m['nom']}", run))

    # ------------------------------------------------------------------ mise en route
    @app.get("/api/setup/machine")
    def setup_machine():
        return machine()

    @app.get("/api/setup/catalog")
    def setup_catalog():
        return catalog()

    @app.post("/api/setup/plan")
    def setup_plan(body: dict = Body(...)):
        try:
            return plan(body["model_id"], body["download_dir"], body["store_dir"], bool(body.get("garder_checkpoint")))
        except KeyError as e:
            raise HTTPException(400, f"paramètre manquant ou modèle inconnu : {e}") from e

    @app.post("/api/setup/install")
    def setup_install(body: dict = Body(...)):
        fn = install_job(library, body["model_id"], body["download_dir"], body["store_dir"],
                         body.get("precisions", "int4"), not bool(body.get("garder_checkpoint")))
        return job_out(engine.spawn("install", f"Installation : {body['model_id']}", fn))

    # ------------------------------------------------------------------ conversations
    @app.get("/api/conversations")
    def conv_list():
        return convs.list()

    @app.post("/api/conversations")
    def conv_new(body: dict = Body({})):
        return convs.create(body.get("titre") or "Nouvelle conversation", bool(body.get("outils")), body.get("systeme"))

    @app.get("/api/conversations/{cid}")
    def conv_get(cid: str):
        try:
            c = convs.get(cid)
        except KeyError as e:
            raise HTTPException(404, "conversation inconnue") from e
        for m in c["messages"]:
            if m["role"] == "assistant":
                m["visible"] = visible_text(m["content"])
        return c

    @app.patch("/api/conversations/{cid}")
    def conv_patch(cid: str, body: dict = Body(...)):
        c = convs.get(cid)
        for k in ("titre", "outils", "systeme"):
            if k in body:
                c[k] = body[k]
        convs.save(c)
        return c

    @app.delete("/api/conversations/{cid}")
    def conv_delete(cid: str):
        convs.delete(cid)
        return {"ok": True}

    @app.post("/api/conversations/{cid}/message")
    def conv_message(cid: str, body: dict = Body(...)):
        try:
            convs.get(cid)
        except KeyError as e:
            raise HTTPException(404, "conversation inconnue") from e
        text = (body.get("texte") or "").strip()
        if not text:
            raise HTTPException(400, "message vide")
        fn = chat_job(convs, toolbox, cid, text, int(body.get("max_tokens") or 1024), float(body.get("temperature", 0.7)))
        return job_out(engine.submit("chat", text[:50], fn))

    # ------------------------------------------------------------------ réglages
    @app.get("/api/options")
    def opts():
        cur = options.current(engine.model) if engine.model is not None else {}
        return {"groupes": options.GROUPS, "options": [o.as_dict() for o in options.catalog()],
                "actuel": json.loads(json.dumps(cur, default=str))}

    @app.post("/api/options")
    def opts_set(body: dict = Body(...)):
        if engine.model is None:
            raise HTTPException(409, "aucun modèle chargé")
        key, value = body.get("cle"), body.get("valeur")
        o = options.option(key)
        if o.apply in ("reload", "store"):
            raise HTTPException(409, f"{o.label} : demande de {'recharger le modèle' if o.apply == 'reload' else 'changer de store'}")
        try:
            msg = engine.run_now("options", f"réglage {key}", lambda j, e: options.apply(e.model, key, value))
        except RuntimeError as e:
            raise HTTPException(400, str(e)) from e
        return {"ok": True, "message": msg}

    # ------------------------------------------------------------------ autopilot
    @app.post("/api/autopilot")
    def autopilot(body: dict = Body({})):
        fn = autopilot_job(int(body.get("essais", 12)), float(body.get("minutes", 60)), body.get("depart", "actuel"),
                           bool(body.get("aveugle")))
        return job_out(engine.submit("autopilot", "Autopilot", fn))

    @app.get("/api/autopilot/reports")
    def reports():
        if results_dir is None or not Path(results_dir).exists():
            return []
        return [{"nom": f.name, "date": f.stat().st_mtime} for f in
                sorted(Path(results_dir).glob("autopilot_*.md"), key=lambda f: -f.stat().st_mtime)[:30]]

    @app.get("/api/autopilot/reports/{name}")
    def report(name: str):
        f = Path(results_dir or ".") / name
        if not name.startswith("autopilot_") or not name.endswith(".md") or not f.exists():
            raise HTTPException(404, "rapport inconnu")
        return {"nom": name, "texte": f.read_text(encoding="utf-8")}

    # ------------------------------------------------------------------ outils et MCP
    @app.get("/api/tools")
    def tools():
        return {"outils": toolbox.describe(engine.model), "dossiers": toolbox.cfg["dossiers"],
                "mcp_serveurs": toolbox.cfg["mcp_serveurs"], "mcp_etat": toolbox.hub.status(),
                "exposer_mcp": toolbox.cfg["exposer_mcp"], "mcp_disponible": mcp is not None, "permissions": PERMS}

    @app.post("/api/tools/permission")
    def tools_perm(body: dict = Body(...)):
        if body.get("permission") not in PERMS:
            raise HTTPException(400, f"permission : {PERMS}")
        toolbox.cfg["permissions"][body["nom"]] = body["permission"]
        toolbox.save()
        return {"ok": True}

    @app.post("/api/tools/folders")
    def tools_folders(body: dict = Body(...)):
        toolbox.cfg["dossiers"] = [str(d).strip() for d in body.get("dossiers", []) if str(d).strip()]
        toolbox.save()
        return {"ok": True}

    @app.post("/api/tools/mcp")
    def tools_mcp(body: dict = Body(...)):
        servers = body.get("serveurs", [])
        for s in servers:
            if not s.get("nom") or not s.get("commande"):
                raise HTTPException(400, "chaque serveur MCP a un nom et une commande")
        old = {s["nom"] for s in toolbox.cfg["mcp_serveurs"]}
        toolbox.cfg["mcp_serveurs"] = servers
        toolbox.save()
        for n in old:
            toolbox.hub.disconnect(n)
        errors = {}
        for s in servers:
            if s.get("actif", True):
                err = toolbox.hub.connect(s)
                if err:
                    errors[s["nom"]] = err
        return {"ok": not errors, "erreurs": errors, "etat": toolbox.hub.status()}

    @app.post("/api/tools/expose")
    def tools_expose(body: dict = Body(...)):
        toolbox.cfg["exposer_mcp"] = bool(body.get("actif"))
        toolbox.save()
        return {"ok": True}

    # ------------------------------------------------------------------ tâches de fond
    @app.post("/api/tasks/batch")
    def tasks_batch(body: dict = Body(...)):
        for k in ("dossier", "instruction", "sortie"):
            if not body.get(k):
                raise HTTPException(400, f"{k} manquant")
        fn = batch_job(body["dossier"], body.get("motif") or "*", body["instruction"], body["sortie"])
        return job_out(engine.submit("task", f"{body['instruction'][:40]} : {Path(body['dossier']).name}", fn))

    # ------------------------------------------------------------------ agents (06/10)
    @app.get("/api/agents")
    def agents_list():
        outils = [t for t in toolbox.describe(engine.model) if t["permission"] != "jamais"]
        return {"agents": agent_lib.list(), "outils": outils, "dossiers": toolbox.cfg["dossiers"]}

    @app.post("/api/agents")
    def agents_save(body: dict = Body(...)):
        try:
            spec = agent_lib.save(AgentSpec(body.get("nom", ""), body.get("role", ""), list(body.get("outils") or []),
                                            body.get("origine") or "utilisateur"))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return spec.as_dict()

    @app.delete("/api/agents/{nom}")
    def agents_remove(nom: str):
        if not agent_lib.remove(nom):
            raise HTTPException(404, "agent inconnu")
        return {"ok": True}

    @app.post("/api/agents/run")
    def agents_run(body: dict = Body(...)):
        objective = (body.get("objectif") or "").strip()
        if not objective:
            raise HTTPException(400, "objectif manquant")
        verify = body.get("verification", "auto")
        if verify not in ("auto", "toujours", "jamais"):
            raise HTTPException(400, "verification : auto, toujours ou jamais")
        fn = orchestrator_job(toolbox, agent_lib, objective, outils=body.get("outils"),
                              max_agents=int(body.get("max_agents", 5)), max_steps=int(body.get("max_etapes", 14)),
                              sub_steps=int(body.get("etapes_par_tache", 10)), minutes=float(body.get("minutes", 60)),
                              runs_dir=runs_dir, lessons=lessons, criteria=bool(body.get("criteres", True)),
                              verify=verify, wait_criteria=float(body.get("attente_criteres", 90)))
        return job_out(engine.submit("agents", "Agents : " + objective.splitlines()[0][:50], fn))

    @app.get("/api/agents/runs")
    def agents_runs():
        return list_runs(runs_dir) if runs_dir.exists() else []

    @app.get("/api/agents/runs/{rid}")
    def agents_run_get(rid: str):
        f = runs_dir / f"{rid}.json"
        if not f.parent.exists() or f.parent.resolve() != runs_dir.resolve() or not f.exists():
            raise HTTPException(404, "run inconnu")
        return json.loads(f.read_text(encoding="utf-8"))

    @app.get("/api/agents/lessons")
    def lessons_list():
        return lessons.list()

    @app.post("/api/agents/lessons")
    def lessons_add(body: dict = Body(...)):
        try:
            return lessons.add(body.get("texte", ""), body.get("objectif", ""), body.get("source", "utilisateur"))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.delete("/api/agents/lessons/{lid}")
    def lessons_remove(lid: str):
        if not lessons.remove(lid):
            raise HTTPException(404, "leçon inconnue")
        return {"ok": True}

    @app.post("/api/agents/runs/{rid}/corriger")
    def agents_correct(rid: str, body: dict = Body(...)):
        """« Corriger » sur la réponse d'un run : la correction devient une leçon pour les runs suivants."""
        rec = agents_run_get(rid)
        try:
            return lessons.add(body.get("texte", ""), rec.get("objectif", ""), body.get("source", "utilisateur"))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/agents/runs/{rid}/garder")
    def agents_keep(rid: str, body: dict = Body(...)):
        rec = agents_run_get(rid)
        a = rec.get("agents", {}).get(body.get("agent", ""))
        if a is None:
            raise HTTPException(404, "agent absent de ce run")
        spec = agent_lib.save(AgentSpec(a["nom"], a["role"], a["outils"], "orchestrateur"))
        return spec.as_dict()

    # ------------------------------------------------------------------ interface
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(KeyError)
    def key_error(request, exc):
        return JSONResponse({"detail": f"introuvable : {exc}"}, status_code=404)

    _ = time
    return app
