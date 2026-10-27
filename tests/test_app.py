"""
L'app (05/10), de bout en bout sur un modèle miniature (CPU) : chargement, conversation en
direct (SSE), outils avec demande d'accord, réglages, tâche de fond découpée, mise en route
(machine, catalogue, plan), bibliothèque, MCP.
"""

from __future__ import annotations

import json
import time

import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from moelab.app.engine import Engine  # noqa: E402
from moelab.app.server import create_app  # noqa: E402
from moelab.standalone import make_standalone  # noqa: E402
from moelab.store import build_store  # noqa: E402
from moelab.testing import make_demo_model  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("app")
    md = make_demo_model(d / "m", "qwen3next")
    st = build_store(md, d / "store", ("int4",), log=lambda *a: None)
    make_standalone(md, st, log=lambda *a: None)
    engine = Engine(d / "data", log=lambda *a: None)
    app = create_app(engine, results_dir=d / "results")
    with TestClient(app) as c:
        c.store, c.engine, c.tmp = str(st), engine, d
        yield c
    engine.close()


def wait(c, jid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] not in ("en attente", "en cours"):
            return j
        time.sleep(0.2)
    raise AssertionError("job trop long")


def test_load_and_status(client):
    r = client.post("/api/models/add", json={"store": client.store, "nom": "demo"}).json()
    assert r["nom"] == "demo"
    job = client.post("/api/models/load", json={"id": r["id"], "ram_gb": 0.05}).json()
    j = wait(client, job["id"])
    assert j["status"] == "terminé", j
    st = client.get("/api/status").json()
    assert st["model"]["name"] == "store" and st["caches"]["ram"]["budget_gio"] == 0.05
    assert st["bibliotheque"][0]["autonome"] and st["dernier"]["store"] == client.store


def test_chat_streams(client):
    c = client.post("/api/conversations", json={}).json()
    job = client.post(f"/api/conversations/{c['id']}/message", json={"texte": "bonjour", "max_tokens": 6}).json()
    with client.stream("GET", f"/api/jobs/{job['id']}/events") as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1]["type"] == "done" and events[-1]["status"] == "terminé"
    assert any(e["type"] == "text" for e in events)
    conv = client.get(f"/api/conversations/{c['id']}").json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"] and conv["titre"] == "bonjour"
    assert conv["messages"][-1]["stats"]["tokens"] <= 6
    assert client.get("/api/conversations").json()[0]["id"] == c["id"]
    # 2e message : l'historique est repris du cache (sur GPU ; sur CPU, passe normale)
    job = client.post(f"/api/conversations/{c['id']}/message", json={"texte": "encore", "max_tokens": 4}).json()
    assert wait(client, job["id"])["status"] == "terminé"
    client.delete(f"/api/conversations/{c['id']}")
    assert all(x["id"] != c["id"] for x in client.get("/api/conversations").json())


def test_tool_with_approval(client, monkeypatch):
    """Un outil en « demander » : le job s'arrête, l'interface répond, l'outil tourne (ou pas)."""
    from moelab.app import chat

    calls = iter(['<tool_call>{"name": "ecrire_fichier", "arguments": {"chemin": "%s", "contenu": "salut"}}</tool_call>'
                  % str(client.tmp / "ok" / "a.txt").replace("\\", "/"), "C'est écrit."])
    monkeypatch.setattr("moelab.agent.moelab_generate_fn",
                        lambda *a, **k: (lambda msgs, tools: next(calls)))
    client.post("/api/tools/folders", json={"dossiers": [str(client.tmp / "ok")]})
    (client.tmp / "ok").mkdir(exist_ok=True)
    c = client.post("/api/conversations", json={"outils": True}).json()
    job = client.post(f"/api/conversations/{c['id']}/message", json={"texte": "écris"}).json()
    aid = None
    for _ in range(100):
        evs = client.get(f"/api/jobs/{job['id']}").json()["events"]
        ap = [e for e in evs if e["type"] == "approval"]
        if ap:
            aid = ap[0]["id"]
            assert ap[0]["outil"] == "ecrire_fichier"
            break
        time.sleep(0.1)
    assert aid
    assert client.post(f"/api/jobs/{job['id']}/approve", json={"id": aid, "accepte": True}).json()["ok"]
    j = wait(client, job["id"])
    assert j["status"] == "terminé"
    assert (client.tmp / "ok" / "a.txt").read_text() == "salut"
    evs = client.get(f"/api/jobs/{job['id']}").json()["events"]
    assert any(e["type"] == "tool" and "écrit" in e["resultat"] for e in evs)
    _ = chat


def test_chat_tools_step_limit_then_conclude(client, monkeypatch):
    """13/10 : le chat s'arrêtait après 8 tours, au milieu (aucune réponse, fichier non écrit). Maintenant 16 tours,
    puis un dernier tour pour conclure."""
    from moelab.app.chat import CHAT_LAST, CHAT_STEPS

    seen = []

    def gen(msgs, tools):
        seen.append(msgs[-1]["content"])
        if msgs[-1]["content"] == CHAT_LAST:
            return "Voici ce que j'ai trouvé, il manque la 3e version."
        return '<tool_call>{"name": "calculer", "arguments": {"expression": "1+1"}}</tool_call>'

    monkeypatch.setattr("moelab.agent.moelab_generate_fn", lambda *a, **k: gen)
    c = client.post("/api/conversations", json={"outils": True}).json()
    job = client.post(f"/api/conversations/{c['id']}/message", json={"texte": "cherche"}).json()
    assert wait(client, job["id"])["status"] == "terminé"
    conv = client.get(f"/api/conversations/{c['id']}").json()
    assert sum(1 for m in conv["messages"] if m["role"] == "tool") == CHAT_STEPS
    assert conv["messages"][-2]["content"] == CHAT_LAST
    assert conv["messages"][-1]["content"].startswith("Voici ce que j'ai trouvé")
    evs = client.get(f"/api/jobs/{job['id']}").json()["events"]
    assert any(e["type"] == "log" and "dernier tour" in e["texte"] for e in evs)


def test_chat_long_tool_results_fit_in_window(client, monkeypatch):
    """13/10 : huit pages lues dans le chat -> CUDA out of memory. Le contexte est maintenant nettoyé comme chez
    les agents ; la conversation enregistrée garde les résultats complets."""
    folder = client.tmp / "ok"
    folder.mkdir(exist_ok=True)
    client.post("/api/tools/folders", json={"dossiers": [str(folder)]})
    big = folder / "page.txt"
    big.write_text(("ligne de texte de la page " * 20 + "\n") * 400, encoding="utf-8")
    path = str(big).replace("\\", "/")
    calls = iter(['<tool_call>{"name": "lire_fichier", "arguments": {"chemin": "%s", "debut": %d}}</tool_call>'
                  % (path, i * 3000) for i in range(5)] + ["Fini."])
    seen_sizes = []

    def gen(msgs, tools):
        seen_sizes.append(sum(len(m.get("content") or "") for m in msgs))
        return next(calls)

    monkeypatch.setattr("moelab.agent.moelab_generate_fn", lambda *a, **k: gen)
    rt = client.engine.rt
    old = rt.cfg.graph_max_len
    rt.cfg.graph_max_len = 1200
    try:
        c = client.post("/api/conversations", json={"outils": True}).json()
        job = client.post(f"/api/conversations/{c['id']}/message", json={"texte": "lis la page"}).json()
        assert wait(client, job["id"])["status"] == "terminé"
    finally:
        rt.cfg.graph_max_len = old
    evs = client.get(f"/api/jobs/{job['id']}").json()["events"]
    assert any(e["type"] == "log" and ("nettoyé" in e["texte"] or "raccourcie" in e["texte"]) for e in evs)
    assert max(seen_sizes[2:]) < 3 * 3000        # le contexte vu par le modèle ne grossit plus sans fin
    conv = client.get(f"/api/conversations/{c['id']}").json()
    tools_saved = [m["content"] for m in conv["messages"] if m["role"] == "tool"]
    assert len(tools_saved) == 5 and all(len(t) > 1000 and "effacé" not in t for t in tools_saved)
    assert conv["messages"][-1]["content"] == "Fini." and not any(k.startswith("_") for m in conv["messages"] for k in m)
    assert tools_saved[0].startswith("[r1] ") and "archive" in [m for m in conv["messages"] if m["role"] == "tool"][0]
    # message suivant : relire retrouve le texte complet d'un résultat du message précédent
    calls2 = iter(['<tool_call>{"name": "relire", "arguments": {"ref": "r1", "debut": 9000}}</tool_call>', "Ok."])
    monkeypatch.setattr("moelab.agent.moelab_generate_fn", lambda *a, **k: (lambda msgs, tools: next(calls2)))
    job = client.post(f"/api/conversations/{c['id']}/message", json={"texte": "relis"}).json()
    assert wait(client, job["id"])["status"] == "terminé"
    conv = client.get(f"/api/conversations/{c['id']}").json()
    assert "ligne de texte" in conv["messages"][-2]["content"] and "inconnue" not in conv["messages"][-2]["content"]


def test_chat_writes_block_the_model_only_showed(client, monkeypatch):
    """13/10 : au dernier tour, le fichier demandé recopié dans la réponse au lieu d'être écrit."""
    folder = client.tmp / "ok"
    folder.mkdir(exist_ok=True)
    client.post("/api/tools/folders", json={"dossiers": [str(folder)]})
    client.post("/api/tools/permission", json={"nom": "ecrire_fichier", "permission": "toujours"})
    target = folder / "v.md"
    answer = "Voici le contenu du fichier :\n```markdown\n# Versions\n- 2.14.0\n```\nRemarque : dates non trouvées."
    monkeypatch.setattr("moelab.agent.moelab_generate_fn", lambda *a, **k: (lambda msgs, tools: answer))
    try:
        c = client.post("/api/conversations", json={"outils": True}).json()
        job = client.post(f"/api/conversations/{c['id']}/message",
                          json={"texte": f"Écris dans {target} les versions."}).json()
        assert wait(client, job["id"])["status"] == "terminé"
    finally:
        client.post("/api/tools/permission", json={"nom": "ecrire_fichier", "permission": "demander"})
    assert target.read_text(encoding="utf-8") == "# Versions\n- 2.14.0\n"
    conv = client.get(f"/api/conversations/{c['id']}").json()
    assert "a été écrit dans" in conv["messages"][-1]["content"]


def test_chat_write_fact_check(client, monkeypatch):
    """13/10 : dates écrites qui n'étaient dans aucune page lue : refusé une fois, réécrit, accepté."""
    folder = client.tmp / "ok"
    folder.mkdir(exist_ok=True)
    client.post("/api/tools/folders", json={"dossiers": [str(folder)]})
    client.post("/api/tools/permission", json={"nom": "ecrire_fichier", "permission": "toujours"})
    target = str(folder / "f.md").replace("\\", "/")
    call = '<tool_call>{"name": "ecrire_fichier", "arguments": {"chemin": "%s", "contenu": "2.14.0 sortie le 2026-08-27"}}</tool_call>' % target
    calls = iter([call, call, "Écrit."])
    monkeypatch.setattr("moelab.agent.moelab_generate_fn", lambda *a, **k: (lambda msgs, tools: next(calls)))
    try:
        c = client.post("/api/conversations", json={"outils": True}).json()
        job = client.post(f"/api/conversations/{c['id']}/message", json={"texte": "écris"}).json()
        assert wait(client, job["id"])["status"] == "terminé"
    finally:
        client.post("/api/tools/permission", json={"nom": "ecrire_fichier", "permission": "demander"})
    tools = [m["content"] for m in client.get(f"/api/conversations/{c['id']}").json()["messages"] if m["role"] == "tool"]
    assert tools[0].startswith("NON ÉCRIT") and "2026-08-27" in tools[0]
    assert not tools[1].startswith("NON ÉCRIT") and (folder / "f.md").exists()


def test_options_and_tools(client):
    o = client.get("/api/options").json()
    assert o["actuel"]["policy"] == "lo" and any(x["key"] == "t1" for x in o["options"])
    assert client.post("/api/options", json={"cle": "t1", "valeur": "0.7"}).json()["ok"]
    assert client.get("/api/options").json()["actuel"]["t1"] == 0.7
    r = client.post("/api/options", json={"cle": "dense", "valeur": "gpu"})
    assert r.status_code == 409
    t = client.get("/api/tools").json()
    names = {x["nom"]: x for x in t["outils"]}
    assert names["etat"]["permission"] == "toujours" and names["ecrire_fichier"]["permission"] == "demander"
    client.post("/api/tools/permission", json={"nom": "etat", "permission": "jamais"})
    assert {x["nom"]: x for x in client.get("/api/tools").json()["outils"]}["etat"]["permission"] == "jamais"
    assert client.post("/api/tools/permission", json={"nom": "etat", "permission": "parfois"}).status_code == 400


def test_batch_task_interleaves(client):
    folder = client.tmp / "docs"
    folder.mkdir()
    for i in range(3):
        (folder / f"f{i}.txt").write_text(f"texte {i}")
    job = client.post("/api/tasks/batch", json={"dossier": str(folder), "motif": "*.txt", "instruction": "Résume",
                                                "sortie": str(client.tmp / "out.md")}).json()
    j = wait(client, job["id"])
    assert j["status"] == "terminé" and j["result"]["fichiers"] == 3
    out = (client.tmp / "out.md").read_text(encoding="utf-8")
    assert out.count("## f") == 3


def test_setup_endpoints(client):
    m = client.get("/api/setup/machine").json()
    assert m["ram_totale_gio"] > 0 and m["disques"]
    cat = client.get("/api/setup/catalog").json()
    q30 = next(x for x in cat if x["id"] == "qwen3-30b-a3b")
    assert q30["gere"] and q30["stores_gio"]["int4"] == 14.3
    p = client.post("/api/setup/plan", json={"model_id": "qwen3-30b-a3b", "download_dir": str(client.tmp / "dl"),
                                             "store_dir": str(client.tmp / "st")}).json()
    assert {c["precisions"] for c in p["choix"]} == {"int4", "int8+int4", "fp16+int4"}
    assert p["telechargement_gio"] > 25 and p["meme_disque"]


def test_mcp_tools_disabled_then_enabled(client):
    from moelab.app.mcp_server import build_mcp

    app = client.app
    mcp = build_mcp(client.engine, app.state.toolbox, app.state.convs)
    tools = {t.name for t in mcp._tool_manager.list_tools()}
    assert {"demander", "etat_moteur", "lancer_autopilot"} <= tools
    fn = mcp._tool_manager.get_tool("etat_moteur").fn
    with pytest.raises(RuntimeError):
        fn()
    client.post("/api/tools/expose", json={"actif": True})
    assert fn()["model"]["name"] == "store"
    client.post("/api/tools/expose", json={"actif": False})


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "moelab" in r.text
    assert client.get("/static/app.js").status_code == 200
    _ = torch


def test_agents_run_end_to_end(client, monkeypatch):
    """Page Agents (06/10) : run complet avec un faux modèle scripté par agent, run enregistré, agent gardé."""

    def call(name, **args):
        return f'<tool_call>{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}</tool_call>'

    scripts = {"orchestrateur": iter([call("creer_agent", nom="Calculateur", role="fait des calculs", outils=["calculer"]),
                                      call("confier", agent="calculateur", tache="Calcule 12*12"),
                                      call("terminer", reponse="12 × 12 = 144")]),
               "calculateur": iter([call("calculer", expression="12*12"), call("rendre", compte_rendu="144")])}

    def fake(*a, **k):
        def fn(msgs, tools):
            who = "orchestrateur" if "orchestrateur de moelab" in msgs[0]["content"] else "calculateur"
            return next(scripts[who])
        return fn

    monkeypatch.setattr("moelab.agent.moelab_generate_fn", fake)
    d = client.get("/api/agents").json()
    assert any(t["nom"] == "executer_python" for t in d["outils"]) and d["agents"] == []
    job = client.post("/api/agents/run", json={"objectif": "Combien font 12 fois 12 ?", "outils": ["calculer"],
                                                "criteres": False, "verification": "jamais"}).json()
    j = wait(client, job["id"])
    assert j["status"] == "terminé", j
    assert j["result"]["reponse"] == "12 × 12 = 144" and j["result"]["agents"] == ["calculateur"]
    ctx = j["result"]["contexte"]  # 08/10 : statistiques de contexte du run
    assert {"generations", "prompt_tokens", "repris", "replis", "nettoyages", "archives"} <= set(ctx)
    evs = client.get(f"/api/jobs/{job['id']}").json()["events"]
    kinds = [e["type"] for e in evs]
    assert {"speaker", "agent_created", "agent_start", "agent_done", "final"} <= set(kinds)
    runs = client.get("/api/agents/runs").json()
    assert runs[0]["id"] == job["id"] and runs[0]["agents"] == ["calculateur"]
    rec = client.get(f"/api/agents/runs/{job['id']}").json()
    assert rec["agents"]["calculateur"]["taches"][0]["compte_rendu"] == "144"
    assert client.get("/api/agents/runs/..%2Fagents").status_code == 404
    kept = client.post(f"/api/agents/runs/{job['id']}/garder", json={"agent": "calculateur"}).json()
    assert kept["nom"] == "calculateur"
    assert [a["nom"] for a in client.get("/api/agents").json()["agents"]] == ["calculateur"]
    assert client.post("/api/agents", json={"nom": "", "role": "x"}).status_code == 400
    assert client.delete("/api/agents/calculateur").json()["ok"]
    assert client.post("/api/agents/run", json={"objectif": " "}).status_code == 400


def test_agents_oom_retry(client, monkeypatch):
    """Une erreur mémoire GPU pendant une génération : on libère, on raccourcit et on réessaie une fois."""
    state = {"n": 0}

    def fake(*a, **k):
        def fn(msgs, tools):
            state["n"] += 1
            if state["n"] == 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulé)")
            return '<tool_call>{"name": "terminer", "arguments": {"reponse": "fini malgré la mémoire"}}</tool_call>'
        return fn

    monkeypatch.setattr("moelab.agent.moelab_generate_fn", fake)
    job = client.post("/api/agents/run", json={"objectif": "test mémoire", "outils": [], "criteres": False,
                                                "verification": "jamais"}).json()
    j = wait(client, job["id"])
    assert j["status"] == "terminé" and j["result"]["reponse"] == "fini malgré la mémoire"
    evs = client.get(f"/api/jobs/{job['id']}").json()["events"]
    assert any(e["type"] == "log" and "mémoire GPU pleine" in e.get("texte", "") for e in evs)


def test_agents_criteria_verifier_lessons_api(client, monkeypatch):
    """Critères validés par l'utilisateur pendant le run, vérification finale, « Corriger » -> leçon réutilisée."""

    def call(name, **args):
        return f'<tool_call>{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}</tool_call>'

    seen = []

    def fake(*a, **k):
        def fn(msgs, tools):
            sysm = msgs[0]["content"]
            seen.append(msgs)
            if "Tu prépares le travail" in sysm:
                return "- la dernière version publiée"
            if "orchestrateur de moelab" in sysm:
                done = any(m["role"] == "tool" and "NON acceptée" in m["content"] for m in msgs)
                return call("terminer", reponse="Version quatorze." if done else "Version treize.")
            return call("rendre", compte_rendu="PROBLÈMES :\n- « Version treize » : une version plus récente existe.")
        return fn

    monkeypatch.setattr("moelab.agent.moelab_generate_fn", fake)
    client.post("/api/agents/lessons", json={"texte": "dernière version = la plus récente publiée", "objectif": "dernière version"})
    job = client.post("/api/agents/run", json={"objectif": "Quelle est la dernière version de PyTorch ?", "outils": [],
                                                "attente_criteres": 30, "verification": "auto"}).json()
    aid = None
    for _ in range(200):
        evs = client.get(f"/api/jobs/{job['id']}").json()["events"]
        ask = [e for e in evs if e["type"] == "criteria_ask"]
        if ask:
            aid = ask[0]["id"]
            assert ask[0]["criteres"] == ["la dernière version publiée"]
            break
        time.sleep(0.05)
    assert aid
    client.post(f"/api/jobs/{job['id']}/approve", json={"id": aid, "accepte": True,
                                                         "valeur": "- la version la plus récente publiée à ce jour"})
    j = wait(client, job["id"])
    assert j["status"] == "terminé", j
    assert j["result"]["reponse"].startswith("Version quatorze.") and j["result"]["verification"]["ok"] is False
    rec = client.get(f"/api/agents/runs/{job['id']}").json()
    assert rec["criteres"] == ["la version la plus récente publiée à ce jour"]
    assert rec["lecons"] == ["dernière version = la plus récente publiée"]
    assert "Leçons des runs précédents" in seen[0][1]["content"]
    les = client.post(f"/api/agents/runs/{job['id']}/corriger", json={"texte": "toujours vérifier la page des releases"}).json()
    assert les["source"] == "utilisateur" and "pytorch" in les["mots"]
    lst = client.get("/api/agents/lessons").json()
    assert len(lst) == 2
    for x in lst:
        assert client.delete(f"/api/agents/lessons/{x['id']}").json()["ok"]
    assert client.post("/api/agents/run", json={"objectif": "x", "verification": "parfois"}).status_code == 400
