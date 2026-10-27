"""
Orchestrateur et sous-agents (06/10), avec une fausse génération scriptée par agent :
création, délégation, bloc-notes, limites, relances, bibliothèque, conversation gardée,
mise de côté de l'état de décodage, chat servi pendant un long job.
"""

from __future__ import annotations

import json
import threading
import time

from moelab.app.agents import AgentLibrary, AgentSpec, Orchestrator, slug
from moelab.app.engine import Engine
from moelab.app.tools import general_tools


def call(name, **args):
    return f'<tool_call>{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}</tool_call>'


class Script:
    """make_gen : une liste de réponses par agent ; garde les messages vus par chaque agent."""

    def __init__(self, **scripts):
        self.scripts = {k: iter(v) for k, v in scripts.items()}
        self.seen: dict[str, list] = {}

    def __call__(self, name):
        def fn(messages, tools):
            self.seen.setdefault(name, []).append([dict(m) for m in messages])
            return next(self.scripts[name])

        return fn


def tools(tmp_path):
    return general_tools([str(tmp_path)], tmp_path / "work")


def test_create_delegate_finish(tmp_path):
    s = Script(
        orchestrateur=[call("creer_agent", nom="Chercheur", role="calcule et note", outils=["calculer"]),
                       call("confier", agent="chercheur", tache="Calcule 6*7."),
                       "Je conclus. " + call("terminer", reponse="La réponse est 42.")],
        chercheur=[call("calculer", expression="6*7"), call("noter", texte="6*7 = 42"),
                   call("rendre", compte_rendu="42")])
    events, states = [], []
    o = Orchestrator("Combien font 6 fois 7 ?", tools(tmp_path), s, emit=lambda t, **d: events.append((t, d)),
                     save_state=lambda: states.append("save") or len(states),
                     restore_state=lambda st: states.append(f"restore {st}"))
    rec = o.run()
    assert rec["reponse"] == "La réponse est 42."
    t = rec["agents"]["chercheur"]["taches"][0]
    assert t["compte_rendu"] == "42" and [c["nom"] for st in t["etapes"] for c in st["appels"]] == ["calculer", "noter", "rendre"]
    assert t["etapes"][0]["appels"][0]["resultat"] == "[r1] 42"  # tout résultat est archivé (08/10)
    assert rec["notes"][0]["texte"] == "6*7 = 42" and rec["notes"][0]["auteur"] == "chercheur"
    kinds = [e[0] for e in events]
    assert kinds.index("agent_created") < kinds.index("agent_start") < kinds.index("agent_done") < kinds.index("final")
    assert any(e[0] == "tool" and e[1]["agent"] == "chercheur" for e in events)
    # l'état de l'orchestrateur est mis de côté avant la tâche, puis remis après
    assert states[0] == "save" and states[-1] == "restore 1"
    # le sous-agent ne voit que son rôle et sa tâche, pas l'objectif
    first = s.seen["chercheur"][0]
    # 08/10 : message système commun à tous les sous-agents ; le rôle est dans le 1er message
    assert first[0]["role"] == "system" and "calcule et note" not in first[0]["content"]
    assert "Tu es « chercheur »" in first[1]["content"] and "calcule et note" in first[1]["content"]
    assert "6 fois 7" not in json.dumps(first, ensure_ascii=False) and "Calcule 6*7." in first[1]["content"]
    # l'orchestrateur reçoit le compte rendu comme résultat de confier
    assert "Compte rendu de chercheur" in s.seen["orchestrateur"][2][-1]["content"]


def test_limits_and_errors(tmp_path):
    s = Script(orchestrateur=[
        call("creer_agent", nom="a", role="r", outils=["executer_tout"]),
        call("creer_agent", nom="a", role="r", outils="calculer, lire_fichier"),
        call("creer_agent", nom="b", role="r", outils=[]),
        call("creer_agent", nom="orchestrateur", role="r", outils=[]),
        call("confier", agent="zz", tache="x"),
        call("terminer", reponse="fin")])
    o = Orchestrator("obj", tools(tmp_path), s, max_agents=1)
    o.run()
    res = [c["resultat"] for st in o.orch_steps for c in st["appels"]]
    assert "outils inconnus" in res[0] and "executer_tout" in res[0]
    # un agent qui lit des fichiers reçoit aussi de quoi naviguer
    assert "créé" in res[1] and o.agents["a"]["spec"].outils == ["calculer", "lire_fichier", "lister_dossier", "chercher_fichiers"]
    assert "Limite atteinte" in res[2]
    assert "Limite" in res[3] or "réservé" in res[3]
    assert "agent inconnu" in res[4]


def test_nudge_then_fallback(tmp_path):
    s = Script(orchestrateur=["Je réfléchis...", call("terminer", reponse="ok")])
    o = Orchestrator("obj", tools(tmp_path), s)
    assert o.run()["reponse"] == "ok"
    assert s.seen["orchestrateur"][1][-1]["role"] == "user" and "sans appeler d'outil" in s.seen["orchestrateur"][1][-1]["content"]
    # plus de relances : la réponse est ce qui a été produit
    s2 = Script(orchestrateur=["a", "b", "c"])
    o2 = Orchestrator("obj", tools(tmp_path), s2, nudges=2)
    rec = o2.run()
    assert "n'a pas appelé terminer" in rec["reponse"] and "c" in rec["reponse"]


def test_library_agent_and_kept_conversation(tmp_path):
    lib = AgentLibrary(tmp_path / "data")
    lib.save(AgentSpec("Rédacteur", "écrit des synthèses", ["calculer", "ecrire_fichier", "outil_disparu"], "utilisateur"))
    assert lib.get("redacteur").outils[0] == "calculer"
    s = Script(
        orchestrateur=[call("confier", agent="redacteur", tache="Tâche 1"),
                       call("confier", agent="redacteur", tache="Tâche 2"),
                       call("terminer", reponse="fini")],
        redacteur=[call("rendre", compte_rendu="un"), call("rendre", compte_rendu="deux")])
    states = iter(range(100))
    restored = []
    o = Orchestrator("obj", tools(tmp_path), s, library=lib, save_state=lambda: next(states),
                     restore_state=restored.append)
    assert "redacteur (écrit des synthèses" in o.system_prompt()
    rec = o.run()
    a = rec["agents"]["redacteur"]
    assert a["origine"] == "bibliotheque" and a["outils"] == ["calculer", "ecrire_fichier", "lister_dossier", "chercher_fichiers"]
    assert [t["compte_rendu"] for t in a["taches"]] == ["un", "deux"]
    # 2e tâche : l'agent voit sa 1re tâche (conversation gardée) et son état de décodage est remis
    second = s.seen["redacteur"][1]
    # repli par tâche (08/10) : la 2e tâche voit la 1re et son compte rendu, pas ses étapes
    assert second[1]["content"].endswith("Tâche : Tâche 1") and second[2] == {"role": "assistant",
                                                                             "content": "(tâche terminée) un"}
    assert second[-1]["content"].startswith("Nouvelle tâche : Tâche 2") and len(second) == 4
    # seul l'état de l'orchestrateur est mis de côté puis remis ; l'agent repart de son préfixe commun
    assert restored == [0, 1]
    assert lib.get("redacteur").runs == 1
    assert slug("Analyste Web!") == "analyste_web"


def test_out_of_time(tmp_path):
    s = Script(orchestrateur=[call("creer_agent", nom="a", role="r", outils=[]), call("terminer", reponse="vite")])
    o = Orchestrator("obj", tools(tmp_path), s, minutes=0)
    o.run()
    assert "Budget de temps épuisé" in o.orch_steps[0]["appels"][0]["resultat"]


def test_sub_agent_without_report(tmp_path):
    s = Script(orchestrateur=[call("creer_agent", nom="a", role="r", outils=["calculer"]),
                              call("confier", agent="a", tache="t"), call("terminer", reponse="r")],
               a=["Voici le résultat : 12."])
    o = Orchestrator("obj", tools(tmp_path), s)
    rec = o.run()
    assert rec["agents"]["a"]["taches"][0]["compte_rendu"] == "Voici le résultat : 12."


def test_serve_waiting_runs_chat_inside_long_job(tmp_path):
    eng = Engine(tmp_path / "d", log=lambda *a: None)
    order = []
    started = threading.Event()

    def long(job, engine):
        started.set()
        time.sleep(0.5)  # le chat arrive pendant ce temps
        n = engine.serve_waiting(job.priority)
        order.append(("servis", n))
        return "fin"

    j = eng.submit("agents", "long", long, needs_model=False)
    started.wait(5)
    c = eng.submit("chat", "chat", lambda job, e: order.append("chat") or "ok", needs_model=False)
    ld = eng.submit("load", "load", lambda job, e: order.append("load") or "ok", needs_model=False)
    t0 = time.time()
    while ld.status != "terminé" and time.time() - t0 < 10:
        time.sleep(0.05)
    assert order[:2] == ["chat", ("servis", 1)] and order[2] == "load"  # le chargement attend la fin du long job
    assert j.status == "terminé" and c.status == "terminé"
    assert eng.serve_waiting(9) == 0  # hors du fil de travail : ne fait rien
    eng.close()


def test_run_of_06_10_replayed(tmp_path):
    """
    Le 1er vrai run (06/10) : appels sans balises (`terminer({...})`), lire_fichier sur un dossier
    (PermissionError sous Windows, pris pour un refus), mauvais noms d'arguments. Rejoué ici.
    """
    d = tmp_path / "test"
    d.mkdir()
    (d / "a.txt").write_text("Texte A", encoding="utf-8")
    s = Script(
        orchestrateur=[
            call("chercher_fichiers", chemin=str(d), pattern="*.txt"),            # mauvais noms d'arguments
            call("lister_dossier", chemin=str(d)),                                 # permis à l'orchestrateur
            call("creer_agent", nom="lecteur", role="lit", outils=["lire_fichier"]),
            call("confier", agent="lecteur", tache=f"Lis les .txt de {d}"),
            'Je termine.\n\nterminer({"reponse": "Résumé : A."})'],              # sans <tool_call>
        lecteur=[call("lire_fichier", chemin=str(d) + "/"),                         # un dossier
                 call("lister_dossier", chemin=str(d)),
                 call("lire_fichier", chemin=str(d / "a.txt")),
                 'rendre({"compte_rendu": "a.txt : Texte A"})'])
    o = Orchestrator("Résume les fichiers", tools(tmp_path), s)
    rec = o.run()
    assert rec["reponse"] == "Résumé : A."
    r = [c["resultat"] for st in o.orch_steps for c in st["appels"]]
    assert "inconnus ['chemin', 'pattern']" in r[0] and "dossier (obligatoire)" in r[0] and "motif" in r[0]
    assert "a.txt" in r[1]
    steps = rec["agents"]["lecteur"]["taches"][0]["etapes"]
    assert "est un dossier" in steps[0]["appels"][0]["resultat"] and "lister_dossier" in steps[0]["appels"][0]["resultat"]
    assert rec["agents"]["lecteur"]["taches"][0]["compte_rendu"] == "a.txt : Texte A"
    assert "<tool_call>" in o.system_prompt() and "lister_dossier, chercher_fichiers" in o.system_prompt()


def test_folders_in_prompts_and_string_calls(tmp_path):
    """2e vrai run (06/10) : liste des dossiers autorisés vide, et `terminer("texte")` (une chaîne, sans JSON)."""
    s = Script(orchestrateur=['Rien à lire.\n\nnoter("Aucun dossier autorisé.")\nterminer("Ajoute le dossier dans Outils et MCP.")'])
    o = Orchestrator("Résume C:/x", general_tools([], tmp_path / "w"), s)
    assert "AUCUN" in o.system_prompt() and "Outils et MCP" in o.system_prompt()
    rec = o.run()
    assert rec["reponse"] == "Ajoute le dossier dans Outils et MCP." and rec["notes"][0]["texte"] == "Aucun dossier autorisé."
    o2 = Orchestrator("x", tools(tmp_path), s, folders=[str(tmp_path)])
    assert str(tmp_path) in o2.system_prompt() and "AUCUN" not in o2.system_prompt()
    o2.create("a", "r", ["calculer"])
    assert str(tmp_path) in o2.agents["a"]["messages"][0]["content"]


def test_third_run_kwargs_and_append(tmp_path):
    """
    3e vrai run (06/10, réussi) : `rendre(compte_rendu="...")` en style Python n'était pas lu, et le
    rédacteur relisait et réécrivait tout resume.md à chaque fichier. Maintenant : appel lu,
    ajouter_fichier écrit à la suite sans relire.
    """
    d = tmp_path / "test"
    d.mkdir()
    out = d / "resume.md"
    s = Script(
        orchestrateur=[call("creer_agent", nom="redacteur", role="résume", outils=["ajouter_fichier"]),
                       call("confier", agent="redacteur", tache="a"), call("confier", agent="redacteur", tache="b"),
                       'terminer(reponse="Fini : resume.md.")'],
        redacteur=[f'ajouter_fichier(chemin="{out}", contenu="## a.txt\\n1. un")',
                   'rendre(compte_rendu="a résumé (une ligne).")',
                   f'ajouter_fichier(chemin=r"{out}", contenu="## b.txt\\n1. deux\\n")',
                   "rendre(compte_rendu='b résumé')"])
    o = Orchestrator("Résume", tools(tmp_path), s)
    rec = o.run()
    assert rec["reponse"] == "Fini : resume.md."
    assert [t["compte_rendu"] for t in rec["agents"]["redacteur"]["taches"]] == ["a résumé (une ligne).", "b résumé"]
    assert out.read_text(encoding="utf-8") == "## a.txt\n1. un\n## b.txt\n1. deux\n"
    assert "lister_dossier" in rec["agents"]["redacteur"]["outils"]  # ajouter_fichier compte comme outil de fichiers


def test_windows_path_keeps_newlines():
    """Sous Windows (06/10) : le chemin à \\ simples forçait une lecture brute, et le contenu gardait des « \\n » littéraux."""
    from moelab.agent import parse_tool_calls
    from moelab.app.tools import general_tools

    known = {t.name: t for t in general_tools([], None)}
    src = r'ajouter_fichier(chemin="C:\Users\Home\AppData\Local\Temp\t\resume.md", contenu="## a.txt\n1. un\tok")'
    _, calls = parse_tool_calls(src, known)
    assert calls[0]["arguments"] == {"chemin": "C:\\Users\\Home\\AppData\\Local\\Temp\\t\\resume.md",
                                     "contenu": "## a.txt\n1. un\tok"}
    _, calls = parse_tool_calls(r'lire_fichier(chemin="D:\\Utilisateurs\\x.txt", debut=3)', known)
    assert calls[0]["arguments"] == {"chemin": "D:\\Utilisateurs\\x.txt", "debut": 3}


def test_fourth_run_dates_sources_truncated_call(tmp_path):
    """
    4e vrai run (06/10) : dates de PyTorch 2.12-2.14 inventées (2024 au lieu de 2026, le modèle ne
    connaissait pas la date du jour), et « rende » + {"compte_rendu": ...} (nom tronqué) non lu.
    """
    s = Script(orchestrateur=[call("creer_agent", nom="redacteur", role="écrit", outils=["calculer"]),
                              call("confier", agent="redacteur", tache="t"),
                              'terminer("ok")'],
               redacteur=['Fini.\nrende\n{"compte_rendu": "Fichier créé (source : D:/x.md)."}'])
    o = Orchestrator("Les 3 dernières versions de PyTorch", tools(tmp_path), s, today="mardi 6 octobre 2026")
    sp = o.system_prompt()
    assert "Nous sommes le mardi 6 octobre 2026" in sp and "non trouvé" in sp and "Ne complète jamais de mémoire" in sp
    rec = o.run()
    sub = rec["agents"]["redacteur"]
    assert sub["taches"][0]["compte_rendu"] == "Fichier créé (source : D:/x.md)."
    sys_sub = s.seen["redacteur"][0][0]["content"]
    assert "Nous sommes le mardi 6 octobre 2026" in sys_sub and "Règle des sources" in sys_sub


def test_json_only_fallback_is_strict(tmp_path):
    from moelab.agent import parse_tool_calls

    known = {t.name: t for t in general_tools([str(tmp_path)], tmp_path / "w")}
    # ecrire_fichier et ajouter_fichier ont les mêmes arguments : ambigu, rien n'est appelé
    assert parse_tool_calls('{"chemin": "a", "contenu": "b"}', known)[1] == []
    # objet au milieu d'une réponse : pas un appel
    assert parse_tool_calls('Exemple {"expression": "1+1"} dans le texte.', known)[1] == []
    assert parse_tool_calls('calc\n{"expression": "1+1"}', known)[1] == [{"name": "calculer", "arguments": {"expression": "1+1"}}]


def test_prefix_cache_pruned(tmp_path):
    import os

    from moelab.graphs import GraphDecoder

    for i in range(30):
        f = tmp_path / f"prefix_{i:02d}.pt"
        f.write_bytes(b"x")
        os.utime(f, (1000 + i, 1000 + i))
    GraphDecoder._prune_prefixes(tmp_path)
    left = sorted(f.name for f in tmp_path.glob("prefix_*.pt"))
    assert len(left) == 24 and left[0] == "prefix_06.pt"  # les 6 plus anciens supprimés


def test_fifth_run_step_limit_and_tool_hint(tmp_path):
    """
    5e vrai run (06/10) : le chercheur atteint sa limite de tours sans appeler rendre (compte rendu vide,
    ses notes perdues) ; l'orchestrateur appelle chercher_web lui-même, lit « outil inconnu » et conclut
    que la recherche web n'existe pas, puis veut recréer le chercheur.
    """
    s = Script(
        orchestrateur=[call("creer_agent", nom="chercheur", role="cherche", outils=["calculer", "executer_python"]),
                       call("confier", agent="chercheur", tache="Trouve 2.14 et 2.13"),
                       call("executer_python", code="print(1)"),                   # outil des agents
                       call("creer_agent", nom="chercheur", role="cherche", outils=["calculer"]),
                       call("terminer", reponse="fin")],
        chercheur=[call("noter", texte="2.14 : septembre 2026 (source : github)"),
                   call("calculer", expression="2"), call("calculer", expression="3"),
                   call("rendre", compte_rendu="2.14 en septembre 2026 ; 2.13 non trouvé")])  # au tour forcé
    o = Orchestrator("x", tools(tmp_path), s, sub_steps=4)
    rec = o.run()
    t = rec["agents"]["chercheur"]["taches"][0]
    # rendu au tour forcé : accepté, mais 2.14 n'a été lu nulle part (seulement des calculs) -> signalé
    assert t["compte_rendu"].startswith("2.14 en septembre 2026 ; 2.13 non trouvé")
    assert "non retrouvé" in t["compte_rendu"] and "2.14" in t["compte_rendu"].split("non retrouvé")[1]
    last_seen = s.seen["chercheur"][-1]
    assert "Limite de tours atteinte" in last_seen[-1]["content"]
    r = [c["resultat"] for st in o.orch_steps for c in st["appels"]]
    assert "outil des agents, pas le tien" in r[2] and "(chercheur)" in r[2]
    assert "existe déjà (outils : calculer" in r[3] and "autre nom" in r[3]


def test_step_limit_without_report_keeps_notes(tmp_path):
    s = Script(orchestrateur=[call("creer_agent", nom="a", role="r", outils=["calculer"]),
                              call("confier", agent="a", tache="t"), call("terminer", reponse="fin")],
               a=[call("noter", texte="fait trouvé (source : x)"), call("calculer", expression="1"),
                  "Je continue à chercher."])  # ne rend toujours pas au tour forcé
    o = Orchestrator("x", tools(tmp_path), s, sub_steps=3)
    rec = o.run()
    cr = rec["agents"]["a"]["taches"][0]["compte_rendu"]
    assert cr.startswith("Je continue à chercher.") and "fait trouvé (source : x)" in cr


def test_sixth_run_fact_checks(tmp_path):
    """
    6e vrai run (06/10) : dates inventées par le chercheur, nouveautés inventées par le rédacteur (qui ne
    voyait pas le compte rendu du chercheur), vérification confiée au rédacteur lui-même. Rejoué avec une
    fausse page GitHub : le code refuse une fois ce qui n'a été lu nulle part.
    """
    from moelab.agent import Tool

    from moelab.app.tools import find_passages

    page = "Releases · pytorch\nPyTorch 2.14.0 Release\n@ethche released this 02 Sep 17:40\nNVGEMM ; Batched linalg.eigh up to 100x faster"

    def lire_page(url: str, chercher: str = "", debut: int = 0) -> str:
        return find_passages(page, chercher) if chercher else page

    work = [t for t in tools(tmp_path) if t.name != "lire_page"]
    work.append(Tool("lire_page", "lit une page", {"type": "object", "properties": {
        "url": {"type": "string"}, "chercher": {"type": "string"}, "debut": {"type": "integer"}}, "required": ["url"]},
        lire_page))
    out = tmp_path / "pytorch.md"
    s = Script(
        orchestrateur=[
            call("creer_agent", nom="chercheur", role="cherche", outils=["lire_page"]),
            call("confier", agent="chercheur", tache="Date de sortie de PyTorch 2.14 ?"),
            call("creer_agent", nom="redacteur", role="écrit", outils=["ecrire_fichier"]),
            call("confier", agent="redacteur", tache=f"Écris {out} à partir du bloc-notes."),
            call("lire_fichier", chemin=str(out)),
            call("terminer", reponse="2.14 : 2 septembre, CUDA 12.4."),          # 12.4 : seulement dans le fichier écrit
            call("terminer", reponse="2.14 : 2 septembre ; linalg.eigh 100x.")],
        chercheur=[
            call("lire_page", url="https://github.com/pytorch/pytorch/releases", chercher="released"),
            call("noter", texte="2.14 sortie le 18 septembre 2026"),            # inventé
            call("rendre", compte_rendu="2.14 : 18 septembre 2026"),            # refusé une fois
            call("rendre", compte_rendu="2.14 : 2 septembre 2026 (github), linalg.eigh 100x")],
        redacteur=[
            call("ecrire_fichier", chemin=str(out), contenu="# 2.14\n- 2 septembre\n- CUDA 12.4"),   # refusé une fois
            call("ecrire_fichier", chemin=str(out), contenu="# 2.14\n- 2 septembre\n- linalg.eigh 100x\n- CUDA 12.4"),
            call("rendre", compte_rendu="écrit")])
    o = Orchestrator("Dernière version de PyTorch", work, s, today="mardi 6 octobre 2026")
    rec = o.run()
    st = rec["agents"]["chercheur"]["taches"][0]["etapes"]
    res = [c["resultat"] for e in st for c in e["appels"]]
    assert "released this 02 Sep" in res[0]
    assert "non retrouvé" in res[1] and "18 septembre" in res[1]                # note signalée
    assert res[2].startswith("Compte rendu NON accepté") and "18 septembre" in res[2]
    assert rec["agents"]["chercheur"]["taches"][0]["compte_rendu"] == "2.14 : 2 septembre 2026 (github), linalg.eigh 100x"
    # le rédacteur voit le compte rendu du chercheur dans le bloc-notes
    assert "Compte rendu (tâche" in s.seen["redacteur"][0][-1]["content"] and "2 septembre 2026" in s.seen["redacteur"][0][-1]["content"]
    rs = [c["resultat"] for e in rec["agents"]["redacteur"]["taches"][0]["etapes"] for c in e["appels"]]
    assert rs[0].startswith("NON ÉCRIT") and "12.4" in rs[0]
    assert "écrit" in rs[1] and "non retrouvé" in rs[1] and "12.4" in rs[1]   # 2e essai : écrit, mais signalé
    assert out.read_text(encoding="utf-8").endswith("CUDA 12.4")
    # l'orchestrateur relit le fichier lui-même ; ce fichier n'est pas une source : 12.4 reste non retrouvé
    r = [c["resultat"] for e in o.orch_steps for c in e["appels"]]
    assert "CUDA 12.4" in r[4]
    assert r[5].startswith("Réponse NON acceptée") and "12.4" in r[5]
    assert rec["reponse"] == "2.14 : 2 septembre ; linalg.eigh 100x."
    where = [c["ou"] for c in rec["verifications"]]
    assert where[:3] == ["note", "compte rendu (refusé une fois)", f"fichier {out} (refusé une fois)"]


def test_lire_page_find_and_chunks(tmp_path, monkeypatch):
    from moelab.app import tools as T

    long = "menu " * 2000 + "@ethche released this 02 Sep 17:40 " + "pied " * 2000
    monkeypatch.setattr(T, "html_to_text", lambda html: long)

    class R:
        headers = type("H", (), {"get_content_charset": staticmethod(lambda: "utf-8")})()

        def read(self, n):
            return b"<html></html>"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: R())
    lp = {t.name: t for t in T.general_tools([], tmp_path)}["lire_page"]
    first = lp.fn(url="https://x.org/p")
    assert len(first) < 4200 and "suite avec debut=4000" in first
    found = lp.fn(url="https://x.org/p", chercher="RELEASED")
    assert found.startswith("1 passage(s)") and "02 Sep 17:40" in found
    assert "n'apparaît pas" in lp.fn(url="https://x.org/p", chercher="zzz")


def test_seventh_run_context_and_companions(tmp_path):
    """
    7e vrai run (06/10) : chercheur créé avec chercher_web sans lire_page (aucune page lue), contexte
    trop long (« CUDA out of memory » dans confier), prompt affiché dans le déroulé après l'erreur.
    """
    from moelab.agent import compact_messages

    o = Orchestrator("x", tools(tmp_path), lambda n: None)
    o.create("chercheur", "cherche", ["chercher_web"])
    assert o.agents["chercheur"]["spec"].outils == ["chercher_web", "lire_page"]
    # bloc-notes limité : les plus récentes d'abord, un renvoi vers lire_notes
    for i in range(50):
        o.notes.append({"auteur": "a", "texte": f"note {i} " + "x" * 200})
    txt = o.notes_text(limit=3000)
    assert len(txt) < 3400 and "note 49" in txt and "note 0 " not in txt and "lire_notes" in txt
    # raccourcir une conversation : système et 1re demande gardés, anciens résultats raccourcis puis retirés
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "objectif"}]
    for i in range(10):
        msgs += [{"role": "assistant", "content": f"appel {i}"}, {"role": "tool", "content": "r" * 2000}]
    count = lambda m: sum(len(x["content"]) for x in m) // 4  # noqa: E731
    n = compact_messages(msgs, count, budget=1500)
    assert n > 0 and count(msgs) <= 1500
    assert msgs[0]["content"] == "S" and msgs[1]["content"] == "objectif" and msgs[-1]["content"] == "r" * 2000
    assert any("[... raccourci" in m["content"] for m in msgs)


def test_text_sink_reset_after_interrupted_generation():
    from moelab.app.engine import Job, TextSink

    class Tok:
        def decode(self, ids, **k):
            return "".join(chr(97 + (int(i) % 26)) for i in ids)

    job = Job("t", "t", None)
    sink = TextSink(Tok(), job)
    import torch

    sink.put(torch.tensor([[1, 2, 3]]))  # prompt : pas affiché
    sink.put(torch.tensor([4]))          # 1er token... puis erreur mémoire : pas de end()
    sink.reset()
    sink.put(torch.tensor([[7, 7, 7, 7]]))  # le prompt suivant ne doit pas s'afficher
    sink.put(torch.tensor([5]))
    sink.end()
    shown = "".join(e["text"] for e in job.events if e["type"] == "text")
    assert "hhhh" not in shown and "f" in shown


def test_shared_prefix_refusal_archive_and_parallel_calls(tmp_path):
    """
    08/10 : même message système et mêmes outils (même ordre) pour tous les sous-agents -> un seul cache de
    préfixe ; un outil non reçu est refusé sans s'exécuter ; un long résultat est archivé ([r1]) et relu
    par relire ; plusieurs appels indépendants dans un même message.
    """
    from moelab.agent import Tool

    page = "début " + "x" * 9000 + " released this 02 Sep 17:40 " + "y" * 3000
    calls_made = []

    def lire_page(url: str, chercher: str = "", debut: int = 0) -> str:
        calls_made.append(url)
        return page

    work = [t for t in tools(tmp_path) if t.name != "lire_page"]
    work.append(Tool("lire_page", "lit", {"type": "object", "properties": {"url": {"type": "string"}},
                                           "required": ["url"]}, lire_page))
    seen_tools: dict[str, list] = {}
    scripts = {
        "orchestrateur": iter([call("creer_agent", nom="a", role="lit", outils=["lire_page"]),
                               call("creer_agent", nom="b", role="calcule", outils=["calculer"]),
                               call("confier", agent="a", tache="lis"), call("confier", agent="b", tache="calcule"),
                               call("terminer", reponse="ok")]),
        "a": iter([call("lire_page", url="https://u/1") + call("lire_page", url="https://u/2"),   # 2 appels, 1 message
                   call("relire", ref="r1", chercher="released"),
                   call("rendre", compte_rendu="vu")]),
        "b": iter([call("ecrire_fichier", chemin=str(tmp_path / "x.txt"), contenu="x")
                   + call("lire_page", url="https://u/3"), call("rendre", compte_rendu="refusé")])}
    seen_msgs: dict[str, list] = {}

    def make_gen(name):
        def fn(messages, tools_):
            seen_tools.setdefault(name, []).append([t["function"]["name"] for t in tools_])
            seen_msgs.setdefault(name, []).append([dict(m) for m in messages])
            return next(scripts[name])
        return fn

    o = Orchestrator("x", work, make_gen)
    rec = o.run()
    assert seen_msgs["a"][0][0] == seen_msgs["b"][0][0]                    # même message système
    assert seen_tools["a"][0] == seen_tools["b"][0]                        # mêmes outils, même ordre
    # 10e run : un outil de lecture non donné est accordé au 1er appel ; l'écriture reste refusée
    assert calls_made == ["https://u/1", "https://u/2", "https://u/3"] and not (tmp_path / "x.txt").exists()
    assert rec["agents"]["b"]["outils"] == ["calculer", "lire_page"]
    st = rec["agents"]["a"]["taches"][0]["etapes"]
    res = [c["resultat"] for c in st[0]["appels"]]
    assert len(res) == 2 and res[0].startswith("[r1] ") and res[1].startswith("[r2] ")
    got = [m["content"] for m in seen_msgs["a"][1] if m["role"] == "tool"]     # ce que l'agent a reçu
    assert "relire('r1', debut=" in got[0] and len(got[0]) <= 4000
    assert "released this 02 Sep" in st[1]["appels"][0]["resultat"]        # retrouvé dans l'archive
    rb = rec["agents"]["b"]["taches"][0]["etapes"][0]["appels"][0]["resultat"]
    assert rb.startswith("Refusé") and "calculer" in rb
    assert rec["contexte"]["archives"] == 3 and rec["contexte"]["replis"] == 2


def test_clean_context_once_then_append_only():
    from moelab.agent import clean_context

    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "objectif"}]
    for i in range(12):
        msgs += [{"role": "assistant", "content": "Je réfléchis longuement. " * 20
                  + f'<tool_call>{{"name": "lire_page", "arguments": {{"url": "u{i}"}}}}</tool_call>'},
                 {"role": "tool", "content": f"[r{i + 1}] " + "texte " * 400}]
    count = lambda m: sum(len(x["content"]) for x in m) // 4  # noqa: E731
    total = count(msgs)
    assert clean_context(msgs, count, trigger=total + 1, target=10) == 0    # sous le seuil : rien
    n = clean_context(msgs, count, trigger=int(total * 0.8), target=int(total * 0.3))
    assert n > 0 and count(msgs) <= int(total * 0.3)
    assert msgs[:2] == [{"role": "system", "content": "S"}, {"role": "user", "content": "objectif"}]
    assert msgs[-1]["content"].startswith("[r12] texte")                    # les derniers messages intacts
    stubs = [m["content"] for m in msgs[2:-2]]
    assert all(c.startswith("<tool_call>") or "archivé : relire('r" in c for c in stubs)
    # ensuite la conversation ne fait que s'allonger : pas de nouveau nettoyage tant que le seuil n'est pas atteint
    before = [dict(m) for m in msgs]
    msgs.append({"role": "assistant", "content": "suite"})
    assert clean_context(msgs, count, trigger=int(total * 0.8), target=int(total * 0.3)) == 0 and msgs[:-1] == before


def test_criteria_verifier_and_lessons(tmp_path):
    """
    08/10 : critères au départ, vérificateur indépendant une fois avant terminer, leçons données à toutes
    les étapes. Rejoue le run du 08/10 (2.14 oubliée) : le vérificateur la signale, l'orchestrateur corrige.
    """
    s = Script(
        criteres=["Voici :\n- les 3 versions les plus récentes publiées au 8 octobre 2026\n- écrites dans pytorch.md"],
        orchestrateur=[call("terminer", reponse="Versions onze, douze et treize."),   # sans chiffres : seul le
                       call("terminer", reponse="Versions douze, treize et quatorze.")],  # vérificateur intervient
        verificateur=[call("rendre", compte_rendu="PROBLÈMES :\n- MANQUE : la version la plus récente "
                                                    "(preuve : [r1] « version quatorze publiée »)")])
    lessons = [{"texte": "« dernières versions » = les plus récentes à la date du jour"}]
    o = Orchestrator("Les trois dernières versions de PyTorch", tools(tmp_path), s, criteria=True, verify="auto",
                     lessons=lessons, today="jeudi 8 octobre 2026")
    o.archive["r1"] = "Releases : version quatorze publiée en septembre"   # une source déjà lue
    o.source_refs.add("r1")
    rec = o.run()
    assert rec["criteres"] == ["les 3 versions les plus récentes publiées au 8 octobre 2026", "écrites dans pytorch.md"]
    crit_prompt = s.seen["criteres"][0]
    assert "plus récentes à la date du jour" in crit_prompt[1]["content"]          # leçon donnée aux critères
    orch_first = s.seen["orchestrateur"][0][1]["content"]
    assert "Critères de réussite" in orch_first and "pytorch.md" in orch_first and "Leçons des runs" in orch_first
    r = [c["resultat"] for st in o.orch_steps for c in st["appels"]]
    assert r[0].startswith("Réponse NON acceptée par le vérificateur") and "MANQUE" in r[0]
    assert rec["reponse"].startswith("Versions douze, treize et quatorze.") and "Le vérificateur avait signalé" in rec["reponse"]
    # le vérificateur voit l'objectif, les critères et la réponse, pas l'historique de l'orchestrateur
    v = s.seen["verificateur"][0]
    assert "Tu es « verificateur »" in v[1]["content"] and "Versions onze, douze et treize." in v[1]["content"]
    assert "les 3 versions les plus récentes" in v[1]["content"] and len(v) == 2
    assert "Bloc-notes partagé" not in v[1]["content"]                             # pas de bloc-notes périmé
    assert rec["verification"]["ok"] is False and o.created == 0                   # pas compté dans les agents créés


def test_verifier_auto_skips_simple_tasks_and_user_criteria(tmp_path):
    s = Script(criteres=["- trois résumés"], orchestrateur=[call("terminer", reponse="Résumés écrits.")])
    o = Orchestrator("Résume ces fichiers", tools(tmp_path), s, criteria=True, verify="auto",
                     ask_criteria=lambda crit: ["un résumé de 5 lignes par fichier"])
    rec = o.run()
    assert rec["verification"] is None and "verificateur" not in rec["agents"]      # rien à vérifier : pas de coût
    assert rec["criteres"] == ["un résumé de 5 lignes par fichier"]                 # corrigés par l'utilisateur


def test_lesson_book(tmp_path):
    from moelab.app.lessons import LessonBook

    b = LessonBook(tmp_path)
    b.add("« dernières versions » = les plus récentes à la date du jour ; vérifier la liste complète", "Les 3 dernières versions de PyTorch")
    b.add("Résumés : exactement 5 lignes", "Résume les fichiers texte")
    found = b.relevant("Quelles sont les dernières versions de Python ?")
    assert len(found) == 1 and found[0]["texte"].startswith("« dernières versions »")
    assert b.relevant("Une question sans rapport") == []
    assert b.list()[0]["utilisations"] in (0, 1)
    lid = b.list()[0]["id"]
    assert b.remove(lid) and not b.remove(lid)


def test_job_ask_value():
    import threading

    from moelab.app.engine import Job

    j = Job("t", "t", None)
    assert j.ask_value("criteria_ask", {"criteres": ["a"]}, timeout=0, default="defaut") == "defaut"

    def answer():
        for _ in range(100):
            ev = [e for e in j.events if e["type"] == "criteria_ask"]
            if ev:
                j.answer(ev[0]["id"], True, "- corrigé")
                return
            threading.Event().wait(0.01)

    threading.Thread(target=answer).start()
    assert j.ask_value("criteria_ask", {"criteres": ["a"]}, timeout=5, default=None) == "- corrigé"



def test_eighth_run_verifier_must_quote(tmp_path):
    """
    Run du 08/10 (62 min) : le fichier contenait les bonnes dates, mais le vérificateur, qui ne voyait que le
    résumé de l'orchestrateur et un bloc-notes périmé, a « vu » 2026-09-10 dans le fichier. Maintenant il voit
    le vrai fichier et chaque problème doit citer le livrable ou une source : les faux problèmes sont rejetés.
    """
    from moelab.agent import Tool

    out = tmp_path / "pytorch.md"
    page = "PyTorch 2.14.0 Release @ethche released this 02 Sep 17:40 NVGEMM"

    def lire_page(url: str, chercher: str = "", debut: int = 0) -> str:
        return page

    work = [t for t in tools(tmp_path) if t.name != "lire_page"]
    work.append(Tool("lire_page", "lit", {"type": "object", "properties": {"url": {"type": "string"}},
                                           "required": ["url"]}, lire_page))
    s = Script(
        orchestrateur=[call("creer_agent", nom="chercheur", role="cherche", outils=["lire_page", "ecrire_fichier"]),
                       call("confier", agent="chercheur", tache="2.14 et écris le fichier"),
                       call("terminer", reponse="Fichier écrit.")],
        chercheur=[call("lire_page", url="https://github.com/pytorch/pytorch/releases"),
                   call("ecrire_fichier", chemin=str(out), contenu="## 2.14.0\n- 02 Sep\n- NVGEMM"),
                   call("rendre", compte_rendu="Fichier écrit, 2.14.0 du 02 Sep (source : github).")],
        verificateur=[call("rendre", compte_rendu="PROBLÈMES :\n- « 2026-09-10 » : mauvaise date (preuve : [r1])\n"
                                                   "- la nouveauté est tronquée\n"
                                                   "- MANQUE : la 2.13 (preuve : [r1] « 2.13.0 released »)")])
    o = Orchestrator("La dernière version de PyTorch dans pytorch.md", work, s, verify="toujours")
    rec = o.run()
    assert rec["reponse"] == "Fichier écrit."                                   # pas de faux refus
    assert rec["verification"]["ok"] is True and len(rec["verification"]["rejetes"]) == 3
    v = s.seen["verificateur"][0][1]["content"]
    assert "02 Sep" in v and "NVGEMM" in v and "=== LIVRABLE" in v                # le vrai fichier
    assert "Bloc-notes partagé" not in v


def test_after_verifier_bounded_and_clean_notes(tmp_path):
    """Après un refus (problème prouvé), 4 tours au plus ; sinon la dernière proposition, avec les remarques."""
    from moelab.app.agents import cut_clean

    s = Script(orchestrateur=[call("terminer", reponse="Réponse A."), call("lire_notes"), call("lire_notes"),
                              call("lire_notes"), call("lire_notes"), call("lire_notes")],
               verificateur=[call("rendre", compte_rendu="PROBLÈMES :\n- « Réponse A » : incomplète (preuve : [r1])")])
    o = Orchestrator("La meilleure option ?", tools(tmp_path), s, verify="toujours")
    rec = o.run()
    assert rec["reponse"].startswith("Réponse A.") and "Remarques du vérificateur, non corrigées" in rec["reponse"]
    assert len(o.orch_steps) == 1 + 4                                            # refus + 4 tours, puis fin
    txt = "La 2.12 apporte torch.accelerator.Graph. Elle accélère linalg.eigh. " * 30
    c = cut_clean(txt, 100)
    assert len(c) <= 104 and c.endswith("[…]") and "torch.ac " not in c and c[:-4].rstrip().endswith(".")


def test_ninth_run_verifier_facts_and_reserved_name(tmp_path):
    """
    Run du 08/10 (68,7 min) : l'orchestrateur, persuadé que 2.13.0 était la plus récente, a demandé « la version
    avant 2.12.1 » à un agent qu'il avait appelé « verificateur ». Le contrôle final a réutilisé cet agent ; il a
    quand même trouvé 2.14.0 (2026-09-02) dans la page, mais sans citation au bon format : remarque ignorée,
    livrable faux accepté. Maintenant : nom réservé, comptes rendus exclus des sources, et une remarque qui
    apporte des faits lus dans les sources et absents du livrable est gardée.
    """
    from moelab.agent import Tool
    from moelab.app.agents import check_problems, found_facts

    out = tmp_path / "pytorch.md"
    page = "PyTorch 2.14.0 Release released this 02 Sep 2026. PyTorch 2.13.0 Release released 08 Jul 2026."

    def lire_page(url: str, chercher: str = "", debut: int = 0) -> str:
        return page

    work = [t for t in tools(tmp_path) if t.name != "lire_page"]
    work.append(Tool("lire_page", "lit", {"type": "object", "properties": {"url": {"type": "string"}},
                                           "required": ["url"]}, lire_page))
    free_form = ("La troisième version stable la plus récente est PyTorch 2.14.0 (v2.14.0), sortie le 2026-09-02 "
                 "(source : r1). La version 2.15.0 n'existe pas encore.")
    s = Script(
        orchestrateur=[call("creer_agent", nom="verificateur", role="vérifie", outils=["lire_page"]),
                       call("creer_agent", nom="chercheur", role="cherche", outils=["lire_page", "ecrire_fichier"]),
                       call("confier", agent="chercheur", tache="Les versions avant 2.12.1, écris le fichier"),
                       call("terminer", reponse="Fichier écrit : 2.13.0 et 2.12.1."),
                       call("terminer", reponse="Fichier corrigé : 2.14.0 (2 septembre 2026), 2.13.0, 2.12.1.")],
        chercheur=[call("lire_page", url="https://github.com/pytorch/pytorch/releases"),
                   call("ecrire_fichier", chemin=str(out), contenu="## 2.13.0\n- 08 Jul 2026\n## 2.12.1"),
                   call("rendre", compte_rendu="2.13.0 (08 Jul 2026) et 2.12.1. Peut-être 2.11.5 aussi.")],
        verificateur=[call("rendre", compte_rendu=free_form)])
    o = Orchestrator("Les trois dernières versions de PyTorch dans pytorch.md", work, s, verify="toujours",
                     today="jeudi 8 octobre 2026")
    rec = o.run()
    first = o.orch_steps[0]["appels"][0]["resultat"]
    assert "réservé au contrôle final" in first and o.created == 1           # le nom « verificateur » est refusé
    assert rec["agents"]["verificateur"]["origine"] == "systeme"
    assert o.source_refs == {"r1"} and "r2" in o.archive                    # r2 : compte rendu, pas une source
    refusal = o.orch_steps[3]["appels"][0]["resultat"]
    assert refusal.startswith("Réponse NON acceptée") and "2.14.0" in refusal and "2026-09-02" in refusal
    assert "2.15.0" not in refusal.split("absent du livrable :")[1].split(")")[0]  # pas lu dans une source
    assert rec["reponse"].startswith("Fichier corrigé : 2.14.0")
    # le compte rendu de l'agent n'est pas une source : « 2.11.5 » n'y devient pas un fait vérifié
    sources = {r: o.archive[r] for r in o.source_refs}
    assert found_facts("Il manque 2.11.5.", "rien", sources) == []
    assert found_facts("Il manque 2.14.0, sortie le 02 Sep.", "## 2.14\n- 2 septembre", sources) == []  # déjà là
    kept, rejected = check_problems("PROBLÈMES :\n- tout va mal", "x", sources)
    assert kept == [] and rejected == ["tout va mal"]


def test_tenth_run_task_kept_tools_granted_orchestrator_writes(tmp_path):
    """
    Run du 08/10 (67,9 min, bon résultat) : 1) au 2e nettoyage, la tâche en cours du chercheur a été effacée et
    il est parti lire 2.12, 2.11, 2.10 (17 min) ; 2) le rédacteur n'avait pas lire_fichier pour modifier son
    fichier ; 3) un rédacteur créé seulement pour recopier un contenu déjà prêt.
    """
    from moelab.agent import clean_context, compact_messages

    task = {"role": "user", "content": "Nouvelle tâche : la 2e nouveauté de 2.13.0 " + "z" * 900, "_garder": True}
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "Tâche précédente : ..."},
            {"role": "assistant", "content": "(tâche terminée) vu"}, task]
    msgs += [{"role": "tool", "content": f"[r{i}] " + "page " * 300} for i in range(6)]
    count = lambda m: sum(len(x["content"]) for x in m) // 4  # noqa: E731
    assert clean_context(msgs, count, trigger=10, target=1) > 0
    assert task in msgs and msgs[msgs.index(task)]["content"] == task["content"]     # jamais réduite ni retirée
    assert compact_messages(msgs, count, budget=1) >= 0 and task in msgs

    out = tmp_path / "pytorch.md"
    out.write_text("## 2.13.0\n- une nouveauté\n", encoding="utf-8")
    events = []
    s = Script(
        orchestrateur=[call("creer_agent", nom="redacteur", role="écrit", outils=["ecrire_fichier"]),
                       call("confier", agent="redacteur", tache=f"Ajoute une 2e nouveauté dans {out}"),
                       call("ecrire_fichier", chemin=str(tmp_path / "direct.md"), contenu="Écrit par l'orchestrateur."),
                       call("terminer", reponse="Fichiers écrits.")],
        redacteur=[call("lire_fichier", chemin=str(out)),
                   call("ecrire_fichier", chemin=str(out), contenu="## 2.13.0\n- une nouveauté\n- une autre\n"),
                   call("rendre", compte_rendu="Fichier mis à jour.")])
    o = Orchestrator("Mets à jour pytorch.md", tools(tmp_path), s, emit=lambda t, **d: events.append((t, d)))
    rec = o.run()
    steps = rec["agents"]["redacteur"]["taches"][0]["etapes"]
    assert "une nouveauté" in steps[0]["appels"][0]["resultat"]                  # lire_fichier accordé
    assert rec["agents"]["redacteur"]["outils"][-1] == "lire_fichier"
    assert any(t == "log" and "lire_fichier ajouté" in d["texte"] for t, d in events)
    assert "une autre" in out.read_text(encoding="utf-8")
    assert (tmp_path / "direct.md").read_text(encoding="utf-8") == "Écrit par l'orchestrateur."
    assert "ecrire_fichier" in o.system_prompt() and o.created == 1
    first = s.seen["redacteur"][0][1]
    assert first.get("_garder") and "toujours permis" in first["content"]


def test_eleventh_run_no_invented_criteria_absence_claims_orchestrator_reads(tmp_path):
    """
    Run du 13/10 (62,5 min, « Échec » alors que le fichier était juste) : les critères avaient inventé « GitHub
    Releases seulement, pas de blogs » ; le vérificateur a affirmé que 2026-07-08 et 2026-05-13 n'étaient dans
    aucune source (elles y étaient) ; l'orchestrateur a essayé lire_page lui-même (refusé), puis abandonné.
    """
    from moelab.app.agents import CRIT_SYSTEM, check_problems

    assert "n'ajoute aucune exigence" in CRIT_SYSTEM and "également valables" in CRIT_SYSTEM
    sources = {"r1": "PyTorch 2.14.0 was released on September 2, 2026. Blog posted September 17, 2026.",
               "r2": "PyTorch 2.13.0 Release ... released this 08 Jul 17:39",
               "r3": "PyTorch 2.12 Release Blog ... May 13, 2026"}
    livrable = "## 2.14.0\n- 2026-09-17\n## 2.13.0\n- 2026-07-08\n## 2.12.0\n- 2026-05-13"
    report = ("PROBLÈMES :\n"
              "« 2026-09-17 » n'apparaît pas dans la page : la date de sortie est « September 2, 2026 » (preuve : [r1])\n"
              "« 2026-07-08 » n'apparaît pas dans la page : date non trouvée (preuve : [r2] ne contient pas de date)\n"
              "« 2026-05-13 » n'apparaît pas dans la page : date non trouvée (preuve : [r3])\n"
              "« 2.13.0 » : la nouveauté citée n'est pas la bonne (preuve : [r2])")
    kept, rejected = check_problems(report, livrable, sources)
    assert len(kept) == 2 and "September 2" in kept[0] and kept[1].startswith("« 2.13.0 »")   # vrai problème gardé
    assert len(rejected) == 2 and "2026-07-08" in rejected[0] and "2026-05-13" in rejected[1]  # fausse absence

    from moelab.agent import Tool

    def lire_page(url: str, chercher: str = "", debut: int = 0) -> str:
        return "PyTorch 2.14.0 released this 02 Sep"

    work = [t for t in tools(tmp_path) if t.name != "lire_page"]
    work.append(Tool("lire_page", "lit", {"type": "object", "properties": {"url": {"type": "string"}},
                                           "required": ["url"]}, lire_page))
    s = Script(orchestrateur=[call("lire_page", url="https://github.com/pytorch/pytorch/releases"),
                              call("ecrire_fichier", chemin=str(tmp_path / "p.md"), contenu="2.14.0 : 02 Sep"),
                              call("terminer", reponse="Fichier écrit.")])
    o = Orchestrator("La dernière version de PyTorch dans p.md", work, s)
    rec = o.run()
    assert "02 Sep" in o.orch_steps[0]["appels"][0]["resultat"] and o.created == 0
    assert (tmp_path / "p.md").read_text(encoding="utf-8") == "2.14.0 : 02 Sep" and rec["reponse"] == "Fichier écrit."
    assert "lire_page" in o.system_prompt().split("regarder toi-même :")[1].split(".")[0]
