"""
Agents (06/10) : un orchestrateur qui crée lui-même ses sous-agents et leur confie le travail.

Un seul modèle tient en mémoire : les agents ne tournent pas en parallèle mais l'un après
l'autre, sur le même modèle. Chacun a son rôle (message système), ses outils (une partie de
ceux que l'utilisateur permet) et sa conversation, gardée d'une tâche à l'autre pendant le run.

    orchestrateur : creer_agent, confier, lister_agents, noter, lire_notes, terminer
    sous-agent    : ses outils + noter, lire_notes, rendre(compte_rendu)

Le code garantit, le modèle propose (comme l'autopilot) :
  - un seul niveau : un sous-agent ne crée pas d'agents ;
  - limites fixées par le code : nombre d'agents, tours de l'orchestrateur, tours par tâche,
    minutes ; budget épuisé -> les outils répondent « appelle terminer » ;
  - relances (3 au plus) si l'orchestrateur s'arrête sans appeler d'outil ;
  - un outil en « demander » attend l'accord de l'utilisateur, dans n'importe quel agent.

Moteur : la conversation de chaque agent (K/V, états) est mise de côté sur CPU quand un autre
agent parle, puis remise : l'orchestrateur reprend après un compte rendu sans recalculer son
historique, un sous-agent qu'on relance reprend la sienne. Entre deux tours, les messages de
chat en attente passent (Engine.serve_waiting).

Bibliothèque : <data>/agents.json. Un agent créé pendant un run peut y être gardé ;
l'orchestrateur voit les agents enregistrés et peut leur confier une tâche sans les recréer.
Chaque run est enregistré dans <data>/agent_runs/<id>.json (l'arbre complet).
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..agent import DATE_RULE, Agent, Tool, today_fr, tool
from .engine import Cancelled, TextSink, read_json, write_json
from .factcheck import FactCheck, items as fact_items, keys as fact_keys

ORCH_TOOLS = ("creer_agent", "confier", "lister_agents", "noter", "lire_notes", "terminer")
SUB_EXTRA = ("noter", "lire_notes", "rendre", "relire")
# Lecture seule, pour que l'orchestrateur prépare des tâches précises (06/10 : sans eux, il essayait
# de lister le dossier lui-même, échouait, puis confiait une tâche à l'aveugle)
NAV_TOOLS = ("lister_dossier", "chercher_fichiers")
SCOUT_TOOLS = NAV_TOOLS + ("lire_fichier",)  # 6e run : l'orchestrateur relit lui-même les fichiers écrits
# Un agent qui lit ou écrit des fichiers reçoit toujours de quoi naviguer
FILE_TOOLS = ("lire_fichier", "ecrire_fichier", "ajouter_fichier")
WRITE_TOOLS = ("ecrire_fichier", "ajouter_fichier")
# Outils sans effet de bord, donnés à tout sous-agent qui les appelle (08/10, 10e run : le rédacteur n'avait pas
# lire_fichier pour modifier son propre fichier ; un aller-retour de 3 générations perdu). Les permissions de
# l'écran Outils (toujours / demander) s'appliquent toujours.
READ_ONLY = ("lire_fichier", "lister_dossier", "chercher_fichiers", "lire_page", "chercher_web", "calculer", "etat",
             "reglages")
# Résultats d'outils qui ne sont pas des sources : ce que le modèle produit lui-même
NOT_SOURCES = ORCH_TOOLS + SUB_EXTRA + WRITE_TOOLS
MAX_SEEN = 4000  # ce que l'agent voit d'un résultat (Agent.max_result_chars)
NOTES_LIMIT = 3000      # bloc-notes joint à une tâche ou rendu par lire_notes (les plus récentes d'abord)
REPORT_IN_NOTES = 900   # part d'un compte rendu recopiée dans le bloc-notes
ARCHIVE_MIN = 1         # tout résultat est archivé (r1, r2...) : relire le retrouve après un nettoyage, et le
                        # vérificateur voit tout ce qui a été lu (08/10 : un rapport court lui échappait)
ARCHIVE_MAX = 200_000   # taille gardée par résultat archivé
# Grand nettoyage du contexte (08/10, d'après DeepSeek-V3.2 : à 80 % de la fenêtre, effacer l'historique des
# outils a mieux marché qu'un résumé) : une seule grosse coupe, puis la conversation ne fait que s'allonger
# et le cache (P1) reprend tout. Raccourcir un peu à chaque tour obligeait à tout recalculer à chaque tour.
CLEAN_TRIGGER = 0.80    # part de la fenêtre utile qui déclenche le nettoyage
CLEAN_TARGET = 0.45     # part visée après le nettoyage
AFTER_VERIFY_STEPS = 4  # tours laissés à l'orchestrateur pour corriger après un refus du vérificateur
# Un outil qui n'a pas de sens sans un autre (06/10, 7e run : chercher_web sans lire_page = extraits seulement)
COMPANIONS = {"chercher_web": ("lire_page",), **{t: ("lister_dossier", "chercher_fichiers") for t in
                                                 ("lire_fichier", "ecrire_fichier", "ajouter_fichier")}}

ORCH_SYSTEM = """Tu es l'orchestrateur de moelab, un assistant qui tourne en local sur ce PC.
{date_rule}
Tu ne fais pas le travail toi-même : tu découpes l'objectif de l'utilisateur, tu crées des agents spécialisés, tu leur confies des tâches précises, puis tu assembles leurs comptes rendus.

Règles :
- Chaque message se termine par au moins un appel d'outil, au format <tool_call>{{"name": ..., "arguments": {{...}}}}</tool_call>. Réfléchis en une ou deux phrases, puis appelle l'outil.
- Pour préparer des tâches précises ou vérifier un point (une date, un lien, un fichier), tu peux regarder toi-même : {scout}. Les recherches longues sont faites par les agents.
- Quand tu as déjà tout le contenu d'un fichier (dans les comptes rendus), écris-le toi-même avec {ecriture} : ne crée pas un agent juste pour recopier. Pour modifier un fichier, réécris-le en entier.
- Ne recopie pas dans ton texte ce que tu mets dans un appel (contenu, tâche) : écris-le une seule fois, dans l'appel.
- creer_agent : un nom court (ex. chercheur, analyste, redacteur), un rôle précis (ce qu'il sait faire, comment travailler, le format de son compte rendu) et les outils dont il a besoin, choisis parmi : {outils}.
- confier : une tâche précise et autonome. L'agent ne voit ni l'objectif ni les autres agents : seulement ta tâche et le bloc-notes. Donne-lui chemins, adresses et contraintes.
- Le bloc-notes (noter, lire_notes) est partagé avec tous les agents : notes-y les faits utiles à la suite.
- Vérifie ce que rendent les agents ; confie une correction si un compte rendu est incomplet. Redonne alors la demande d'origine et ce qui manque, sans y mettre ta propre hypothèse (« la version avant X », « dans telle période ») : l'agent doit pouvoir constater que la réalité est différente.
- Les comptes rendus vont automatiquement dans le bloc-notes : un agent rédacteur y trouve ce qu'ont rendu les autres.
- Si un agent a écrit un fichier, relis-le toi-même (lire_fichier) et compare-le aux comptes rendus avant de terminer. Ne demande jamais à un agent de vérifier son propre travail.
- Le code vérifie les dates et les chiffres contre ce qui a été réellement lu pendant le run ; un élément « non retrouvé » doit être vérifié à la source ou remplacé par « non trouvé ».
- Les longs résultats sont archivés sous une référence [r1], [r2]… : quand ta conversation est nettoyée, relire(ref, chercher=…) les retrouve.
- Une date, un chiffre ou un nom de ta réponse finale doit venir d'un compte rendu qui cite sa source (page lue, fichier lu). S'il manque ou paraît douteux (par exemple une date incohérente avec aujourd'hui), confie une vérification ; sinon écris « non trouvé ». Ne complète jamais de mémoire.
- Quand l'objectif est atteint : terminer(reponse) avec la réponse complète pour l'utilisateur, en français.

Dossiers que les outils de fichiers peuvent lire et écrire : {dossiers}

Limites : {max_agents} agents créés, {max_steps} tours pour toi, {minutes} minutes au total.
Agents enregistrés, utilisables directement avec confier (sans les créer) : {bibliotheque}"""

# Message système COMMUN à tous les sous-agents (08/10) : même début de prompt, donc un seul cache de
# préfixe (P9) pour tous ; le nom, le rôle et les outils permis de chaque agent sont dans son 1er message.
SUB_SYSTEM = """Tu es un agent de moelab, un assistant qui tourne en local sur ce PC. Ton nom, ton rôle et les outils que tu as le droit d'utiliser sont dans le premier message de l'orchestrateur ; les autres outils te seront refusés.
{date_rule}

Dossiers que les outils de fichiers peuvent lire et écrire : {dossiers}

Tu reçois des tâches de l'orchestrateur. Travaille avec tes outils : réfléchis en une ou deux phrases (ne recopie jamais dans ton texte le contenu que tu vas écrire : mets-le directement dans l'appel), puis appelle les outils, chacun dans son bloc <tool_call>{{"name": ..., "arguments": {{...}}}}</tool_call>. Quand des appels ne dépendent pas l'un de l'autre (lire trois pages, deux fichiers), mets-les dans le même message : ça économise un tour. Quand la tâche est faite, appelle rendre(compte_rendu) : ce que tu as trouvé ou fait, les chiffres, les fichiers créés, ce qui reste incertain. Concis mais complet. Réponds en français.

Les longs résultats sont archivés sous une référence [r1], [r2]… : relire(ref, chercher=…) les retrouve, même après un nettoyage de ta conversation. Pour une date ou un chiffre dans une longue page ou un long fichier : lire_page ou lire_fichier avec l'argument chercher.

Les dates et chiffres que tu notes, rends ou écris sont vérifiés automatiquement contre ce que tu as lu pendant ce run.

Si ce que tu lis contredit ce que suppose ta tâche (un élément plus récent, plus ancien ou différent de ce qu'on t'a dit), écris-le en tête de ton compte rendu au lieu de l'ignorer : l'orchestrateur peut se tromper.

Règle des sources : chaque date, chiffre, nom ou citation de ton compte rendu (et de ce que tu notes ou écris) vient d'une page ou d'un fichier que tu as lu pendant ce run, avec sa source (adresse ou chemin). Si tu ne l'as pas trouvé, écris « non trouvé » : n'invente jamais et ne complète pas de mémoire."""

AGENT_INTRO = """Tu es « {nom} ». Ton rôle : {role}
Tes outils : {outils} (plus noter, lire_notes, relire et rendre ; les outils de lecture {lecture} sont toujours permis). Tout autre outil te sera refusé.

"""

# -- Critères, vérificateur final, leçons (08/10) -----------------------------------------------------------
CRIT_SYSTEM = """Tu prépares le travail d'un orchestrateur d'agents. {date_rule}
Transforme l'objectif de l'utilisateur en 2 à 5 critères de réussite, vérifiables un par un à la fin : ce qu'il faut livrer, où, et comment interpréter les mots qui demandent un choix (« dernier », « meilleur », « principal »…) par rapport à la date du jour. Reformule seulement ce que l'utilisateur demande : n'ajoute aucune exigence qu'il n'a pas exprimée (sources imposées ou interdites, format précis, vérifications, cas d'échec). Le site, le blog et le dépôt officiels d'un projet sont des sources également valables. Tiens compte des leçons données. Réponds uniquement par la liste, une ligne par critère, chaque ligne commençant par « - »."""

VERIF_NAME = "verificateur"  # réservé au contrôle final (08/10)

VERIF_ROLE = ("Vérificateur : contrôle une réponse proposée contre l'objectif et ses critères, avec les sources "
              "déjà lues. Ne refait pas le travail ; cherche ce qui manque, ce qui est faux ou mal compris.")

VERIF_TASK = """Contrôle ce livrable avant qu'il soit donné à l'utilisateur.

Objectif de l'utilisateur : {objectif}

Critères de réussite :
{criteres}
{lecons}
=== LIVRABLE (ce que l'utilisateur va recevoir) ===
{livrable}
=== FIN DU LIVRABLE ===

Sources lues pendant le run (relire(ref, chercher=mot-clé) pour vérifier) :
{archive}

Vérifie chaque critère et chaque affirmation du livrable (date, chiffre, nom, nouveauté) en la retrouvant dans une source avec relire, pas avec ta mémoire. Sois bref. Puis appelle rendre :
- « OK » si tout est bon ;
- sinon « PROBLÈMES : » puis une ligne par problème, sous l'une de ces deux formes exactement :
  - « passage exact copié du livrable » : ce qui ne va pas (preuve : [rN])
  - MANQUE : ce qui manque (preuve : [rN] « passage exact copié de la source »)
Un problème sans citation exacte sera ignoré."""

VERIFY_WORDS = re.compile(r"derni|r[ée]cent|nouveau|nouvelle|meilleur|\bplus\b|\btop\b|classement|compar|version|"
                          r"prix|date|combien|liste|principa", re.I)

LAST_TURN = ("Limite de tours atteinte pour cette tâche. Appelle rendre MAINTENANT avec ce que tu as trouvé "
             "(y compris tes notes), en disant ce qui manque. Aucun autre outil.")

NUDGE = ("Tu t'es arrêté sans appeler d'outil. Continue avec un appel au format exact :\n"
         '<tool_call>\n{"name": "terminer", "arguments": {"reponse": "..."}}\n</tool_call>\n'
         "(ou confier / creer_agent si le travail n'est pas fini).")


def cut_clean(text: str, n: int) -> str:
    """Coupe `text` à `n` caractères au plus, en fin de phrase ou de ligne si possible, jamais au milieu d'un mot."""
    text = (text or "").strip()
    if len(text) <= n:
        return text
    head = text[:n]
    cut = max(head.rfind(". "), head.rfind("\n"), head.rfind(" ; "))
    if cut < n // 2:
        cut = head.rfind(" ")
    return head[: cut + 1 if cut > 0 else n].rstrip() + " […]"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[«»“”\"'’`*_]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_QUOTE = re.compile(r"[«“\"]\s*(.+?)\s*[»”\"]")


_VERSION3 = re.compile(r"(?<![\w.])v?(\d+\.\d+\.\d+(?:\.\d+)?)(?![\w]|\.\d)")


def found_facts(text: str, deliverable: str, sources: dict, ignore: str = "") -> list[str]:
    """
    Faits d'une remarque (versions x.y.z, dates, chiffres) qui sont dans une source lue pendant le run mais pas
    dans le livrable (08/10, run de 68 min : le vérificateur avait trouvé la version 2.14.0 et sa date, mais sans
    citation au bon format ; sa remarque a été ignorée et le livrable faux accepté).
    """
    src = "\n".join(sources.values())
    src_n, deliv_n = _norm(src), _norm(deliverable)
    known, have, skip = FactCheck(src).known, fact_keys(deliverable), fact_keys(ignore)
    out = []
    for m in _VERSION3.finditer(text or ""):
        v = m.group(1)
        short = re.sub(r"\.0$", "", v)
        if v in src_n and v not in deliv_n and not re.search(rf"(?<![\w.]){re.escape(short)}(?![\w]|\.\d)", deliv_n):
            out.append(v)
    spans = [m.span() for m in _VERSION3.finditer(text or "")]
    for k, raw in fact_items(text or ""):
        if k in known and k not in have and k not in skip and raw not in out:
            pos = (text or "").find(raw)
            if not any(a <= pos < b for a, b in spans):  # « 2.14 » dans « 2.14.0 » : déjà compté
                out.append(raw)
    return list(dict.fromkeys(out))


_ABSENT = re.compile(r"n'appara[iî]t pas|non trouv|introuvable|absent|ne contient pas|aucune (?:source|preuve|date)|"
                     r"pas de date|non v[ée]rifi", re.I)


def _facts_in_sources(quote: str, known: set) -> bool:
    """Le passage cité avance des faits (dates, chiffres) et tous sont dans les sources lues."""
    facts = [k for k, _ in fact_items(quote)]
    return bool(facts) and all(k in known for k in facts)


def check_problems(report: str, deliverable: str, archive: dict, ignore: str = "") -> tuple[list[str], list[str]]:
    """
    Problèmes du vérificateur -> (gardés, rejetés). Une ligne est gardée si sa citation existe vraiment :
      « passage du livrable » : ...       -> le passage est dans le livrable
      MANQUE : ... [rN] « passage »       -> le passage est dans la source rN
    ou, sans citation valable, si elle avance des faits (version, date, chiffre) présents dans une source et
    absents du livrable. Le reste est rejeté : c'est le code qui contrôle le contrôleur.
    `archive` : les SOURCES seulement (résultats d'outils), jamais les comptes rendus des agents.
    """
    text = report or ""
    if text.strip().upper().startswith("OK") and "PROBL" not in text.upper():
        return [], []
    body = text.split(":", 1)[1] if "PROBL" in text.upper()[:30] else text
    kept, rejected = [], []
    cache: dict = {}

    def known():
        if "k" not in cache:
            cache["k"] = FactCheck("\n".join(archive.values())).known
        return cache["k"]

    deliv = _norm(deliverable)
    for line in body.splitlines():
        line = line.strip().lstrip("-•*").strip()
        if len(line) < 6:
            continue
        quotes = [q for q in _QUOTE.findall(line) if len(_norm(q)) >= 3]
        if line.upper().startswith("MANQUE"):
            refs = re.findall(r"\[(r\d+)\]", line)
            ok = any(_norm(q) in _norm(archive.get(r, "")) for r in refs for q in quotes)
        else:
            ok = bool(quotes) and _norm(quotes[0]) in deliv
            if ok and _ABSENT.search(line) and _facts_in_sources(quotes[0], known()) \
                    and not found_facts(line, deliverable, archive, ignore):
                ok = False  # 13/10 : « 2026-07-08 n'apparaît pas » alors que les pages lues le donnent
        if not ok:
            facts = found_facts(line, deliverable, archive, ignore)
            if facts:
                line = f"MANQUE (relevé dans les sources, absent du livrable : {', '.join(facts)}) : {cut_clean(line, 500)}"
                ok = True
        (kept if ok else rejected).append(line)
    return kept, rejected


def parse_criteria(text: str, limit: int = 8) -> list[str]:
    """Lignes « - critère » (ou numérotées) -> liste ; le reste est ignoré."""
    out = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)", line)
        if m:
            out.append(m.group(1).strip())
    return out[:limit]


def criteria_from_text(text) -> list[str]:
    """Critères corrigés par l'utilisateur : une ligne par critère, avec ou sans tiret ou numéro."""
    if isinstance(text, list):
        text = "\n".join(map(str, text))
    return [c for c in (re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", x).strip() for x in str(text or "").splitlines()) if c][:8]


def slug(name: str) -> str:
    """Nom d'agent : minuscules sans accents, chiffres, _ et - (« Rédacteur Web » -> redacteur_web)."""
    name = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9_-]+", "_", (name or "").strip().lower().replace(" ", "_"))
    return s.strip("_")[:32]


# ----------------------------------------------------------------------------
# Bibliothèque d'agents
# ----------------------------------------------------------------------------
@dataclass
class AgentSpec:
    nom: str
    role: str
    outils: list = field(default_factory=list)
    origine: str = "orchestrateur"      # orchestrateur | utilisateur
    cree: float = field(default_factory=time.time)
    runs: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class AgentLibrary:
    def __init__(self, data_dir):
        self.path = Path(data_dir) / "agents.json"
        self.data = read_json(self.path, {"agents": {}})
        self.data.setdefault("agents", {})

    def save_file(self) -> None:
        write_json(self.path, self.data)

    def list(self) -> list[dict]:
        return sorted(self.data["agents"].values(), key=lambda a: a["nom"])

    def get(self, nom: str) -> AgentSpec | None:
        d = self.data["agents"].get(slug(nom))
        return AgentSpec(**d) if d else None

    def save(self, spec: AgentSpec) -> AgentSpec:
        spec.nom = slug(spec.nom)
        if not spec.nom:
            raise ValueError("nom vide")
        if not spec.role.strip():
            raise ValueError("rôle vide")
        old = self.data["agents"].get(spec.nom)
        if old:
            spec.runs = max(spec.runs, old.get("runs", 0))
        self.data["agents"][spec.nom] = spec.as_dict()
        self.save_file()
        return spec

    def remove(self, nom: str) -> bool:
        out = self.data["agents"].pop(slug(nom), None) is not None
        self.save_file()
        return out

    def count_use(self, nom: str) -> None:
        d = self.data["agents"].get(slug(nom))
        if d:
            d["runs"] = d.get("runs", 0) + 1
            self.save_file()


# ----------------------------------------------------------------------------
# L'orchestrateur (indépendant du moteur : testable avec une fausse génération)
# ----------------------------------------------------------------------------
class Orchestrator:
    """
    objective     : l'objectif de l'utilisateur
    work_tools    : outils que les sous-agents peuvent recevoir (déjà filtrés par permission)
    make_gen      : make_gen(nom_agent) -> generate_fn(messages, schemas) -> texte brut
    save_state    : () -> état de décodage (ou None) ; restore_state(état)
    between       : appelé entre deux tours de l'orchestrateur (laisser passer le chat)
    emit          : emit(type, **données) : événements pour l'interface
    """

    def __init__(self, objective: str, work_tools: list[Tool], make_gen, library: AgentLibrary | None = None,
                 emit=None, save_state=None, restore_state=None, between=None, max_agents: int = 5,
                 max_steps: int = 14, sub_steps: int = 10, minutes: float = 60, nudges: int = 3,
                 cancelled=None, folders: list | None = None, today: str | None = None,
                 lessons: list | None = None, criteria: bool = False, verify: str = "jamais", ask_criteria=None):
        self.objective = objective.strip()
        self.work_tools = {t.name: t for t in work_tools if t.name not in ORCH_TOOLS + SUB_EXTRA}
        self.make_gen = make_gen
        self.library = library
        self.emit = emit or (lambda *a, **k: None)
        self.save_state = save_state or (lambda: None)
        self.restore_state = restore_state or (lambda s: None)
        self.between = between or (lambda: None)
        self.max_agents, self.max_steps, self.sub_steps = max_agents, max_steps, sub_steps
        self.minutes, self.nudges = minutes, nudges
        self.cancelled = cancelled or (lambda: False)
        self.folders = [str(f) for f in (folders or []) if f]
        self.date_rule = DATE_RULE.format(date=today or today_fr())
        self.fc = FactCheck(self.objective, self.date_rule)   # ce qui a été réellement lu pendant le run
        self.written: set[str] = set()                        # fichiers écrits par les agents (pas des sources)
        self.checks: list[dict] = []                          # éléments non retrouvés, pour le rapport
        self.t0 = time.time()
        self.agents: dict[str, dict] = {}   # nom -> {spec, messages, state, taches}
        self.created = 0
        self.notes: list[dict] = []
        self.final: str | None = None
        self._final_bounced = False
        self.orch_steps: list[dict] = []
        self.gens: dict = {}
        self.archive: dict[str, str] = {}   # r1, r2... -> résultat complet (relire)
        self.source_refs: set[str] = set()  # celles qui viennent d'un outil (pas les comptes rendus des agents)
        self.lessons = [x["texte"] if isinstance(x, dict) else str(x) for x in (lessons or [])]
        self.use_criteria, self.verify_mode, self.ask_criteria = criteria, verify, ask_criteria
        self.criteria: list[str] = []
        self.verification: dict | None = None   # {"ok", "problemes", "rejetes", "s"}
        self.written_paths: dict[str, str] = {}  # clé normalisée -> chemin tel qu'écrit
        self.last_proposed: str | None = None    # dernière réponse proposée à terminer
        self.after_verify: int | None = None     # tours de l'orchestrateur depuis un refus du vérificateur
        self.timing = {"criteres_s": 0.0, "verification_s": 0.0}
        self.ctx = {"nettoyages": 0, "replis": 0, "archives": 0}

    # -- budget -------------------------------------------------------------
    def elapsed_min(self) -> float:
        return (time.time() - self.t0) / 60

    def out_of_time(self) -> bool:
        return self.elapsed_min() >= self.minutes

    def _check(self):
        if self.cancelled():
            raise Cancelled()

    # -- vérification des faits (06/10, 6e run) -------------------------------------
    @staticmethod
    def _norm_path(p) -> str:
        return os.path.normcase(os.path.normpath(str(p or "")))

    def _observe(self, name: str, args: dict, result) -> None:
        """Le résultat d'un outil de travail devient une source (sauf un fichier écrit pendant le run)."""
        if name in NOT_SOURCES:
            return
        if name == "lire_fichier" and self._norm_path((args or {}).get("chemin")) in self.written:
            return
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        self.fc.add(text[:MAX_SEEN])

    def _source_tool(self, t: Tool) -> Tool:
        orch = self

        def fn(**kw):
            out = t.fn(**kw)
            orch._observe(t.name, kw, out)
            return orch._archived(out)

        return Tool(t.name, t.description, t.parameters, fn)

    def _archived(self, out):
        """Un long résultat reçoit une référence [rN] et reste disponible pour relire, quoi qu'il arrive au contexte."""
        text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
        if len(text) < ARCHIVE_MIN:
            return out
        ref = f"r{len(self.archive) + 1}"
        self.archive[ref] = text[:ARCHIVE_MAX]
        self.source_refs.add(ref)
        self.ctx["archives"] += 1
        if len(text) <= MAX_SEEN - 120:
            return f"[{ref}] {text}"
        return (f"[{ref}] {text[:MAX_SEEN - 220]}\n[... {len(text)} caractères au total : suite avec "
                f"relire('{ref}', debut={MAX_SEEN - 220}), ou relire('{ref}', chercher='mot')]")

    def _relire_tool(self) -> Tool:
        orch = self

        @tool
        def relire(ref: str, chercher: str = "", debut: int = 0) -> str:
            """Relit un résultat archivé [r1], [r2]… (page, fichier, recherche), en entier par morceaux ou seulement les passages autour d'un mot.

            Args:
                ref: la référence, ex. r3
                chercher: mot ou expression à trouver ; vide = lire dans l'ordre
                debut: position de départ en caractères, pour lire la suite
            """
            from .tools import find_passages

            text = orch.archive.get(str(ref).strip().strip("[]"))
            if text is None:
                return f"Référence inconnue « {ref} ». Références : {', '.join(orch.archive) or 'aucune'}."
            if chercher.strip():
                return find_passages(text, chercher.strip(), limit=MAX_SEEN - 200)
            debut = max(0, int(debut))
            part = text[debut: debut + MAX_SEEN - 200]
            if debut + MAX_SEEN - 200 < len(text):
                part += f"\n[... suite avec relire('{ref}', debut={debut + MAX_SEEN - 200})]"
            return part or "(fin du résultat)"

        return relire

    def _flag(self, where: str, who: str, missing: list[str]) -> str:
        self.checks.append({"ou": where, "agent": who, "elements": missing, "t": round(self.elapsed_min(), 2)})
        self.emit("verification", agent=who, ou=where, elements=missing)
        return ("Vérification automatique : non retrouvé dans les pages et fichiers lus pendant ce run : "
                + ", ".join(missing))

    def _write_tool(self, t: Tool, who: str) -> Tool:
        """Écriture de fichier : refusée une fois si le contenu avance des dates ou chiffres jamais lus."""
        orch, refused = self, set()

        def fn(**kw):
            path, content = kw.get("chemin"), kw.get("contenu", "")
            missing = orch.fc.missing(content)
            key = orch._norm_path(path)
            if missing and key not in refused:
                refused.add(key)
                orch._flag(f"fichier {path} (refusé une fois)", who, missing)
                return ("NON ÉCRIT. Le contenu avance des éléments qui ne viennent d'aucune page ni d'aucun fichier lus "
                        f"pendant ce run : {', '.join(missing)}. Utilise seulement ta tâche, le bloc-notes et ce que tu as "
                        "lu ; remplace ce qui manque par « non trouvé », puis réécris.")
            out = t.fn(**kw)
            orch.written.add(key)
            orch.written_paths[key] = str(path)
            if missing:
                out = f"{out}\n{orch._flag(f'fichier {path}', who, missing)}"
            return out

        return Tool(t.name, t.description, t.parameters, fn)

    # -- outils partagés ---------------------------------------------------------
    def _note_tools(self, author: str) -> list[Tool]:
        orch = self

        @tool
        def noter(texte: str) -> str:
            """Ajoute un fait utile au bloc-notes partagé par tous les agents de ce run.

            Args:
                texte: le fait, court et précis (avec la source si besoin)
            """
            missing = orch.fc.missing(texte)
            orch.notes.append({"auteur": author, "texte": texte.strip(), "t": round(orch.elapsed_min(), 2),
                               **({"non_retrouve": missing} if missing else {})})
            orch.emit("note", agent=author, texte=texte.strip())
            if missing:
                return (f"noté ({len(orch.notes)} notes). Attention, {orch._flag('note', author, missing)}. "
                        "Vérifie-les à la source (lire_page avec chercher) et corrige la note, ou écris « non trouvé ».")
            return f"noté ({len(orch.notes)} notes)"

        @tool
        def lire_notes() -> str:
            """Relit le bloc-notes partagé (faits notés par tous les agents de ce run)."""
            return orch.notes_text(limit=NOTES_LIMIT) or "(bloc-notes vide)"

        return [noter, lire_notes]

    def notes_text(self, limit: int | None = None) -> str:
        """Bloc-notes ; avec `limit`, seulement les notes les plus récentes qui tiennent dans `limit` caractères."""
        lines = [f"- [{n['auteur']}] {n['texte']}" for n in self.notes]
        if limit is None:
            return "\n".join(lines)
        out, n = [], 0
        for line in reversed(lines):
            if n + len(line) > limit and out:
                out.append(f"[... {len(lines) - len(out)} notes plus anciennes : lire_notes]")
                break
            out.append(line)
            n += len(line)
        return "\n".join(reversed(out))

    def folders_text(self) -> str:
        if self.folders:
            return ", ".join(self.folders) + " (et leurs sous-dossiers ; rien d'autre)"
        if any(t in self.work_tools for t in FILE_TOOLS + SCOUT_TOOLS):
            return ("AUCUN. Les outils de fichiers ne peuvent rien lire ni écrire. Si l'objectif demande des fichiers, "
                    "termine tout de suite : l'utilisateur doit ajouter le dossier dans l'écran Outils et MCP "
                    "(Dossiers autorisés).")
        return "aucun"

    def tool_names(self) -> list[str]:
        return sorted(self.work_tools)

    # -- outils de l'orchestrateur ------------------------------------------------
    def orchestrator_tools(self) -> list[Tool]:
        orch = self

        @tool
        def creer_agent(nom: str, role: str, outils: list) -> str:
            """Crée un agent spécialisé (il ne travaille que quand tu lui confies une tâche).

            Args:
                nom: nom court, ex. chercheur
                role: ce qu'il sait faire, comment il travaille, le format de son compte rendu
                outils: liste des noms d'outils qu'il peut utiliser
            """
            return orch.create(nom, role, outils)

        @tool
        def confier(agent: str, tache: str) -> str:
            """Confie une tâche précise à un agent ; renvoie son compte rendu quand il a fini.

            Args:
                agent: nom de l'agent (créé ou enregistré)
                tache: la tâche, autonome : l'agent ne voit pas l'objectif global
            """
            return orch.delegate(agent, tache)

        @tool
        def lister_agents() -> list:
            """Agents de ce run et agents enregistrés : rôle, outils, tâches déjà faites."""
            out = [{"nom": n, "role": a["spec"].role, "outils": a["spec"].outils, "taches_faites": len(a["taches"])}
                   for n, a in orch.agents.items()]
            for d in (orch.library.list() if orch.library else []):
                if d["nom"] not in orch.agents:
                    out.append({"nom": d["nom"], "role": d["role"], "outils": d["outils"], "enregistre": True})
            return out

        @tool
        def terminer(reponse: str) -> str:
            """Termine le run avec la réponse complète pour l'utilisateur.

            Args:
                reponse: la réponse finale, complète, en français
            """
            orch.last_proposed = reponse.strip()
            missing = orch.fc.missing(reponse)
            if missing and not orch._final_bounced:
                orch._final_bounced = True
                orch._flag("réponse finale (refusée une fois)", "orchestrateur", missing)
                return ("Réponse NON acceptée : elle avance des éléments qui ne viennent d'aucune page ni d'aucun fichier "
                        f"lus pendant ce run : {', '.join(missing)}. Fais-les vérifier par un agent (à la source) ou "
                        "écris « non trouvé », puis rappelle terminer.")
            if orch.verification is None and orch.needs_verification(reponse):
                problems = orch.verify(reponse)
                if problems:
                    orch.after_verify = -1  # ce tour-ci (le refus) ne compte pas
                    return ("Réponse NON acceptée par le vérificateur :\n" + problems
                            + f"\nCorrige ces points seulement (tu as {AFTER_VERIFY_STEPS} tours), puis rappelle terminer "
                              "avec la réponse corrigée.")
            orch.final = reponse.strip()
            if missing:
                orch.final += "\n\n⚠ " + orch._flag("réponse finale", "orchestrateur", missing) + " (à vérifier)."
            if orch.verification and orch.verification["ok"] is False:
                orch.final += ("\n\n⚠ Le vérificateur avait signalé : " + orch.verification["problemes"][:600]
                               + " (corrigé ensuite par l'orchestrateur, non revérifié).")
            orch.emit("final", reponse=orch.final)
            return "terminé"

        scout = [self._source_tool(self.work_tools[t]) for t in self.scout_names()]
        # 08/10, 10e run : un rédacteur créé seulement pour recopier un contenu déjà prêt (6 générations, ~10 min)
        scout += [self._write_tool(self.work_tools[t], "orchestrateur") for t in WRITE_TOOLS if t in self.work_tools]
        return [*scout, creer_agent, confier, lister_agents, *self._note_tools("orchestrateur"), self._relire_tool(),
                terminer]

    def scout_names(self) -> list[str]:
        """Outils de lecture de l'orchestrateur (13/10 : à chaque run il essayait chercher_web ou lire_page,
        refusés : un tour perdu à chaque fois ; une date à vérifier demandait un agent entier)."""
        return [t for t in dict.fromkeys(SCOUT_TOOLS + READ_ONLY) if t in self.work_tools]

    def _parse_tools(self, outils) -> tuple[list[str], list[str]]:
        if isinstance(outils, str):
            outils = [x for x in re.split(r"[,;\s]+", outils) if x]
        names = [str(x).strip() for x in (outils or []) if str(x).strip()]
        ok = [n for n in names if n in self.work_tools]
        bad = [n for n in names if n not in self.work_tools]
        return list(dict.fromkeys(ok)), bad

    def _with_navigation(self, names: list[str]) -> list[str]:
        out = list(names)
        for t in names:
            for c in COMPANIONS.get(t, ()):
                if c in self.work_tools and c not in out:
                    out.append(c)
        return out

    def create(self, nom: str, role: str, outils, origine: str = "orchestrateur") -> str:
        if self.out_of_time():
            return "Budget de temps épuisé : appelle terminer maintenant avec ce que tu as."
        n = slug(nom)
        if not n:
            return "Erreur : nom vide."
        if n in ("orchestrateur", VERIF_NAME):
            return ("Erreur : ce nom est réservé" + (" au contrôle final automatique ; choisis-en un autre (ex. controleur)."
                                                     if n == VERIF_NAME else "."))
        if n in self.agents:
            return (f"L'agent « {n} » existe déjà (outils : {', '.join(self.agents[n]['spec'].outils) or 'aucun'}). "
                    "Confie-lui la tâche avec confier ; pour d'autres outils, crée un agent avec un autre nom.")
        if self.created >= self.max_agents:
            return (f"Limite atteinte ({self.max_agents} agents). Réutilise un agent existant : "
                    + ", ".join(self.agents))
        if not (role or "").strip():
            return "Erreur : donne un rôle précis."
        ok, bad = self._parse_tools(outils)
        if bad:
            return f"Erreur : outils inconnus ou non permis : {', '.join(bad)}. Outils possibles : {', '.join(self.tool_names()) or 'aucun'}."
        ok = self._with_navigation(ok)
        spec = AgentSpec(n, role.strip(), ok, origine)
        self._add(spec)
        self.created += 1
        return f"Agent « {n} » créé, outils : {', '.join(ok) or 'aucun (bloc-notes seulement)'}."

    def sub_system(self) -> str:
        return SUB_SYSTEM.format(dossiers=self.folders_text(), date_rule=self.date_rule)

    def _add(self, spec: AgentSpec) -> None:
        self.agents[spec.nom] = {"spec": spec, "messages": [{"role": "system", "content": self.sub_system()}],
                                 "state": None, "taches": []}
        self.emit("agent_created", agent=spec.nom, role=spec.role, outils=spec.outils, origine=spec.origine)

    def delegate(self, agent: str, tache: str, steps: int | None = None, notes: bool = True, system: bool = False) -> str:
        self._check()
        if self.out_of_time():
            return "Budget de temps épuisé : appelle terminer maintenant avec ce que tu as."
        n = slug(agent)
        if n == VERIF_NAME and not system:
            # 08/10 : l'orchestrateur avait créé son propre « verificateur » ; le contrôle final l'a réutilisé, avec
            # ses outils et son historique : il n'était plus indépendant
            return ("Erreur : « verificateur » est réservé au contrôle final automatique. Crée un agent avec un autre "
                    "nom (ex. controleur) pour une vérification en cours de route.")
        if n not in self.agents:
            spec = self.library.get(n) if self.library else None
            if spec is None:
                return f"Erreur : agent inconnu « {agent} ». Agents : {', '.join(self.agents) or 'aucun'} ; crée-le d'abord avec creer_agent."
            spec.outils = self._with_navigation([t for t in spec.outils if t in self.work_tools])  # jamais plus que permis
            spec.origine = "bibliotheque"
            self._add(spec)
            if self.library:
                self.library.count_use(n)
        if not (tache or "").strip():
            return "Erreur : tâche vide."
        a = self.agents[n]
        spec = a["spec"]
        orch_state = self.save_state()                  # la conversation de l'orchestrateur, mise de côté
        task = {"tache": tache.strip(), "etapes": [], "compte_rendu": None, "debut_min": round(self.elapsed_min(), 2)}
        a["taches"].append(task)
        self.emit("agent_start", agent=n, tache=task["tache"], numero=len(a["taches"]))
        box: dict = {}
        orch = self

        @tool
        def rendre(compte_rendu: str) -> str:
            """Rend ton compte rendu à l'orchestrateur : la tâche est finie.

            Args:
                compte_rendu: ce que tu as trouvé ou fait, chiffres, fichiers créés, incertitudes
            """
            # le vérificateur cite des affirmations douteuses pour les contester : pas de contrôle des faits sur lui
            missing = orch.fc.missing(compte_rendu) if n != VERIF_NAME else []
            if missing and not box.get("bounced") and not box.get("forced"):
                box["bounced"] = True
                orch._flag("compte rendu (refusé une fois)", n, missing)
                return ("Compte rendu NON accepté : il avance des éléments qui ne viennent d'aucune page ni d'aucun fichier "
                        f"lus pendant ce run : {', '.join(missing)}. Vérifie-les à la source (lire_page avec chercher=…) "
                        "ou remplace-les par « non trouvé », puis rappelle rendre.")
            box["cr"] = compte_rendu.strip()
            if missing:
                box["cr"] += "\n" + orch._flag("compte rendu", n, missing)
            return "rendu"

        tools = self._sub_tools(spec, n) + [rendre]
        first = len(a["messages"]) == 1
        lecture = ", ".join(t for t in READ_ONLY if t in self.work_tools and t not in spec.outils) or "—"
        intro = (AGENT_INTRO.format(nom=n, role=spec.role, outils=", ".join(spec.outils) or "aucun", lecture=lecture)
                 if first else "")
        msg = intro + ("Tâche : " if first else "Nouvelle tâche : ") + task["tache"]
        if self.notes and notes:
            msg += "\n\nBloc-notes partagé :\n" + self.notes_text(limit=NOTES_LIMIT)
        start = len(a["messages"])
        a["messages"].append({"role": "user", "content": msg, "_garder": True})

        def on_step(step):
            rec = {"texte": step.text, "appels": []}
            for c, r in zip(step.calls, step.results):
                rec["appels"].append({"nom": c.get("name"), "arguments": c.get("arguments", {}), "resultat": r[:3000]})
                self.emit("tool", agent=n, nom=c.get("name"), arguments=c.get("arguments", {}), resultat=r[:3000])
            task["etapes"].append(rec)
            self.emit("turn", agent=n)
            self._check()

        gen = self._gen(n)
        notes_before = len(self.notes)
        report = None
        try:
            final = Agent(tools, gen, max_steps=max(1, (steps or self.sub_steps) - 1), on_step=on_step, append_stop=False,
                          stop_when=lambda: "cr" in box).run(a["messages"])
            if "cr" not in box and final.startswith("(arrêt"):
                # 06/10, 5e run : le chercheur avait tout trouvé et noté, mais n'a jamais appelé rendre ;
                # son compte rendu était vide. Dernier tour : rendre seulement.
                box["forced"] = True
                a["messages"].append({"role": "user", "content": LAST_TURN})
                self.emit("log", texte=f"{n} : limite de tours, dernier tour pour rendre")
                final = Agent([rendre], gen, max_steps=1, on_step=on_step, append_stop=False,
                              stop_when=lambda: "cr" in box).run(a["messages"])
            report = box.get("cr")
            if report is None:
                if final and not final.startswith("(arrêt"):
                    report = final  # réponse en texte, sans rendre : c'est le compte rendu
                else:
                    last = (task["etapes"][-1]["texte"] if task["etapes"] else "").strip()
                    report = "(l'agent n'a pas rendu de compte rendu)" + (f"\nDernier texte : {last[:1500]}" if last else "")
                mine = [x["texte"] for x in self.notes[notes_before:] if x["auteur"] == n]
                if mine:
                    report += "\nNotes de l'agent pendant cette tâche :\n" + "\n".join(f"- {t}" for t in mine)
        finally:
            # Repli par tâche (08/10, d'après DeepSeek-V3.2 : le raisonnement et les résultats d'outils servent
            # pendant la tâche ; la suivante n'en garde que la demande et le compte rendu). L'agent repartira de
            # son préfixe commun (cache P9) + ce court historique : pas d'état de décodage à garder sur CPU.
            done = report or box.get("cr") or "(pas de compte rendu)"
            a["messages"][start:] = [
                {"role": "user", "content": intro + ("Tâche : " if first else "Tâche précédente : ") + task["tache"]},
                {"role": "assistant", "content": f"(tâche terminée) {str(done)[:REPORT_IN_NOTES]}"}]
            self.ctx["replis"] += 1
            self.restore_state(orch_state)
        task["compte_rendu"] = report
        task["fin_min"] = round(self.elapsed_min(), 2)
        # 6e run : le rédacteur ne voyait pas le compte rendu du chercheur et a inventé les nouveautés
        ref = f"r{len(self.archive) + 1}"
        self.archive[ref] = report  # le compte rendu complet (relire ; ce n'est pas une source pour la vérification)
        short = cut_clean(report, REPORT_IN_NOTES)
        if short != report:
            short += f" (complet : relire('{ref}'))"
        self.notes.append({"auteur": n, "texte": f"Compte rendu (tâche « {cut_clean(task['tache'], 80)} ») : {short}",
                           "t": round(self.elapsed_min(), 2), "type": "compte_rendu"})
        self.emit("agent_done", agent=n, compte_rendu=report, numero=len(a["taches"]))
        return f"Compte rendu de {n} :\n{report}"

    def _sub_tools(self, spec: AgentSpec, who: str) -> list[Tool]:
        """Les MÊMES outils (même schéma, même ordre) pour tous les sous-agents : préfixe commun. Ceux que l'agent
        n'a pas reçus répondent par un refus, sans rien exécuter (ni demander l'accord de l'utilisateur)."""
        out = []
        for name in sorted(self.work_tools):
            t = self.work_tools[name]
            if name in spec.outils:
                out.append(self._write_tool(t, who) if name in WRITE_TOOLS else self._source_tool(t))
            elif name in READ_ONLY:
                out.append(self._granted(self._source_tool(t), spec))
            else:
                out.append(Tool(name, t.description, t.parameters, self._refused(name, spec)))
        return out + self._note_tools(who) + [self._relire_tool()]

    def _granted(self, t: Tool, spec: AgentSpec) -> Tool:
        """Outil de lecture que l'orchestrateur n'avait pas donné : accordé au premier appel, et gardé."""
        orch = self

        def fn(**kw):
            if t.name not in spec.outils:
                spec.outils.append(t.name)
                orch.emit("log", texte=f"{spec.nom} : outil {t.name} ajouté (lecture seule)")
            return t.fn(**kw)

        return Tool(t.name, t.description, t.parameters, fn)

    @staticmethod
    def _refused(name: str, spec: AgentSpec):
        def fn(**kw):
            return (f"Refusé : « {name} » n'est pas dans tes outils ({', '.join(spec.outils) or 'aucun'}). "
                    "Fais sans, ou rends ton compte rendu en disant ce qu'il te faudrait.")
        return fn

    # -- critères et vérification (08/10) ------------------------------------------
    def lessons_text(self) -> str:
        return ("\nLeçons des runs précédents (à respecter) :\n" + "\n".join(f"- {x}" for x in self.lessons) + "\n"
                if self.lessons else "")

    def make_criteria(self) -> list[str]:
        t0 = time.time()
        msgs = [{"role": "system", "content": CRIT_SYSTEM.format(date_rule=self.date_rule)},
                {"role": "user", "content": self.objective + self.lessons_text()}]
        try:
            text = self._gen("criteres")(msgs, [])
        except Cancelled:
            raise
        except Exception as e:  # pas de critères : le run continue sans
            self.emit("log", texte=f"critères non générés : {e}")
            text = ""
        crit = parse_criteria(text)
        if self.ask_criteria is not None:
            crit = self.ask_criteria(crit) or crit
        self.timing["criteres_s"] = round(time.time() - t0, 1)
        self.emit("criteria", criteres=crit)
        return crit

    def needs_verification(self, answer: str) -> bool:
        if self.verify_mode == "toujours":
            return True
        if self.verify_mode != "auto":
            return False
        return bool(fact_items(answer)) or bool(VERIFY_WORDS.search(self.objective + " " + " ".join(self.criteria)))

    def deliverable(self, answer: str) -> str:
        """Ce que l'utilisateur va recevoir : la réponse + le contenu réel des fichiers écrits pendant le run."""
        parts = [f"Réponse :\n{answer.strip()}"]
        for key, path in list(self.written_paths.items())[-3:]:
            try:
                text = Path(path).expanduser().read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                text = f"(illisible : {e})"
            parts.append(f"Fichier {path} :\n{text[:6000]}")
        return "\n\n".join(parts)

    def verify(self, answer: str) -> str | None:
        """Vérificateur indépendant, une fois par run : objectif + critères + LIVRABLE réel (réponse et fichiers
        écrits) + sources archivées, sans l'historique du run ni le bloc-notes (08/10 : le bloc-notes gardait un
        compte rendu périmé, le vérificateur a « vu » dans le fichier des dates qui n'y étaient pas). Chaque
        problème doit citer le livrable ou une source mot pour mot ; le code rejette les citations introuvables.
        Retourne None si tout va bien, sinon la liste des problèmes retenus."""
        if self.out_of_time():
            self.emit("log", texte="vérification finale sautée : limite de temps du run dépassée")
            self.verification = {"ok": None, "problemes": "", "rejetes": [], "s": 0, "saute": "temps"}
            return None
        t0 = time.time()
        if VERIF_NAME not in self.agents:
            outils = [t for t in ("lire_fichier", "lire_page", "lister_dossier", "chercher_fichiers") if t in self.work_tools]
            self._add(AgentSpec(VERIF_NAME, VERIF_ROLE, outils, "systeme"))
        livrable = self.deliverable(answer)
        # 08/10 : les comptes rendus des agents étaient listés comme sources ; seuls les résultats d'outils le sont
        sources = {r: t for r, t in self.archive.items() if r in self.source_refs}
        arch = "\n".join(f"[{r}] {txt[:90].replace(chr(10), ' ')}…" for r, txt in list(sources.items())[-30:])
        task = VERIF_TASK.format(objectif=self.objective,
                                 criteres="\n".join(f"- {c}" for c in self.criteria) or "- (aucun : relis l'objectif)",
                                 lecons=self.lessons_text(), livrable=livrable, archive=arch or "(aucune)")
        report = self.delegate(VERIF_NAME, task, steps=6, notes=False, system=True)
        report = report.split(":", 1)[1].strip() if report.startswith("Compte rendu de") else report
        ignore = "\n".join([self.objective, *self.criteria, self.date_rule])
        kept, rejected = check_problems(report, livrable, sources, ignore)
        rescued = [k for k in kept if k.startswith("MANQUE (relevé dans les sources")]
        if rescued:
            self.emit("log", texte=f"vérificateur : {len(rescued)} remarque(s) sans citation exacte gardée(s) : elle(s) "
                                   "apporte(nt) des faits lus dans les sources et absents du livrable")
        ok = not kept
        self.timing["verification_s"] = round(time.time() - t0, 1)
        self.verification = {"ok": ok, "problemes": "\n".join(kept), "rejetes": rejected,
                             "s": self.timing["verification_s"]}
        if rejected:
            self.emit("log", texte=f"vérificateur : {len(rejected)} problème(s) ignoré(s), citation introuvable dans le "
                                   "livrable ou la source")
        self.emit("verified", ok=ok, problemes=self.verification["problemes"], rejetes=rejected)
        return None if ok else "PROBLÈMES :\n" + "\n".join(kept)

    def _gen(self, name: str):
        if name not in self.gens:
            self.gens[name] = self.make_gen(name)
        return self.gens[name]

    # -- le run -----------------------------------------------------------------
    def system_prompt(self) -> str:
        lib = [f"{d['nom']} ({d['role'][:120]}; outils : {', '.join(t for t in d['outils'] if t in self.work_tools) or 'aucun'})"
               for d in (self.library.list() if self.library else [])]
        scout = self.scout_names()
        ecriture = " ou ".join(t for t in WRITE_TOOLS if t in self.work_tools) or "(écriture non permise : un agent)"
        return ORCH_SYSTEM.format(ecriture=ecriture, outils=", ".join(self.tool_names()) or "aucun", max_agents=self.max_agents,
                                  scout=", ".join(scout) if scout else "aucun outil (rien n'est permis)",
                                  max_steps=self.max_steps, minutes=int(self.minutes),
                                  bibliotheque="; ".join(lib) or "aucun", dossiers=self.folders_text(),
                                  date_rule=self.date_rule)

    def run(self) -> dict:
        tools = self.orchestrator_tools()
        if self.use_criteria:
            self.criteria = self.make_criteria()
        first = self.objective
        if self.criteria:
            first += ("\n\nCritères de réussite (contrôlés par un vérificateur avant la réponse finale) :\n"
                      + "\n".join(f"- {c}" for c in self.criteria))
        first += self.lessons_text()
        messages = [{"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": first}]
        left, nudges, last = self.max_steps, self.nudges, ""

        def on_step(step):
            self.orch_steps.append({"texte": step.text, "appels": [
                {"nom": c.get("name"), "arguments": c.get("arguments", {}), "resultat": r[:3000]}
                for c, r in zip(step.calls, step.results)]})
            for c, r in zip(step.calls, step.results):
                if c.get("name") != "confier":  # le compte rendu est déjà montré dans le nœud de l'agent
                    self.emit("tool", agent="orchestrateur", nom=c.get("name"), arguments=c.get("arguments", {}),
                              resultat=r[:3000])
            self.emit("turn", agent="orchestrateur")
            self._check()
            if self.final is None and self.after_verify is not None:
                self.after_verify += 1
                if self.after_verify >= AFTER_VERIFY_STEPS:
                    # 08/10 : après un refus, l'orchestrateur s'est perdu 30 min et n'a jamais terminé.
                    self._finish_with_last("le temps de correction après le vérificateur est écoulé")
            if self.final is None:
                self.between()

        def hint(name):
            if name in self.work_tools:  # 06/10, 5e run : « outil inconnu » compris comme « cet outil n'existe pas »
                who = [x for x, a in self.agents.items() if name in a["spec"].outils]
                return (f"« {name} » est un outil des agents, pas le tien : confie la tâche à un agent qui l'a"
                        + (f" ({', '.join(who)})" if who else "") + ", ou crée-en un avec cet outil.")
            return None

        while left > 0 and self.final is None:
            agent = Agent(tools, self._gen("orchestrateur"), max_steps=left, on_step=on_step,
                          stop_when=lambda: self.final is not None, unknown_hint=hint)
            last = agent.run(messages)
            left -= len(agent.steps)
            if self.final is not None:
                break
            if self.out_of_time() or left <= 0 or nudges <= 0:
                break
            nudges -= 1
            messages.append({"role": "user", "content": NUDGE})
            self.emit("log", texte="relance : l'orchestrateur s'est arrêté sans appeler d'outil")
        if self.final is None and self.last_proposed:
            self._finish_with_last("l'orchestrateur n'a pas rappelé terminer")
        if self.final is None:  # arrêt sans terminer : on rend ce qu'on a
            parts = [last] if last and not last.startswith("(arrêt") else []
            for n, a in self.agents.items():
                for t in a["taches"]:
                    if t["compte_rendu"]:
                        parts.append(f"**{n}** : {t['compte_rendu']}")
            self.final = ("(L'orchestrateur n'a pas appelé terminer ; voici ce qui a été produit.)\n\n"
                          + "\n\n".join(parts)) if parts else "(aucune réponse)"
            self.emit("final", reponse=self.final)
        return self.record()

    def _finish_with_last(self, why: str) -> None:
        """Rendre la dernière réponse proposée, avec les remarques du vérificateur, plutôt qu'un vrac de comptes rendus."""
        text = self.last_proposed or "(aucune réponse)"
        if self.verification and self.verification["ok"] is False:
            text += ("\n\n⚠ Remarques du vérificateur, non corrigées (" + why + ") :\n" + self.verification["problemes"])
        self.final = text
        self.emit("log", texte=f"réponse finale : dernière proposition ({why})")
        self.emit("final", reponse=self.final)

    def record(self) -> dict:
        return {"objectif": self.objective, "reponse": self.final, "minutes": round(self.elapsed_min(), 1),
                "orchestrateur": self.orch_steps, "notes": self.notes, "verifications": self.checks,
                "contexte": dict(self.ctx), "criteres": self.criteria, "lecons": self.lessons,
                "verification": self.verification, "temps": dict(self.timing),
                "agents": {n: {**a["spec"].as_dict(), "taches": a["taches"]} for n, a in self.agents.items()}}


# ----------------------------------------------------------------------------
# Le job de l'app
# ----------------------------------------------------------------------------
def orchestrator_job(toolbox, library: AgentLibrary, objective: str, outils: list | None = None,
                     max_agents: int = 5, max_steps: int = 14, sub_steps: int = 10, minutes: float = 60,
                     max_new_tokens: int = 1200, runs_dir=None, lessons=None, criteria: bool = True,
                     verify: str = "auto", wait_criteria: float = 90):
    def run(job, engine):
        import torch

        from .. import agent as agent_mod

        model, tok = engine.model, engine.tok
        work = toolbox.for_job(model, job)
        if outils is not None:
            allowed = set(outils)
            work = [t for t in work if t.name in allowed]
        sink = TextSink(tok, job)

        def budget() -> int | None:
            d = dec()
            max_len = getattr(d, "max_len", None) or getattr(engine.rt.cfg, "graph_max_len", None)
            return None if not max_len else max_len - min(max_new_tokens, 700) - 32

        stats = {"generations": 0, "prompt_tokens": 0, "repris": 0, "prefixes": 0}

        def make_gen(name):
            timings: list = []
            g = agent_mod.moelab_generate_fn(model, tok, max_new_tokens, temperature=0.2, streamer=sink,
                                             timings=timings)

            def fit(messages, tools, factor=1.0):
                b = budget()
                if not b:
                    return
                count = lambda m: agent_mod.render(tok, m, tools).shape[1]  # noqa: E731
                n = agent_mod.clean_context(messages, count, int(b * CLEAN_TRIGGER * factor),
                                            int(b * CLEAN_TARGET * factor))
                if n:
                    orch.ctx["nettoyages"] += 1
                    job.emit("log", texte=f"{name} : grand nettoyage du contexte ({n} messages réduits ou retirés), "
                                          "résultats toujours accessibles avec relire")
                # filet de sécurité : jamais au-delà de la fenêtre
                if agent_mod.compact_messages(messages, count, int(b * factor)):
                    job.emit("log", texte=f"{name} : conversation raccourcie pour tenir dans la fenêtre")

            def counted(messages, tools):
                n0 = len(timings)
                out = g(messages, tools)
                for t in timings[n0:]:
                    stats["generations"] += 1
                    stats["prompt_tokens"] += int(t.get("prompt_tokens", 0))
                    stats["repris"] += int(t.get("reused", 0))
                    stats["prefixes"] += int(bool(t.get("prefix_hit")))
                return out

            def fn(messages, tools):
                if job.cancel_requested:
                    raise Cancelled()
                job.emit("speaker", agent=name)
                fit(messages, tools)
                sink.reset()
                try:
                    return counted(messages, tools)
                except torch.cuda.OutOfMemoryError:
                    # 06/10, 7e run : 2 fois « CUDA out of memory » dans confier. On libère, on raccourcit, on réessaie.
                    torch.cuda.empty_cache()
                    d = dec()
                    if d is not None:
                        d.tokens = None  # rien à reprendre : la génération interrompue a laissé un cache partiel
                    job.emit("log", texte=f"{name} : mémoire GPU pleine, conversation raccourcie et nouvel essai")
                    fit(messages, tools, factor=0.6)
                    sink.reset()
                    try:
                        return counted(messages, tools)
                    except torch.cuda.OutOfMemoryError as e:
                        torch.cuda.empty_cache()
                        raise RuntimeError("mémoire GPU pleine même après avoir raccourci la conversation : "
                                           "baisse vram_gb ou relance") from e

            return fn

        def dec():
            return getattr(model, "moelab_graph_decoder", None)

        def save_state():
            d = dec()
            return d.export_state() if d is not None and hasattr(d, "export_state") else None

        def restore_state(s):
            d = dec()
            if s is not None and d is not None:
                d.import_state(s)

        def between():
            s = save_state()
            served = engine.serve_waiting(job.priority)
            if served:
                job.emit("log", texte=f"{served} message(s) de chat servi(s) entre deux tours")
                restore_state(s)

        def ask_criteria(crit):
            """L'utilisateur peut corriger les critères dans l'écran Agents ; sans réponse, on continue."""
            v = job.ask_value("criteria_ask", {"criteres": crit}, timeout=wait_criteria, default=None)
            return criteria_from_text(v) or crit if v else crit

        found = lessons.relevant(objective) if lessons is not None else []
        if found:
            job.emit("log", texte=f"{len(found)} leçon(s) des runs précédents utilisée(s)")
        orch = Orchestrator(objective, work, make_gen, library, emit=job.emit, save_state=save_state,
                            restore_state=restore_state, between=between, max_agents=max_agents,
                            max_steps=max_steps, sub_steps=sub_steps, minutes=minutes,
                            cancelled=lambda: job.cancel_requested, folders=toolbox.cfg.get("dossiers"),
                            lessons=found, criteria=criteria, verify=verify,
                            ask_criteria=ask_criteria if wait_criteria > 0 else None)
        job.emit("log", texte=f"outils permis aux agents : {', '.join(orch.tool_names()) or 'aucun'}")
        try:
            rec = orch.run()
        finally:
            if engine.rt is not None:
                engine.rt.end_of_request()
            partial = orch.record()
            partial["contexte"].update(stats, calcules=stats["prompt_tokens"] - stats["repris"])
            partial.update(id=job.id, date=time.time(), modele=engine.model_info.get("name"))
            if runs_dir is not None:
                write_json(Path(runs_dir) / f"{job.id}.json", partial)
        return {"reponse": rec["reponse"], "agents": list(rec["agents"]), "minutes": rec["minutes"], "run": job.id,
                "contexte": partial["contexte"], "temps": partial["temps"], "verification": partial["verification"]}

    return run


def list_runs(runs_dir) -> list[dict]:
    out = []
    for p in sorted(Path(runs_dir).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
        d = read_json(p, None)
        if d:
            out.append({"id": d.get("id", p.stem), "objectif": d.get("objectif", "")[:200], "date": d.get("date"),
                        "minutes": d.get("minutes"), "agents": list(d.get("agents", {})),
                        "reponse": (d.get("reponse") or "")[:300]})
    return out
