"""
Les outils du modèle dans l'app (05/10) : outils intégrés, outils des serveurs MCP branchés,
et une permission par outil (« toujours », « demander », « jamais »).

Réglages enregistrés dans <data>/tools.json :
    permissions      {outil: toujours | demander | jamais}
    dossiers         dossiers que les outils de fichiers ont le droit de lire / écrire
    mcp_serveurs     [{nom, commande, args, env, actif}] : serveurs MCP lancés par l'app (stdio)
    exposer_mcp      l'app expose elle-même un serveur MCP sur /mcp (Claude Desktop, VS Code...)
"""

from __future__ import annotations

import ast
import asyncio
import html as htmllib
import json
import operator
import re
import subprocess
import sys
import threading
import urllib.parse
from pathlib import Path

from .. import options
from ..agent import Tool, tool
from .engine import read_json, write_json

PERMS = ("toujours", "demander", "jamais")
DEFAULT_PERMS = {"etat": "toujours", "reglages": "toujours", "calculer": "toujours", "lister_dossier": "toujours",
                 "lire_fichier": "toujours", "chercher_fichiers": "toujours", "changer_reglage": "demander",
                 "ecrire_fichier": "demander", "ajouter_fichier": "demander", "lire_page": "demander", "chercher_web": "demander",
                 "executer_python": "demander"}
MAX_READ = 12000

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
        ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("seuls les nombres et + - * / // % ** sont permis")

    return ev(ast.parse(expr.replace(",", ".").replace("^", "**"), mode="eval").body)


def moelab_tools(model) -> list[Tool]:
    """Le moteur se décrit et se règle lui-même (comme scripts/agent.py)."""
    rt = model.moelab_runtime

    @tool
    def etat() -> dict:
        """État du moteur : budgets et remplissage des caches VRAM/RAM, politique, statistiques de la question en cours."""
        s = rt.stats.as_dict()
        fast = rt.fast
        out = {"politique": rt.policy.describe(),
               "vram_experts": {"utilise_mio": round(fast.used / 2**20) if fast else 0,
                                "budget_mio": round(fast.budget / 2**20) if fast else 0,
                                "experts": len(fast.entries) if fast else 0},
               "ram_experts": {"utilise_mio": round(rt.host.used / 2**20), "budget_mio": round(rt.host.budget / 2**20),
                               "experts": len(rt.host.entries)},
               "question_en_cours": {
                   "_note": "depuis le début de la question, lecture du prompt comprise",
                   "tokens_traites": s["tokens"],
                   "experts_trouves_en_vram_pourcent": round(100 * s["fast_hit_rate"], 1),
                   "experts_lus_sur_disque_par_token": round(s["disk_per_token"], 1),
                   "experts_calcules_sur_cpu_pourcent": round(100 * s["cpu_share"], 1)}}
        if rt.pool.fallbacks:
            out["pool_ram"] = f"{rt.pool.fallbacks} tampons pris hors du pool épinglé (copies vers la VRAM plus lentes)"
        return out

    @tool
    def reglages(groupe: str = "") -> list:
        """Réglages de moelab : valeur actuelle, application (live = à chaud, reload = rechargement), effet mesuré.

        Args:
            groupe: precision, prefill, memory, gpu, io ou store ; vide = tous
        """
        cur = options.current(model)
        out = []
        for o in options.catalog():
            if groupe and o.group != groupe:
                continue
            d = {"cle": o.key, "valeur": cur.get(o.key, o.default), "application": o.apply}
            if o.measured:
                d["effet_mesure"] = o.measured
            out.append(d)
        return out

    @tool
    def changer_reglage(cle: str, valeur: str) -> str:
        """Change un réglage à chaud (application « live » ou « caches » seulement).

        Args:
            cle: la clé du réglage, ex. policy, t1, prefill_skip
            valeur: la nouvelle valeur, en texte, ex. hobbit, 0.7, on, off
        """
        return options.apply(model, cle, valeur)

    return [etat, reglages, changer_reglage]


def general_tools(folders: list[str], workdir=None) -> list[Tool]:
    roots = [Path(f).expanduser().resolve() for f in folders if f]
    work = Path(workdir).expanduser().resolve() if workdir else None

    def allowed(p: str) -> Path:
        path = Path(p).expanduser().resolve()
        if not roots:
            raise PermissionError("aucun dossier autorisé : ajoute-en un dans l'écran Outils")
        if not any(path == r or r in path.parents for r in roots):
            raise PermissionError(f"{path} n'est pas dans un dossier autorisé par l'utilisateur. Dossiers autorisés : "
                                  f"{', '.join(map(str, roots))}")
        return path

    @tool
    def calculer(expression: str) -> float:
        """Calcule une expression arithmétique exacte.

        Args:
            expression: ex. (6.8 - 3.35) / 3.35 * 100
        """
        return safe_eval(expression)

    @tool
    def lister_dossier(chemin: str) -> list:
        """Liste le contenu d'un dossier autorisé (nom, type, taille).

        Args:
            chemin: chemin du dossier
        """
        p = allowed(chemin)
        if not p.is_dir():
            raise NotADirectoryError(f"{p} n'est pas un dossier (ou n'existe pas)")
        return [{"chemin": str(f), "nom": f.name, "type": "dossier" if f.is_dir() else "fichier",
                 "taille_ko": round(f.stat().st_size / 1024, 1) if f.is_file() else None}
                for f in sorted(p.iterdir())[:200]]

    @tool
    def chercher_fichiers(dossier: str, motif: str) -> list:
        """Cherche des fichiers par nom (motif glob, ex. *.pdf) dans un dossier autorisé et ses sous-dossiers.

        Args:
            dossier: chemin du dossier
            motif: ex. *.md ou *rapport*
        """
        p = allowed(dossier)
        return [str(f) for f in list(p.rglob(motif))[:100]]

    @tool
    def lire_fichier(chemin: str, debut: int = 0, chercher: str = "") -> str:
        """Lit un fichier texte (ou PDF) d'un dossier autorisé, par morceaux. Avec « chercher », renvoie seulement les passages qui contiennent ce mot.

        Args:
            chemin: chemin du fichier
            debut: position de départ, en caractères (pour lire la suite)
            chercher: mot ou expression à trouver dans le fichier ; vide = lire dans l'ordre
        """
        p = allowed(chemin)
        if p.is_dir():  # sous Windows, ouvrir un dossier donne PermissionError : le modèle croyait à un refus
            raise IsADirectoryError(f"{p} est un dossier, pas un fichier : utilise lister_dossier pour voir ses fichiers")
        if not p.exists():
            raise FileNotFoundError(f"{p} n'existe pas : vérifie avec lister_dossier ou chercher_fichiers")
        text = read_text_file(p)
        if chercher.strip():
            return find_passages(text, chercher.strip(), str(p))
        part = text[debut: debut + MAX_READ]
        if debut + MAX_READ < len(text):
            part += f"\n[... suite à partir de debut={debut + MAX_READ}, {len(text)} caractères au total]"
        return part

    @tool
    def ecrire_fichier(chemin: str, contenu: str) -> str:
        """Écrit (ou remplace entièrement) un fichier texte dans un dossier autorisé. Pour ajouter à la fin d'un fichier, utiliser ajouter_fichier.

        Args:
            chemin: chemin du fichier
            contenu: le texte à écrire
        """
        p = allowed(chemin)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(contenu, encoding="utf-8")
        return f"écrit : {p} ({len(contenu)} caractères)"

    @tool
    def ajouter_fichier(chemin: str, contenu: str) -> str:
        """Ajoute du texte à la fin d'un fichier d'un dossier autorisé (le crée s'il n'existe pas), sans avoir à le relire : pour remplir un fichier morceau par morceau.

        Args:
            chemin: chemin du fichier
            contenu: le texte à ajouter à la fin
        """
        p = allowed(chemin)
        p.parent.mkdir(parents=True, exist_ok=True)
        sep = ""
        if p.exists() and p.stat().st_size:
            with open(p, "rb") as fh:
                fh.seek(-1, 2)
                sep = "" if fh.read(1) == b"\n" else "\n"
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(sep + contenu if contenu.endswith("\n") else sep + contenu + "\n")
        return f"ajouté à {p} ({len(contenu)} caractères, {p.stat().st_size} octets au total)"

    pages: dict[str, str] = {}  # pages déjà téléchargées pendant cette session d'outils (lecture par morceaux)

    def page_text(url: str) -> str:
        import urllib.request

        if url in pages:
            return pages[url]
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (moelab)"})
        with urllib.request.urlopen(req, timeout=20) as r:
            html = r.read(3_000_000).decode(r.headers.get_content_charset() or "utf-8", "replace")
        text = html_to_text(html)
        if len(pages) > 20:
            pages.pop(next(iter(pages)))
        pages[url] = text
        return text

    @tool
    def lire_page(url: str, chercher: str = "", debut: int = 0) -> str:
        """Lit une page web (texte sans balises), par morceaux de 4 000 caractères. Avec « chercher », renvoie seulement les passages qui contiennent ce mot ou cette expression : le plus efficace pour trouver une date, un chiffre ou un nom dans une longue page.

        Args:
            url: adresse http(s)
            chercher: mot ou expression à trouver dans la page (ex. released, 2.14.0) ; plusieurs à la fois séparés par | (ex. 2.14.0 | 2.13.0 | 2.12.0) ; vide = lire dans l'ordre
            debut: position de départ en caractères, pour lire la suite
        """
        if not re.match(r"^https?://", url):
            raise ValueError("adresse http(s) attendue")
        text = page_text(url)
        if chercher.strip():
            return find_passages(text, chercher.strip(), url)
        debut = max(0, int(debut))
        part = text[debut: debut + PAGE_CHUNK]
        if debut + PAGE_CHUNK < len(text):
            part += (f"\n[... {len(text)} caractères au total : suite avec debut={debut + PAGE_CHUNK}, "
                     "ou chercher un mot précis]")
        return part or "(page vide ou fin de page)"

    @tool
    def chercher_web(requete: str, nombre: int = 5) -> list:
        """Cherche sur le web (DuckDuckGo) : titre, adresse et extrait des premiers résultats. Lire une page ensuite avec lire_page.

        Args:
            requete: les mots à chercher
            nombre: nombre de résultats, 10 au plus
        """
        import urllib.request

        data = urllib.parse.urlencode({"q": requete}).encode()
        req = urllib.request.Request("https://html.duckduckgo.com/html/", data=data,
                                     headers={"User-Agent": "Mozilla/5.0 (moelab)"})
        with urllib.request.urlopen(req, timeout=20) as r:
            page = r.read(1_000_000).decode("utf-8", "replace")
        res = parse_ddg(page)[: max(1, min(int(nombre), 10))]
        return res or [{"note": "aucun résultat (ou page de résultats illisible)"}]

    @tool
    def executer_python(code: str, delai_s: int = 60) -> str:
        """Exécute un script Python (calculs, analyse de données, graphiques, conversions) dans le dossier de travail des agents ; renvoie ce qu'il affiche. Les fichiers créés restent dans ce dossier.

        Args:
            code: le script complet ; afficher les résultats avec print
            delai_s: temps maximum en secondes (300 au plus)
        """
        if work is None:
            raise RuntimeError("pas de dossier de travail")
        work.mkdir(parents=True, exist_ok=True)
        script = work / "_script.py"
        script.write_text(code, encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", str(script)], cwd=work, capture_output=True, text=True,
                               timeout=max(1, min(int(delai_s), 300)), encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return f"arrêté après {delai_s} s (trop long)"
        out = (p.stdout or "")[-8000:]
        if p.returncode != 0:
            out += f"\n[erreur, code {p.returncode}]\n" + (p.stderr or "")[-4000:]
        return out.strip() or "(rien d'affiché ; utilise print)"

    return [calculer, lister_dossier, chercher_fichiers, lire_fichier, ecrire_fichier, ajouter_fichier, lire_page,
            chercher_web, executer_python]


PAGE_CHUNK = 4000  # = ce que l'agent voit d'un résultat d'outil (Agent.max_result_chars)


_TIME_TAG = re.compile(r'(?is)<(relative-time|local-time|time)\b[^>]*\bdatetime="(\d{4}-\d{2}-\d{2})[^"]*"[^>]*>(.*?)</\1>')


def html_to_text(html: str) -> str:
    # 13/10 : GitHub affiche « 02 Sep » sans l'année ; la date complète est dans l'attribut datetime. Le modèle
    # devinait l'année (2.13.0 « sortie en 2023 »). On garde la date complète du code de la page.
    # le texte affiché reste (« released this 02 Sep ») : il dit de quelle date il s'agit
    html = _TIME_TAG.sub(lambda m: f" {re.sub(r'<[^>]+>', ' ', m.group(3)).strip()} ({m.group(2)}) ", html)
    html = re.sub(r"(?is)<(script|style|noscript|svg|nav|header|footer|template)\b.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr|section|article)>", "\n", html)
    text = htmllib.unescape(re.sub(r"<[^>]+>", " ", html))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def _spans(text: str, terms: list[str], width: int) -> list[tuple[int, int]]:
    low, found = text.lower(), []
    for term in terms:
        t = term.lower()
        i = low.find(t)
        while i != -1:
            found.append((max(0, i - width), min(len(text), i + len(t) + width)))
            i = low.find(t, i + len(t))
    spans: list[tuple[int, int]] = []
    for a, b in sorted(found):
        if spans and a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(b, spans[-1][1]))
        else:
            spans.append((a, b))
    return spans


_STOP = {"dans", "avec", "pour", "les", "des", "une", "the", "and", "with", "from", "that", "this", "release", "page"}


def find_passages(text: str, term: str, url: str = "", width: int = 300, limit: int = PAGE_CHUNK) -> str:
    """
    Passages de `text` autour de chaque occurrence de `term` (sans tenir compte de la casse). 13/10 : le chat a
    perdu neuf tours à deviner des titres (« New Features », « Highlights ») absents des pages ; maintenant, si
    l'expression manque, on cherche ses mots un par un, et sinon on rend le début de la page : jamais un tour vide.
    """
    terms = [t.strip() for t in term.split("|") if t.strip()] or [term]
    spans = _spans(text, terms, width)
    header = (f"{{n}} passage(s) contenant « {term} » :" if len(terms) == 1 else
              "{n} passage(s) contenant " + " ou ".join(f"« {t} »" for t in terms) + " :")
    if not spans:
        words = [w for w in dict.fromkeys(re.findall(r"[\w.+-]{4,}", term)) if w.lower() not in _STOP and w != term]
        spans = _spans(text, words, width) if words else []
        if spans:
            header = (f"« {term} » n'apparaît pas tel quel ; {{n}} passage(s) avec "
                      + ", ".join(f"« {w} »" for w in words) + " :")
    if not spans:
        start = text[:min(limit, 2500)]
        more = (f"\n[... {len(text)} caractères au total : suite avec debut={len(start)}]" if len(text) > len(start) else "")
        return (f"« {term} » n'apparaît pas dans la page ({len(text)} caractères). Voici son début, pour choisir un "
                f"autre mot :\n{start}{more}")
    out, n = [], 0
    for a, b in spans:
        piece = text[a:b].replace("\n", " ").strip()
        if n + len(piece) > limit:
            out.append(f"[... {len(spans) - len(out)} autres passages : cherche un mot plus précis]")
            break
        out.append(f"[{a}] …{piece}…")
        n += len(piece) + 10
    return header.format(n=len(spans)) + "\n" + "\n".join(out)


def parse_ddg(page: str) -> list[dict]:
    """Résultats de la page HTML de DuckDuckGo -> [{titre, url, extrait}]."""
    out = []
    links = re.findall(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.S)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)>', page, re.S)
    for i, (href, title) in enumerate(links):
        href = htmllib.unescape(href)
        q = urllib.parse.urlparse(href if "://" in href else "https:" + href)
        if "uddg" in (qs := urllib.parse.parse_qs(q.query)):
            href = qs["uddg"][0]
        if "duckduckgo.com/y.js" in href:  # annonce
            continue
        clean = lambda t: re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", "", t))).strip()  # noqa: E731
        out.append({"titre": clean(title), "url": href, "extrait": clean(snippets[i]) if i < len(snippets) else ""})
    return out


def read_text_file(p: Path) -> str:
    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("lecture des PDF : pip install pypdf") from e
        return "\n".join((page.extract_text() or "") for page in PdfReader(str(p)).pages)
    return p.read_text(encoding="utf-8", errors="replace")


class McpHub:
    """
    Client MCP : lance les serveurs configurés (stdio), garde leurs sessions ouvertes dans une
    boucle asyncio à part, et présente leurs outils comme des outils ordinaires
    (« serveur__outil »).
    """

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="moelab-mcp", daemon=True)
        self.thread.start()
        self.servers: dict[str, dict] = {}  # nom -> {session, tools, status, stop}

    def _run(self, coro, timeout=60):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _serve(self, cfg: dict, ready: asyncio.Future):
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except ImportError:
            ready.set_result("le paquet mcp n'est pas installé (pip install mcp)")
            return
        stop = asyncio.Event()
        try:
            params = StdioServerParameters(command=cfg["commande"], args=list(cfg.get("args") or []),
                                           env=cfg.get("env") or None)
            async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                self.servers[cfg["nom"]] = {"session": session, "tools": tools, "status": "connecté", "stop": stop}
                ready.set_result(None)
                await stop.wait()
        except Exception as e:
            msg = f"{e.__class__.__name__}: {e}"
            self.servers[cfg["nom"]] = {"session": None, "tools": [], "status": f"erreur : {msg}", "stop": stop}
            if not ready.done():
                ready.set_result(msg)

    def connect(self, cfg: dict, timeout: float = 30) -> str | None:
        self.disconnect(cfg["nom"])

        async def start():
            ready = self.loop.create_future()
            self.loop.create_task(self._serve(cfg, ready))
            return await asyncio.wait_for(ready, timeout)

        try:
            return self._run(start(), timeout + 5)
        except Exception as e:
            self.servers[cfg["nom"]] = {"session": None, "tools": [], "status": f"erreur : {e}", "stop": None}
            return str(e)

    def disconnect(self, name: str) -> None:
        s = self.servers.pop(name, None)
        if s and s.get("stop") is not None:
            self.loop.call_soon_threadsafe(s["stop"].set)

    def status(self) -> dict:
        return {n: {"status": s["status"], "outils": [t.name for t in s["tools"]]} for n, s in self.servers.items()}

    def tools(self) -> list[Tool]:
        out = []
        for name, s in self.servers.items():
            if s.get("session") is None:
                continue
            for t in s["tools"]:
                out.append(self._wrap(name, s["session"], t))
        return out

    def _wrap(self, server: str, session, t) -> Tool:
        hub = self

        def call(**kwargs):
            res = hub._run(session.call_tool(t.name, kwargs), timeout=120)
            parts = [getattr(c, "text", None) or str(c) for c in (res.content or [])]
            text = "\n".join(parts)
            return f"Erreur : {text}" if getattr(res, "isError", False) else text

        schema = t.inputSchema or {"type": "object", "properties": {}}
        return Tool(f"{server}__{t.name}", f"[MCP {server}] {t.description or ''}"[:500], schema, call)


class ToolBox:
    """Réglages des outils + liste des outils permis pour un job, avec demande d'accord si besoin."""

    def __init__(self, data_dir):
        self.path = Path(data_dir) / "tools.json"
        self.cfg = read_json(self.path, {})
        self.cfg.setdefault("permissions", {})
        self.cfg.setdefault("dossiers", [])
        self.cfg.setdefault("mcp_serveurs", [])
        self.cfg.setdefault("exposer_mcp", False)
        self.workdir = Path(data_dir) / "espace_agents"  # dossier de travail de executer_python
        self.hub = McpHub()
        self._started = False

    def start_mcp(self) -> None:
        if self._started:
            return
        self._started = True
        for s in self.cfg["mcp_serveurs"]:
            if s.get("actif", True):
                threading.Thread(target=self.hub.connect, args=(s,), daemon=True).start()

    def save(self) -> None:
        write_json(self.path, self.cfg)

    def permission(self, name: str) -> str:
        p = self.cfg["permissions"].get(name)
        if p in PERMS:
            return p
        return DEFAULT_PERMS.get(name, "demander")

    def all_tools(self, model) -> list[Tool]:
        tools = moelab_tools(model) if model is not None else []
        return tools + general_tools(self.cfg["dossiers"], self.workdir) + self.hub.tools()

    def describe(self, model) -> list[dict]:
        out = []
        for t in self.all_tools(model):
            out.append({"nom": t.name, "description": t.description, "permission": self.permission(t.name),
                        "source": "mcp" if "__" in t.name else ("moelab" if t.name in ("etat", "reglages", "changer_reglage")
                                                                 else "intégré")})
        return out

    def for_job(self, model, job) -> list[Tool]:
        """Outils permis (pas « jamais »), ceux en « demander » enveloppés d'une demande d'accord. Les écritures de
        fichier passent d'abord par le contrôle des dates du futur (avant de te demander ton accord)."""
        out = []
        for t in self.all_tools(model):
            perm = self.permission(t.name)
            if perm == "jamais":
                continue
            if perm == "demander":
                t = Tool(t.name, t.description, t.parameters, _asking(job, t))
            if t.name in ("ecrire_fichier", "ajouter_fichier"):
                t = Tool(t.name, t.description, t.parameters, future_guard(t.fn))
            out.append(t)
        return out


def future_guard(fn, today=None):
    """13/10 : un contenu qui cite des dates du futur est refusé une fois, avec la question ; réécrit, il passe."""
    import datetime as dt

    from .factcheck import future_warning

    warned: set = set()

    def wrapped(**kw):
        content = str(kw.get("contenu", ""))
        msg = future_warning(content, today or dt.date.today())
        key = str(kw.get("chemin", ""))
        if msg and key not in warned:
            warned.add(key)
            return msg
        return fn(**kw)

    return wrapped


def _asking(job, t: Tool):
    def fn(**kwargs):
        if not job.ask({"outil": t.name, "arguments": kwargs}):
            return "refusé par l'utilisateur"
        return t.fn(**kwargs)

    return fn


# ----------------------------------------------------------------------------
# Archive des résultats pour le chat (13/10) : comme chez les agents, un long résultat garde une référence [rN] ;
# quand le contexte est nettoyé, le modèle le relit avec relire au lieu de l'avoir perdu.
# ----------------------------------------------------------------------------
ARCHIVED_TOOLS = ("lire_page", "lire_fichier", "chercher_web", "lister_dossier", "chercher_fichiers")
SEEN = 4000
_REF = re.compile(r"^\[(r\d+)\] ", re.S)


def rebuild_archive(messages: list[dict]) -> dict:
    """Archive d'une conversation enregistrée : les résultats d'outils qui commencent par [rN]."""
    out = {}
    for m in messages:
        if m.get("role") == "tool" and (r := _REF.match(m.get("content") or "")):
            out[r.group(1)] = m.get("archive") or m["content"][r.end():]
    return out


def archived(t: Tool, archive: dict) -> Tool:
    def fn(**kw):
        out = t.fn(**kw)
        text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
        ref = f"r{max([int(k[1:]) for k in archive] + [0]) + 1}"
        archive[ref] = text
        if len(text) <= SEEN - 120:
            return f"[{ref}] {text}"
        return (f"[{ref}] {text[:SEEN - 220]}\n[... {len(text)} caractères : suite avec relire('{ref}', "
                f"debut={SEEN - 220}), ou relire('{ref}', chercher='mot')]")

    return Tool(t.name, t.description, t.parameters, fn)


def relire_tool(archive: dict) -> Tool:
    @tool
    def relire(ref: str, chercher: str = "", debut: int = 0) -> str:
        """Relit un résultat déjà obtenu [r1], [r2]… (page, fichier, recherche), en entier par morceaux ou seulement les passages autour d'un mot. Bien plus rapide que de relire la page.

        Args:
            ref: la référence, ex. r3
            chercher: mot ou expression à trouver, plusieurs séparés par | ; vide = lire dans l'ordre
            debut: position de départ en caractères, pour lire la suite
        """
        text = archive.get(str(ref).strip().strip("[]"))
        if text is None:
            return f"Référence inconnue « {ref} ». Références : {', '.join(archive) or 'aucune'}."
        if chercher.strip():
            return find_passages(text, chercher.strip(), limit=SEEN - 200)
        debut = max(0, int(debut))
        part = text[debut: debut + SEEN - 200]
        if debut + SEEN - 200 < len(text):
            part += f"\n[... suite avec relire('{ref}', debut={debut + SEEN - 200})]"
        return part or "(fin du résultat)"

    return relire
