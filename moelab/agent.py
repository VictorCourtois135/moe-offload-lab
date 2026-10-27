"""
Agent local (05/10) : le modèle chargé par moelab appelle des outils.

Trois briques, sans dépendance à un service en ligne :
  - `tool` : une fonction Python devient un outil (nom, description, schéma JSON des
    arguments tiré des annotations et de la docstring) ;
  - `parse_tool_calls` : lit les appels dans le texte du modèle, au format des modèles
    Qwen / Hermes : <tool_call>{"name": ..., "arguments": {...}}</tool_call> ;
  - `Agent` : la boucle. Le modèle répond ; s'il demande des outils, le code les exécute
    et lui renvoie les résultats ; on recommence jusqu'à une réponse sans appel.

Principe : le code garantit, le modèle propose. Un outil inconnu, des arguments faux ou
une exception deviennent un message d'erreur rendu au modèle, jamais un plantage ; la
boucle s'arrête après `max_steps` tours.

`moelab_generate_fn` branche la boucle sur un modèle moelab. Le modèle de chat du
tokenizer décrit les outils s'il sait le faire (Qwen3 : oui) ; sinon moelab les décrit
lui-même dans le message système, au même format. Les messages de l'assistant gardent
le texte brut du modèle (appels compris) : d'un tour à l'autre, le début du prompt est
identique et P1 reprend tout l'historique du cache.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def schema(self) -> dict:
        """Format « function » d'OpenAI, compris par les modèles de chat de transformers."""
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


def _doc_parts(doc: str) -> tuple[str, dict[str, str]]:
    """Docstring -> (description, {argument: description}) ; arguments listés après « Args: »."""
    lines = inspect.cleandoc(doc or "").splitlines()
    desc, args, in_args = [], {}, False
    for line in lines:
        s = line.strip()
        if s.lower() in ("args:", "arguments:"):
            in_args = True
            continue
        if in_args and ":" in s:
            k, _, v = s.partition(":")
            args[k.strip()] = v.strip()
        elif not in_args and s:
            desc.append(s)
    return " ".join(desc), args


def tool(fn: Callable | None = None, *, name: str | None = None):
    """Décorateur : `@tool` ou `@tool(name="...")` -> objet Tool."""

    def make(f: Callable) -> Tool:
        desc, arg_docs = _doc_parts(f.__doc__)
        props, required = {}, []
        for p in inspect.signature(f).parameters.values():
            ann = p.annotation if p.annotation is not inspect.Parameter.empty else str
            if isinstance(ann, str):  # annotations différées (from __future__ import annotations)
                ann = {"str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict}.get(
                    ann.split("|")[0].strip(), str)
            prop = {"type": _JSON_TYPES.get(ann, "string")}
            if p.name in arg_docs:
                prop["description"] = arg_docs[p.name]
            props[p.name] = prop
            if p.default is inspect.Parameter.empty:
                required.append(p.name)
        params = {"type": "object", "properties": props, "required": required}
        return Tool(name or f.__name__, desc, params, f)

    return make(fn) if fn is not None else make


_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)


def _loads(s: str):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    return json.loads(s)


def _bare_calls(text: str, known) -> tuple[str, list[dict]]:
    """
    Appels écrits en texte simple pour un outil connu (06/10 : Qwen3-Next, plusieurs fois de suite) :
        nom({"arg": ...})      nom {"arg": ...}      nom("texte")  -> 1er argument obligatoire
    `known` : {nom: Tool} (ou une simple liste de noms : alors seulement la forme JSON).
    """
    calls, spans = [], []
    dec = json.JSONDecoder()
    pat = re.compile(r"(?<![\w.])(" + "|".join(re.escape(k) for k in sorted(known, key=len, reverse=True))
                     + r")\s*(\(\s*)?(?=[{\"])")
    pos = 0
    while (m := pat.search(text, pos)) is not None:
        try:
            obj, end = dec.raw_decode(text, m.end())
        except ValueError:
            pos = m.end()
            continue
        name = m.group(1)
        if isinstance(obj, str):  # nom("texte") : seulement entre parenthèses, et vers le 1er argument obligatoire
            t = known.get(name) if isinstance(known, dict) else None
            req = (getattr(t, "parameters", None) or {}).get("required") or list(
                (getattr(t, "parameters", None) or {}).get("properties", {}))[:1]
            if not m.group(2) or not req:
                pos = end
                continue
            obj = {req[0]: obj}
        if isinstance(obj, dict):
            if set(obj) <= {"name", "arguments"} and obj.get("name") == name and isinstance(obj.get("arguments"), dict):
                obj = obj["arguments"]
            close = re.match(r"\s*\)", text[end:])
            end += close.end() if close else 0
            calls.append({"name": name, "arguments": obj})
            spans.append((m.start(), end))
        pos = end
    kw_calls, kw_spans = _kwarg_calls(text, known, spans)
    calls += kw_calls
    spans += kw_spans
    order = sorted(range(len(calls)), key=lambda i: spans[i][0])
    calls = [calls[i] for i in order]
    for a, b in sorted(spans, reverse=True):
        text = text[:a] + text[b:]
    return text, calls


def _kwarg_calls(text: str, known, taken) -> tuple[list[dict], list[tuple[int, int]]]:
    """
    Style Python (06/10, 3e vrai run) : `rendre(compte_rendu="...")`, `lire_fichier(chemin="C:\\x", debut=0)`.
    Lu avec ast : seules des valeurs littérales sont acceptées (rien n'est exécuté).
    """
    calls, spans = [], []
    pat = re.compile(r"(?<![\w.])(" + "|".join(re.escape(k) for k in sorted(known, key=len, reverse=True))
                     + r")\s*\(\s*(?=[A-Za-z_]\w*\s*=)")
    for m in pat.finditer(text):
        if any(a <= m.start() < b for a, b in taken + spans):
            continue
        j = m.end()
        while (j := text.find(")", j)) != -1:
            src = text[m.start():j + 1].strip()
            node, raw = None, False
            # 2e essai : chaînes lues « brutes » (chemins Windows : C:\Users ferait planter \U) ; voir _unraw
            for raw, cand in ((False, src), (True, re.sub(r"\\(?![\\\"'])", r"\\\\", src))):
                try:
                    node = ast.parse(cand, mode="eval").body
                    break
                except (SyntaxError, ValueError):
                    node = None
            if node is None:
                j += 1
                continue
            if isinstance(node, ast.Call) and not node.args:
                try:
                    args = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
                except ValueError:
                    break
                if raw:
                    args = {k: _unraw(v) for k, v in args.items()}
                calls.append({"name": m.group(1), "arguments": args})
                spans.append((m.start(), j + 1))
            break
    return calls, spans


_JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MOIS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
         "novembre", "décembre")


def today_fr(d=None) -> str:
    """« mardi 6 octobre 2026 » : la date du jour pour les consignes (06/10 : le modèle datait de 2024 des
    versions sorties en 2026, ses connaissances s'arrêtant avant)."""
    import datetime

    d = d or datetime.date.today()
    return f"{_JOURS[d.weekday()]} {d.day} {_MOIS[d.month - 1]} {d.year}"


DATE_RULE = ("Nous sommes le {date}. Tes connaissances s'arrêtent avant cette date : pour tout ce qui est récent "
             "(versions, dates, prix, actualité, personnes), vérifie avec les outils au lieu de répondre de mémoire. "
             "Ce qui est daté après aujourd'hui est prévu, pas sorti : « dernier » veut dire déjà paru.")


_WIN_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _unraw(v):
    r"""
    Lecture brute (2e essai) : un chemin Windows garde ses \ tels quels ; dans un autre texte, \n et \t
    redeviennent retour à la ligne et tabulation (06/10 : sous Windows, le contenu d'un fichier avait
    gardé des « \n » littéraux parce que le chemin, dans le même appel, forçait la lecture brute).
    """
    if not isinstance(v, str) or _WIN_PATH.match(v):
        return v
    return v.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _json_only_call(text: str, known: dict) -> tuple[str, list[dict]]:
    """
    Dernier recours (06/10, 4e vrai run : « rende » puis {"compte_rendu": ...}, nom tronqué) : un objet JSON
    dont les clés correspondent aux arguments d'UN SEUL outil (obligatoires présents, aucune clé inconnue).
    Deux outils aux mêmes arguments (ecrire_fichier / ajouter_fichier) : ambigu, rien n'est appelé.
    """
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            continue
        if not isinstance(obj, dict) or not obj or text[end:].strip():
            continue  # seulement un objet qui termine le message, là où se place un appel
        keys = set(obj)
        match = [n for n, t in known.items()
                 if (props := set((getattr(t, "parameters", None) or {}).get("properties", {})))
                 and keys <= props and set(t.parameters.get("required", [])) <= keys]
        if len(match) == 1:
            start = i
            m = re.search(r"[A-Za-z_]\w*\s*$", text[:i])  # le nom tronqué juste avant (« rende »)
            if m:
                start = m.start()
            return (text[:start] + text[end:]).strip(), [{"name": match[0], "arguments": obj}]
    return text, []


def parse_tool_calls(text: str, known=None) -> tuple[str, list[dict]]:
    """
    -> (texte hors appels, [{"name", "arguments"}]). Un appel illisible devient
    {"name": None, "error": ...} : l'agent le signale au modèle au lieu de l'ignorer.
    `known` (noms d'outils) : sans balise <tool_call>, on cherche aussi `nom({...})`.
    """
    if known and "<tool_call>" not in text:
        rest, bare = _bare_calls(text, known)
        if bare:
            return rest.strip(), bare
        if isinstance(known, dict):
            rest, only = _json_only_call(text, known)
            if only:
                return rest, only
    calls = []
    for m in _CALL_RE.finditer(text):
        raw = m.group(1)
        try:
            obj = _loads(raw)
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                args = _loads(args) if args.strip() else {}
            calls.append({"name": obj["name"], "arguments": args if isinstance(args, dict) else {}})
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            calls.append({"name": None, "error": f"appel illisible ({e.__class__.__name__}) : {raw[:200]}"})
    content = _CALL_RE.sub("", text).strip()
    return content, calls


@dataclass
class Step:
    text: str
    calls: list[dict] = field(default_factory=list)
    results: list[str] = field(default_factory=list)


class Agent:
    """
    generate_fn(messages, tools_schema) -> texte brut du modèle (appels compris).
    on_step(step) : appelé après chaque tour (affichage).
    """

    def __init__(self, tools: list[Tool], generate_fn, max_steps: int = 8, max_result_chars: int = 4000,
                 on_step=None, stop_when=None, unknown_hint=None, append_stop: bool = True):
        self.tools = {t.name: t for t in tools}
        self.unknown_hint = unknown_hint  # nom -> phrase d'aide pour un outil inconnu (ou None)
        self.append_stop = append_stop    # ajouter « (arrêt : ...) » à la conversation quand les tours sont épuisés
        self.generate_fn = generate_fn
        self.max_steps = max_steps
        self.max_result_chars = max_result_chars
        self.on_step = on_step
        self.stop_when = stop_when  # ex. un outil « terminer » a été appelé : on ne relance pas le modèle
        self.steps: list[Step] = []

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, c: dict) -> str:
        if c.get("name") is None:
            return f"Erreur : {c.get('error', 'appel illisible')}"
        t = self.tools.get(c["name"])
        if t is None:
            hint = self.unknown_hint(c["name"]) if self.unknown_hint else None
            return (f"Erreur : outil inconnu « {c['name']} »." + (f" {hint}" if hint else "")
                    + f" Tes outils : {', '.join(self.tools)}")
        args = c.get("arguments", {})
        props = t.parameters.get("properties", {})
        if props and isinstance(args, dict):
            unknown = [k for k in args if k not in props]
            missing = [k for k in t.parameters.get("required", []) if k not in args]
            if unknown or missing:
                return (f"Erreur d'arguments pour {t.name} : "
                        + (f"inconnus {unknown} ; " if unknown else "") + (f"manquants {missing} ; " if missing else "")
                        + "arguments attendus : " + ", ".join(
                            f"{k} ({'obligatoire' if k in t.parameters.get('required', []) else 'facultatif'})" for k in props))
        try:
            out = t.fn(**args)
        except TypeError as e:
            return f"Erreur d'arguments pour {t.name} : {e}. Arguments attendus : {', '.join(props)}"
        except Exception as e:  # l'outil a échoué : le modèle le saura
            return f"Erreur dans {t.name} : {e.__class__.__name__}: {e}"
        s = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
        if len(s) > self.max_result_chars:
            s = s[: self.max_result_chars] + f"\n[... {len(s) - self.max_result_chars} caractères coupés]"
        return s

    def run(self, messages: list[dict]) -> str:
        """Fait tourner la boucle sur `messages` (complétée en place). Retourne la réponse finale."""
        self.steps = []
        for _ in range(self.max_steps):
            text = self.generate_fn(messages, self.schemas())
            content, calls = parse_tool_calls(text, known=self.tools)
            messages.append({"role": "assistant", "content": text})
            step = Step(content, calls)
            self.steps.append(step)
            if not calls:
                if self.on_step:
                    self.on_step(step)
                return content
            for c in calls:
                r = self.call(c)
                step.results.append(r)
                messages.append({"role": "tool", "name": c.get("name") or "?", "content": r})
            if self.on_step:
                self.on_step(step)
            if self.stop_when is not None and self.stop_when():
                return step.results[-1] if step.results else content
        final = f"(arrêt : {self.max_steps} tours sans réponse finale)"
        if self.append_stop:
            messages.append({"role": "assistant", "content": final})
        return final


# ----------------------------------------------------------------------------
# Branchement sur un modèle moelab
# ----------------------------------------------------------------------------
FALLBACK_TOOLS_PROMPT = (
    "Tu peux appeler des outils. Les voici, au format JSON :\n<tools>\n{tools}\n</tools>\n\n"
    "Pour appeler un outil, écris exactement :\n<tool_call>\n"
    '{{"name": "nom_de_l_outil", "arguments": {{...}}}}\n</tool_call>\n'
    "puis arrête-toi : le résultat te sera donné dans un message <tool_response>. "
    "Quand tu as ce qu'il faut, réponds normalement, sans appel."
)


def native_tools(tokenizer) -> bool:
    """Le modèle de chat du tokenizer sait-il décrire des outils (Qwen3 : oui) ?"""
    tpl = getattr(tokenizer, "chat_template", None) or ""
    if isinstance(tpl, dict):
        tpl = " ".join(tpl.values())
    return "tools" in tpl and "tool_call" in tpl


def render(tokenizer, messages: list[dict], tools: list[dict], add_generation_prompt: bool = True):
    """Messages + outils -> ids du prompt (tenseur 1 x T)."""
    import torch

    if native_tools(tokenizer):
        msgs, kw = messages, {"tools": tools or None}
    else:  # description des outils dans le message système, réponses des outils en messages user
        sys = FALLBACK_TOOLS_PROMPT.format(tools="\n".join(json.dumps(t, ensure_ascii=False) for t in tools))
        msgs = list(messages)
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n\n" + sys}
        elif tools:
            msgs.insert(0, {"role": "system", "content": sys})
        msgs = [{"role": "user", "content": f"<tool_response>\n{m['content']}\n</tool_response>"}
                if m["role"] == "tool" else m for m in msgs]
        kw = {}
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=add_generation_prompt, return_tensors="pt",
                                        return_dict=False, **kw)
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return ids


def prefix_length(tokenizer, messages: list[dict], tools: list[dict], ids) -> int | None:
    """
    P9 : longueur du début fixe du prompt = message système + description des outils, s'il
    est bien le début exact du prompt complet (sinon None : pas de préfixe gardé).
    """
    if not messages or messages[0]["role"] != "system":
        return None
    pre = render(tokenizer, messages[:1], tools, add_generation_prompt=False)[0].tolist()
    full = ids[0].tolist()
    return len(pre) if 0 < len(pre) < len(full) and full[: len(pre)] == pre else None


_REF_RE = re.compile(r"^\[(r\d+)\]")
_REPORT_RE = re.compile(r"^Compte rendu de (\S+) :")


def _stub(m: dict) -> dict | None:
    """Version nettoyée d'un message (None : inchangé). Résultats d'outils -> leur référence d'archive ;
    texte libre de l'assistant -> ses seuls appels d'outils."""
    c = m.get("content") or ""
    if m.get("_nettoye") or m.get("_garder"):
        return None
    if m.get("role") == "tool":
        if (r := _REF_RE.match(c)):
            new = f"[{r.group(1)} archivé : relire('{r.group(1)}', chercher=…)]"
        elif (r := _REPORT_RE.match(c)):
            new = f"[compte rendu de {r.group(1)} : dans le bloc-notes, lire_notes]"
        elif len(c) > 240:
            new = c[:160] + " [... effacé au nettoyage]"
        else:
            return None
    elif m.get("role") == "assistant":
        calls = re.findall(r"<tool_call>.*?</tool_call>", c, re.DOTALL)
        if calls:
            new = "\n".join(calls)
            if new == c:
                return None
        elif len(c) > 240:
            new = c[:160] + " [...]"
        else:
            return None
    else:
        if len(c) <= 600:
            return None
        new = c[:400] + " [... effacé au nettoyage]"
    return {**m, "content": new, "_nettoye": True}


def clean_context(messages: list[dict], count, trigger: int, target: int, keep_last: int = 2) -> int:
    """
    Grand nettoyage (08/10, d'après DeepSeek-V3.2 : à 80 % de la fenêtre, effacer l'historique des outils
    battait le résumé) : seulement au-delà de `trigger` tokens, et alors en une fois jusqu'à `target` :
      1. tout l'ancien historique est réduit à l'essentiel : résultats -> référence d'archive (relire),
         comptes rendus -> renvoi au bloc-notes, texte de l'assistant -> ses appels d'outils ;
      2. s'il le faut, les plus anciens échanges sont retirés.
    Le système, la 1re demande, les messages marqués `_garder` (la tâche en cours d'un agent : 08/10, elle avait
    été effacée et l'agent est parti lire d'autres pages) et les `keep_last` derniers messages restent intacts. Ensuite la conversation ne
    fait que s'allonger : le cache reprend tout jusqu'au prochain nettoyage. Retourne le nombre de messages
    touchés (0 : rien à faire).
    """
    if count(messages) <= trigger:
        return 0
    start = 2 if len(messages) > 1 and messages[0].get("role") == "system" else 1
    end = len(messages) - keep_last
    touched = 0
    for i in range(start, end):
        new = _stub(messages[i])
        if new is not None:
            messages[i] = new
            touched += 1
    while count(messages) > target:
        i = _oldest_removable(messages, start, len(messages) - keep_last)
        if i is None:
            break
        del messages[i]
        touched += 1
    return touched


def _oldest_removable(messages: list[dict], start: int, end: int) -> int | None:
    for i in range(start, end):
        if not messages[i].get("_garder"):
            return i
    return None


def compact_messages(messages: list[dict], count, budget: int, keep_last: int = 4, keep_chars: int = 300) -> int:
    """
    Fait tenir une conversation dans `budget` tokens (count(messages) -> tokens) : d'abord les anciens
    messages longs (résultats d'outils surtout) sont raccourcis, puis les plus anciens échanges sont retirés.
    Le message système, la 1re demande et les `keep_last` derniers messages sont gardés tels quels.
    Modifie la liste en place ; retourne le nombre de changements. (06/10 : au-delà du cache statique,
    moelab repasse en passe normale, qui ne tient pas dans 6 Go : « CUDA out of memory ».)
    """
    changes = 0
    start = 2 if len(messages) > 1 and messages[0].get("role") == "system" else 1
    while count(messages) > budget:
        end = len(messages) - keep_last
        long_ = [i for i in range(start, end) if len(messages[i].get("content") or "") > keep_chars + 80
                 and not messages[i].get("_raccourci") and not messages[i].get("_garder")]
        if long_:
            i = long_[0]
            c = messages[i]["content"]
            messages[i] = {**messages[i], "content": c[:keep_chars] + f"\n[... raccourci : {len(c)} caractères, ancien]",
                           "_raccourci": True}
        elif (i := _oldest_removable(messages, start, end)) is not None:
            del messages[i]
        else:
            break
        changes += 1
    return changes


def moelab_generate_fn(model, tokenizer, max_new_tokens: int = 768, temperature: float = 0.0, streamer=None,
                       timings: list | None = None, prefix_cache: bool = True):
    """
    Fonction de génération pour Agent, sur un modèle chargé par moelab. P1 : l'historique est
    repris du cache d'un tour à l'autre. P9 (prefix_cache) : le message système et les outils
    sont calculés une fois, gardés en mémoire et dans <store>/prefix_cache, rechargés ensuite
    (même après un redémarrage).
    """
    from pathlib import Path

    from .generate import generate

    rt = model.moelab_runtime
    # tokens de contrôle à retirer du texte (fin de tour, remplissage…) ; <tool_call> n'en fait pas partie
    specials = {tokenizer.eos_token, tokenizer.pad_token, tokenizer.unk_token, "<|im_end|>", "<|endoftext|>",
                "<|eot_id|>", *getattr(tokenizer, "all_special_tokens", [])}
    specials = sorted((s for s in specials if s and "tool_call" not in s), key=len, reverse=True)
    prefix_dir = str(Path(rt.store.path) / "prefix_cache") if prefix_cache else None
    known: dict = {}

    def fn(messages, tools):
        ids = render(tokenizer, messages, tools)
        key = (messages[0].get("content") if messages else None, json.dumps(tools, sort_keys=True))
        if key not in known:
            known[key] = prefix_length(tokenizer, messages, tools, ids) if prefix_cache else None
        ids = ids.to(model.moelab_device)
        tm: dict = {}
        new = generate(model, ids, max_new_tokens, temperature=temperature, top_p=0.9, streamer=streamer,
                       reuse=rt.cfg.kv_reuse, timings=tm, prefix_len=known[key], prefix_dir=prefix_dir)
        if timings is not None:
            timings.append({**tm, "new_tokens": len(new)})
        text = tokenizer.decode(new, skip_special_tokens=False)
        for s in specials:
            text = text.replace(s, "")
        return text.strip()

    return fn
