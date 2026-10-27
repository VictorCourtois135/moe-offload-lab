"""
Vérification automatique des faits (06/10, 6e run des agents).

Le modèle local invente quand une information lui manque, même avec la consigne de ne pas le
faire : des dates de sortie plausibles mais fausses, des nouveautés et des numéros de version
qui n'étaient dans aucune page. Ce module ne juge pas le sens d'une phrase ; il vérifie ce qui
se vérifie mécaniquement, les dates et les chiffres :

    18 septembre 2026, Sep 02, 2026-09-02, 02/09/2026   -> (jour, mois)
    12.4, 3,35                                         -> nombre décimal
    20 %, 100x, 12×                                    -> pourcentage, multiplicateur

Un élément d'un texte produit par le modèle (note, compte rendu, fichier écrit, réponse finale)
est « retrouvé » s'il apparaît, sous une forme ou une autre, dans ce qui a été réellement lu
pendant le run (résultats des outils), dans l'objectif de l'utilisateur ou la date du jour.
"""

from __future__ import annotations

import re

_MONTHS = {
    1: ("janvier", "janv", "january", "jan"),
    2: ("février", "fevrier", "févr", "fevr", "fév", "fev", "february", "feb"),
    3: ("mars", "march", "mar"),
    4: ("avril", "avr", "april", "apr"),
    5: ("mai", "may"),
    6: ("juin", "june", "jun"),
    7: ("juillet", "juil", "july", "jul"),
    8: ("août", "aout", "aoû", "august", "aug"),
    9: ("septembre", "sept", "september", "sep"),
    10: ("octobre", "october", "oct"),
    11: ("novembre", "november", "nov"),
    12: ("décembre", "decembre", "déc", "dec", "december"),
}
_NAME_TO_MONTH = {n: m for m, names in _MONTHS.items() for n in names}
_MONTH_RE = "|".join(sorted((re.escape(n) for n in _NAME_TO_MONTH), key=len, reverse=True))

_DAY_MONTH = re.compile(rf"(?<![\w.])(\d{{1,2}})(?:er|st|nd|rd|th)?\s+({_MONTH_RE})\.?(?![a-zà-ÿ])", re.I)
_MONTH_DAY = re.compile(rf"(?<![\w])({_MONTH_RE})\.?\s+(\d{{1,2}})(?![\d:])(?:st|nd|rd|th)?", re.I)
_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_DMY = re.compile(r"(?<![\d.])(\d{1,2})[/.](\d{1,2})[/.](\d{4})(?!\d)")
_DECIMAL = re.compile(r"(?<![\w.,])(\d+[.,]\d+)(?![\w]|[.,]\d)")
_PERCENT = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s*%")
_MULT = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s*[x×](?![a-z])(?!\s*\d)", re.I)  # « 12 × 12 » : un produit
_VERSION = re.compile(r"(?<![\w.])v?(\d+(?:\.\d+){2,})(?![\w])")


def _num(s: str) -> str:
    s = s.replace(",", ".")
    return s.rstrip("0").rstrip(".") if "." in s else s


def items(text: str) -> list[tuple[tuple, str]]:
    """[(clé, texte trouvé)] : dates (jour, mois), décimaux, pourcentages, multiplicateurs."""
    out: list[tuple[tuple, str]] = []
    if not text:
        return out
    date_spans = []
    for m in _ISO.finditer(text):
        d, mo = int(m.group(3)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            out.append((("date", d, mo), m.group(0)))
            date_spans.append(m.span())
    for m in _DMY.finditer(text):
        d, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            out.append((("date", d, mo), m.group(0)))
            date_spans.append(m.span())
    for m in _DAY_MONTH.finditer(text):
        d, mo = int(m.group(1)), _NAME_TO_MONTH[m.group(2).lower()]
        if 1 <= d <= 31:
            out.append((("date", d, mo), m.group(0)))
    for m in _MONTH_DAY.finditer(text):
        d, mo = int(m.group(2)), _NAME_TO_MONTH[m.group(1).lower()]
        if 1 <= d <= 31:
            out.append((("date", d, mo), m.group(0)))
    for m in _PERCENT.finditer(text):
        out.append((("pct", _num(m.group(1))), m.group(0).strip()))
    for m in _MULT.finditer(text):
        out.append((("mult", _num(m.group(1))), m.group(0).strip()))
    for m in _DECIMAL.finditer(text):
        if any(a <= m.start() < b for a, b in date_spans):
            continue
        out.append((("num", _num(m.group(1))), m.group(0)))
    return out


def keys(text: str) -> set:
    """Clés d'une source. Un pourcentage ou un multiplicateur y vaut aussi pour le nombre seul, et
    inversement (« 12 times » ne serait pas reconnu, « 12x » et « 12 x » oui)."""
    out = set()
    for m in _VERSION.finditer(text or ""):  # « v2.14.0 » vaut aussi pour « 2.14 »
        parts = m.group(1).split(".")
        for i in range(2, len(parts) + 1):
            out.add(("num", _num(".".join(parts[:i]))))
    for k, _ in items(text):
        out.add(k)
        if k[0] in ("pct", "mult"):
            out.add(("num", k[1]))
    return out


class FactCheck:
    """Accumule ce qui a été lu (sources) et dit ce qu'un texte avance sans l'avoir lu."""

    def __init__(self, *sources: str):
        self.known: set = set()
        for s in sources:
            self.add(s)

    def add(self, text: str) -> None:
        self.known |= keys(text or "")

    def missing(self, text: str) -> list[str]:
        """Éléments du texte non retrouvés dans les sources (sans doublons, dans l'ordre)."""
        seen, out = set(), []
        for k, raw in items(text or ""):
            if k in self.known or (k[0] in ("pct", "mult") and ("num", k[1]) in self.known):
                continue
            if k not in seen:
                seen.add(k)
                out.append(raw)
        return out


# ----------------------------------------------------------------------------
# Dates du futur (13/10) : « les trois dernières versions » écrites avec une version annoncée pour dans
# deux semaines. Ce qui est daté après aujourd'hui est prévu, pas sorti.
# ----------------------------------------------------------------------------
_Y_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_Y_DMY = re.compile(r"(?<![\d.])(\d{1,2})[/.](\d{1,2})[/.](\d{4})(?!\d)")
_Y_DAY_MONTH = re.compile(rf"(?<![\w.])(\d{{1,2}})(?:er|st|nd|rd|th)?\s+({_MONTH_RE})\.?,?\s+(\d{{4}})(?!\d)", re.I)
_Y_MONTH_DAY = re.compile(rf"(?<![\w])({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})(?!\d)", re.I)


def full_dates(text: str) -> list[tuple]:
    """[(date, texte trouvé)] pour les dates complètes (avec l'année)."""
    import datetime as dt

    out = []

    def add(y, m, d, raw, pos=0):
        try:
            out.append((pos, dt.date(int(y), int(m), int(d)), raw))
        except ValueError:
            pass

    for m in _Y_ISO.finditer(text or ""):
        add(m.group(1), m.group(2), m.group(3), m.group(0), m.start())
    for m in _Y_DMY.finditer(text or ""):
        add(m.group(3), m.group(2), m.group(1), m.group(0), m.start())
    for m in _Y_DAY_MONTH.finditer(text or ""):
        add(m.group(3), _NAME_TO_MONTH[m.group(2).lower()], m.group(1), m.group(0), m.start())
    for m in _Y_MONTH_DAY.finditer(text or ""):
        add(m.group(3), _NAME_TO_MONTH[m.group(1).lower()], m.group(2), m.group(0), m.start())
    return [(d, raw) for _, d, raw in sorted(out, key=lambda x: x[0])]


def future_dates(text: str, today) -> list[str]:
    """Dates complètes du texte postérieures à `today` (sans doublons, dans l'ordre)."""
    return list(dict.fromkeys(raw for d, raw in full_dates(text) if d > today))


def future_warning(text: str, today) -> str | None:
    fut = future_dates(text, today)
    if not fut:
        return None
    return (f"NON ÉCRIT (vérification, une seule fois) : le contenu cite des dates postérieures à aujourd'hui "
            f"({today.isoformat()}) : {', '.join(fut)}. Ce qui est daté du futur est prévu, pas encore sorti. Si la "
            "demande porte sur ce qui existe déjà (« dernières versions », « récents »…), retire ces éléments et "
            "remplace-les par les bons ; sinon réécris le même contenu, il sera accepté.")
