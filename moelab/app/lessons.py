"""
Mémoire de leçons des agents (08/10).

Quand l'utilisateur corrige un run (« Corriger » sur la réponse) ou garde un problème trouvé par le
vérificateur, une leçon est enregistrée dans <data>/lessons.json. Aux runs suivants, les leçons
proches de l'objectif (mots en commun, choisis par le code : aucun appel au modèle) sont données
à l'étape des critères, à l'orchestrateur et au vérificateur. Le système s'améliore avec l'usage
sans qu'il faille coder chaque cas.

    {"id", "texte", "mots", "source": utilisateur | verificateur, "objectif", "cree", "utilisations"}
"""

from __future__ import annotations

import re
import time
import unicodedata
import uuid
from pathlib import Path

from .engine import read_json, write_json

_STOP = set("""
avec dans pour plus moins sans sous entre leur leurs elle elles nous vous ils sont etre avoir fait faire
cette cela ceci tout tous toute toutes chaque comme mais donc alors aussi tres bien quand quel quelle
quels quelles celui celle ceux depuis apres avant pendant encore deja ainsi chez puis votre notre
fichier dossier ecris ecrire mettre donne donner the and for with from that this what which
""".split())


def words(text: str) -> set[str]:
    t = unicodedata.normalize("NFKD", (text or "").lower()).encode("ascii", "ignore").decode()
    out = set()
    for w in re.findall(r"[a-z0-9][a-z0-9._-]{2,}", t):
        w = w.strip("._-")
        if len(w) < 4 or w in _STOP or w.isdigit():
            continue
        out.add(w[:-1] if w.endswith("s") and len(w) > 4 else w)  # pluriel simple
    return out


class LessonBook:
    def __init__(self, data_dir):
        self.path = Path(data_dir) / "lessons.json"
        self.data = read_json(self.path, {"lecons": []})
        self.data.setdefault("lecons", [])

    def save(self) -> None:
        write_json(self.path, self.data)

    def list(self) -> list[dict]:
        return sorted(self.data["lecons"], key=lambda x: -x.get("cree", 0))

    def add(self, texte: str, objectif: str = "", source: str = "utilisateur") -> dict:
        texte = (texte or "").strip()
        if not texte:
            raise ValueError("leçon vide")
        d = {"id": uuid.uuid4().hex[:8], "texte": texte[:600], "mots": sorted(words(texte) | words(objectif)),
             "source": source, "objectif": (objectif or "")[:300], "cree": time.time(), "utilisations": 0}
        self.data["lecons"].append(d)
        self.save()
        return d

    def remove(self, lid: str) -> bool:
        n = len(self.data["lecons"])
        self.data["lecons"] = [x for x in self.data["lecons"] if x["id"] != lid]
        self.save()
        return len(self.data["lecons"]) < n

    def relevant(self, objectif: str, k: int = 5, count: bool = True) -> list[dict]:
        """Les leçons qui partagent le plus de mots avec l'objectif (au moins un)."""
        ow = words(objectif)
        scored = []
        for x in self.data["lecons"]:
            score = len(ow & set(x.get("mots", [])))
            if score:
                scored.append((score, x.get("cree", 0), x))
        out = [x for _, _, x in sorted(scored, key=lambda t: (-t[0], -t[1]))[:k]]
        if count and out:
            for x in out:
                x["utilisations"] = x.get("utilisations", 0) + 1
            self.save()
        return out
