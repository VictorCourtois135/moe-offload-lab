"""
Caches d'experts (un par niveau de mémoire : VRAM, RAM) et historique d'usage persistant.

Score LRFU (Least Recently/Frequently Used, Lee et al. 2001) — c'est le "LFRU" de
Colibrì. Chaque entrée garde un compteur f qui décroît avec le temps :

    à chaque accès au temps t :   f <- f * gamma^(t - t_dernier) + 1
    score au temps "maintenant" : f * gamma^(maintenant - t_dernier)

gamma = 0.5^(1/demi_vie). Un seul paramètre couvre tout le spectre : demi-vie
courte -> se comporte comme LRU (seule la récence compte) ; demi-vie longue ->
comme LFU (la fréquence domine). Le "temps" avance d'un cran par passe avant du
modèle (un token en génération).

Clé d'une entrée : (couche, expert, précision). Les versions haute et basse
précision d'un même expert sont des entrées distinctes, avec leur propre score :
c'est l'équivalent du LHU de HOBBIT (on suit séparément l'usage en haute précision).

Épinglage (Colibrì) : une entrée épinglée n'est jamais évincée par le chargement
à la demande. `repin()` échange des entrées épinglées froides contre des entrées
non épinglées plus chaudes, avec une hystérésis (25%) et au plus 4 échanges par
appel, pour éviter que le cache change d'avis en permanence.

Éviction « least-stale » (S3, 02/10, d'après SpecMD, Apple 2026) : `eviction="stale"`.
Un modèle MoE parcourt ses couches toujours dans le même ordre ; pendant une passe,
les experts utilisés ou préchargés POUR CETTE PASSE (touchés au temps courant) vont
encore servir (couches suivantes, prefetch en vol), alors que ceux des passes
précédentes ne serviront que si le routeur les choisit à nouveau. On évince donc
d'abord les entrées « périmées » (dernier accès avant la passe courante), par score
LRFU, puis seulement les entrées de la passe courante. SpecMD mesure jusqu'à 85 fois
moins de « collisions » (un expert préchargé évincé avant d'avoir servi) qu'en LRU.

Éviction en O(log n) (03/10, Qwen3-Next) : trier toutes les entrées à chaque éviction
coûtait 0,3 à 0,5 ms avec 1 200 à 2 500 entrées, plusieurs fois par couche. L'ordre des
scores LRFU ne dépend pas de l'instant où on les compare : log(score) = log f + (clock - t)
log gamma, et clock est le même pour tous. La clé log f - t log gamma est donc fixe tant que
l'entrée n'est pas touchée : un tas (heapq) avec invalidation paresseuse donne la même
victime que le tri, en O(log n).
"""

from __future__ import annotations

import heapq
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

Key = tuple[int, int, str]  # (couche, expert, précision)


@dataclass
class Entry:
    key: Key
    blob: torch.Tensor
    nbytes: int
    f: float = 0.0
    t: int = 0
    pinned: bool = False
    event: object = None  # événement CUDA si la copie vers ce niveau est encore en cours
    prefetched: bool = False  # chargée par anticipation et pas encore utilisée
    ver: int = 0  # version de la clé de tas (TierCache._push)


@dataclass
class TierStats:
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    evictions: int = 0
    rejected: int = 0
    bytes_in: int = 0


class TierCache:
    def __init__(self, name: str, device: torch.device | str, budget_bytes: int, rank_of, half_life: float = 32.0,
                 pin_fraction: float = 0.5, on_evict=None, eviction: str = "lrfu"):
        if eviction not in ("lrfu", "stale"):
            raise ValueError(f"politique d'éviction inconnue : {eviction} (lrfu ou stale)")
        self.eviction = eviction
        self.listener = None  # A5 : prévenu de chaque insertion / éviction (table GPU des experts résidents)
        self.name = name
        self.on_evict = on_evict  # appelé avec le tampon de chaque entrée évincée (retour au pool RAM)
        self.device = torch.device(device)
        self.budget = int(budget_bytes)
        self.rank_of = rank_of  # précision -> rang (0 = la plus précise)
        self.gamma = 0.5 ** (1.0 / half_life)
        self.pin_budget = int(self.budget * pin_fraction)
        self.entries: dict[Key, Entry] = {}
        self.by_expert: dict[tuple[int, int], set[str]] = {}
        self.used = 0
        self.pinned_bytes = 0
        self.clock = 0
        self.stats = TierStats()
        self._lg = math.log(self.gamma)
        self._heap: list = []
        self._seq = 0

    # -- score LRFU -------------------------------------------------------
    def _push(self, e: Entry) -> None:
        """(Re)place l'entrée dans le tas des victimes possibles (les épinglées n'y sont pas)."""
        e.ver += 1
        if e.pinned:
            return
        self._seq += 1
        heapq.heappush(self._heap, (math.log(e.f) - e.t * self._lg, self._seq, e.ver, e.key))
        if len(self._heap) > 4 * len(self.entries) + 64:  # trop d'éléments périmés : on reconstruit
            self._heap = [(math.log(x.f) - x.t * self._lg, i, x.ver, x.key)
                          for i, x in enumerate(self.entries.values()) if not x.pinned]
            heapq.heapify(self._heap)

    def score(self, e: Entry) -> float:
        return e.f * self.gamma ** (self.clock - e.t)

    def touch(self, e: Entry) -> None:
        e.f = self.score(e) + 1.0
        e.t = self.clock
        if e.key in self.entries:
            self._push(e)

    def tick(self, n: int = 1) -> None:
        self.clock += n

    def _victim_key(self, e: Entry):
        """Ordre d'éviction : score LRFU croissant ; avec "stale", les entrées de la passe courante en dernier."""
        if self.eviction == "stale":
            return (e.t >= self.clock, self.score(e))
        return self.score(e)

    # -- recherche ----------------------------------------------------------
    def lookup(self, layer: int, expert: int, need: str, exact: bool = False, count: bool = True) -> Entry | None:
        """
        Meilleure entrée disponible pour (couche, expert) d'une précision au moins
        aussi bonne que `need` (une version fp16 peut servir une demande int4, pas
        l'inverse). `exact=True` : uniquement la précision demandée.
        `count=False` : simple vérification, sans compter de hit/miss (prefetch).
        """
        precs = self.by_expert.get((layer, expert))
        if not precs:
            self.stats.misses += count
            return None
        if exact:
            best = need if need in precs else None
        else:
            ok = [p for p in precs if self.rank_of(p) <= self.rank_of(need)]
            best = min(ok, key=self.rank_of) if ok else None
        if best is None:
            self.stats.misses += count
            return None
        self.stats.hits += count
        return self.entries[(layer, expert, best)]

    def contains(self, key: Key) -> bool:
        return key in self.entries

    # -- insertion / éviction -------------------------------------------
    def insert(self, key: Key, blob: torch.Tensor, pinned: bool = False, protect: frozenset = frozenset(),
               event=None) -> Entry | None:
        """
        Ajoute une entrée (évince les entrées non épinglées de plus faible score si
        besoin). Retourne None si la place ne peut pas être libérée.
        """
        if key in self.entries:
            e = self.entries[key]
            self.touch(e)
            return e
        need = blob.numel() * blob.element_size()
        if pinned and self.pinned_bytes + need > self.pin_budget:
            pinned = False
        if need > self.budget - self.pinned_bytes:
            self.stats.rejected += 1
            return None
        if self.used + need > self.budget:
            chosen = self._victims(need, protect)
            if chosen is None:
                self.stats.rejected += 1
                return None
            for v in chosen:
                self.evict(v.key)
        e = Entry(key, blob, need, pinned=pinned, event=event)
        self.touch(e)
        self.entries[key] = e
        self._push(e)
        self.by_expert.setdefault(key[:2], set()).add(key[2])
        self.used += need
        if pinned:
            self.pinned_bytes += need
        self.stats.inserts += 1
        self.stats.bytes_in += need
        if self.listener is not None:
            self.listener.added(e)
        return e

    def _victims(self, need: int, protect) -> list[Entry] | None:
        """Entrées à évincer (score croissant) pour faire `need` octets de place ; None si impossible."""
        if self.eviction == "stale":  # ordre qui dépend de l'instant : tri complet (rejeté le 02/10, gardé pour mesure)
            victims = sorted((e for e in self.entries.values() if not e.pinned and e.key not in protect),
                             key=self._victim_key)
        else:
            victims = None
        freed, chosen, skipped = 0, [], []
        if victims is not None:
            for v in victims:
                if self.used - freed + need <= self.budget:
                    break
                chosen.append(v)
                freed += v.nbytes
        else:
            seen = set()
            while self.used - freed + need > self.budget and self._heap:
                item = heapq.heappop(self._heap)
                _, _, ver, key = item
                e = self.entries.get(key)
                if e is None or e.ver != ver or e.pinned or key in seen:
                    continue  # périmé (touchée, évincée ou épinglée depuis)
                if key in protect:
                    skipped.append(item)
                    continue
                seen.add(key)
                chosen.append(e)
                freed += e.nbytes
            for item in skipped:
                heapq.heappush(self._heap, item)
        if self.used - freed + need > self.budget:
            for e in chosen:  # rien n'est évincé : les candidats retournent dans le tas
                self._push(e)
            return None
        return chosen

    def evict(self, key: Key) -> None:
        e = self.entries.pop(key)
        if e.event is not None and self.device.type == "cuda":
            # Une copie encore en vol vers ce tampon : on attend qu'elle finisse avant de le rendre.
            torch.cuda.current_stream().wait_event(e.event)
        precs = self.by_expert[key[:2]]
        precs.discard(key[2])
        if not precs:
            del self.by_expert[key[:2]]
        self.used -= e.nbytes
        if e.pinned:
            self.pinned_bytes -= e.nbytes
        self.stats.evictions += 1
        if self.listener is not None:
            self.listener.removed(e)
        if self.on_evict is not None:
            self.on_evict(e.blob)

    def clear(self) -> None:
        for k in list(self.entries):
            self.evict(k)
        self._heap = []
        self.stats = TierStats()

    # -- épinglage --------------------------------------------------------
    def pin(self, key: Key) -> bool:
        e = self.entries.get(key)
        if e is None or e.pinned:
            return e is not None
        if self.pinned_bytes + e.nbytes > self.pin_budget:
            return False
        e.pinned = True
        self.pinned_bytes += e.nbytes
        return True

    def unpin(self, key: Key) -> None:
        e = self.entries.get(key)
        if e is not None and e.pinned:
            e.pinned = False
            self.pinned_bytes -= e.nbytes
            self._push(e)

    def repin(self, hysteresis: float = 0.25, max_swaps: int = 4) -> list[tuple[Key, Key | None]]:
        """
        À appeler entre deux requêtes (point sûr). Épingle les entrées chaudes non
        épinglées, en échange d'entrées épinglées plus froides, seulement si le
        candidat dépasse la victime de `hysteresis` (25%), et au plus `max_swaps` fois.
        Retourne la liste des échanges (candidat, victime ou None si place libre).
        """
        swaps: list[tuple[Key, Key | None]] = []
        cands = sorted((e for e in self.entries.values() if not e.pinned), key=self.score, reverse=True)
        victims = sorted((e for e in self.entries.values() if e.pinned), key=self.score)
        vi = 0
        for c in cands:
            if len(swaps) >= max_swaps:
                break
            if self.pinned_bytes + c.nbytes <= self.pin_budget:
                self.pin(c.key)
                swaps.append((c.key, None))
                continue
            while vi < len(victims) and not victims[vi].pinned:
                vi += 1
            if vi >= len(victims):
                break
            v = victims[vi]
            if self.score(c) <= self.score(v) * (1.0 + hysteresis):
                break  # les candidats suivants sont encore moins chauds
            self.unpin(v.key)
            if self.pin(c.key):
                swaps.append((c.key, v.key))
                vi += 1
            else:
                self.pin(v.key)
                break
        return swaps

    def describe(self) -> str:
        return (f"{self.name}: {self.used / 2**20:.0f}/{self.budget / 2**20:.0f} Mio, "
                f"{len(self.entries)} entrées ({self.pinned_bytes / 2**20:.0f} Mio épinglés)")


class UsageLog:
    """
    Historique d'usage persistant entre les sessions (le `.coli_usage` de Colibrì).
    Compte combien de fois chaque (couche, expert) a été utilisé, et combien de fois
    en haute précision. Au démarrage, les plus utilisés sont préchargés et épinglés.
    """

    def __init__(self, path: str | Path | None, store_id: str):
        self.path = Path(path) if path else None
        self.store_id = store_id
        self.counts: Counter = Counter()
        self.hi_counts: Counter = Counter()
        self.sessions = 0
        self.ignored_reason: str | None = None
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("store_id") != store_id:
                self.ignored_reason = f"historique d'un autre store ({data.get('store_id')}), ignoré"
            else:
                self.counts = Counter({self._key(k): v for k, v in data["counts"].items()})
                self.hi_counts = Counter({self._key(k): v for k, v in data.get("hi_counts", {}).items()})
                self.sessions = data.get("sessions", 0)

    @staticmethod
    def _key(s: str) -> tuple[int, int]:
        a, b = s.split(":")
        return int(a), int(b)

    def record(self, layer: int, expert: int, hi: bool) -> None:
        self.counts[(layer, expert)] += 1
        if hi:
            self.hi_counts[(layer, expert)] += 1

    def score(self, layer: int, expert: int) -> int:
        return self.counts.get((layer, expert), 0)

    def top(self, n: int | None = None) -> list[tuple[int, int]]:
        return [k for k, _ in self.counts.most_common(n)]

    def prefers_hi(self, layer: int, expert: int) -> bool:
        c = self.counts.get((layer, expert), 0)
        return c > 0 and self.hi_counts.get((layer, expert), 0) * 2 >= c

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "store_id": self.store_id,
            "sessions": self.sessions + 1,
            "counts": {f"{l}:{e}": v for (l, e), v in self.counts.items()},
            "hi_counts": {f"{l}:{e}": v for (l, e), v in self.hi_counts.items()},
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.path)
