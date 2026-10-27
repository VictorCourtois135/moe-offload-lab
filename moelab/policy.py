"""
Décisions : en quelle précision calculer chaque expert (HOBBIT), et comment
répartir les experts manquants entre transfert vers le GPU et calcul direct sur
CPU (q* de FreeToken).

Les précisions sont des NIVEAUX : 0 = la plus précise du store (fp16), 1 = la
suivante (int8 si le store en a trois), 2 = la plus basse (int4). Avec un store à
deux précisions, le niveau 1 est ramené à la plus basse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

MODES = ("hi", "lo", "hobbit")


@dataclass
class PrecisionPolicy:
    """
    mode :
      - "hi"     : tout en haute précision (le modèle d'origine, référence de qualité)
      - "lo"     : tout en basse précision (quantification statique uniforme, ce que
                   fait Colibrì aujourd'hui)
      - "hobbit" : précision choisie par token et par expert selon son importance
    t1 : seuil HOBBIT. Score d'"unimportance" s <= t1 -> haute précision, sinon basse.
    t_mid : (E2, doc 18) seuil du niveau intermédiaire : t1 < s <= t_mid -> niveau 1
         (int8 dans un store fp16/int8/int4). None = pas de niveau intermédiaire.
    t2 : seuil de saut. s > t2 -> l'expert n'est pas calculé du tout. None = jamais
         (par défaut : sauter un expert change les décisions du routeur, on ne le
         fait que si on le demande explicitement).
    t1_layers : (E1) un seuil t1 par couche, calibré par measure_routing.py ; les
         couches absentes gardent t1.
    miss_t1 : (E3) seuil plus strict appliqué aux experts ABSENTS de la VRAM : un
         expert demandé en haute précision mais manquant passe au niveau suivant si
         s > miss_t1. C'est l'idée d'origine de HOBBIT : la basse précision sert
         surtout à rendre les défauts de cache moins chers. None = désactivé.
    protect_layers : couches toujours en haute précision (idée DyMoE / pattern en U
         mesuré dans docs/07).
    upgrade : un fp16 déjà en cache peut servir une demande int4 (voir allow_upgrade).
    hi_vote : un expert est chargé UNE fois par couche, pour tous les tokens qui
         l'ont choisi. Il prend le niveau le plus précis demandé par au moins cette
         fraction de ses tokens. En génération (1 token), c'est exactement la règle de
         HOBBIT ; en prefill (des centaines de tokens), sans ce vote, presque tous les
         experts finiraient en haute précision (il suffirait d'un token).
    """

    mode: str = "hobbit"
    t1: float = 0.6
    t2: float | None = None
    protect_layers: frozenset[int] = field(default_factory=frozenset)
    hi_vote: float = 0.5
    upgrade: bool = True
    t_mid: float | None = None
    t1_layers: dict[int, float] | None = None
    miss_t1: float | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode inconnu : {self.mode} (choix : {', '.join(MODES)})")
        self.protect_layers = frozenset(self.protect_layers)
        if self.t2 is not None and self.t2 < self.t1:
            raise ValueError("t2 doit être >= t1")
        if self.t_mid is not None and self.t_mid < self.t1:
            raise ValueError("t_mid doit être >= t1")
        if self.t1_layers is not None:
            self.t1_layers = {int(k): float(v) for k, v in self.t1_layers.items()}

    def t1_for(self, layer: int) -> float:
        return self.t1_layers.get(layer, self.t1) if self.t1_layers else self.t1

    # ------------------------------------------------------------------
    # Score d'importance
    # ------------------------------------------------------------------
    def unimportance(self, weights: torch.Tensor) -> torch.Tensor:
        """
        weights : (T, k) poids du routeur pour les k experts choisis par chaque token.
        Retourne s (T, k) : pour chaque expert, la somme des poids NORMALISÉS des
        experts mieux classés que lui (équation 2 de HOBBIT). Le mieux classé a s = 0.
        """
        w = weights.float()
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sw, order = w.sort(dim=-1, descending=True, stable=True)
        s_sorted = sw.cumsum(dim=-1) - sw
        return torch.empty_like(s_sorted).scatter_(-1, order, s_sorted)

    @staticmethod
    def unimportance_one(weights: list[float]) -> list[float]:
        """Même calcul pour UN token, en Python pur (optimisation A2)."""
        total = sum(weights) or 1e-12
        s = [0.0] * len(weights)
        acc = 0.0
        for j in sorted(range(len(weights)), key=lambda j: -weights[j]):  # tri stable, comme unimportance()
            s[j] = acc
            acc += weights[j] / total
        return s

    # ------------------------------------------------------------------
    # Niveaux de précision
    # ------------------------------------------------------------------
    def _fixed(self, layer: int) -> int | None:
        if self.mode == "hi" or layer in self.protect_layers:
            return 0
        if self.mode == "lo":
            return 2
        return None

    def levels(self, layer: int, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retourne (niveau, skip) de forme (T, k) : niveau 0/1/2, skip booléen."""
        T, k = weights.shape
        none = torch.zeros(T, k, dtype=torch.bool)
        fixed = self._fixed(layer)
        if fixed is not None:
            return torch.full((T, k), fixed, dtype=torch.int8), none
        s = self.unimportance(weights)
        lvl = torch.full((T, k), 2, dtype=torch.int8)
        if self.t_mid is not None:
            lvl[s <= self.t_mid + 1e-6] = 1
        lvl[s <= self.t1_for(layer) + 1e-6] = 0
        skip = (s > self.t2 + 1e-6) & (lvl != 0) if self.t2 is not None else none
        return lvl, skip

    def levels_one(self, layer: int, weights: list[float]) -> tuple[list[int], list[bool], list[float]]:
        """Même règle pour UN token, en Python pur (A2). Retourne aussi s (pour E3)."""
        k = len(weights)
        fixed = self._fixed(layer)
        if fixed is not None:
            return [fixed] * k, [False] * k, [0.0] * k
        s = self.unimportance_one(weights)
        t1 = self.t1_for(layer) + 1e-6
        tm = None if self.t_mid is None else self.t_mid + 1e-6
        lvl = [0 if v <= t1 else (1 if tm is not None and v <= tm else 2) for v in s]
        if self.t2 is None:
            return lvl, [False] * k, s
        return lvl, [v > self.t2 + 1e-6 and lv != 0 for v, lv in zip(s, lvl)], s

    def decide(self, layer: int, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(need_hi, skip), deux masques (T, k) : la vue à deux niveaux de `levels`."""
        lvl, skip = self.levels(layer, weights)
        return lvl == 0, skip

    def decide_one(self, layer: int, weights: list[float]) -> tuple[list[bool], list[bool]]:
        lvl, skip, _ = self.levels_one(layer, weights)
        return [v == 0 for v in lvl], skip

    def expert_level(self, token_levels: torch.Tensor) -> int:
        """Vote : le niveau le plus précis demandé par au moins hi_vote des tokens de l'expert."""
        n = token_levels.numel()
        for level in (0, 1):
            if (token_levels <= level).sum().item() >= (self.hi_vote - 1e-9) * n:
                return level
        return 2

    def expert_is_hi(self, votes: torch.Tensor) -> bool:
        """votes : booléens need_hi des tokens qui ont choisi cet expert."""
        return bool(votes.float().mean() >= self.hi_vote - 1e-9)

    def describe(self) -> str:
        if self.mode != "hobbit":
            d = self.mode
        else:
            parts = [f"t1={self.t1}" if not self.t1_layers else "t1 par couche"]
            if self.t_mid is not None:
                parts.append(f"t_mid={self.t_mid}")
            if self.t2 is not None:
                parts.append(f"t2={self.t2}")
            if self.miss_t1 is not None:
                parts.append(f"défaut de cache : t1={self.miss_t1}")
            d = f"hobbit({', '.join(parts)})"
        if self.protect_layers:
            d += f" + couches protégées {sorted(self.protect_layers)}"
        return d

    @property
    def allow_upgrade(self) -> bool:
        """
        Une version plus précise déjà en cache peut-elle servir une demande basse
        précision ? Oui pour hobbit (gain de qualité gratuit), non pour "lo", qui doit
        reproduire exactement une quantification statique. `upgrade=False` l'interdit
        aussi pour hobbit : la qualité mesurée est alors celle de la règle seule,
        indépendante du contenu des caches (mesure de qualité du benchmark).
        """
        return self.upgrade and self.mode != "lo"


def level_to_precision(level: int, precisions: list[str]) -> str:
    """Niveau (0 = plus précis) -> précision disponible dans le store (liste, plus précise d'abord)."""
    if len(precisions) >= 3:
        return precisions[min(level, len(precisions) - 1)]
    return precisions[0] if level == 0 else precisions[-1]


def qstar(m: int, b_pcie: float, b_host: float, min_fill: int = 1) -> int:
    """
    FreeToken (§3.2) : parmi m experts manquants, combien transférer vers le GPU.

    Transfert et calcul CPU lisent la même RAM. Pendant que le PCIe est saturé, il
    reste au CPU b_host - b_pcie. On égalise :
        q * S / b_pcie = (m - q) * S / (b_host - b_pcie)   ->   q* = m * b_pcie / b_host
    Si b_host <= b_pcie, tout est transféré. On garde au moins `min_fill` transfert
    pour que le cache GPU continue de se réchauffer.
    """
    if m <= 0:
        return 0
    if b_host <= b_pcie:
        return m
    return min(m, max(min_fill, round(m * b_pcie / b_host)))
