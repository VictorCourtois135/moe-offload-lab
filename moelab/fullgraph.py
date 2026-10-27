"""
A5 (doc 18, 02/10) : un token entier en UN CUDA graph, décisions prises sur le GPU.

Le profil `lo` du 02/10 : 8,4 ms par token, dont ~0,3 ms par couche à attendre la
lecture du routage par le CPU (16 couches : ~4,8 ms). Le CPU a besoin du routage pour
décider (HOBBIT, caches, disque), donc chaque couche MoE coupe le graph en deux.

Quand tous les experts demandés sont DÉJÀ en VRAM, ces décisions se réduisent à :
la précision voulue (HOBBIT, quelques opérations sur k nombres), puis l'adresse de
l'expert en VRAM. Les deux peuvent se faire sur le GPU :

  - `DeviceTable` : pour chaque précision, une table (couches x experts) des adresses
    des experts résidents en VRAM (0 = absent), tenue à jour par le cache VRAM (chaque
    insertion / éviction) et recopiée vers le GPU avant un token si elle a changé ;
  - `FullMoE.layer` : dans le graph, à la place de la couche MoE : niveaux HOBBIT,
    choix de la précision résidente (même règle que TierCache.lookup), table du noyau
    groupé, noyaux groupés. Un expert absent est compté dans `flag`.

Le token entier (partie dense + 16 couches MoE) devient un seul graph. Après le
replay, le CPU lit une fois `flag` et le routage de toutes les couches (une seule
synchronisation par token, au lieu d'une par couche) :
  - flag = 0 : tout était en VRAM, le résultat est exact ; le CPU met à jour ses
    compteurs (scores des caches, historique, statistiques) d'après le routage lu ;
  - flag > 0 : un expert manquait. Le token est refait par le chemin normal (graphs
    par segments, qui chargent ce qui manque). Le cache KV statique est réécrit aux
    mêmes positions (rien à défaire) ; les états récurrents (attention linéaire de
    Qwen3-Next) sont restaurés depuis une copie faite au début du graph.

Le chemin complet n'est tenté que si le token précédent n'a eu aucun défaut de cache
(régime `lo`, ou hobbit quand tout est chaud) : en régime de défauts fréquents, il ne
coûte rien de plus. Pas de prefetch dans ce chemin : il ne sert à rien quand tout est là.
"""

from __future__ import annotations

import torch

from .kernels import _Plan, _pow2_groups, _pow2_groups8
from .policy import level_to_precision
from .store import PRECISIONS


class DeviceTable:
    """Adresses des experts résidents en VRAM, lisibles par le GPU : (précisions, couches, experts) int64."""

    def __init__(self, rt):
        st = rt.store
        self.rt = rt
        self.precs = list(st.precisions)
        dev = rt.fast.device
        pin = dev.type == "cuda"
        self.host = torch.zeros(len(self.precs), len(st.layers), st.num_experts, dtype=torch.int64, pin_memory=pin)
        self.np = self.host.numpy()
        self.dev = torch.zeros(self.host.shape, dtype=torch.int64, device=dev)
        self.dirty = True
        self.ev = None
        self.waiting: dict = {}  # entrées dont la copie RAM -> VRAM est encore en vol
        self.uploads = 0
        for e in rt.fast.entries.values():
            self.added(e)

    def _guard(self) -> None:
        """La table hôte est peut-être encore en cours de copie vers le GPU : on attend avant de la modifier."""
        if self.ev is not None:
            self.ev.synchronize()
            self.ev = None

    def added(self, e) -> None:
        self._guard()
        l, x, p = e.key
        self.np[self.precs.index(p), l, x] = e.blob.data_ptr()
        self.dirty = True
        if e.event is not None:
            self.waiting[e.key] = e

    def removed(self, e) -> None:
        self._guard()
        l, x, p = e.key
        self.np[self.precs.index(p), l, x] = 0
        self.dirty = True
        self.waiting.pop(e.key, None)

    def upload(self) -> None:
        """Avant un replay : copies en vol attendues (sur le stream courant), table recopiée si elle a changé."""
        for e in self.waiting.values():
            if e.event is not None:
                self.rt._ready(e.blob, e.event)
                e.event = None
        self.waiting.clear()
        if self.dirty:
            if self.dev.device.type == "cuda":
                self.dev.copy_(self.host, non_blocking=True)
                self.ev = torch.cuda.Event()
                self.ev.record()
            else:
                self.dev.copy_(self.host)
            self.dirty = False
            self.uploads += 1


def gpu_levels(policy, layer: int, wf: torch.Tensor):
    """PrecisionPolicy.levels pour UN token, en opérations de tenseurs (capturables). wf : (k,) float32."""
    k = wf.shape[0]
    fixed = policy._fixed(layer)
    none = torch.zeros(k, dtype=torch.bool, device=wf.device)
    if fixed is not None:
        return torch.full((k,), fixed, dtype=torch.int64, device=wf.device), none
    s = policy.unimportance(wf.unsqueeze(0))[0]
    lvl = torch.full((k,), 2, dtype=torch.int64, device=wf.device)
    if policy.t_mid is not None:
        lvl = torch.where(s <= policy.t_mid + 1e-6, torch.ones_like(lvl), lvl)
    lvl = torch.where(s <= policy.t1_for(layer) + 1e-6, torch.zeros_like(lvl), lvl)
    skip = (s > policy.t2 + 1e-6) & (lvl != 0) if policy.t2 is not None else none
    return lvl, skip


class FullMoE:
    """Les couches MoE calculées dans le graph du token entier (A5)."""

    def __init__(self, rt, k: int, x: torch.Tensor, precs_ok: set[str], version: int):
        st = rt.store
        self.rt = rt
        self.k = k
        self.dev = x.device
        self.precs = list(st.precisions)
        self.table = rt.device_table()
        L, H = len(st.layers), st.hidden
        self.plans: dict[int, _Plan] = {}
        for pi, p in enumerate(self.precs):
            lay = rt.fast_layouts[p]
            g = lay.get("group_sizes", {})
            supported = (PRECISIONS[p]["kind"] == "float" or (p == "int4" and _pow2_groups(g))
                         or (p == "int8" and _pow2_groups8(g)))
            if p in precs_ok and supported:
                self.plans[pi] = _Plan(lay, version)
        self.tab = {pi: torch.zeros(k, 4, dtype=torch.int64, device=self.dev) for pi in self.plans}
        self.w = {pi: torch.zeros(k, dtype=torch.float32, device=self.dev) for pi in self.plans}
        self.h = {pi: torch.empty(k, pl.I, dtype=torch.float32, device=self.dev) for pi, pl in self.plans.items()}
        self.offs = {pi: torch.tensor(pl.offs, dtype=torch.int64, device=self.dev) for pi, pl in self.plans.items()}
        self.acc = torch.zeros(H, dtype=torch.float32, device=self.dev)
        self.flag = torch.zeros(1, dtype=torch.float32, device=self.dev)
        self.info = torch.zeros(L, 3, k, dtype=torch.float32, device=self.dev)
        self.lv2pi = torch.tensor([self.precs.index(level_to_precision(v, self.precs)) for v in range(3)],
                                  dtype=torch.int64, device=self.dev)
        planned = torch.zeros(len(self.precs), dtype=torch.bool)
        planned[list(self.plans)] = True
        self.planned = planned.to(self.dev)

    def layer(self, layer: int, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, call) -> torch.Tensor:
        """Remplace run_layer pendant la capture du graph complet. `call(fn)` : lancement opaque (noyaux Triton)."""
        pol = self.rt.policy
        k = self.k
        ix = idx.reshape(-1)
        wf = w.reshape(-1).float()
        lvl, skip = gpu_levels(pol, layer, wf)
        req = self.lv2pi[lvl]
        chosen = torch.full((k,), -1, dtype=torch.int64, device=self.dev)
        upgrade = pol.allow_upgrade
        for pi in reversed(range(len(self.precs))):  # du moins précis au plus précis : le plus précis gagne
            avail = self.table.dev[pi, layer].index_select(0, ix) != 0
            ok = avail & ((req >= pi) if upgrade else (req == pi))
            chosen = torch.where(ok, torch.full_like(chosen, pi), chosen)
        usable = (chosen >= 0) & self.planned[chosen.clamp(min=0)]
        miss = (~skip) & (~usable)
        self.flag.add_(miss.sum().float().reshape(1))
        for pi in self.plans:
            sel = (chosen == pi) & (~skip)
            base = self.table.dev[pi, layer].index_select(0, ix)
            self.tab[pi].copy_(torch.where(sel.unsqueeze(1), base.unsqueeze(1) + self.offs[pi].unsqueeze(0),
                                           torch.zeros_like(self.tab[pi])))
            self.w[pi].copy_(torch.where(sel, wf, torch.zeros_like(wf)))
        xc = x.reshape(-1).contiguous()

        def launch():
            if not self.plans:
                self.acc.zero_()
            first = True
            for pi, plan in self.plans.items():
                plan.up(k, self.tab[pi], self.w[pi], xc, self.h[pi], self.acc, zero=first)
                first = False
            for pi, plan in self.plans.items():
                plan.down(k, self.tab[pi], self.w[pi], self.h[pi], self.acc)

        call(launch)
        self.info[layer, 0].copy_(ix.float())
        self.info[layer, 1].copy_(torch.where(skip, torch.full_like(chosen, -2), chosen).float())
        self.info[layer, 2].copy_(lvl.float())
        return self.acc.view(1, -1).to(x.dtype, copy=True).view_as(x)

    def readout(self) -> list[float]:
        """Une seule lecture GPU -> CPU par token : défauts + routage de toutes les couches."""
        return torch.cat((self.flag, self.info.reshape(-1))).tolist()
