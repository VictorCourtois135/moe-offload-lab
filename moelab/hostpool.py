"""
Pool de tampons en RAM pour les records d'experts (niveau "hôte").

Pourquoi ne pas simplement faire torch.empty(n, pin_memory=True) à chaque lecture ?
Parce que PyTorch passe alors par son allocateur de mémoire épinglée, qui :
  - arrondit chaque demande à la puissance de 2 supérieure (un record fp16 de
    12 Mio occupe 16 Mio, un record int4 de 3,19 Mio en occupe 4) ;
  - ne rend jamais la mémoire libérée au système : il la garde pour la réutiliser,
    et seulement pour une demande de la même taille arrondie.
Résultat : la mémoire épinglée réelle dépasse de loin le budget RAM du cache, et
Windows finit par refuser (CUDA_ERROR_OUT_OF_MEMORY dans cuMemHostAlloc).

Ici, comme les "slots" de Colibrì :
  - la mémoire est allouée par plaques ("slabs") de taille puissance de 2 (donc
    sans perte à l'arrondi, jusqu'à 256 Mio, environ 16 records), découpées en
    slots de la taille exacte d'un record, chaque plaque pour une seule précision ;
  - un slot libéré retourne dans la liste libre de sa précision ;
  - le total ne dépasse jamais `cap` octets. Au-delà (ou si le système refuse
    d'épingler davantage), on donne un tampon ordinaire, non épinglé : plus lent à
    copier vers le GPU, mais jamais d'erreur.

Recyclage des plaques (02/10) : une plaque appartient à une précision. Quand le
plafond est atteint et qu'une précision manque de slots, une plaque ENTIÈREMENT libre
d'une autre précision est redécoupée pour elle. Sans ça, un réglage `lo` (int4)
remplissait le pool de plaques int4, et le réglage `hobbit` lancé ensuite dans le même
processus n'avait plus de place pour ses records int8 : copies depuis de la RAM non
épinglée, et ~19 tok/s au lieu de ~29 au benchmark (mesures du 02/10).

Libération différée (quarantaine) : un slot libéré peut encore être lu par une
copie RAM -> VRAM en vol, ou par la branche CPU de la couche en cours. Il ne
redevient disponible que lorsque (1) l'événement CUDA enregistré à sa libération
est passé et (2) on est passé à la couche suivante (`advance()`).
"""

from __future__ import annotations

import threading

import torch

SLAB_BYTES = 256 * 2**20


def _pow2_ceil(n: int) -> int:
    return 1 << (max(n, 1) - 1).bit_length()


def _pow2_floor(n: int) -> int:
    return 1 << (max(n, 1).bit_length() - 1)


class HostPool:
    def __init__(self, record_bytes: dict[str, int], cap_bytes: int, pin: bool, log=print):
        self.record_bytes = dict(record_bytes)
        self.cap = int(cap_bytes)
        self.pin = pin
        self.log = log
        self.lock = threading.Lock()
        self.slabs: list[torch.Tensor] = []
        self.slab_prec: list[str] = []                       # précision de chaque plaque
        self.slab_slots: list[list[torch.Tensor]] = []       # ses slots
        self.allocated = 0                                   # octets de plaques allouées
        self.free: dict[str, list[torch.Tensor]] = {p: [] for p in record_bytes}
        self.quarantine: list[tuple[torch.Tensor, str, object, int]] = []  # (slot, précision, événement, époque)
        self.owner: dict[int, str] = {}                      # data_ptr d'un slot -> précision
        self.in_use: set[int] = set()                        # slots actuellement prêtés
        self.epoch = 0
        self.exhausted = False                               # le système a refusé d'épingler plus
        self.fallbacks = 0                                   # tampons ordinaires donnés faute de place
        self.reclaims = 0                                    # plaques redécoupées pour une autre précision

    # ------------------------------------------------------------------
    def _new_slab(self, prec: str) -> bool:
        rb = self.record_bytes[prec]
        room = self.cap - self.allocated
        size = max(_pow2_ceil(rb), min(SLAB_BYTES, _pow2_ceil(16 * rb)))
        if size > room:
            size = _pow2_floor(room) if room >= rb else 0
            if size < rb:
                return False
        try:
            if self.pin:  # cudaHostAlloc rend des pages alignées sur 4096
                slab = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            else:  # RAM ordinaire : on aligne à la main (lectures directes, optimisation D1)
                raw = torch.empty(size + 4096, dtype=torch.uint8)
                off = (-raw.data_ptr()) % 4096
                slab = raw[off : off + size]
        except RuntimeError as e:  # torch.AcceleratorError en hérite
            self.exhausted = True
            self.log(f"  mémoire épinglée refusée par le système après {self.allocated / 2**30:.2f} Gio ({e.__class__.__name__}) : "
                     "la suite utilisera de la RAM ordinaire (copies vers le GPU plus lentes)")
            return False
        self.slabs.append(slab)
        self.slab_prec.append(prec)
        self.slab_slots.append([])
        self.allocated += size
        self._carve(len(self.slabs) - 1, prec)
        return True

    def _carve(self, i: int, prec: str) -> None:
        """Découpe la plaque i en slots de la taille d'un record `prec`."""
        slab, rb = self.slabs[i], self.record_bytes[prec]
        slots = []
        for j in range(slab.numel() // rb):
            s = slab[j * rb : (j + 1) * rb]
            self.owner[s.data_ptr()] = prec
            self.free[prec].append(s)
            slots.append(s)
        self.slab_prec[i] = prec
        self.slab_slots[i] = slots

    def _reclaim(self, prec: str) -> bool:
        """Redécoupe pour `prec` une plaque entièrement libre d'une autre précision."""
        rb = self.record_bytes[prec]
        for i, other in enumerate(self.slab_prec):
            if other == prec or self.slabs[i].numel() < rb:
                continue
            ptrs = {s.data_ptr() for s in self.slab_slots[i]}
            free_ptrs = {s.data_ptr() for s in self.free[other]}
            if not ptrs <= free_ptrs:
                continue
            self.free[other] = [s for s in self.free[other] if s.data_ptr() not in ptrs]
            for ptr in ptrs:
                self.owner.pop(ptr, None)
            self._carve(i, prec)
            self.reclaims += 1
            return True
        return False

    def _sweep(self) -> None:
        keep = []
        for slot, prec, ev, epoch in self.quarantine:
            if epoch < self.epoch and (ev is None or ev.query()):
                self.free[prec].append(slot)
            else:
                keep.append((slot, prec, ev, epoch))
        self.quarantine = keep

    # ------------------------------------------------------------------
    def acquire(self, prec: str) -> torch.Tensor:
        """Un tampon de la taille exacte d'un record `prec`. Appelable depuis les threads de lecture disque."""
        with self.lock:
            if not self.free[prec]:
                self._sweep()
            if not self.free[prec] and not self.exhausted:
                self._new_slab(prec)
            if not self.free[prec]:
                self._sweep()  # toutes les précisions : une plaque entière peut s'être libérée
                self._reclaim(prec)
            if self.free[prec]:
                slot = self.free[prec].pop()
                self.in_use.add(slot.data_ptr())
                return slot
            self.fallbacks += 1
        return torch.empty(self.record_bytes[prec], dtype=torch.uint8)

    def release(self, blob: torch.Tensor, event=None) -> None:
        """Rend un tampon. Sans effet pour un tampon ordinaire (le ramasse-miettes s'en charge)."""
        ptr = blob.data_ptr()
        with self.lock:
            prec = self.owner.get(ptr)
            if prec is None or ptr not in self.in_use:  # tampon ordinaire, ou déjà rendu
                return
            self.in_use.discard(ptr)
            self.quarantine.append((blob, prec, event, self.epoch))

    def advance(self) -> None:
        """À appeler au début de chaque couche : les slots libérés avant deviennent réutilisables."""
        with self.lock:
            self.epoch += 1

    def is_pooled(self, blob: torch.Tensor) -> bool:
        return blob.data_ptr() in self.owner

    def describe(self) -> str:
        n_free = sum(len(v) for v in self.free.values())
        kind = "épinglée" if self.pin else "ordinaire"
        s = f"pool RAM {kind} : {self.allocated / 2**30:.2f}/{self.cap / 2**30:.2f} Gio alloués, {n_free} slots libres"
        if self.fallbacks:
            s += f", {self.fallbacks} tampons hors pool"
        if self.reclaims:
            s += f", {self.reclaims} plaques recyclées"
        return s

    def close(self) -> None:
        with self.lock:
            self.free = {p: [] for p in self.record_bytes}
            self.quarantine.clear()
            self.owner.clear()
            self.in_use.clear()
            self.slabs.clear()
            self.slab_prec.clear()
            self.slab_slots.clear()
            self.allocated = 0
        if self.pin and hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()  # rend vraiment la mémoire épinglée au système
