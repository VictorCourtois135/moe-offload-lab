"""
Optimisation B1 (doc 18) : produit matrice-vecteur qui lit l'int4 directement.

Sans ce noyau, un expert int4 est d'abord décompressé en fp16 (6-7 opérations GPU par
matrice, 12 Mio écrits puis relus), puis multiplié. Ici, UN noyau Triton lit les
octets int4 (3,2 Mio par expert), décompresse dans les registres et accumule, pour le
cas de la génération (un seul token : un produit matrice-vecteur).

Format lu (celui du store, voir quant.py) : q de forme (O, I/2) en uint8, quartet bas =
colonne paire, valeur stockée = q + 8 ; échelles (O, I/gs) en float16.

Triton est optionnel : sans lui (ou si le noyau échoue au premier appel), le moteur
garde le chemin PyTorch. Le mode interprète de Triton (TRITON_INTERPRET=1) exécute le
même noyau sur CPU : c'est ce que font les tests pour vérifier qu'il est exact.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pas installé, ou installation cassée
    HAVE_TRITON = False

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
BLOCK_O = 16      # lignes de sortie par programme
BLOCK_P = 128     # octets int4 lus par ligne et par tour de boucle (= 256 colonnes)


if HAVE_TRITON:

    @triton.jit
    def _gemv_int4_kernel(q_ptr, s_ptr, x_ptr, y_ptr, O, P, stride_q, stride_s,
                          GS: tl.constexpr, BO: tl.constexpr, BP: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BO + tl.arange(0, BO)
        rmask = rows < O
        acc = tl.zeros((BO,), dtype=tl.float32)
        for p0 in range(0, P, BP):
            pc = p0 + tl.arange(0, BP)                      # colonnes « octet »
            pmask = pc < P
            m2 = rmask[:, None] & pmask[None, :]
            q = tl.load(q_ptr + rows[:, None] * stride_q + pc[None, :], mask=m2, other=8)
            lo = (q & 15).to(tl.float32) - 8.0                # colonne paire
            hi = (q >> 4).to(tl.float32) - 8.0                # colonne impaire
            xe = tl.load(x_ptr + 2 * pc, mask=pmask, other=0.0).to(tl.float32)
            xo = tl.load(x_ptr + 2 * pc + 1, mask=pmask, other=0.0).to(tl.float32)
            g = (2 * pc) // GS                                 # groupe de chaque octet
            s = tl.load(s_ptr + rows[:, None] * stride_s + g[None, :], mask=m2, other=0.0).to(tl.float32)
            acc += tl.sum((lo * xe[None, :] + hi * xo[None, :]) * s, axis=1)
        tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=rmask)


def gemv_int4(q: torch.Tensor, s: torch.Tensor, x: torch.Tensor, group_size: int) -> torch.Tensor:
    """y = W @ x avec W int4 (q, s) ; x de forme (I,) ; retourne (O,) dans le dtype de x."""
    O, P = q.shape
    y = torch.empty(O, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(O, BLOCK_O),)
    _gemv_int4_kernel[grid](q, s, x.contiguous(), y, O, P, q.stride(0), s.stride(0),
                            GS=group_size, BO=BLOCK_O, BP=BLOCK_P)
    return y


def expert_ffn_int4(x_row: torch.Tensor, v: dict, groups: dict, act_fn) -> torch.Tensor:
    """Un expert int4 pour UN token : down( act(gate(x)) * up(x) ), sans décompresser les poids."""
    h = gemv_int4(v["gate_up.q"], v["gate_up.s"], x_row.reshape(-1), groups["gate_up"])
    gate, up = h.chunk(2)
    return gemv_int4(v["down.q"], v["down.s"], act_fn(gate) * up, groups["down"]).reshape(1, -1)


def triton_usable(device: torch.device | str) -> bool:
    """Le noyau peut-il tourner sur ce device ? (GPU CUDA, ou CPU en mode interprète pour les tests)."""
    if not HAVE_TRITON:
        return False
    dev = torch.device(device)
    return dev.type == "cuda" or (INTERPRET and dev.type == "cpu")


# ----------------------------------------------------------------------------
# A4 + B1 (doc 18) : TOUS les experts d'une couche en 2 lancements.
#
# Le profil sur la RTX 2060 (Windows) a montré qu'un lancement de noyau coûte ~50 µs :
# le moteur passait l'essentiel de son temps à lancer ~15 petits noyaux par expert.
# Ici, une table d'adresses (une ligne par expert) est copiée vers le GPU, et :
#   noyau A : h[e] = silu(gate_e @ x) * (up_e @ x)          grille (experts, blocs de lignes)
#   noyau B : out += somme_e w_e * (down_e @ h[e])          grille (blocs de lignes)
# Deux variantes : poids int4 (lus sans décompression) et poids flottants (fp16/fp32).
# ----------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _grp_int4_up(tab_ptr, x_ptr, h_ptr, I, PH, stride_s, GS: tl.constexpr, BO: tl.constexpr, BP: tl.constexpr):
        e = tl.program_id(0)
        rows = tl.program_id(1) * BO + tl.arange(0, BO)
        rmask = rows < I
        q_addr = tl.load(tab_ptr + e * 4 + 0).to(tl.pointer_type(tl.uint8))
        s_addr = tl.load(tab_ptr + e * 4 + 1).to(tl.pointer_type(tl.float16))
        acc_g = tl.zeros((BO,), dtype=tl.float32)
        acc_u = tl.zeros((BO,), dtype=tl.float32)
        for p0 in range(0, PH, BP):
            pc = p0 + tl.arange(0, BP)
            pmask = pc < PH
            m2 = rmask[:, None] & pmask[None, :]
            xe = tl.load(x_ptr + 2 * pc, mask=pmask, other=0.0).to(tl.float32)
            xo = tl.load(x_ptr + 2 * pc + 1, mask=pmask, other=0.0).to(tl.float32)
            g = (2 * pc) // GS
            for half in tl.static_range(2):  # 0 = lignes gate, 1 = lignes up (décalées de I)
                r = rows + half * I
                q = tl.load(q_addr + r[:, None] * PH + pc[None, :], mask=m2, other=8)
                s = tl.load(s_addr + r[:, None] * stride_s + g[None, :], mask=m2, other=0.0).to(tl.float32)
                v = tl.sum((((q & 15).to(tl.float32) - 8.0) * xe[None, :] + ((q >> 4).to(tl.float32) - 8.0) * xo[None, :]) * s, axis=1)
                if half == 0:
                    acc_g += v
                else:
                    acc_u += v
        h = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
        tl.store(h_ptr + e * I + rows, h, mask=rmask)

    @triton.jit
    def _grp_int4_down(tab_ptr, w_ptr, h_ptr, out_ptr, n, H, I, PI, stride_s, GS: tl.constexpr, BO: tl.constexpr,
                       BP: tl.constexpr):
        rows = tl.program_id(0) * BO + tl.arange(0, BO)
        rmask = rows < H
        acc = tl.zeros((BO,), dtype=tl.float32)
        for e in range(0, n):
            q_addr = tl.load(tab_ptr + e * 4 + 2).to(tl.pointer_type(tl.uint8))
            s_addr = tl.load(tab_ptr + e * 4 + 3).to(tl.pointer_type(tl.float16))
            we = tl.load(w_ptr + e)
            part = tl.zeros((BO,), dtype=tl.float32)
            for p0 in range(0, PI, BP):
                pc = p0 + tl.arange(0, BP)
                pmask = pc < PI
                m2 = rmask[:, None] & pmask[None, :]
                he = tl.load(h_ptr + e * I + 2 * pc, mask=pmask, other=0.0)
                ho = tl.load(h_ptr + e * I + 2 * pc + 1, mask=pmask, other=0.0)
                q = tl.load(q_addr + rows[:, None] * PI + pc[None, :], mask=m2, other=8)
                s = tl.load(s_addr + rows[:, None] * stride_s + ((2 * pc) // GS)[None, :], mask=m2, other=0.0).to(tl.float32)
                part += tl.sum((((q & 15).to(tl.float32) - 8.0) * he[None, :] + ((q >> 4).to(tl.float32) - 8.0) * ho[None, :]) * s, axis=1)
            acc += we * part
        prev = tl.load(out_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
        tl.store(out_ptr + rows, (prev + acc).to(out_ptr.dtype.element_ty), mask=rmask)

    @triton.jit
    def _grp_float_up(tab_ptr, x_ptr, h_ptr, I, H, WDT: tl.constexpr, BO: tl.constexpr, BK: tl.constexpr):
        e = tl.program_id(0)
        rows = tl.program_id(1) * BO + tl.arange(0, BO)
        rmask = rows < I
        w_addr = tl.load(tab_ptr + e * 4 + 0).to(tl.pointer_type(WDT))
        acc_g = tl.zeros((BO,), dtype=tl.float32)
        acc_u = tl.zeros((BO,), dtype=tl.float32)
        for k0 in range(0, H, BK):
            kc = k0 + tl.arange(0, BK)
            kmask = kc < H
            m2 = rmask[:, None] & kmask[None, :]
            xv = tl.load(x_ptr + kc, mask=kmask, other=0.0).to(tl.float32)
            wg = tl.load(w_addr + rows[:, None] * H + kc[None, :], mask=m2, other=0.0).to(tl.float32)
            wu = tl.load(w_addr + (rows + I)[:, None] * H + kc[None, :], mask=m2, other=0.0).to(tl.float32)
            acc_g += tl.sum(wg * xv[None, :], axis=1)
            acc_u += tl.sum(wu * xv[None, :], axis=1)
        h = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
        tl.store(h_ptr + e * I + rows, h, mask=rmask)

    @triton.jit
    def _grp_float_down(tab_ptr, w_ptr, h_ptr, out_ptr, n, H, I, WDT: tl.constexpr, BO: tl.constexpr, BK: tl.constexpr):
        rows = tl.program_id(0) * BO + tl.arange(0, BO)
        rmask = rows < H
        acc = tl.zeros((BO,), dtype=tl.float32)
        for e in range(0, n):
            w_addr = tl.load(tab_ptr + e * 4 + 2).to(tl.pointer_type(WDT))
            we = tl.load(w_ptr + e)
            part = tl.zeros((BO,), dtype=tl.float32)
            for k0 in range(0, I, BK):
                kc = k0 + tl.arange(0, BK)
                kmask = kc < I
                hv = tl.load(h_ptr + e * I + kc, mask=kmask, other=0.0)
                wd = tl.load(w_addr + rows[:, None] * I + kc[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0).to(tl.float32)
                part += tl.sum(wd * hv[None, :], axis=1)
            acc += we * part
        prev = tl.load(out_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
        tl.store(out_ptr + rows, (prev + acc).to(out_ptr.dtype.element_ty), mask=rmask)


_WDT = {}
if HAVE_TRITON:
    _WDT = {torch.float16: tl.float16, torch.float32: tl.float32, torch.bfloat16: tl.bfloat16}


class _Staging:
    """
    Petits tampons hôte épinglés, réutilisés en anneau, pour envoyer les tables des noyaux
    groupés SANS synchronisation. Une copie depuis de la mémoire non épinglée
    (non_blocking=False) attend que le GPU ait fini tout ce qui précède : le CPU ne
    pouvait plus préparer la couche suivante pendant que le GPU calculait (profil du 01/10).
    """

    def __init__(self, slots: int = 32, nbytes: int = 4096):
        self.nbytes = nbytes
        self.bufs = [torch.empty(nbytes, dtype=torch.uint8, pin_memory=True) for _ in range(slots)]
        self.events: list = [None] * slots
        self.i = 0

    def send(self, host: torch.Tensor, device) -> torch.Tensor:
        n = host.numel()
        if n > self.nbytes:
            return host.to(device)
        i = self.i
        self.i = (i + 1) % len(self.bufs)
        if self.events[i] is not None:
            self.events[i].synchronize()  # en pratique déjà passé : 32 couches plus tôt
        buf = self.bufs[i][:n]
        buf.copy_(host)
        dev = torch.empty(n, dtype=torch.uint8, device=device)
        dev.copy_(buf, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.events[i] = ev
        return dev


_staging: _Staging | None = None


def _table(entries, keys, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Table d'adresses (n, 4) int64 + poids (n,) float32, envoyées en UNE copie asynchrone vers le GPU."""
    global _staging
    n = len(entries)
    buf = torch.empty(n * 4 * 8 + n * 4, dtype=torch.uint8)
    buf[: n * 32].view(torch.int64).copy_(torch.tensor([[v[k].data_ptr() for k in keys] for v, _ in entries],
                                                       dtype=torch.int64).reshape(-1))
    buf[n * 32:].view(torch.float32).copy_(torch.tensor([w for _, w in entries], dtype=torch.float32))
    if torch.device(device).type == "cuda":
        if _staging is None:
            _staging = _Staging()
        dev = _staging.send(buf, device)
    else:
        dev = buf.to(device)
    return dev[: n * 32].view(torch.int64), dev[n * 32:].view(torch.float32)


# ----------------------------------------------------------------------------
# Noyaux groupés v2 (01/10). Le profil du 01/10 mesurait la v1 à ~35-50 Go/s effectifs
# pour ~285 Go/s possibles. Corrections :
#   - adresses alignées déclarées (tl.multiple_of) : lectures vectorisées ;
#   - pas de masques quand les dimensions tombent juste (EVEN) ;
#   - une échelle par GROUPE (blocs 3D lignes x groupes x octets) ;
#   - noyau B réparti sur (blocs de lignes, experts), addition atomique en float32 ;
#   - tailles de blocs et nombre de warps choisis par triton.autotune sur la carte ;
#   - un expert de poids 0 ne fait rien (permet des tables de taille fixe, voir GroupedGraph).
# v3 = v2 + conversion des quartets sans instruction de conversion entier -> flottant
# (MAGIC) : (q | 0x4B000000) relu comme un float32 vaut 2^23 + q exactement ; un
# soustraction donne q - 8. Sur Turing, la conversion I2F tourne à 1/4 du débit des
# additions : c'était probablement le facteur limitant (le noyau fp16, sans conversion,
# atteint 263 Go/s ; l'int4 plafonnait à 69 Go/s).
# ----------------------------------------------------------------------------

_CONFIGS = [(8, 128, 2), (16, 128, 4), (16, 256, 4), (32, 128, 4), (32, 256, 8), (64, 128, 8)]


class _Fixed:
    """Noyau sans autotune : ajoute la config par défaut (BO=16, BPW=128, 4 warps)."""

    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        def run(*args, **kw):
            meta = {"BO": 16, "BPW": 128, **kw}
            g = grid(meta) if callable(grid) else grid
            return self.fn[g](*args, num_warps=4, **meta)
        return run


def _tuned(key, restore=None):
    """triton.autotune sur GPU ; config fixe sous l'interprète (tests)."""
    if INTERPRET or not HAVE_TRITON:
        return lambda f: f
    cfgs = [triton.Config({"BO": bo, "BPW": bpw}, num_warps=w) for bo, bpw, w in _CONFIGS]
    try:
        # restore_value : le noyau B ajoute dans l'accumulateur ; les essais de l'autotune
        # ne doivent pas s'y additionner
        return triton.autotune(configs=cfgs, key=key, **({"restore_value": restore} if restore else {}))
    except TypeError:  # Triton trop ancien : config fixe
        return lambda f: _Fixed(f)


_FIXED = {"BO": 16, "BPW": 64}  # sous l'interprète


if HAVE_TRITON:

    @triton.jit
    def _nib(v, MAGIC: tl.constexpr):
        """Quartet v (int32, 0..15) -> float32 (v - 8)."""
        if MAGIC:
            return (v | 0x4B000000).to(tl.float32, bitcast=True) - 8388616.0
        else:
            return v.to(tl.float32) - 8.0

    @triton.jit
    def _zero_acc(acc_ptr, H, j, nj):
        for z0 in range(j * 128, H, nj * 128):
            z = z0 + tl.arange(0, 128)
            tl.store(acc_ptr + z, tl.zeros((128,), dtype=tl.float32), mask=z < H)

    @_tuned(key=["I", "PH", "H", "MAGIC"])
    @triton.jit
    def _grp4_up(tab_ptr, w_ptr, x_ptr, h_ptr, acc_ptr, I, PH, H, stride_s,
                 GB: tl.constexpr, EVEN: tl.constexpr, ZERO: tl.constexpr, MAGIC: tl.constexpr,
                 BO: tl.constexpr, BPW: tl.constexpr):
        e = tl.program_id(0)
        j = tl.program_id(1)
        if ZERO:  # remet à zéro l'accumulateur du noyau B (évite un lancement de plus)
            if e == 0:
                _zero_acc(acc_ptr, H, j, tl.num_programs(1))
        we = tl.load(w_ptr + e)
        if we != 0.0:
            NG: tl.constexpr = BPW // GB
            rows = j * BO + tl.arange(0, BO)
            gi = tl.arange(0, NG)
            bi = tl.arange(0, GB)
            q_addr = tl.multiple_of(tl.load(tab_ptr + e * 4 + 0).to(tl.pointer_type(tl.uint8)), 16)
            s_addr = tl.multiple_of(tl.load(tab_ptr + e * 4 + 1).to(tl.pointer_type(tl.float16)), 16)
            acc_g = tl.zeros((BO,), dtype=tl.float32)
            acc_u = tl.zeros((BO,), dtype=tl.float32)
            for p0 in range(0, PH, BPW):
                col = p0 + gi[:, None] * GB + bi[None, :]           # (NG, GB) colonnes « octet »
                sg = p0 // GB + gi                                   # (NG,) indice de groupe
                if EVEN:
                    xe = tl.load(x_ptr + 2 * col).to(tl.float32)
                    xo = tl.load(x_ptr + 2 * col + 1).to(tl.float32)
                else:
                    cm = col < PH
                    xe = tl.load(x_ptr + 2 * col, mask=cm, other=0.0).to(tl.float32)
                    xo = tl.load(x_ptr + 2 * col + 1, mask=cm, other=0.0).to(tl.float32)
                for half in tl.static_range(2):  # 0 = lignes gate, 1 = lignes up (décalées de I)
                    r = rows + half * I
                    qo = r[:, None, None] * PH + col[None, :, :]
                    so = r[:, None] * stride_s + sg[None, :]
                    if EVEN:
                        q = tl.load(q_addr + qo)
                        sc = tl.load(s_addr + so).to(tl.float32)
                    else:
                        m3 = (rows < I)[:, None, None] & (col < PH)[None, :, :]
                        q = tl.load(q_addr + qo, mask=m3, other=8)
                        sc = tl.load(s_addr + so, mask=(rows < I)[:, None] & (sg * GB < PH)[None, :], other=0.0).to(tl.float32)
                    qi = q.to(tl.int32)
                    part = tl.sum(_nib(qi & 15, MAGIC) * xe[None, :, :] + _nib(qi >> 4, MAGIC) * xo[None, :, :], axis=2)
                    v = tl.sum(part * sc, axis=1)
                    if half == 0:
                        acc_g += v
                    else:
                        acc_u += v
            h = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
            tl.store(h_ptr + e * I + rows, h, mask=rows < I)

    @_tuned(key=["H", "PI", "I", "MAGIC"], restore=["acc_ptr"])
    @triton.jit
    def _grp4_down(tab_ptr, w_ptr, h_ptr, acc_ptr, H, I, PI, stride_s,
                   GB: tl.constexpr, EVEN: tl.constexpr, MAGIC: tl.constexpr, BO: tl.constexpr, BPW: tl.constexpr):
        e = tl.program_id(1)
        we = tl.load(w_ptr + e)
        if we != 0.0:
            rows = tl.program_id(0) * BO + tl.arange(0, BO)
            NG: tl.constexpr = BPW // GB
            gi = tl.arange(0, NG)
            bi = tl.arange(0, GB)
            q_addr = tl.multiple_of(tl.load(tab_ptr + e * 4 + 2).to(tl.pointer_type(tl.uint8)), 16)
            s_addr = tl.multiple_of(tl.load(tab_ptr + e * 4 + 3).to(tl.pointer_type(tl.float16)), 16)
            acc = tl.zeros((BO,), dtype=tl.float32)
            for p0 in range(0, PI, BPW):
                col = p0 + gi[:, None] * GB + bi[None, :]
                sg = p0 // GB + gi
                qo = rows[:, None, None] * PI + col[None, :, :]
                so = rows[:, None] * stride_s + sg[None, :]
                if EVEN:
                    he = tl.load(h_ptr + e * I + 2 * col)
                    ho = tl.load(h_ptr + e * I + 2 * col + 1)
                    q = tl.load(q_addr + qo)
                    sc = tl.load(s_addr + so).to(tl.float32)
                else:
                    cm = col < PI
                    he = tl.load(h_ptr + e * I + 2 * col, mask=cm, other=0.0)
                    ho = tl.load(h_ptr + e * I + 2 * col + 1, mask=cm, other=0.0)
                    q = tl.load(q_addr + qo, mask=(rows < H)[:, None, None] & cm[None, :, :], other=8)
                    sc = tl.load(s_addr + so, mask=(rows < H)[:, None] & (sg * GB < PI)[None, :], other=0.0).to(tl.float32)
                qi = q.to(tl.int32)
                part = tl.sum(_nib(qi & 15, MAGIC) * he[None, :, :] + _nib(qi >> 4, MAGIC) * ho[None, :, :], axis=2)
                acc += tl.sum(part * sc, axis=1)
            tl.atomic_add(acc_ptr + rows, acc * we, mask=rows < H)

    @_tuned(key=["I", "H"])
    @triton.jit
    def _grpf_up(tab_ptr, w_ptr, x_ptr, h_ptr, acc_ptr, I, H, WDT: tl.constexpr, EVEN: tl.constexpr,
                 ZERO: tl.constexpr, BO: tl.constexpr, BPW: tl.constexpr):
        e = tl.program_id(0)
        j = tl.program_id(1)
        if ZERO:
            if e == 0:
                _zero_acc(acc_ptr, H, j, tl.num_programs(1))
        we = tl.load(w_ptr + e)
        if we != 0.0:
            rows = j * BO + tl.arange(0, BO)
            w_addr = tl.multiple_of(tl.load(tab_ptr + e * 4 + 0).to(tl.pointer_type(WDT)), 16)
            acc_g = tl.zeros((BO,), dtype=tl.float32)
            acc_u = tl.zeros((BO,), dtype=tl.float32)
            for k0 in range(0, H, BPW):
                kc = k0 + tl.arange(0, BPW)
                if EVEN:
                    xv = tl.load(x_ptr + kc).to(tl.float32)
                    wg = tl.load(w_addr + rows[:, None] * H + kc[None, :]).to(tl.float32)
                    wu = tl.load(w_addr + (rows + I)[:, None] * H + kc[None, :]).to(tl.float32)
                else:
                    km = kc < H
                    m2 = (rows < I)[:, None] & km[None, :]
                    xv = tl.load(x_ptr + kc, mask=km, other=0.0).to(tl.float32)
                    wg = tl.load(w_addr + rows[:, None] * H + kc[None, :], mask=m2, other=0.0).to(tl.float32)
                    wu = tl.load(w_addr + (rows + I)[:, None] * H + kc[None, :], mask=m2, other=0.0).to(tl.float32)
                acc_g += tl.sum(wg * xv[None, :], axis=1)
                acc_u += tl.sum(wu * xv[None, :], axis=1)
            h = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
            tl.store(h_ptr + e * I + rows, h, mask=rows < I)

    @_tuned(key=["H", "I"], restore=["acc_ptr"])
    @triton.jit
    def _grpf_down(tab_ptr, w_ptr, h_ptr, acc_ptr, H, I, WDT: tl.constexpr, EVEN: tl.constexpr,
                   BO: tl.constexpr, BPW: tl.constexpr):
        e = tl.program_id(1)
        we = tl.load(w_ptr + e)
        if we != 0.0:
            rows = tl.program_id(0) * BO + tl.arange(0, BO)
            w_addr = tl.multiple_of(tl.load(tab_ptr + e * 4 + 2).to(tl.pointer_type(WDT)), 16)
            acc = tl.zeros((BO,), dtype=tl.float32)
            for k0 in range(0, I, BPW):
                kc = k0 + tl.arange(0, BPW)
                if EVEN:
                    hv = tl.load(h_ptr + e * I + kc)
                    wd = tl.load(w_addr + rows[:, None] * I + kc[None, :]).to(tl.float32)
                else:
                    km = kc < I
                    hv = tl.load(h_ptr + e * I + kc, mask=km, other=0.0)
                    wd = tl.load(w_addr + rows[:, None] * I + kc[None, :], mask=(rows < H)[:, None] & km[None, :],
                                 other=0.0).to(tl.float32)
                acc += tl.sum(wd * hv[None, :], axis=1)
            tl.atomic_add(acc_ptr + rows, acc * we, mask=rows < H)



# ----------------------------------------------------------------------------
# v4 (01/10, après la mesure de v3) : v3 n'a rien gagné (68 Go/s) : la conversion des
# quartets n'était pas le frein. Ce qui distingue l'int4 v2 du noyau fp16 (265 Go/s) :
# des blocs 3D, DEUX réductions par tour de boucle (sur les octets du groupe, puis sur
# les groupes avec l'échelle), et des changements de disposition des données entre les
# deux (passage par la mémoire partagée). v4 reprend la forme du noyau fp16 :
#   - blocs 2D (lignes x octets d'UN groupe) : une seule échelle par ligne et par tour ;
#   - accumulateur 2D (lignes x octets) : acc += échelle * (lo * x_pair + hi * x_impair),
#     une seule réduction, après la boucle ;
#   - U groupes par tour (déroulé), tailles choisies par l'autotune.
# ----------------------------------------------------------------------------

_CONFIGS4 = [(16, 1, 2), (32, 1, 4), (32, 2, 4), (64, 1, 4), (64, 2, 4), (64, 4, 8), (128, 1, 8), (128, 2, 8)]


def _tuned4(key, restore=None):
    if INTERPRET or not HAVE_TRITON:
        return lambda f: f
    cfgs = [triton.Config({"BO": bo, "U": u}, num_warps=w) for bo, u, w in _CONFIGS4]
    try:
        return triton.autotune(configs=cfgs, key=key, **({"restore_value": restore} if restore else {}))
    except TypeError:
        return lambda f: _Fixed4(f)


class _Fixed4(_Fixed):
    def __getitem__(self, grid):
        def run(*args, **kw):
            meta = {"BO": 32, "U": 1, **kw}
            g = grid(meta) if callable(grid) else grid
            return self.fn[g](*args, num_warps=4, **meta)
        return run


_FIXED4 = {"BO": 16, "U": 2}  # sous l'interprète


if HAVE_TRITON:

    @_tuned4(key=["I", "PH", "H"])
    @triton.jit
    def _grp4_up_v4(tab_ptr, w_ptr, x_ptr, h_ptr, acc_ptr, I, PH, H, stride_s,
                    GB: tl.constexpr, EVEN: tl.constexpr, ZERO: tl.constexpr, BO: tl.constexpr, U: tl.constexpr):
        e = tl.program_id(0)
        j = tl.program_id(1)
        if ZERO:
            if e == 0:
                _zero_acc(acc_ptr, H, j, tl.num_programs(1))
        we = tl.load(w_ptr + e)
        if we != 0.0:
            rows = j * BO + tl.arange(0, BO)
            rmask = rows < I
            bi = tl.arange(0, GB)
            q_addr = tl.multiple_of(tl.load(tab_ptr + e * 4 + 0).to(tl.pointer_type(tl.uint8)), 16)
            s_addr = tl.multiple_of(tl.load(tab_ptr + e * 4 + 1).to(tl.pointer_type(tl.float16)), 16)
            ag = tl.zeros((BO, GB), dtype=tl.float32)
            au = tl.zeros((BO, GB), dtype=tl.float32)
            NGT = PH // GB
            for g0 in range(0, NGT, U):
                for u in tl.static_range(U):
                    g = g0 + u
                    col = g * GB + bi
                    if EVEN:
                        xe = tl.load(x_ptr + 2 * col).to(tl.float32)
                        xo = tl.load(x_ptr + 2 * col + 1).to(tl.float32)
                        qg = tl.load(q_addr + rows[:, None] * PH + col[None, :])
                        qu = tl.load(q_addr + (rows + I)[:, None] * PH + col[None, :])
                        sg = tl.load(s_addr + rows * stride_s + g).to(tl.float32)
                        su = tl.load(s_addr + (rows + I) * stride_s + g).to(tl.float32)
                    else:
                        gm = g < NGT
                        cm = (col < PH) & gm
                        m2 = rmask[:, None] & cm[None, :]
                        xe = tl.load(x_ptr + 2 * col, mask=cm, other=0.0).to(tl.float32)
                        xo = tl.load(x_ptr + 2 * col + 1, mask=cm, other=0.0).to(tl.float32)
                        qg = tl.load(q_addr + rows[:, None] * PH + col[None, :], mask=m2, other=8)
                        qu = tl.load(q_addr + (rows + I)[:, None] * PH + col[None, :], mask=m2, other=8)
                        sg = tl.load(s_addr + rows * stride_s + g, mask=rmask & gm, other=0.0).to(tl.float32)
                        su = tl.load(s_addr + (rows + I) * stride_s + g, mask=rmask & gm, other=0.0).to(tl.float32)
                    a = qg.to(tl.int32)
                    b = qu.to(tl.int32)
                    ag += sg[:, None] * (((a & 15).to(tl.float32) - 8.0) * xe[None, :] + ((a >> 4).to(tl.float32) - 8.0) * xo[None, :])
                    au += su[:, None] * (((b & 15).to(tl.float32) - 8.0) * xe[None, :] + ((b >> 4).to(tl.float32) - 8.0) * xo[None, :])
            acc_g = tl.sum(ag, axis=1)
            acc_u = tl.sum(au, axis=1)
            h = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
            tl.store(h_ptr + e * I + rows, h, mask=rmask)

    @_tuned4(key=["H", "PI", "I"], restore=["acc_ptr"])
    @triton.jit
    def _grp4_down_v4(tab_ptr, w_ptr, h_ptr, acc_ptr, H, I, PI, stride_s,
                      GB: tl.constexpr, EVEN: tl.constexpr, BO: tl.constexpr, U: tl.constexpr):
        e = tl.program_id(1)
        we = tl.load(w_ptr + e)
        if we != 0.0:
            rows = tl.program_id(0) * BO + tl.arange(0, BO)
            rmask = rows < H
            bi = tl.arange(0, GB)
            q_addr = tl.multiple_of(tl.load(tab_ptr + e * 4 + 2).to(tl.pointer_type(tl.uint8)), 16)
            s_addr = tl.multiple_of(tl.load(tab_ptr + e * 4 + 3).to(tl.pointer_type(tl.float16)), 16)
            a2 = tl.zeros((BO, GB), dtype=tl.float32)
            NGT = PI // GB
            for g0 in range(0, NGT, U):
                for u in tl.static_range(U):
                    g = g0 + u
                    col = g * GB + bi
                    if EVEN:
                        he = tl.load(h_ptr + e * I + 2 * col)
                        ho = tl.load(h_ptr + e * I + 2 * col + 1)
                        q = tl.load(q_addr + rows[:, None] * PI + col[None, :])
                        sc = tl.load(s_addr + rows * stride_s + g).to(tl.float32)
                    else:
                        gm = g < NGT
                        cm = (col < PI) & gm
                        he = tl.load(h_ptr + e * I + 2 * col, mask=cm, other=0.0)
                        ho = tl.load(h_ptr + e * I + 2 * col + 1, mask=cm, other=0.0)
                        q = tl.load(q_addr + rows[:, None] * PI + col[None, :], mask=rmask[:, None] & cm[None, :], other=8)
                        sc = tl.load(s_addr + rows * stride_s + g, mask=rmask & gm, other=0.0).to(tl.float32)
                    a = q.to(tl.int32)
                    a2 += sc[:, None] * (((a & 15).to(tl.float32) - 8.0) * he[None, :] + ((a >> 4).to(tl.float32) - 8.0) * ho[None, :])
            tl.atomic_add(acc_ptr + rows, tl.sum(a2, axis=1) * we, mask=rmask)


def _launch4_up(n, tab, w, x, h, acc, I, ph, H, ss, gb, zero):
    meta = dict(_FIXED4) if INTERPRET else {}
    even = _even((I, 128), (ph // gb, 4))
    _grp4_up_v4[lambda m: (n, triton.cdiv(I, m["BO"]))](tab, w, x, h, acc, I, ph, H, ss, GB=gb, EVEN=even, ZERO=zero, **meta)


def _launch4_down(n, tab, w, h, acc, H, I, pi, ss, gb):
    meta = dict(_FIXED4) if INTERPRET else {}
    even = _even((H, 128), (pi // gb, 4))
    _grp4_down_v4[lambda m: (triton.cdiv(H, m["BO"]), n)](tab, w, h, acc, H, I, pi, ss, GB=gb, EVEN=even, **meta)


def grouped_graph_gpu_ms(layout: dict, x, out, k: int, version: int, items: dict, repeat: int = 50) -> float:
    """Temps GPU d'un calcul groupé (k experts), rejoué en graph : le temps des noyaux seuls."""
    gg = GroupedGraph({layout["precision"]: layout}, x, out, k, version=version)
    gg.run(items)
    gg.run(items)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(repeat):
        gg.graph.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / repeat


# Choix automatique de la version (grouped_kernels = 0) : on chronomètre v2, v3 et v4 sur la
# carte, aux dimensions du modèle, avec des poids aléatoires, et on garde la plus rapide.
_AUTO: dict = {}
AUTO_CANDIDATES = (2, 3, 4)


def pick_int4_version(layout: dict, device, k: int = 8) -> int:
    dev = torch.device(device)
    if dev.type != "cuda" or not HAVE_TRITON:
        return 4
    key = (tuple(tuple(f[1]) for f in layout["fields"]), k)
    if key in _AUTO:
        return _AUTO[key]
    from .store import views

    nbytes = layout["record_bytes"]
    recs = []
    for _ in range(k):
        b = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=dev)
        v = views(layout, b)
        v["gate_up.s"].copy_(torch.rand_like(v["gate_up.s"], dtype=torch.float32).to(torch.float16) * 0.01)
        v["down.s"].copy_(torch.rand_like(v["down.s"], dtype=torch.float32).to(torch.float16) * 0.01)
        recs.append(b)
    H = views(layout, recs[0])["down.q"].shape[0]
    x = torch.randn(1, H, device=dev, dtype=torch.float16)
    out = torch.zeros(1, H, device=dev, dtype=torch.float16)
    items = {"int4": [(b, 1.0 / k) for b in recs]}
    best, best_ms = 2, float("inf")
    for ver in AUTO_CANDIDATES:
        # chronométré en graph : sinon on mesure le coût de lancement côté CPU, pas les noyaux
        try:
            ms = grouped_graph_gpu_ms(layout, x, out, k, ver, items)
        except Exception:  # noqa: BLE001 — version indisponible sur cette carte
            continue
        if ms < best_ms:
            best, best_ms = ver, ms
    _AUTO[key] = best
    return best


def _even(*pairs) -> bool:
    """Toutes les dimensions tombent-elles juste pour TOUTES les configs d'autotune (blocs max 64 x 256) ?"""
    return all(d % m == 0 for d, m in pairs)


def _meta() -> dict:
    return dict(_FIXED) if INTERPRET else {}


def _pow2_groups(groups: dict) -> bool:
    return all((g // 2) and not ((g // 2) & (g // 2 - 1)) and g // 2 <= 64 for g in (groups["gate_up"], groups["down"]))



# ----------------------------------------------------------------------------
# int8 groupé (01/10) : pour un store int8 + int4, où int8 remplace fp16 comme haute
# précision (2 fois moins d'octets à copier et à garder en cache, qualité quasi fp16).
# Même forme que v2 : blocs (lignes x groupes x valeurs), une échelle par groupe.
# ----------------------------------------------------------------------------

if HAVE_TRITON:

    @_tuned(key=["I", "H"])
    @triton.jit
    def _grp8_up(tab_ptr, w_ptr, x_ptr, h_ptr, acc_ptr, I, H, stride_s,
                 GB: tl.constexpr, EVEN: tl.constexpr, ZERO: tl.constexpr, BO: tl.constexpr, BPW: tl.constexpr):
        e = tl.program_id(0)
        j = tl.program_id(1)
        if ZERO:
            if e == 0:
                _zero_acc(acc_ptr, H, j, tl.num_programs(1))
        we = tl.load(w_ptr + e)
        if we != 0.0:
            NG: tl.constexpr = BPW // GB
            rows = j * BO + tl.arange(0, BO)
            gi = tl.arange(0, NG)
            bi = tl.arange(0, GB)
            q_addr = tl.multiple_of(tl.load(tab_ptr + e * 4 + 0).to(tl.pointer_type(tl.int8)), 16)
            s_addr = tl.multiple_of(tl.load(tab_ptr + e * 4 + 1).to(tl.pointer_type(tl.float16)), 16)
            acc_g = tl.zeros((BO,), dtype=tl.float32)
            acc_u = tl.zeros((BO,), dtype=tl.float32)
            for k0 in range(0, H, BPW):
                col = k0 + gi[:, None] * GB + bi[None, :]
                sg = k0 // GB + gi
                if EVEN:
                    xv = tl.load(x_ptr + col).to(tl.float32)
                else:
                    xv = tl.load(x_ptr + col, mask=col < H, other=0.0).to(tl.float32)
                for half in tl.static_range(2):
                    r = rows + half * I
                    qo = r[:, None, None] * H + col[None, :, :]
                    so = r[:, None] * stride_s + sg[None, :]
                    if EVEN:
                        q = tl.load(q_addr + qo).to(tl.float32)
                        sc = tl.load(s_addr + so).to(tl.float32)
                    else:
                        q = tl.load(q_addr + qo, mask=(rows < I)[:, None, None] & (col < H)[None, :, :], other=0).to(tl.float32)
                        sc = tl.load(s_addr + so, mask=(rows < I)[:, None] & (sg * GB < H)[None, :], other=0.0).to(tl.float32)
                    v = tl.sum(tl.sum(q * xv[None, :, :], axis=2) * sc, axis=1)
                    if half == 0:
                        acc_g += v
                    else:
                        acc_u += v
            h = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
            tl.store(h_ptr + e * I + rows, h, mask=rows < I)

    @_tuned(key=["H", "I"], restore=["acc_ptr"])
    @triton.jit
    def _grp8_down(tab_ptr, w_ptr, h_ptr, acc_ptr, H, I, stride_s,
                   GB: tl.constexpr, EVEN: tl.constexpr, BO: tl.constexpr, BPW: tl.constexpr):
        e = tl.program_id(1)
        we = tl.load(w_ptr + e)
        if we != 0.0:
            rows = tl.program_id(0) * BO + tl.arange(0, BO)
            NG: tl.constexpr = BPW // GB
            gi = tl.arange(0, NG)
            bi = tl.arange(0, GB)
            q_addr = tl.multiple_of(tl.load(tab_ptr + e * 4 + 2).to(tl.pointer_type(tl.int8)), 16)
            s_addr = tl.multiple_of(tl.load(tab_ptr + e * 4 + 3).to(tl.pointer_type(tl.float16)), 16)
            acc = tl.zeros((BO,), dtype=tl.float32)
            for k0 in range(0, I, BPW):
                col = k0 + gi[:, None] * GB + bi[None, :]
                sg = k0 // GB + gi
                qo = rows[:, None, None] * I + col[None, :, :]
                so = rows[:, None] * stride_s + sg[None, :]
                if EVEN:
                    hv = tl.load(h_ptr + e * I + col)
                    q = tl.load(q_addr + qo).to(tl.float32)
                    sc = tl.load(s_addr + so).to(tl.float32)
                else:
                    cm = col < I
                    hv = tl.load(h_ptr + e * I + col, mask=cm, other=0.0)
                    q = tl.load(q_addr + qo, mask=(rows < H)[:, None, None] & cm[None, :, :], other=0).to(tl.float32)
                    sc = tl.load(s_addr + so, mask=(rows < H)[:, None] & (sg * GB < I)[None, :], other=0.0).to(tl.float32)
                acc += tl.sum(tl.sum(q * hv[None, :, :], axis=2) * sc, axis=1)
            tl.atomic_add(acc_ptr + rows, acc * we, mask=rows < H)


def _pow2_groups8(groups: dict) -> bool:
    return all(g and not (g & (g - 1)) and g <= 64 for g in (groups["gate_up"], groups["down"]))


def grouped_ffn_int8(x_row: torch.Tensor, entries: list, layout: dict, out: torch.Tensor) -> None:
    """out (1, H) += somme des experts int8 (entries = [(views(record), poids), ...])."""
    if not _pow2_groups8(layout["group_sizes"]):  # groupes hors noyau : calcul PyTorch
        import torch.nn.functional as F

        from .quant import dequantize

        for v, wt in entries:
            gu = dequantize(v["gate_up.q"], v["gate_up.s"], 8, torch.float32)
            dn = dequantize(v["down.q"], v["down.s"], 8, torch.float32)
            a, b = F.linear(x_row.float(), gu).chunk(2, dim=-1)
            out.add_((wt * F.linear(F.silu(a) * b, dn)).to(out.dtype))
        return
    plan = _Plan(layout)
    n = len(entries)
    dev = x_row.device
    tab, w = _table(entries, ("gate_up.q", "gate_up.s", "down.q", "down.s"), dev)
    h = torch.empty(n, plan.I, dtype=torch.float32, device=dev)
    acc = torch.empty(plan.H, dtype=torch.float32, device=dev)
    x = x_row.reshape(-1).contiguous()
    plan.up(n, tab, w, x, h, acc, zero=True)
    plan.down(n, tab, w, h, acc)
    out.add_(acc.view_as(out))


class _Plan:
    """Dimensions et lancements des noyaux v2/v3 pour une précision (un layout de record)."""

    def __init__(self, layout: dict, version: int = 2):
        prec = layout["precision"]
        self.prec = prec
        self.kind = prec if prec in ("int4", "int8") else "float"
        f = {name: (shape, dtype, off) for name, shape, dtype, off in layout["fields"]}
        if self.kind == "int8":
            (two_i, self.H), _, _ = f["gate_up.q"]
            self.offs = [f["gate_up.q"][2], f["gate_up.s"][2], f["down.q"][2], f["down.s"][2]]
            self.ss_up, self.ss_dn = f["gate_up.s"][0][1], f["down.s"][0][1]
            g = layout["group_sizes"]
            self.g_up, self.g_dn = g["gate_up"], g["down"]
        elif self.kind == "int4":
            (two_i, self.ph), _, _ = f["gate_up.q"]
            (self.H, self.pi), _, _ = f["down.q"]
            self.offs = [f["gate_up.q"][2], f["gate_up.s"][2], f["down.q"][2], f["down.s"][2]]
            self.ss_up, self.ss_dn = f["gate_up.s"][0][1], f["down.s"][0][1]
            g = layout["group_sizes"]
            self.gb_up, self.gb_dn = g["gate_up"] // 2, g["down"] // 2
            self.version = version
            self.magic = version == 3
        else:
            (two_i, self.H), dt, _ = f["gate_up"]
            self.offs = [f["gate_up"][2], f["gate_up"][2], f["down"][2], f["down"][2]]
            self.wdt = _WDT[{"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dt]]
        self.I = two_i // 2

    def up(self, n, tab, w, x, h, acc, zero: bool):
        meta = _meta()
        if self.kind == "int8":
            _grp8_up[lambda m: (n, triton.cdiv(self.I, m["BO"]))](
                tab, w, x, h, acc, self.I, self.H, self.ss_up, GB=self.g_up,
                EVEN=_even((self.I, 64), (self.H, 256)), ZERO=zero, **meta)
        elif self.kind == "int4" and self.version == 4:
            _launch4_up(n, tab, w, x, h, acc, self.I, self.ph, self.H, self.ss_up, self.gb_up, zero)
        elif self.kind == "int4":
            _grp4_up[lambda m: (n, triton.cdiv(self.I, m["BO"]))](
                tab, w, x, h, acc, self.I, self.ph, self.H, self.ss_up, GB=self.gb_up,
                EVEN=_even((self.I, 64), (self.ph, 256)), ZERO=zero, MAGIC=self.magic, **meta)
        else:
            _grpf_up[lambda m: (n, triton.cdiv(self.I, m["BO"]))](
                tab, w, x, h, acc, self.I, self.H, WDT=self.wdt, EVEN=_even((self.I, 64), (self.H, 256)), ZERO=zero, **meta)

    def down(self, n, tab, w, h, acc):
        meta = _meta()
        if self.kind == "int8":
            _grp8_down[lambda m: (triton.cdiv(self.H, m["BO"]), n)](
                tab, w, h, acc, self.H, self.I, self.ss_dn, GB=self.g_dn,
                EVEN=_even((self.H, 64), (self.I, 256)), **meta)
        elif self.kind == "int4" and self.version == 4:
            _launch4_down(n, tab, w, h, acc, self.H, self.I, self.pi, self.ss_dn, self.gb_dn)
        elif self.kind == "int4":
            _grp4_down[lambda m: (triton.cdiv(self.H, m["BO"]), n)](
                tab, w, h, acc, self.H, self.I, self.pi, self.ss_dn, GB=self.gb_dn,
                EVEN=_even((self.H, 64), (self.pi, 256)), MAGIC=self.magic, **meta)
        else:
            _grpf_down[lambda m: (triton.cdiv(self.H, m["BO"]), n)](
                tab, w, h, acc, self.H, self.I, WDT=self.wdt, EVEN=_even((self.H, 64), (self.I, 256)), **meta)


def grouped_ffn_int4(x_row: torch.Tensor, entries: list, groups: dict, out: torch.Tensor, version: int = 2) -> None:
    """
    out (1, H) += somme des experts int4 : entries = [(views(record), poids du routeur), ...].
    Tous les records doivent avoir le même layout (même précision, mêmes formes).
    version : 1 = noyaux du 30/09, 2 = v2, 3 = v2 + conversion MAGIC des quartets,
    4 = blocs 2D, une réduction (voir plus haut), 0 = la plus rapide sur cette carte.
    """
    v0 = entries[0][0]
    two_i, ph = v0["gate_up.q"].shape
    H, pi = v0["down.q"].shape
    I = two_i // 2
    n = len(entries)
    dev = x_row.device
    tab, w = _table(entries, ("gate_up.q", "gate_up.s", "down.q", "down.s"), dev)
    h = torch.empty(n, I, dtype=torch.float32, device=dev)
    x = x_row.reshape(-1).contiguous()
    if version >= 2 and not _pow2_groups(groups):
        version = 1  # groupes qui ne sont pas des puissances de 2 (ou > 128 colonnes) : noyaux v1
    if version == 1:
        _grp_int4_up[(n, triton.cdiv(I, BLOCK_O))](tab, x, h, I, ph, v0["gate_up.s"].stride(0),
                                                   GS=groups["gate_up"], BO=BLOCK_O, BP=BLOCK_P)
        _grp_int4_down[(triton.cdiv(H, BLOCK_O),)](tab, w, h, out, n, H, I, pi, v0["down.s"].stride(0),
                                                   GS=groups["down"], BO=BLOCK_O, BP=BLOCK_P)
        return
    acc = torch.empty(H, dtype=torch.float32, device=dev)
    if version == 4:
        _launch4_up(n, tab, w, x, h, acc, I, ph, H, v0["gate_up.s"].stride(0), groups["gate_up"] // 2, True)
        _launch4_down(n, tab, w, h, acc, H, I, pi, v0["down.s"].stride(0), groups["down"] // 2)
        out.add_(acc.view_as(out))
        return
    meta = _meta()
    magic = version == 3
    _grp4_up[lambda m: (n, triton.cdiv(I, m["BO"]))](tab, w, x, h, acc, I, ph, H, v0["gate_up.s"].stride(0),
                                                    GB=groups["gate_up"] // 2, EVEN=_even((I, 64), (ph, 256)),
                                                    ZERO=True, MAGIC=magic, **meta)
    _grp4_down[lambda m: (triton.cdiv(H, m["BO"]), n)](tab, w, h, acc, H, I, pi, v0["down.s"].stride(0),
                                                      GB=groups["down"] // 2, EVEN=_even((H, 64), (pi, 256)),
                                                      MAGIC=magic, **meta)
    out.add_(acc.view_as(out))


def grouped_ffn_float(x_row: torch.Tensor, entries: list, out: torch.Tensor) -> None:
    """Même chose pour des experts en fp16 / bf16 / fp32 (produit matrice-vecteur, sans cuBLAS)."""
    v0 = entries[0][0]
    two_i, H = v0["gate_up"].shape
    I = two_i // 2
    n = len(entries)
    dev = x_row.device
    tab, w = _table(entries, ("gate_up", "gate_up", "down", "down"), dev)
    h = torch.empty(n, I, dtype=torch.float32, device=dev)
    acc = torch.empty(H, dtype=torch.float32, device=dev)
    x = x_row.reshape(-1).contiguous()
    wdt = _WDT[v0["gate_up"].dtype]
    meta = _meta()
    _grpf_up[lambda m: (n, triton.cdiv(I, m["BO"]))](tab, w, x, h, acc, I, H, WDT=wdt, EVEN=_even((I, 64), (H, 256)),
                                                    ZERO=True, **meta)
    _grpf_down[lambda m: (triton.cdiv(H, m["BO"]), n)](tab, w, h, acc, H, I, WDT=wdt, EVEN=_even((H, 64), (I, 256)), **meta)
    out.add_(acc.view_as(out))


# ----------------------------------------------------------------------------
# A6 bis : le calcul groupé d'une couche MoE rejoué en UN CUDA graph.
#
# Le profil du 01/10 (CUDA graphs) : par couche, ~0,7 ms de temps CPU pour préparer et
# lancer les noyaux groupés (vues des records, table, 2-4 lancements Triton, chacun
# coûteux en Python), plus que les noyaux eux-mêmes (~0,4 ms). Ici, tout ce qui ne
# change pas d'un token à l'autre est figé dans un graph par couche :
#     table hôte (épinglée, adresse fixe) -> table GPU, noyaux A et B pour chaque
#     précision du store (int4, fp16), out = accumulateur + part CPU (q*)
# Par token, le CPU écrit seulement k adresses et k poids dans la table hôte (numpy),
# puis relance le graph : un seul lancement. La table a une taille fixe de k lignes par
# précision ; les lignes inutilisées ont un poids 0 et leurs programmes ne font rien.
# ----------------------------------------------------------------------------


class GroupedGraph:
    def __init__(self, layouts: dict, x: torch.Tensor, out: torch.Tensor, k: int, version: int = 2,
                 capture: bool = True):
        self.dev = x.device
        self.x, self.out, self.k, self.version = x.reshape(-1), out, k, version
        self.plans: dict[str, _Plan] = {p: _Plan(lay, version) for p, lay in layouts.items()}
        H = out.shape[-1]
        pin = self.dev.type == "cuda"
        self.tabs = {}
        for p, plan in self.plans.items():
            host = torch.zeros(k * 36, dtype=torch.uint8, pin_memory=pin)
            dev = torch.empty_like(host, device=self.dev)
            self.tabs[p] = {
                "host": host, "dev": dev,
                "tab_np": host[: k * 32].view(torch.int64).numpy(), "w_np": host[k * 32:].view(torch.float32).numpy(),
                "tab": dev[: k * 32].view(torch.int64), "w": dev[k * 32:].view(torch.float32),
                "h": torch.empty(k, plan.I, dtype=torch.float32, device=self.dev),
            }
        self.acc = torch.empty(H, dtype=torch.float32, device=self.dev)
        self.cpu_part = torch.zeros_like(out)
        self.cpu_dirty = False
        self.capture = capture and self.dev.type == "cuda"
        self.graph = None
        self.replays = 0
        self.runs = 0

    def supports(self, prec: str) -> bool:
        return prec in self.plans

    def _sequence(self) -> None:
        for t in self.tabs.values():
            t["dev"].copy_(t["host"], non_blocking=True)
        first = True
        for p, plan in self.plans.items():
            t = self.tabs[p]
            plan.up(self.k, t["tab"], t["w"], self.x, t["h"], self.acc, zero=first)
            first = False
        for p, plan in self.plans.items():
            t = self.tabs[p]
            plan.down(self.k, t["tab"], t["w"], t["h"], self.acc)
        self.out.copy_(self.acc.view_as(self.out))
        self.out.add_(self.cpu_part)

    def run(self, items: dict, cpu_out: torch.Tensor | None = None) -> None:
        """items : {précision: [(record sur le GPU, poids), ...]} ; cpu_out : somme des experts calculés sur CPU."""
        self.runs += 1
        for p, t in self.tabs.items():
            tab, w = t["tab_np"], t["w_np"]
            tab[:] = 0
            w[:] = 0.0
            offs = self.plans[p].offs
            for j, (blob, wt) in enumerate(items.get(p, ())):
                base = blob.data_ptr()
                tab[4 * j: 4 * j + 4] = [base + o for o in offs]
                w[j] = wt
        if cpu_out is not None:
            self.cpu_part.copy_(cpu_out)
            self.cpu_dirty = True
        elif self.cpu_dirty:
            self.cpu_part.zero_()
            self.cpu_dirty = False
        if not self.capture:
            self._sequence()
            return
        if self.graph is not None:
            self.graph.replay()
            self.replays += 1
            return
        # 1er appel : passe normale (autotune, résultat de ce token), puis capture pour les suivants
        self._sequence()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._sequence()
        self.graph = g
        torch.cuda.synchronize()
