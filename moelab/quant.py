"""
Quantification des poids d'experts.

Deux formats entiers, symétriques, avec une échelle par groupe de `group_size`
colonnes (gs64 par défaut, le format que Colibrì a mesuré comme le meilleur
compromis : ~44% d'erreur de quantification en moins qu'une échelle par ligne) :

- int8 : une valeur par octet, dans [-127, 127]
- int4 : deux valeurs par octet, dans [-8, 7], stockées décalées de +8 dans [0, 15]
         (quartet bas = colonne paire, quartet haut = colonne impaire)

Pour une matrice W de forme (sorties, entrées) :
    q = arrondi(w / échelle)            (entier)
    w_approx = q * échelle              (déquantification)

On quantifie le long de la dimension d'entrée (celle qu'on multiplie par x), groupe
par groupe, ligne par ligne.

Choix de l'échelle d'un groupe (`method`) :
  - "absmax" : max(|w|) / qmax. Le plus simple : en int4, la valeur -8 ne sert jamais
    et un seul poids extrême fixe le pas de tout le groupe.
  - "mse" (Q1, 02/10, défaut des nouveaux stores int4) : on essaie plusieurs échelles
    et on garde celle qui minimise l'erreur du groupe. Candidats : max|w|/7 et
    « max signé / -8 » (le plus grand poids tombe sur -8, comme le Q4_0 de llama.cpp :
    un niveau de plus), chacun multiplié par 1,00, 0,98 ... 0,70 (écrêter un poids
    extrême pour affiner le pas de tous les autres). Jamais pire que "absmax" (le
    candidat 1,00 en fait partie). Même format, mêmes noyaux : seule l'échelle change.
    L'échelle peut être négative (max signé / -8) : la déquantification la multiplie.
  - GPTQ (Q2, `gptq_quantize`) : en plus, l'erreur de chaque colonne est compensée sur
    les colonnes suivantes, d'après les vraies entrées de la couche (calibration).

Ce module contient aussi la déquantification FP8 "par blocs" utilisée par les
checkpoints officiels *-FP8 de Qwen et DeepSeek (poids float8_e4m3 + une échelle
float32 `weight_scale_inv` par bloc de 128x128).
"""

from __future__ import annotations

import math

import torch

QMAX = {8: 127, 4: 7}
QMIN = {8: -127, 4: -8}
SCALE_METHODS = ("absmax", "mse")
_GRID = [1.0 - 0.02 * i for i in range(16)]  # 1,00 ... 0,70


def pick_group_size(in_features: int, preferred: int = 64) -> int:
    """Plus grande taille de groupe <= preferred qui divise in_features (et paire, pour int4)."""
    for gs in range(min(preferred, in_features), 1, -1):
        if in_features % gs == 0 and gs % 2 == 0:
            return gs
    raise ValueError(f"impossible de trouver une taille de groupe paire qui divise {in_features}")


def _safe(s: torch.Tensor) -> torch.Tensor:
    """Échelle arrondie en float16 (celle qui sera stockée), jamais nulle."""
    s16 = s.to(torch.float16)
    tiny = torch.tensor(1e-7, dtype=torch.float16, device=s.device)
    return torch.where(s16 == 0, tiny, s16)


def choose_scales(wf: torch.Tensor, bits: int, method: str = "absmax") -> torch.Tensor:
    """
    wf : (..., gs) float32, un groupe par ligne de la dernière dimension.
    Retourne l'échelle float16 de chaque groupe, forme (..., 1). On quantifie ensuite
    avec exactement cette échelle (arrondie en float16 AVANT de quantifier : sinon
    l'arrondi ajoute une erreur évitable).
    """
    if method not in SCALE_METHODS:
        raise ValueError(f"méthode d'échelle inconnue : {method} (choix : {', '.join(SCALE_METHODS)})")
    qmin, qmax = QMIN[bits], QMAX[bits]
    absmax = wf.abs().amax(dim=-1, keepdim=True)
    if method == "absmax":
        return _safe((absmax / qmax).clamp(min=1e-7))
    bases = [absmax / qmax]
    if qmin == -qmax - 1:  # int4 : le plus grand poids (en valeur absolue) sur -8
        mx = wf.gather(-1, wf.abs().argmax(dim=-1, keepdim=True))
        bases.append(mx / qmin)
    best_s, best_e = None, None
    for b in bases:
        for f in _GRID:
            s = _safe(b * f)
            sf = s.float()
            err = ((torch.round(wf / sf).clamp(qmin, qmax) * sf - wf) ** 2).sum(dim=-1, keepdim=True)
            if best_s is None:
                best_s, best_e = s, err
            else:
                better = err < best_e
                best_s = torch.where(better, s, best_s)
                best_e = torch.where(better, err, best_e)
    return best_s


def _pack(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Entiers (out, in) -> format du store (int8, ou int4 deux par octet)."""
    if bits == 8:
        return q.to(torch.int8).contiguous()
    u = (q + 8).to(torch.uint8)  # [-8, 7] -> [0, 15]
    return (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()


def quantize(w: torch.Tensor, bits: int, group_size: int, method: str = "absmax",
             chunk_rows: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """
    w : (out, in) en n'importe quel dtype flottant (le calcul se fait sur son device).
    Retourne (q, scales) :
      - int8 : q de forme (out, in) en torch.int8
      - int4 : q de forme (out, in // 2) en torch.uint8 (deux valeurs par octet)
      - scales : (out, in // group_size) en float16
    """
    if bits not in QMAX:
        raise ValueError(f"bits doit valoir 4 ou 8, pas {bits}")
    out_f, in_f = w.shape
    if in_f % group_size:
        raise ValueError(f"in_features={in_f} n'est pas divisible par group_size={group_size}")
    qs, ss = [], []
    for r0 in range(0, out_f, chunk_rows):  # par paquets de lignes : la recherche d'échelle garde la mémoire bornée
        wf = w[r0:r0 + chunk_rows].float().reshape(-1, in_f // group_size, group_size)
        s = choose_scales(wf, bits, method)
        q = torch.round(wf / s.float()).clamp(QMIN[bits], QMAX[bits]).to(torch.int16)
        qs.append(q.reshape(-1, in_f))
        ss.append(s.reshape(-1, in_f // group_size))
    q = torch.cat(qs) if len(qs) > 1 else qs[0]
    s = torch.cat(ss) if len(ss) > 1 else ss[0]
    return _pack(q, bits), s.contiguous()


def gptq_quantize(w: torch.Tensor, hessian: torch.Tensor, bits: int, group_size: int, method: str = "mse",
                  damp: float = 0.01, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPTQ (Frantar et al., 2023) au format du store : mêmes entiers, mêmes échelles par
    groupe, donc mêmes noyaux. Les colonnes sont quantifiées une à une ; l'erreur de
    chacune est reportée sur les colonnes pas encore quantifiées, pondérée par l'inverse
    de H = somme des x xᵀ (les vraies entrées de la couche) : ce qui compte, c'est
    l'erreur sur la SORTIE W·x, pas sur les poids. L'échelle d'un groupe est choisie
    (`method`) au moment où l'on atteint sa première colonne, sur les poids déjà corrigés.

    w : (out, in) ; hessian : (in, in), même device (GPU conseillé). Retourne (q, scales)
    comme `quantize`.
    """
    if bits not in QMAX:
        raise ValueError(f"bits doit valoir 4 ou 8, pas {bits}")
    W = w.float().clone()
    out_f, in_f = W.shape
    if in_f % group_size or block % group_size:
        raise ValueError(f"in={in_f}, bloc={block} doivent être multiples de group_size={group_size}")
    qmin, qmax = QMIN[bits], QMAX[bits]
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H.diagonal().add_(damp * torch.mean(torch.diag(H)))
    # Hinv = chol(H^-1), triangulaire supérieure ; on augmente l'amortissement si besoin
    for _ in range(6):
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
            break
        except RuntimeError:
            H.diagonal().add_(damp * torch.mean(torch.diag(H)))
    else:
        return quantize(w, bits, group_size, method)

    Q = torch.zeros(out_f, in_f, dtype=torch.int16, device=W.device)
    S = torch.zeros(out_f, in_f // group_size, dtype=torch.float16, device=W.device)
    for i1 in range(0, in_f, block):
        i2 = min(i1 + block, in_f)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        sf = None
        for i in range(i2 - i1):
            c = i1 + i
            if c % group_size == 0:
                g = c // group_size
                s = choose_scales(W1[:, i:i + group_size].unsqueeze(1), bits, method).reshape(-1)
                S[:, g] = s
                sf = s.float()
            col = W1[:, i]
            q = torch.round(col / sf).clamp(qmin, qmax)
            Q[:, c] = q.to(torch.int16)
            err = (col - q * sf) / Hi[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hi[i, i:].unsqueeze(0)
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return _pack(Q, bits), S.contiguous()


def dequantize(q: torch.Tensor, scales: torch.Tensor, bits: int, dtype: torch.dtype) -> torch.Tensor:
    """Inverse de `quantize`. Fonctionne sur CPU comme sur GPU (le calcul se fait là où sont q et scales)."""
    out_f = q.shape[0]
    if bits == 8:
        vals = q.to(dtype)
    elif bits == 4:
        lo = q & 0x0F
        hi = q >> 4
        vals = torch.stack((lo, hi), dim=-1).reshape(out_f, q.shape[1] * 2).to(dtype) - 8
    else:
        raise ValueError(f"bits doit valoir 4 ou 8, pas {bits}")
    in_f = vals.shape[1]
    group_size = in_f // scales.shape[1]
    vals = vals.reshape(out_f, in_f // group_size, group_size) * scales.to(dtype).unsqueeze(-1)
    return vals.reshape(out_f, in_f)


def dequantize_fp8_block(w: torch.Tensor, scale_inv: torch.Tensor, block_size: tuple[int, int] = (128, 128)) -> torch.Tensor:
    """
    Déquantifie un poids FP8 "par blocs" (format des checkpoints Qwen/DeepSeek *-FP8).
    Malgré son nom, `weight_scale_inv` se MULTIPLIE : W = w_fp8 * scale(bloc).
    """
    bo, bi = block_size
    out_f, in_f = w.shape
    expected = (math.ceil(out_f / bo), math.ceil(in_f / bi))
    if tuple(scale_inv.shape) != expected:
        raise ValueError(f"échelles FP8 de forme {tuple(scale_inv.shape)}, attendu {expected} pour {tuple(w.shape)} et blocs {block_size}")
    s = scale_inv.float().repeat_interleave(bo, dim=0)[:out_f].repeat_interleave(bi, dim=1)[:, :in_f]
    return w.float() * s


def bits_per_weight(bits: int, group_size: int) -> float:
    """Coût réel en bits par poids, échelle float16 comprise (int4 gs64 -> 4.25)."""
    return bits + 16 / group_size
