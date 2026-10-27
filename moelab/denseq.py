"""
D5 (doc 18) : la partie dense en int8 (couches Linear de l'attention, experts partagés,
tête de sortie). Option `dense_int8` au chargement.

Pourquoi : une fois les experts optimisés, la partie dense est relue en entier à chaque
token. Sur OLMoE, ~0,75 Go par token en fp16 (projections de l'attention + tête de
sortie de 206 Mo) ; sur Qwen3-Next-80B, ~4 Go de partie dense en fp16 dont une tête de
622 Mo, à faire tenir dans 6 Go de VRAM avec les experts. En int8 par groupes de 64
(quasi sans perte, comme pour les experts) : 2 fois moins d'octets lus par token, et
la VRAM libérée sert au cache d'experts.

QLinear remplace nn.Linear :
  - génération (1 token) : un noyau Triton GEMV qui lit l'int8 directement (capturable
    dans les CUDA graphs) ;
  - prefill (plusieurs tokens) : déquantification par blocs de lignes puis F.linear.
Le routeur des couches MoE n'est pas touché (petit, et ses décisions comptent).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import HAVE_TRITON, INTERPRET, _even, _tuned
from .quant import dequantize, pick_group_size, quantize

if HAVE_TRITON:
    import triton
    import triton.language as tl

    @_tuned(key=["O", "K"])
    @triton.jit
    def _gemv8(q_ptr, s_ptr, x_ptr, y_ptr, O, K, stride_s,
               GB: tl.constexpr, EVEN: tl.constexpr, BO: tl.constexpr, BPW: tl.constexpr):
        NG: tl.constexpr = BPW // GB
        rows = tl.program_id(0) * BO + tl.arange(0, BO)
        gi = tl.arange(0, NG)
        bi = tl.arange(0, GB)
        acc = tl.zeros((BO,), dtype=tl.float32)
        for k0 in range(0, K, BPW):
            col = k0 + gi[:, None] * GB + bi[None, :]
            sg = k0 // GB + gi
            qo = rows[:, None, None] * K + col[None, :, :]
            so = rows[:, None] * stride_s + sg[None, :]
            if EVEN:
                xv = tl.load(x_ptr + col).to(tl.float32)
                q = tl.load(q_ptr + qo).to(tl.float32)
                sc = tl.load(s_ptr + so).to(tl.float32)
            else:
                cm = col < K
                xv = tl.load(x_ptr + col, mask=cm, other=0.0).to(tl.float32)
                q = tl.load(q_ptr + qo, mask=(rows < O)[:, None, None] & cm[None, :, :], other=0).to(tl.float32)
                sc = tl.load(s_ptr + so, mask=(rows < O)[:, None] & (sg * GB < K)[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(tl.sum(q * xv[None, :, :], axis=2) * sc, axis=1)
        tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=rows < O)


def gemv_int8(q: torch.Tensor, s: torch.Tensor, x: torch.Tensor, group_size: int) -> torch.Tensor:
    """y = (q * s) @ x pour UN vecteur x (K,) ; q (O, K) int8, s (O, K/gs) fp16. Retourne (O,) dans le dtype de x."""
    O, K = q.shape
    y = torch.empty(O, dtype=x.dtype, device=x.device)
    meta = {"BO": 16, "BPW": 64} if INTERPRET else {}
    _gemv8[lambda m: (triton.cdiv(O, m["BO"]),)](q, s, x, y, O, K, s.stride(0), GB=group_size,
                                                 EVEN=_even((O, 64), (K, 256)), **meta)
    return y


class QLinear(nn.Module):
    """nn.Linear en int8 par groupes (poids seuls ; calcul en float)."""

    def __init__(self, q: torch.Tensor, s: torch.Tensor, group_size: int, bias: torch.Tensor | None, out_dtype):
        super().__init__()
        self.register_buffer("q", q, persistent=False)
        self.register_buffer("s", s, persistent=False)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.group_size = group_size
        self.in_features, self.out_features = q.shape[1], q.shape[0]
        self.out_dtype = out_dtype
        pow2 = group_size & (group_size - 1) == 0 and group_size <= 64
        self.kernel = HAVE_TRITON and pow2 and (q.is_cuda or INTERPRET)

    @classmethod
    def from_linear(cls, lin: nn.Linear, device, group_size: int = 64) -> "QLinear":
        w = lin.weight.data
        gs = pick_group_size(w.shape[1], group_size)
        q, s = quantize(w.to(device), 8, gs)
        bias = None if lin.bias is None else lin.bias.data.to(device)
        return cls(q, s, gs, bias, w.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        n = x.numel() // self.in_features
        if n == 1 and self.kernel and (x.is_cuda or INTERPRET):
            y = gemv_int8(self.q, self.s, x.reshape(-1).contiguous(), self.group_size).reshape(*lead, -1)
        else:
            # plusieurs tokens (prefill) : déquantification par blocs de lignes (borne la mémoire temporaire)
            parts = []
            step = max(1, (64 * 2**20) // (self.in_features * x.element_size()))
            for o in range(0, self.out_features, step):
                w = dequantize(self.q[o: o + step], self.s[o: o + step], 8, x.dtype)
                parts.append(F.linear(x, w))
            y = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int8 gs{self.group_size}"


def quantizable(model, min_out: int = 64) -> list[tuple[str, nn.Linear]]:
    """Les nn.Linear de la partie dense à passer en int8 (pas les petites têtes comme shared_expert_gate)."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and m.out_features >= min_out and m.weight.dim() == 2 and m.in_features % 2 == 0]


def linear_bytes(model, min_out: int = 64) -> int:
    return sum(m.weight.numel() * m.weight.element_size() for _, m in quantizable(model, min_out))


@torch.no_grad()
def quantize_dense(model, device, log=print) -> dict:
    """Remplace les nn.Linear de la partie dense par des QLinear (un à la fois : pic mémoire = une matrice)."""
    before = after = 0
    n = 0
    for name, lin in quantizable(model):
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        before += lin.weight.numel() * lin.weight.element_size()
        ql = QLinear.from_linear(lin, device)
        after += ql.q.numel() + ql.s.numel() * 2
        setattr(parent, attr, ql)
        n += 1
        del lin
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"  partie dense en int8 : {n} couches Linear, {before / 2**30:.2f} -> {after / 2**30:.2f} Gio")
    return {"layers": n, "bytes_before": before, "bytes_after": after}
