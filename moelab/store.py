"""
Store d'experts sur disque.

Organisation :
    <store>/manifest.json          description du store (modèle, formes, précisions, offsets)
    <store>/layer000.fp16.bin      les E experts de la couche MoE 0, en fp16
    <store>/layer000.int4.bin      les mêmes experts, en int4
    ...

Chaque fichier est une suite de "records" de taille fixe, un par expert. Un record
contient TOUT ce qu'il faut pour calculer l'expert (gate_up + down, et leurs
échelles s'il est quantifié), rangé de façon contiguë : charger un expert = UNE
lecture disque (idée reprise de Colibrì : "each expert's three matrices are stored
adjacent and read in one pread"). Les records sont alignés sur 4096 octets.

Le même record, une fois en RAM ou en VRAM, est un simple tenseur uint8 1D :
`views()` en tire des tenseurs typés sans copie, `materialize()` les déquantifie.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .quant import bits_per_weight, dequantize, pick_group_size, quantize

ALIGN = 4096
FIELD_ALIGN = 64

PRECISIONS = {
    "fp32": {"kind": "float", "dtype": "float32"},
    "fp16": {"kind": "float", "dtype": "float16"},
    "bf16": {"kind": "float", "dtype": "bfloat16"},
    "int8": {"kind": "int", "bits": 8},
    "int4": {"kind": "int", "bits": 4},
}

TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "uint8": torch.uint8,
}
ELEM_SIZE = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "uint8": 1}


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


@dataclass(frozen=True)
class Field:
    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int

    @property
    def nbytes(self) -> int:
        n = ELEM_SIZE[self.dtype]
        for s in self.shape:
            n *= s
        return n


def record_layout(prec: str, hidden: int, inter: int, group_size: int = 64, scale: str | None = None) -> dict:
    """
    Décrit le contenu d'un record d'expert pour une précision donnée. `scale` : choix de
    l'échelle des formats entiers (quant.py) ; None = "mse" pour l'int4 (Q1, 02/10),
    "absmax" pour l'int8 (déjà presque sans perte). "gptq" : records réécrits par
    scripts/requant_store.py (même format).
    """
    spec = PRECISIONS[prec]
    if spec["kind"] == "int" and scale is None:
        scale = "mse" if spec["bits"] == 4 else "absmax"
    mats = {"gate_up": (2 * inter, hidden), "down": (hidden, inter)}
    raw: list[tuple[str, tuple[int, ...], str]] = []
    groups = {}
    if spec["kind"] == "float":
        for name, shape in mats.items():
            raw.append((name, shape, spec["dtype"]))
    else:
        bits = spec["bits"]
        for name, (o, i) in mats.items():
            gs = pick_group_size(i, group_size)
            groups[name] = gs
            if bits == 8:
                raw.append((f"{name}.q", (o, i), "int8"))
            else:
                raw.append((f"{name}.q", (o, i // 2), "uint8"))
            raw.append((f"{name}.s", (o, i // gs), "float16"))
    fields, off = [], 0
    for name, shape, dtype in raw:
        f = Field(name, shape, dtype, off)
        fields.append(f)
        off = _align(off + f.nbytes, FIELD_ALIGN)
    lay = {
        "precision": prec,
        "fields": [[f.name, list(f.shape), f.dtype, f.offset] for f in fields],
        "record_bytes": _align(off, ALIGN),
        "group_sizes": groups,
    }
    if spec["kind"] == "int":
        lay["scale"] = scale
    return lay


def _fields(layout: dict) -> list[Field]:
    return [Field(n, tuple(s), d, o) for n, s, d, o in layout["fields"]]


def encode_record(layout: dict, gate_up: torch.Tensor, down: torch.Tensor,
                  quantized: dict | None = None) -> np.ndarray:
    """
    Construit les octets d'un record à partir des poids en float32. `quantized` :
    {"gate_up": (q, s), "down": (q, s)} déjà quantifiés ailleurs (GPTQ), même format.
    """
    prec = layout["precision"]
    spec = PRECISIONS[prec]
    buf = np.zeros(layout["record_bytes"], dtype=np.uint8)
    tensors: dict[str, torch.Tensor] = {}
    if spec["kind"] == "float":
        dt = TORCH_DTYPES[spec["dtype"]]
        tensors["gate_up"] = gate_up.to(dt).cpu()
        tensors["down"] = down.to(dt).cpu()
    else:
        method = layout.get("scale") or "absmax"
        for name, w in (("gate_up", gate_up), ("down", down)):
            if quantized and name in quantized:
                q, s = quantized[name]
            else:
                q, s = quantize(w, spec["bits"], layout["group_sizes"][name], "absmax" if method == "gptq" else method)
            tensors[f"{name}.q"], tensors[f"{name}.s"] = q.cpu(), s.cpu()
    for f in _fields(layout):
        t = tensors[f.name].contiguous()
        if tuple(t.shape) != f.shape:
            raise ValueError(f"{f.name}: forme {tuple(t.shape)} au lieu de {f.shape}")
        raw = t.view(torch.uint8).reshape(-1).numpy() if t.dtype != torch.uint8 else t.reshape(-1).numpy()
        buf[f.offset : f.offset + f.nbytes] = raw
    return buf


def views(layout: dict, blob: torch.Tensor) -> dict[str, torch.Tensor]:
    """Tenseurs typés qui pointent dans le record (aucune copie). Marche sur CPU et GPU."""
    out = {}
    for f in _fields(layout):
        part = blob[f.offset : f.offset + f.nbytes]
        if f.dtype != "uint8":
            part = part.view(TORCH_DTYPES[f.dtype])
        out[f.name] = part.view(f.shape)
    return out


def materialize(layout: dict, blob: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Record -> (gate_up, down) dans `dtype`, sur le device du record (déquantifie si besoin)."""
    v = views(layout, blob)
    spec = PRECISIONS[layout["precision"]]
    if spec["kind"] == "float":
        return v["gate_up"].to(dtype), v["down"].to(dtype)
    bits = spec["bits"]
    return (
        dequantize(v["gate_up.q"], v["gate_up.s"], bits, dtype),
        dequantize(v["down.q"], v["down.s"], bits, dtype),
    )


def unpacked_layout(layout: dict) -> dict | None:
    """
    B2 (doc 18) : la forme « déballée » d'un record int4 = le layout int8, mêmes groupes.
    Les valeurs int4 y sont rangées une par octet (dans [-8, 7]) : la déquantification ne
    coûte plus que 2 opérations par matrice au lieu de 6-7, pour 2 fois plus de place.
    """
    if layout["precision"] != "int4":
        return None
    (o, i2) = [f for f in _fields(layout) if f.name == "gate_up.q"][0].shape
    hidden, inter = i2 * 2, o // 2
    gs = max(layout["group_sizes"].values())
    l8 = record_layout("int8", hidden, inter, gs)
    return l8 if l8["group_sizes"] == layout["group_sizes"] else None


def unpack_int4_record(blob: torch.Tensor, layout4: dict, layout8: dict) -> torch.Tensor:
    """Record int4 -> record au format int8 (valeurs identiques), sur le device de `blob`."""
    v4 = views(layout4, blob)
    out = torch.empty(layout8["record_bytes"], dtype=torch.uint8, device=blob.device)
    v8 = views(layout8, out)
    for name in ("gate_up", "down"):
        q = v4[f"{name}.q"]
        dst = v8[f"{name}.q"].view(q.shape[0], q.shape[1], 2)
        dst[..., 0] = (q & 0x0F).to(torch.int8) - 8
        dst[..., 1] = (q >> 4).to(torch.int8) - 8
        v8[f"{name}.s"].copy_(v4[f"{name}.s"])
    return out


class _DirectFile:
    """
    Lecture d'un fichier SANS le cache de fichiers du système (optimisation D1, doc 18) :
    Windows : FILE_FLAG_NO_BUFFERING ; Linux : O_DIRECT. Deux contraintes : la position et
    la taille lues sont des multiples de 4096 (les records le sont), et l'adresse du
    tampon aussi (les slots du pool le sont). Sinon, `read_into` renvoie False et
    l'appelant fait une lecture ordinaire.
    Pourquoi : le cache de Windows faussait les benchmarks (le store finissait en RAM
    côté système), et il ajoute une copie mémoire à chaque lecture.
    """

    def __init__(self, path: Path):
        self.path = path
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class OVERLAPPED(ctypes.Structure):
                _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p),
                            ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateFileW.restype = wintypes.HANDLE
            k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
            k32.ReadFile.restype = wintypes.BOOL
            k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                     ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(OVERLAPPED)]
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, NO_BUFFERING = 0x80000000, 0x1, 3, 0x20000000
            h = k32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, NO_BUFFERING, None)
            if h is None or h == wintypes.HANDLE(-1).value:
                raise OSError(ctypes.get_last_error(), f"CreateFileW({path}) sans cache")
            self._k32, self._h, self._ov, self._ct, self._wt = k32, h, OVERLAPPED, ctypes, wintypes
        else:
            self._fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))

    def read_into(self, out: torch.Tensor, offset: int) -> bool:
        n = out.numel()
        if out.data_ptr() % ALIGN or offset % ALIGN or n % ALIGN:
            return False
        if sys.platform == "win32":
            ct, wt = self._ct, self._wt
            ov = self._ov(Offset=offset & 0xFFFFFFFF, OffsetHigh=offset >> 32)
            got = wt.DWORD(0)
            ok = self._k32.ReadFile(self._h, ct.c_void_p(out.data_ptr()), n, ct.byref(got), ct.byref(ov))
            if not ok:
                raise OSError(ct.get_last_error(), f"ReadFile sans cache ({self.path})")
            return got.value == n
        return os.preadv(self._fd, [memoryview(out.numpy())], offset) == n

    def close(self) -> None:
        if sys.platform == "win32":
            self._k32.CloseHandle(self._h)
        else:
            os.close(self._fd)


class ExpertStore:
    """Accès en lecture au store. Thread-safe : chaque thread a ses propres descripteurs de fichier."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        mpath = self.path / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(f"pas de manifest.json dans {self.path} (le store a-t-il été construit ?)")
        self.manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if not self.manifest.get("complete"):
            raise RuntimeError(f"le store {self.path} est incomplet : relance build_store.py avec --resume")
        self.layers: list[str] = self.manifest["layers"]
        self.num_experts: int = self.manifest["num_experts"]
        self.hidden: int = self.manifest["hidden"]
        self.inter: int = self.manifest["inter"]
        self.layouts: dict[str, dict] = {l["precision"]: l for l in self.manifest["layouts"]}
        self.precisions: list[str] = [l["precision"] for l in self.manifest["layouts"]]  # plus précise d'abord
        self._local = threading.local()
        self._all_handles: list = []
        self.bytes_read = 0
        self.read_seconds = 0.0
        self._lock = threading.Lock()
        self.direct = False          # D1 : lectures sans cache du système (activé par le runtime)
        self.direct_reads = 0
        self.direct_fallbacks = 0
        self._direct_error: str | None = None
        self.hot: dict[str, dict[tuple[int, int], int]] = {}  # D4 : experts copiés sur un disque rapide
        self.hot_path: Path | None = None
        self.hot_reads = 0
        self.read_parts = 1          # morceaux simultanés par record (_split_reads), mesuré par measure_bandwidth.py
        self._split_pool = None

    @property
    def hi(self) -> str:
        return self.precisions[0]

    @property
    def lo(self) -> str:
        return self.precisions[-1]

    @property
    def store_id(self) -> str:
        return self.manifest["store_id"]

    def rank(self, prec: str) -> int:
        """0 = la plus précise. Une précision de rang plus petit peut remplacer une de rang plus grand."""
        return self.precisions.index(prec)

    def record_bytes(self, prec: str) -> int:
        return self.layouts[prec]["record_bytes"]

    def layer_index(self, prefix: str) -> int:
        return self.layers.index(prefix)

    def file(self, layer: int, prec: str) -> Path:
        return self.path / f"layer{layer:03d}.{prec}.bin"

    def _fh(self, layer: int, prec: str):
        return self._fh_path(self.file(layer, prec))

    def _fh_path(self, path: Path):
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = self._local.handles = {}
        fh = handles.get(path)
        if fh is None:
            fh = handles[path] = open(path, "rb", buffering=0)
            with self._lock:
                self._all_handles.append(fh)
        return fh

    def _direct(self, path: Path) -> _DirectFile | None:
        files = getattr(self._local, "direct", None)
        if files is None:
            files = self._local.direct = {}
        df = files.get(path)
        if df is None and self._direct_error is None:
            try:
                df = files[path] = _DirectFile(path)
                with self._lock:
                    self._all_handles.append(df)
            except OSError as e:  # système de fichiers sans lecture directe : lectures ordinaires
                self._direct_error = f"{e.__class__.__name__}: {e}"
        return df

    def close(self) -> None:
        """Ferme tous les fichiers ouverts (sous Windows, un fichier ouvert ne peut pas être supprimé)."""
        if self._split_pool is not None:
            self._split_pool.shutdown(wait=True)
            self._split_pool = None
        with self._lock:
            for fh in self._all_handles:
                fh.close()
            self._all_handles.clear()
        self._local = threading.local()

    # -- D4 : un « store chaud » sur un disque rapide ---------------------------
    def attach_hot(self, hot_dir: str | Path) -> int:
        """Branche un store chaud (build_hot_store.py) : ses experts seront lus là-bas. Retourne leur nombre."""
        hot_dir = Path(hot_dir)
        meta = json.loads((hot_dir / "hot.json").read_text(encoding="utf-8"))
        if meta.get("store_id") != self.store_id:
            raise ValueError(f"le store chaud {hot_dir} a été construit pour un autre store")
        self.hot = {p: {tuple(map(int, k.split(":"))): slot for k, slot in m.items()} for p, m in meta["experts"].items()}
        self.hot_path = hot_dir
        return sum(len(m) for m in self.hot.values())

    def _locate(self, layer: int, expert: int, prec: str) -> tuple[Path, int]:
        rb = self.record_bytes(prec)
        slot = self.hot.get(prec, {}).get((layer, expert))
        if slot is not None:
            return self.hot_path / f"hot.{prec}.bin", slot * rb
        return self.file(layer, prec), expert * rb

    def _read_range(self, path: Path, offset: int, out: torch.Tensor) -> bool:
        """Lit out.numel() octets à `offset` ; True si la lecture directe (sans cache) a servi."""
        n = out.numel()
        done = False
        if self.direct:
            df = self._direct(path)
            if df is not None:
                try:
                    done = df.read_into(out, offset)
                except OSError as e:
                    self._direct_error = str(e)
                    self.direct = False
            with self._lock:
                if done:
                    self.direct_reads += 1
                else:
                    self.direct_fallbacks += 1
        if not done:
            fh = self._fh_path(path)
            fh.seek(offset)
            got = fh.readinto(memoryview(out.numpy()))
            if got != n:
                raise IOError(f"lecture tronquée : {path} à {offset} ({got}/{n} octets)")
        return done

    def _split_reads(self, path: Path, offset: int, out: torch.Tensor, parts: int) -> None:
        """
        Un record lu en `parts` morceaux simultanés (03/10, Qwen3-Next) : un SSD ne donne son
        débit qu'avec plusieurs lectures en vol. Mesuré sur la machine de Victor : 0,33 Go/s
        avec une lecture, 1,15 Go/s avec quatre. Un expert manquant attendu seul (le cas des
        16 % que le prefetch ne prévoit pas) se lit alors ~3 fois plus vite. Morceaux alignés
        sur 4 096 octets (lecture directe).
        """
        from concurrent.futures import ThreadPoolExecutor

        n = out.numel()
        chunk = -(-n // parts)
        chunk = -(-chunk // ALIGN) * ALIGN
        if self._split_pool is None:
            self._split_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="moelab-split")
        futs = [self._split_pool.submit(self._read_range, path, offset + o, out[o:min(o + chunk, n)])
                for o in range(0, n, chunk)]
        for f in futs:
            f.result()

    def read(self, layer: int, expert: int, prec: str, out: torch.Tensor | None = None,
             parts: int | None = None) -> torch.Tensor:
        """Lit un record depuis le disque. `out` : tampon uint8 à remplir (ex. mémoire épinglée).
        `parts` : morceaux simultanés (défaut : self.read_parts)."""
        rb = self.record_bytes(prec)
        if out is None:
            out = torch.empty(rb, dtype=torch.uint8)
        t0 = time.perf_counter()
        path, offset = self._locate(layer, expert, prec)
        hot = path.parent == self.hot_path
        parts = self.read_parts if parts is None else parts
        if parts > 1 and rb >= 2 * ALIGN:
            self._split_reads(path, offset, out, parts)
        else:
            self._read_range(path, offset, out)
        dt = time.perf_counter() - t0
        if hot:
            with self._lock:
                self.hot_reads += 1
        with self._lock:
            self.bytes_read += rb
            self.read_seconds += dt
        return out

    def total_bytes(self, prec: str) -> int:
        return self.record_bytes(prec) * self.num_experts * len(self.layers)

    def summary(self) -> str:
        parts = []
        for p in self.precisions:
            parts.append(f"{p}: {self.record_bytes(p) / 2**20:.2f} Mio/expert, {self.total_bytes(p) / 2**30:.2f} Gio au total")
        return (
            f"store {self.path} — {len(self.layers)} couches MoE x {self.num_experts} experts "
            f"(H={self.hidden}, I={self.inter}) — " + " ; ".join(parts)
        )


def json_ident(info: dict, E: int, H: int, I: int, layouts: list[dict]) -> str:
    """Ce qui identifie le contenu d'un store (store_id) : modèle, formes, précisions et choix d'échelle."""
    precs = [l["precision"] if l.get("scale") in (None, "absmax") else f"{l['precision']}:{l['scale']}" for l in layouts]
    return json.dumps([info["model_type"], info["layers"], E, H, I, precs])


def build_store(
    model_dir: str | Path,
    out_dir: str | Path,
    precisions: tuple[str, ...] = ("fp16", "int4"),
    group_size: int = 64,
    resume: bool = True,
    log=print,
    int4_scale: str = "mse",
    device: str | None = None,
    progress=None,
) -> Path:
    """
    Convertit les experts d'un checkpoint Hugging Face en store. Lit un expert à la
    fois : la RAM nécessaire ne dépend pas de la taille du modèle. Reprenable : les
    couches déjà terminées (progress.json) sont sautées avec --resume.
    """
    from .checkpoint import Checkpoint
    from .loader import discover_experts

    for p in precisions:
        if p not in PRECISIONS:
            raise ValueError(f"précision inconnue : {p} (choix : {', '.join(PRECISIONS)})")
    order = ["fp32", "fp16", "bf16", "int8", "int4"]
    precisions = tuple(sorted(dict.fromkeys(precisions), key=order.index))

    info = discover_experts(model_dir)
    ckpt = Checkpoint(model_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    H, I, E = info["hidden"], info["inter"], info["num_experts"]
    layouts = [record_layout(p, H, I, group_size, int4_scale if p == "int4" else None) for p in precisions]
    ident = json_ident(info, E, H, I, layouts)
    manifest = {
        "format": "moelab-expert-store",
        "version": 1,
        "store_id": hashlib.sha1(ident.encode()).hexdigest()[:12],
        "source": str(Path(model_dir).resolve()),
        "model_type": info["model_type"],
        "layers": info["layers"],
        "num_experts": E,
        "hidden": H,
        "inter": I,
        "layouts": layouts,
        "complete": False,
    }
    progress_path = out / "progress.json"
    done: set[int] = set()
    old_manifest = out / "manifest.json"
    if resume and progress_path.exists() and old_manifest.exists():
        old = json.loads(old_manifest.read_text(encoding="utf-8"))
        if old.get("store_id") == manifest["store_id"]:
            done = set(json.loads(progress_path.read_text(encoding="utf-8"))["done"])
            if done:
                log(f"reprise : {len(done)} couche(s) déjà converties")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    for l in layouts:
        bpw = bits_per_weight(PRECISIONS[l["precision"]]["bits"], group_size) if PRECISIONS[l["precision"]]["kind"] == "int" else None
        extra = f" ({bpw:.2f} bits/poids)" if bpw else ""
        log(f"  {l['precision']}: {l['record_bytes'] / 2**20:.2f} Mio par expert{extra}, "
            f"{l['record_bytes'] * E * len(info['layers']) / 2**30:.2f} Gio au total")

    # la recherche d'échelle (int4 "mse") est ~30 fois plus coûteuse que "absmax" : sur le GPU s'il y en a un
    qdev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    todo = [i for i in range(len(info["layers"])) if i not in done]
    for n_done, li in enumerate(todo):
        prefix = info["layers"][li]
        handles = {}
        for l in layouts:
            f = open(out / f"layer{li:03d}.{l['precision']}.bin", "wb")
            f.truncate(l["record_bytes"] * E)
            handles[l["precision"]] = (f, l)
        try:
            for e in range(E):
                gate_up, down = ckpt.read_expert(prefix, e, H, I)
                gate_up, down = gate_up.to(qdev), down.to(qdev)
                for f, l in handles.values():
                    f.seek(e * l["record_bytes"])
                    f.write(encode_record(l, gate_up, down).tobytes())
        finally:
            for f, _ in handles.values():
                f.close()
        done.add(li)
        progress_path.write_text(json.dumps({"done": sorted(done)}), encoding="utf-8")
        elapsed = time.time() - t_start
        eta = elapsed / (n_done + 1) * (len(todo) - n_done - 1)
        log(f"  couche {li + 1}/{len(info['layers'])} ({prefix}) convertie — {elapsed:.0f} s écoulées, reste ~{eta:.0f} s")
        if progress is not None:  # (couches faites, total, secondes restantes estimées) : pour l'app
            progress(len(done), len(info["layers"]), eta)

    manifest["complete"] = True
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    ckpt.close()
    return out


def build_hot_store(store_dir, hot_dir, budget_bytes: int, usage_path=None, log=print) -> dict:
    """
    D4 : copie sur un disque rapide les experts les plus utilisés (d'après usage.json),
    jusqu'à `budget_bytes`. Le store d'origine reste complet et intact : le store chaud
    n'est qu'un raccourci, lu en priorité par ExpertStore.read une fois branché.
    Chaque expert est copié dans la précision où il a le plus servi.
    """
    store = ExpertStore(store_dir)
    usage_path = Path(usage_path) if usage_path else Path(store_dir) / "usage.json"
    if not usage_path.exists():
        raise FileNotFoundError(f"pas d'historique d'usage ({usage_path}) : discute d'abord avec le modèle (chat.py)")
    data = json.loads(usage_path.read_text(encoding="utf-8"))
    if data.get("store_id") != store.store_id:
        raise ValueError("usage.json provient d'un autre store")
    counts = {tuple(map(int, k.split(":"))): v for k, v in data["counts"].items()}
    hi_counts = {tuple(map(int, k.split(":"))): v for k, v in data.get("hi_counts", {}).items()}
    ranked = sorted(counts, key=lambda k: -counts[k])
    hot_dir = Path(hot_dir)
    hot_dir.mkdir(parents=True, exist_ok=True)
    chosen: dict[str, list[tuple[int, int]]] = {p: [] for p in store.precisions}
    used = 0
    for key in ranked:
        prec = store.hi if hi_counts.get(key, 0) * 2 >= counts[key] else store.lo
        rb = store.record_bytes(prec)
        if used + rb > budget_bytes:
            continue
        chosen[prec].append(key)
        used += rb
    meta = {"store_id": store.store_id, "experts": {}}
    for prec, keys in chosen.items():
        if not keys:
            continue
        with open(hot_dir / f"hot.{prec}.bin", "wb") as f:
            for (layer, expert) in keys:
                f.write(store.read(layer, expert, prec).numpy().tobytes())
        meta["experts"][prec] = {f"{l}:{e}": i for i, (l, e) in enumerate(keys)}
    (hot_dir / "hot.json").write_text(json.dumps(meta), encoding="utf-8")
    store.close()
    n = sum(len(k) for k in chosen.values())
    log(f"store chaud : {n} experts, {used / 2**30:.2f} Gio dans {hot_dir}")
    return {"experts": n, "bytes": used}
