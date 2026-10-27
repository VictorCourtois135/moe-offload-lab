"""
Lecture directe d'un checkpoint Hugging Face (.safetensors), tenseur par tenseur.

On n'appelle jamais `from_pretrained` sur le modèle complet : on ouvre les fichiers
.safetensors et on lit seulement les tenseurs demandés. Un .safetensors est un
en-tête JSON suivi des données brutes, donc lire un tenseur = lire une zone du
fichier. C'est ce qui permet de construire le store d'experts d'un modèle de 80B
sur une machine à 16 Go de RAM.

Formats d'experts gérés :
- "par expert" (format sur disque d'OLMoE, Qwen3-MoE, Qwen3-Next...) :
      <prefix>.<e>.gate_proj.weight  (I, H)
      <prefix>.<e>.up_proj.weight    (I, H)
      <prefix>.<e>.down_proj.weight  (H, I)
- "fusionné" (format mémoire de transformers 5) :
      <prefix>.gate_up_proj  (E, 2I, H)
      <prefix>.down_proj     (E, H, I)
- FP8 par blocs (checkpoints *-FP8) : chaque poids float8 a un compagnon
  `<clé>_scale_inv`, déquantifié à la volée.

Lecteur maison (02/10) : la bibliothèque safetensors projette chaque fichier en mémoire
(mmap en copie privée). Sous Windows, ça a planté net (« access violation », sans
message Python) en chargeant la partie dense de Qwen3-Next : 82 Go de fichiers projetés
pendant que le modèle occupe déjà la RAM. Ici, on lit l'en-tête JSON nous-mêmes, puis
chaque tenseur par un simple seek + read : pas de projection, la mémoire utilisée est
exactement celle du tenseur demandé.
"""

from __future__ import annotations

import json
import struct
import threading
from math import prod
from pathlib import Path

import torch

from .quant import dequantize_fp8_block

FP8_DTYPES = {getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, n)}

ST_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool,
}
if hasattr(torch, "float8_e4m3fn"):
    ST_DTYPES["F8_E4M3"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    ST_DTYPES["F8_E5M2"] = torch.float8_e5m2


class SafeFile:
    """Un fichier .safetensors lu sans projection mémoire : en-tête JSON, puis seek + read par tenseur."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.fh = open(self.path, "rb")
        n = struct.unpack("<Q", self.fh.read(8))[0]
        header = json.loads(self.fh.read(n))
        header.pop("__metadata__", None)
        self.header = header
        self.base = 8 + n
        self.lock = threading.Lock()

    def keys(self):
        return self.header.keys()

    def _info(self, key: str):
        h = self.header[key]
        if h["dtype"] not in ST_DTYPES:
            raise TypeError(f"{key} : type safetensors {h['dtype']} non géré")
        return ST_DTYPES[h["dtype"]], list(h["shape"]), h["data_offsets"]

    def _read(self, offset: int, nbytes: int) -> bytearray:
        buf = bytearray(nbytes)
        with self.lock:
            self.fh.seek(self.base + offset)
            got = self.fh.readinto(buf)
        if got != nbytes:
            raise IOError(f"lecture tronquée dans {self.path} ({got}/{nbytes} octets)")
        return buf

    @staticmethod
    def _tensor(buf: bytearray, dtype: torch.dtype, shape: list[int]) -> torch.Tensor:
        if not buf:
            return torch.empty(shape, dtype=dtype)
        return torch.frombuffer(buf, dtype=torch.uint8).view(dtype).reshape(shape)

    def get_tensor(self, key: str) -> torch.Tensor:
        dtype, shape, (b, e) = self._info(key)
        return self._tensor(self._read(b, e - b), dtype, shape)

    def get_row(self, key: str, index: int) -> torch.Tensor:
        """t[index] seulement (première dimension) : une seule lecture de la taille d'une ligne."""
        dtype, shape, (b, e) = self._info(key)
        row = (e - b) // shape[0]
        return self._tensor(self._read(b + index * row, row), dtype, shape[1:])

    def close(self) -> None:
        self.fh.close()


class Checkpoint:
    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"dossier du modèle introuvable : {self.dir}")
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            self.key_to_file = {k: self.dir / f for k, f in weight_map.items()}
        else:
            files = sorted(self.dir.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"aucun fichier .safetensors dans {self.dir}")
            self.key_to_file = {}
            for f in files:
                sf = SafeFile(f)
                for k in sf.keys():
                    self.key_to_file[k] = f
                sf.close()
        self._handles: dict[Path, object] = {}

        self.block_size = (128, 128)
        cfg_path = self.dir / "config.json"
        if cfg_path.exists():
            qc = json.loads(cfg_path.read_text(encoding="utf-8")).get("quantization_config") or {}
            if qc.get("weight_block_size"):
                self.block_size = tuple(qc["weight_block_size"])

    # -- accès bas niveau ---------------------------------------------------
    def _handle(self, path: Path):
        h = self._handles.get(path)
        if h is None:
            h = SafeFile(path)
            self._handles[path] = h
        return h

    def has(self, key: str) -> bool:
        return key in self.key_to_file

    def keys(self):
        return self.key_to_file.keys()

    def get_raw(self, key: str) -> torch.Tensor:
        if key not in self.key_to_file:
            raise KeyError(f"clé absente du checkpoint : {key}")
        return self._handle(self.key_to_file[key]).get_tensor(key)

    def get(self, key: str) -> torch.Tensor:
        """Tenseur prêt à l'emploi : un poids FP8 est déquantifié en float32, le reste garde son dtype."""
        t = self.get_raw(key)
        if t.dtype in FP8_DTYPES:
            scale_key = key + "_scale_inv"
            if not self.has(scale_key):
                raise ValueError(f"{key} est en FP8 mais {scale_key} est absent : format FP8 non géré")
            return dequantize_fp8_block(t, self.get_raw(scale_key), self.block_size)
        return t

    def get_row(self, key: str, index: int) -> torch.Tensor:
        """Lit seulement t[index] d'un tenseur (utile pour les experts fusionnés : une tranche = un expert)."""
        return self._handle(self.key_to_file[key]).get_row(key, index)

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles.clear()

    # -- experts ------------------------------------------------------------
    def expert_layout(self, prefix: str) -> str:
        if self.has(f"{prefix}.0.gate_proj.weight"):
            return "per_expert"
        if self.has(f"{prefix}.gate_up_proj"):
            return "fused"
        raise KeyError(
            f"aucun expert trouvé sous '{prefix}' (ni '{prefix}.0.gate_proj.weight', ni '{prefix}.gate_up_proj')"
        )

    def read_expert(self, prefix: str, e: int, hidden: int, inter: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne (gate_up, down) en float32, dans la convention de transformers :
        gate_up (2I, H) = [gate; up] empilés, down (H, I).
        """
        if self.expert_layout(prefix) == "per_expert":
            gate = self.get(f"{prefix}.{e}.gate_proj.weight").float()
            up = self.get(f"{prefix}.{e}.up_proj.weight").float()
            down = self.get(f"{prefix}.{e}.down_proj.weight").float()
            gate_up = torch.cat([gate, up], dim=0)
        else:
            gu_key, dn_key = f"{prefix}.gate_up_proj", f"{prefix}.down_proj"
            gate_up = self.get_row(gu_key, e)
            down = self.get_row(dn_key, e)
            if gate_up.dtype in FP8_DTYPES:
                raise NotImplementedError("experts fusionnés en FP8 : format non géré (utiliser un checkpoint par expert)")
            gate_up, down = gate_up.float(), down.float()
            if tuple(down.shape) == (inter, hidden) and hidden != inter:
                # Variante "transposée" (entrées, sorties) : on revient à (sorties, entrées).
                gate_up, down = gate_up.t().contiguous(), down.t().contiguous()
        if tuple(gate_up.shape) != (2 * inter, hidden) or tuple(down.shape) != (hidden, inter):
            raise ValueError(
                f"expert {prefix}[{e}] : formes {tuple(gate_up.shape)} / {tuple(down.shape)}, "
                f"attendu {(2 * inter, hidden)} / {(hidden, inter)}"
            )
        return gate_up, down
