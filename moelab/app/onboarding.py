"""
Mise en route (05/10) : ce qu'on a fait à la main pour Qwen3-Next, en assistant.

    machine()            GPU, VRAM, RAM, disques et place libre
    CATALOG / catalog()  modèles MoE gérés, tailles de téléchargement et de store calculées
    plan(...)            disque et précisions selon la place libre (comme le 02/10, « 50 Go sur C: »)
    install_job(...)     téléchargement (reprenable), store, store autonome, mesure du disque,
                         inscription dans la bibliothèque ; option : supprimer le checkpoint
    Library              la bibliothèque des modèles installés (<data>/models.json)
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

import psutil
import torch

from ..store import PRECISIONS, record_layout
from .engine import Cancelled, read_json, write_json

GIB = 2**30

# Architectures que moelab fait tourner (tests : test_qwen3next, test_qwen3moe, OLMoE partout).
SUPPORTED = {"olmoe", "qwen3_moe", "qwen3_next"}

CATALOG = [
    {"id": "olmoe-1b-7b", "nom": "OLMoE 1B-7B Instruct", "repo": "allenai/OLMoE-1B-7B-0125-Instruct", "arch": "olmoe",
     "couches": 16, "experts": 64, "actifs": 8, "hidden": 2048, "inter": 1024, "telechargement_go": 13.8,
     "dense_go": 0.9, "params": "7 B (1,3 B actifs)",
     "note": "Petit et rapide (100+ tok/s) : idéal pour découvrir, tient presque entier en VRAM."},
    {"id": "qwen3-30b-a3b", "nom": "Qwen3 30B-A3B Instruct (FP8)", "repo": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
     "arch": "qwen3_moe", "couches": 48, "experts": 128, "actifs": 8, "hidden": 2048, "inter": 768,
     "telechargement_go": 31.0, "dense_go": 2.2, "params": "30 B (3,3 B actifs)",
     "note": "Le bon compromis : bien plus capable qu'OLMoE, 4 fois moins d'experts que Qwen3-Next."},
    {"id": "qwen3-next-80b", "nom": "Qwen3-Next 80B-A3B Instruct (FP8)", "repo": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
     "arch": "qwen3_next", "couches": 48, "experts": 512, "actifs": 10, "hidden": 2048, "inter": 512,
     "telechargement_go": 82.0, "dense_go": 2.6, "params": "80 B (3 B actifs)",
     "note": "Le plus capable. Sur une RTX 2060 : ~7 tok/s, 1er token ~4 s."},
]

PRECISION_SETS = {"int4": ("int4",), "int8+int4": ("int8", "int4"), "fp16+int4": ("fp16", "int4")}


def store_bytes(entry: dict, precs) -> int:
    n = entry["couches"] * entry["experts"]
    return sum(record_layout(p, entry["hidden"], entry["inter"], 64)["record_bytes"] * n for p in precs)


def catalog() -> list[dict]:
    out = []
    for e in CATALOG:
        d = dict(e)
        d["stores_gio"] = {k: round(store_bytes(e, v) / GIB, 1) for k, v in PRECISION_SETS.items()}
        d["gere"] = e["arch"] in SUPPORTED
        out.append(d)
    return out


def machine() -> dict:
    disks = []
    seen = set()
    for p in psutil.disk_partitions(all=False):
        if p.mountpoint in seen or "cdrom" in p.opts or p.fstype in ("", "squashfs", "tmpfs", "overlay"):
            continue
        try:
            u = psutil.disk_usage(p.mountpoint)
        except OSError:
            continue
        if u.total < 8 * GIB:
            continue
        seen.add(p.mountpoint)
        disks.append({"chemin": p.mountpoint, "fs": p.fstype, "total_gio": round(u.total / GIB, 1),
                      "libre_gio": round(u.free / GIB, 1)})
    out = {"systeme": f"{os.name}", "cpu_threads": psutil.cpu_count(), "ram_totale_gio": round(psutil.virtual_memory().total / GIB, 1),
           "ram_dispo_gio": round(psutil.virtual_memory().available / GIB, 1), "disques": disks, "gpu": None}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        out["gpu"] = {"nom": torch.cuda.get_device_name(0), "vram_totale_gio": round(total / GIB, 1),
                      "vram_libre_gio": round(free / GIB, 1)}
    return out


def free_gib(path: str) -> float:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free / GIB


def same_disk(a: str, b: str) -> bool:
    return Path(a).resolve().anchor.lower() == Path(b).resolve().anchor.lower()


def checkpoint_ready(model_dir: str) -> bool:
    d = Path(model_dir)
    if not (d / "config.json").exists():
        return False
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        import json

        files = set(json.loads(idx.read_text(encoding="utf-8"))["weight_map"].values())
        return all((d / f).exists() for f in files)
    return any(d.glob("*.safetensors"))


def plan(model_id: str, download_dir: str, store_dir: str, keep_checkpoint: bool = False) -> dict:
    """Ce qu'il faut de place, sur quel disque, et les précisions possibles (la meilleure qui tient)."""
    e = next((x for x in CATALOG if x["id"] == model_id), None)
    if e is None:
        raise KeyError(model_id)
    dl_gib = 0.0 if checkpoint_ready(download_dir) else e["telechargement_go"] * 1e9 / GIB
    dense_gib = e["dense_go"] * 1e9 / GIB
    free_dl, free_st = free_gib(download_dir), free_gib(store_dir)
    same = same_disk(download_dir, store_dir)
    choices = []
    for key, precs in PRECISION_SETS.items():
        st = store_bytes(e, precs) / GIB + dense_gib
        if same:
            peak_ok = dl_gib + st <= 0.97 * free_dl
        else:
            peak_ok = dl_gib <= 0.97 * free_dl and st <= 0.97 * free_st
        choices.append({"precisions": key, "store_gio": round(st, 1), "tient": peak_ok,
                        "final_gio": round(st + (e["telechargement_go"] * 1e9 / GIB if keep_checkpoint else 0), 1)})
    fits = [c for c in choices if c["tient"]]
    # int4 seul : le plus rapide (politique lo), suffisant d'après nos mesures ; int8+int4 si la place le permet
    # largement (politique hobbit, meilleure qualité sur OLMoE).
    rec = None
    if fits:
        rec = "int8+int4" if any(c["precisions"] == "int8+int4" and c["tient"] for c in choices) and \
            free_st > 1.5 * next(c["store_gio"] for c in choices if c["precisions"] == "int8+int4") else "int4"
        if not any(c["precisions"] == rec and c["tient"] for c in choices):
            rec = fits[0]["precisions"]
    m = machine()
    warn = []
    if m["gpu"] is None:
        warn.append("Pas de GPU NVIDIA détecté : moelab tournera sur CPU, très lentement.")
    elif m["gpu"]["vram_totale_gio"] < e["dense_go"] * 0.6 + 1.5:
        warn.append("Peu de VRAM pour la partie dense de ce modèle : la tête et les embeddings resteront en RAM.")
    if m["ram_totale_gio"] < 12 and model_id != "olmoe-1b-7b":
        warn.append("Moins de 12 Go de RAM : préférer OLMoE.")
    if not e["arch"] in SUPPORTED:
        warn.append("Architecture pas encore gérée par moelab.")
    return {"modele": e, "telechargement_gio": round(dl_gib, 1), "deja_telecharge": dl_gib == 0,
            "libre_telechargement_gio": round(free_dl, 1), "libre_store_gio": round(free_st, 1), "meme_disque": same,
            "choix": choices, "recommande": rec, "avertissements": warn,
            "pic_gio": round(dl_gib + (next((c["store_gio"] for c in choices if c["precisions"] == rec), 0) if rec else 0), 1)}


class Library:
    def __init__(self, data_dir):
        self.path = Path(data_dir) / "models.json"
        self.data = read_json(self.path, {"modeles": [], "dernier": None})

    def save(self):
        write_json(self.path, self.data)

    def list(self) -> list[dict]:
        out = []
        for m in self.data["modeles"]:
            st = Path(m["store"])
            d = dict(m)
            d["present"] = (st / "manifest.json").exists()
            d["autonome"] = (st / "model" / "model.safetensors.index.json").exists()
            d["checkpoint_present"] = bool(m.get("checkpoint")) and Path(m["checkpoint"]).exists()
            d["profil_autopilot"] = (st / "autopilot_profile.json").exists()
            out.append(d)
        return out

    def add(self, store: str, nom: str | None = None, checkpoint: str | None = None, catalogue: str | None = None) -> dict:
        from ..store import ExpertStore

        st = ExpertStore(store)
        entry = {"id": st.store_id, "nom": nom or Path(store).name, "store": str(Path(store).resolve()),
                 "checkpoint": checkpoint, "catalogue": catalogue, "model_type": st.manifest.get("model_type"),
                 "precisions": list(st.precisions), "taille_gio": round(sum(st.total_bytes(p) for p in st.precisions) / GIB, 1),
                 "ajoute": time.time()}
        st.close()
        self.data["modeles"] = [m for m in self.data["modeles"] if m["store"] != entry["store"]] + [entry]
        self.save()
        return entry

    def get(self, mid: str) -> dict:
        m = next((m for m in self.data["modeles"] if m["id"] == mid), None)
        if m is None:
            raise KeyError(mid)
        return m

    def remove(self, mid: str, delete_store: bool = False, delete_checkpoint: bool = False) -> None:
        m = self.get(mid)
        if delete_store:
            shutil.rmtree(m["store"], ignore_errors=True)
        if delete_checkpoint and m.get("checkpoint"):
            shutil.rmtree(m["checkpoint"], ignore_errors=True)
            m["checkpoint"] = None
        if delete_store:
            self.data["modeles"] = [x for x in self.data["modeles"] if x["id"] != mid]
        self.save()


def _dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def install_job(library: Library, model_id: str, download_dir: str, store_dir: str, precisions: str = "int4",
                delete_checkpoint: bool = False):
    """
    Installation (job dans son propre fil, Engine.spawn), étapes reprenables :
      1. téléchargement depuis Hugging Face : le chat reste utilisable pendant ce temps ;
      2. construction du store, dans le fil du moteur : le modèle chargé est déchargé (la
         quantification se fait sur le GPU), puis rechargé ;
      3. store autonome ; 4. mesure du disque (bandwidth.json) ; 5. bibliothèque ;
      6. suppression du checkpoint si demandé.
    """
    e = next(x for x in CATALOG if x["id"] == model_id)
    precs = PRECISION_SETS[precisions]

    def run(job, engine):
        if not checkpoint_ready(download_dir):
            job.emit("step", etape="telechargement", texte=f"Téléchargement de {e['repo']} (~{e['telechargement_go']} Go)")
            _download(job, e, download_dir)
        if job.cancel_requested:
            raise Cancelled()

        def build(eng):
            again = eng.model_info.get("store") if eng.model is not None else None
            if again:
                job.emit("log", texte="modèle déchargé pendant la construction du store")
                eng.unload()
            from ..store import build_store

            def prog(done, total, eta):
                job.progress = done / total
                job.emit("progress", etape="store", fait=done, total=total, reste_s=round(eta))
                if job.cancel_requested:
                    raise Cancelled()

            try:
                build_store(download_dir, store_dir, precs, log=lambda m: job.emit("log", texte=m), progress=prog)
            finally:
                if again:
                    job.emit("log", texte="rechargement du modèle précédent")
                    eng.load(again, log=lambda m: job.emit("log", texte=m))

        job.emit("step", etape="store", texte="Construction du store (le modèle chargé est déchargé pendant ce temps)")
        engine.in_worker("construction du store", build)
        from ..standalone import make_standalone

        job.emit("step", etape="autonome", texte="Copie de la partie dense dans le store (store autonome)")
        make_standalone(download_dir, store_dir, log=lambda m: job.emit("log", texte=m))
        from ..hw import measure_all, save_bandwidth
        from ..store import ExpertStore

        job.emit("step", etape="disque", texte="Mesure du disque et du PCIe (bandwidth.json)")
        st = ExpertStore(store_dir)
        save_bandwidth(store_dir, measure_all(st, log=lambda m: job.emit("log", texte=m)))
        st.close()
        entry = library.add(store_dir, nom=e["nom"], checkpoint=None if delete_checkpoint else download_dir, catalogue=e["id"])
        if delete_checkpoint:
            job.emit("step", etape="nettoyage", texte=f"Suppression du checkpoint {download_dir}")
            shutil.rmtree(download_dir, ignore_errors=True)
        job.progress = 1.0
        job.emit("step", etape="fin", texte=f"{e['nom']} est installé : charge-le depuis la bibliothèque, puis lance l'autopilot.")
        return entry

    return run


def _download(job, e: dict, local_dir: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as err:
        raise RuntimeError("pip install huggingface_hub") from err
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    expected = e["telechargement_go"] * 1e9
    stop = threading.Event()

    def watch():
        while not stop.wait(2.0):
            got = _dir_size(Path(local_dir))
            job.progress = min(0.99, got / expected)
            job.emit("progress", etape="telechargement", fait=round(got / 1e9, 2), total=e["telechargement_go"])

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    try:
        snapshot_download(e["repo"], local_dir=local_dir, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                                           "*.tiktoken", "*.jinja", "*.py", "*.md"])
    finally:
        stop.set()
