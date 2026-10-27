"""
Le moteur partagé de l'app (05/10).

Un seul gros modèle tient en mémoire (6 Go de VRAM, 16 Go de RAM) : tout ce qui s'en sert
(chat, agent, autopilot, tâches de fond, changement de réglage, chargement d'un autre
modèle) passe par UNE file d'attente et UN fil de travail. Chaque demande est un `Job` :
  - priorité : 0 = immédiat (réglages), 1 = chat, 5 = tâches de fond, 9 = installations ;
  - événements (texte généré, appel d'outil, journal...) que l'interface lit en direct ;
  - un job long peut se découper : sa fonction renvoie CONTINUE et il repasse dans la file,
    ce qui laisse passer un message de chat entre deux fichiers d'une tâche de fond ;
  - un job peut demander la permission à l'utilisateur (outil « demander ») et attendre ;
  - un job long qui ne peut pas se découper (agents, 06/10) appelle `serve_waiting` entre deux
    étapes : les messages de chat en attente passent tout de suite, dans le même fil.
"""

from __future__ import annotations

import gc
import itertools
import json
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import torch

CONTINUE = object()  # valeur de retour d'un job qui veut repasser dans la file


class Cancelled(Exception):
    """Levée dans un job (par exemple pendant la génération) quand l'utilisateur l'annule."""

PRIORITY = {"options": 0, "chat": 1, "load": 2, "autopilot": 3, "agents": 4, "task": 5, "install": 9}


@dataclass
class Job:
    kind: str
    title: str
    fn: object
    priority: int = 5
    needs_model: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "en attente"          # en attente, en cours, terminé, erreur, annulé
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: object = None
    error: str | None = None
    progress: float | None = None       # 0..1 si connu
    state: dict = field(default_factory=dict)  # pour les jobs découpés (CONTINUE)
    events: list = field(default_factory=list)
    cancel_requested: bool = False

    def __post_init__(self):
        self.cond = threading.Condition()
        self._approvals: dict[str, tuple[threading.Event, list]] = {}

    def emit(self, type: str, **data) -> None:
        with self.cond:
            self.events.append({"seq": len(self.events), "type": type, "t": round(time.time(), 3), **data})
            self.cond.notify_all()

    def wait_events(self, after: int, timeout: float = 15.0) -> list:
        """Événements de numéro >= after (attend qu'il y en ait, ou la fin du job)."""
        with self.cond:
            if len(self.events) <= after and self.status in ("en attente", "en cours"):
                self.cond.wait(timeout)
            return self.events[after:]

    def ask(self, what: dict, timeout: float = 600) -> bool:
        """Demande l'accord de l'utilisateur (événement « approval ») et attend sa réponse."""
        aid = uuid.uuid4().hex[:8]
        ev, box = threading.Event(), []
        self._approvals[aid] = (ev, box)
        self.emit("approval", id=aid, **what)
        ok = ev.wait(timeout) and bool(box and box[0])
        self.emit("approval_done", id=aid, accepted=ok)
        return ok

    def ask_value(self, kind: str, what: dict, timeout: float, default):
        """Demande une valeur à l'utilisateur (ex. critères corrigés) ; sans réponse après `timeout` s : `default`.
        Événement « <kind> » avec id ; réponse par answer(id, accepted, valeur)."""
        if timeout <= 0:
            return default
        aid = uuid.uuid4().hex[:8]
        ev, box = threading.Event(), []
        self._approvals[aid] = (ev, box)
        self.emit(kind, id=aid, delai_s=timeout, **what)
        got = ev.wait(timeout)
        self._approvals.pop(aid, None)
        value = box[1] if got and len(box) > 1 and box[0] and box[1] is not None else default
        self.emit(kind + "_done", id=aid, repondu=bool(got))
        return value

    def answer(self, aid: str, accepted: bool, value=None) -> bool:
        e = self._approvals.pop(aid, None)
        if e is None:
            return False
        e[1].append(accepted)
        e[1].append(value)
        e[0].set()
        return True

    def summary(self) -> dict:
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
                "priority": self.priority, "created": self.created, "started": self.started,
                "finished": self.finished, "progress": self.progress, "error": self.error,
                "events": len(self.events),
                "result": self.result if isinstance(self.result, (dict, str, int, float, list, type(None))) else None}


class Engine:
    def __init__(self, data_dir, log=print):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.model = self.rt = self.tok = None
        self.model_info: dict = {}
        self.load_error: str | None = None
        self.jobs: dict[str, Job] = {}
        self._q: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = itertools.count()
        self.current: Job | None = None
        self._stop = False
        self.worker = threading.Thread(target=self._loop, name="moelab-engine", daemon=True)
        self.worker.start()

    # -- file d'attente ---------------------------------------------------
    def submit(self, kind: str, title: str, fn, priority: int | None = None, needs_model: bool = True) -> Job:
        job = Job(kind, title, fn, PRIORITY.get(kind, 5) if priority is None else priority, needs_model)
        self.jobs[job.id] = job
        self._trim()
        self._q.put((job.priority, next(self._seq), job))
        return job

    def spawn(self, kind: str, title: str, fn) -> Job:
        """Job qui tourne dans son propre fil, hors de la file (téléchargement, mesures disque) : le chat
        reste disponible. Pour une étape qui a besoin du GPU, la fonction appelle `in_worker`."""
        job = Job(kind, title, fn, 9, needs_model=False)
        self.jobs[job.id] = job

        def target():
            job.started, job.status = time.time(), "en cours"
            try:
                job.result = fn(job, self)
                job.status = "annulé" if job.cancel_requested else "terminé"
            except Cancelled:
                job.status = "annulé"
            except Exception as e:
                job.status, job.error = "erreur", f"{e.__class__.__name__}: {e}"
                self.log(f"[job {job.kind}] {job.error}\n{traceback.format_exc()}")
            job.finished = time.time()
            job.emit("done", status=job.status, error=job.error,
                     result=job.result if isinstance(job.result, (dict, str, list, int, float, type(None))) else None)

        threading.Thread(target=target, name=f"moelab-{kind}", daemon=True).start()
        return job

    def in_worker(self, title: str, fn, priority: int = 2):
        """Exécute fn(engine) dans le fil du moteur (accès exclusif au GPU et au modèle), attend, renvoie."""
        job = self.submit("exclusive", title, lambda j, e: fn(e), priority=priority, needs_model=False)
        while job.status in ("en attente", "en cours"):
            job.wait_events(len(job.events), timeout=1.0)
        if job.status == "erreur":
            raise RuntimeError(job.error)
        return job.result

    def run_now(self, kind: str, title: str, fn, timeout: float = 120):
        """Soumet un job prioritaire et attend son résultat (réglages, petites lectures)."""
        job = self.submit(kind, title, fn, priority=0)
        t0 = time.time()
        while job.status in ("en attente", "en cours") and time.time() - t0 < timeout:
            job.wait_events(len(job.events), timeout=0.5)
        if job.status == "erreur":
            raise RuntimeError(job.error)
        return job.result

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status not in ("en attente", "en cours"):
            return False
        job.cancel_requested = True
        if job.status == "en attente":
            job.status = "annulé"
            job.finished = time.time()
            job.emit("done", status="annulé")
        return True

    def _trim(self, keep: int = 200):
        if len(self.jobs) > keep:
            old = sorted((j for j in self.jobs.values() if j.status not in ("en attente", "en cours")),
                         key=lambda j: j.created)[: len(self.jobs) - keep]
            for j in old:
                self.jobs.pop(j.id, None)

    def _loop(self):
        while not self._stop:
            try:
                _, _, job = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._run_job(job)

    def _run_job(self, job: Job) -> None:
        if job.status == "annulé":
            return
        if job.needs_model and self.model is None:
            job.status, job.error = "erreur", "aucun modèle chargé (écran Mise en route)"
            job.finished = time.time()
            job.emit("done", status="erreur", error=job.error)
            return
        outer = self.current
        self.current = job
        if job.started is None:
            job.started = time.time()
        job.status = "en cours"
        try:
            out = job.fn(job, self)
            if out is CONTINUE and not job.cancel_requested:
                job.status = "en attente"
                self._q.put((job.priority, next(self._seq), job))
                return
            job.result = out
            job.status = "annulé" if job.cancel_requested else "terminé"
        except Cancelled:
            job.status = "annulé"
        except Exception as e:  # le moteur continue, l'erreur est montrée dans l'interface
            job.status, job.error = "erreur", f"{e.__class__.__name__}: {e}"
            self.log(f"[job {job.kind}] {job.error}\n{traceback.format_exc()}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            self.current = outer
        job.finished = time.time()
        job.emit("done", status=job.status, error=job.error,
                 result=job.result if isinstance(job.result, (dict, str, list, int, float, type(None))) else None)

    def serve_waiting(self, below: int, kinds=("chat", "options")) -> int:
        """
        Appelé DANS le fil de travail par un job long, entre deux étapes : exécute tout de suite les jobs
        en attente de priorité < `below` et d'un genre sûr (chat, réglages : jamais un chargement, qui
        changerait le modèle sous le job), puis rend la main. Le job appelant doit
        mettre de côté et remettre son état de décodage (le chat se sert du même décodeur).
        Retourne le nombre de jobs servis.
        """
        if threading.current_thread() is not self.worker:
            return 0
        n = 0
        while True:
            with self._q.mutex:
                head = self._q.queue[0] if self._q.queue else None
            if head is None or head[0] >= below or head[2].kind not in kinds:
                return n
            try:
                _, _, job = self._q.get_nowait()
            except queue.Empty:
                return n
            self._run_job(job)
            n += 1

    def close(self):
        self._stop = True
        self.unload()

    # -- modèle ---------------------------------------------------------------
    def unload(self) -> None:
        if self.rt is not None:
            try:
                self.rt.end_of_request()
                self.rt.close()
            except Exception:
                pass
        self.model = self.rt = self.tok = None
        self.model_info = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, store: str, model_dir: str | None = None, ram_gb: float | None = None, vram_gb: float | None = None,
             policy: str = "lo", log=None) -> dict:
        """Charge un modèle (dans le fil de travail). Même chemin que les scripts : budgets, bandwidth.json,
        profil autopilot, échauffement."""
        from transformers import AutoTokenizer

        from .. import PrecisionPolicy, RuntimeConfig, load_offloaded_model
        from ..autopilot import apply_profile
        from ..generate import warmup
        from ..hw import load_bandwidth
        from ..plan import preload_advised
        from ..standalone import resolve_model_dir
        from ..store import ExpertStore

        log = log or self.log
        self.unload()
        t0 = time.time()
        mdir = str(resolve_model_dir(model_dir, store))
        st = ExpertStore(store)
        lo_total = st.total_bytes(st.lo)
        name = st.manifest.get("model_type", "?") if hasattr(st, "manifest") else "?"
        st.close()
        bw = load_bandwidth(store) or {}
        cfg = RuntimeConfig(policy=PrecisionPolicy(policy), fast_budget_bytes=None if vram_gb is None else int(vram_gb * 2**30),
                            host_budget_bytes=None if ram_gb is None else int(ram_gb * 2**30),
                            io_workers=int(bw.get("io_workers", 4)), read_parts=int(bw.get("read_parts", 1)),
                            graph_max_len=None, usage_path=str(Path(store) / "usage.json"))
        if bw.get("b_pcie") and bw.get("b_host"):
            cfg.b_pcie, cfg.b_host = bw["b_pcie"], bw["b_host"]
        cuda = torch.cuda.is_available()
        model, rt = load_offloaded_model(mdir, store, cfg, dense="auto", dtype=torch.float16 if cuda else torch.float32,
                                         log=log, dense_int8=cuda)
        rt.warm_start(preload_lo=preload_advised(lo_total, rt.cfg.host_budget_bytes), log=log)
        tok = AutoTokenizer.from_pretrained(mdir)
        self.model, self.rt, self.tok = model, rt, tok
        skip = tuple(k for k, v in (("ram_gb", ram_gb), ("vram_gb", vram_gb)) if v is not None)
        profile = apply_profile(model, store, skip=skip, log=log)
        wt = warmup(model, tok, log=log)
        self.model_info = {"store": str(store), "model_dir": mdir, "name": Path(store).name, "model_type": name,
                           "policy": policy, "loaded_at": time.time(), "load_s": round(time.time() - t0, 1),
                           "warmup_s": round(wt, 1), "profile": profile, "ram_gb": ram_gb, "vram_gb": vram_gb}
        return self.model_info

    # -- état (lecture depuis n'importe quel fil) --------------------------------
    def status(self) -> dict:
        import psutil

        out = {"model": self.model_info or None, "busy": self.current.summary() if self.current else None,
               "queue": sum(1 for j in self.jobs.values() if j.status == "en attente"),
               "ram": {"total_gio": round(psutil.virtual_memory().total / 2**30, 1),
                       "dispo_gio": round(psutil.virtual_memory().available / 2**30, 1)}}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            out["gpu"] = {"nom": torch.cuda.get_device_name(0), "vram_totale_gio": round(total / 2**30, 2),
                          "vram_libre_gio": round(free / 2**30, 2)}
        rt = self.rt
        if rt is not None:
            s = rt.stats.as_dict()
            out["caches"] = {
                "vram": {"utilise_gio": round((rt.fast.used if rt.fast else 0) / 2**30, 2),
                         "budget_gio": round((rt.fast.budget if rt.fast else 0) / 2**30, 2),
                         "experts": len(rt.fast.entries) if rt.fast else 0},
                "ram": {"utilise_gio": round(rt.host.used / 2**30, 2), "budget_gio": round(rt.host.budget / 2**30, 2),
                        "experts": len(rt.host.entries)},
                "pool_hors_pool": rt.pool.fallbacks}
            out["stats"] = {"tokens": s["tokens"], "hit_vram": round(s["fast_hit_rate"], 3),
                            "disque_par_token": round(s["disk_per_token"], 1), "cpu": round(s["cpu_share"], 3),
                            "prefetch_rappel": None if s["prefetch_recall"] is None else round(s["prefetch_recall"], 3),
                            "full_graph_tokens": s["full_tokens"]}
            out["store"] = {"couches": len(rt.store.layers), "experts": rt.store.num_experts,
                            "precisions": list(rt.store.precisions),
                            "total_gio": round(rt.store.total_bytes(rt.store.lo) / 2**30, 1)}
        out["last"] = getattr(self, "last_answer", None)
        return out


class TextSink:
    """Streamer de transformers -> événements « text » d'un job (texte décodé au fil de l'eau)."""

    def __init__(self, tokenizer, job: Job, kind: str = "text"):
        from transformers import TextStreamer

        sink = self

        class _S(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                if text:
                    sink.job.emit(sink.kind, text=text)

        self.job, self.kind = job, kind
        self.streamer = _S(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def reset(self) -> None:
        """Avant chaque génération : la prochaine entrée est un prompt (à ne pas afficher). 06/10 : après une
        erreur mémoire en pleine génération, le prompt suivant s'affichait en entier dans le déroulé."""
        st = self.streamer
        st.token_cache, st.print_len, st.next_tokens_are_prompt = [], 0, True

    def put(self, v):
        if self.job.cancel_requested:
            raise Cancelled()
        self.streamer.put(v)

    def end(self):
        self.streamer.end()


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, data) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(p)
