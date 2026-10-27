"""
Moteur d'exécution des couches MoE : le cœur du projet.

À chaque couche MoE, le routeur d'origine a déjà choisi ses k experts par token
(on ne touche jamais à ses décisions, sauf saut explicite avec t2). Ensuite :

  1. HOBBIT   : pour chaque expert demandé, haute ou basse précision selon son importance
  2. VRAM     : les experts déjà en VRAM sont calculés sur GPU (asynchrone)
  3. q*       : parmi les m experts manquants, q* sont transférés vers la VRAM,
                les m - q* autres sont calculés directement sur CPU, EN MÊME TEMPS
  4. disque   : ce qui n'est pas en RAM est lu depuis le store par un pool de
                threads, pendant que le reste calcule
  5. prefetch : le routeur de la couche suivante, appliqué à l'état courant, prédit
                ses experts ; on commence à les charger avant d'en avoir besoin

Niveaux de mémoire :  VRAM (cache "rapide")  >  RAM (cache "hôte")  >  disque (store)
Le store est la source de vérité : la place disponible ne change que la vitesse,
jamais les poids utilisés pour une précision donnée.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import TierCache, UsageLog
from .hostpool import HostPool
from .policy import PrecisionPolicy, level_to_precision, qstar
from .kernels import (GroupedGraph, _pow2_groups, _pow2_groups8, expert_ffn_int4, grouped_ffn_float, grouped_ffn_int4,
                      grouped_ffn_int8, pick_int4_version, triton_usable)
from .store import PRECISIONS, ExpertStore, materialize, unpack_int4_record, unpacked_layout, views


@dataclass
class RuntimeConfig:
    policy: PrecisionPolicy = field(default_factory=PrecisionPolicy)
    fast_device: str | None = None       # "cuda" ; "cpu" = niveau rapide simulé (tests) ; None = pas de niveau rapide
    fast_budget_bytes: int = 0           # octets de VRAM réservés aux experts
    host_budget_bytes: int = 4 * 2**30   # octets de RAM réservés aux experts
    half_life: float = 32.0              # demi-vie du score LRFU, en passes avant
    pin_fraction: float = 0.5            # part du budget qu'on peut épingler
    use_qstar: bool = True               # False = tous les manquants vont en VRAM (cache pur)
    b_pcie: float = 12e9                 # octets/s, mesurés par scripts/measure_bandwidth.py
    b_host: float = 12e9
    cpu_dtype: torch.dtype = torch.float32
    prefetch: bool = True
    prefetch_max_tokens: int = 1         # prefetch seulement en génération (1 token par passe)
    io_workers: int = 4
    pin_memory: bool | None = None       # mémoire hôte épinglée (transferts asynchrones) ; None = si CUDA
    usage_path: str | None = None        # historique d'usage persistant (usage.json)
    # Optimisations (doc 18), activables une par une pour mesurer leur effet.
    # Les valeurs par défaut suivent les mesures sur la RTX 2060 (30/09 et 01/10, doc 18).
    fast_decode: bool = True             # A2 : génération (1 token) sans opérations de tenseurs pour décider, 1 seule synchro
    prefetch_adaptive: bool = True       # saute le prefetch tant qu'aucun expert ne manque (tout est déjà résident)
    prefetch_depth: int = 1              # C2 : précharger aussi la couche L+2 (routeur de L+2 appliqué à l'état de L)
    read_parts: int = 1                  # morceaux simultanés par expert lu sur disque (mesuré : bandwidth.json)
    prefetch_disk_ahead: int = 0         # Q-Next (02/10) : couches L+2.. dont les experts prédits sont lus du disque
                                         # vers la RAM seulement (jamais en VRAM : pas la pollution de C2)
    direct_io: bool = True               # D1 : lectures disque sans le cache du système (repli automatique si impossible)
    sort_reads: bool = True              # D3 : lectures d'une couche lancées dans l'ordre du fichier (HDD)
    hot_store: str | None = None         # D4 : store chaud sur un disque rapide (build_hot_store.py)
    fast_int4_unpacked: bool = False     # B2 : int4 rangé « déballé » (1 valeur par octet) dans la VRAM
    triton_int4: bool = True             # B1 : noyau Triton qui lit l'int4 directement (si Triton est installé)
    grouped_experts: bool = True         # A4+B1 : tous les experts GPU d'une couche en 2 noyaux Triton par précision
    # A6 : partie dense de la génération en CUDA graphs (moelab/graphs.py), cache KV statique
    cuda_graphs: bool = True             # mesuré le 01/10 : lo 20,9 -> 41,9 tok/s, hobbit 6,8 -> 8,0
    grouped_kernels: int = 0             # noyaux groupés int4 : 0 = le plus rapide sur la carte (v2 ou v4), 1-4 = forcé
    moe_graphs: bool = True              # A6 bis : avec cuda_graphs, le calcul groupé de chaque couche en un CUDA graph
    graph_max_len: int | None = 2048     # taille du cache KV statique (prompt + réponse) ; None = auto au chargement
    graph_pack_routing: bool = True      # A9 : routage + prédiction du prefetch lus en une fois, prédiction calculée dans le graph
    graph_buckets: bool = True           # graphs par paliers de longueur (128, 256...) : l'attention ne lit que le nécessaire
    # S3 (02/10, d'après SpecMD) : à mesurer avant d'activer par défaut
    cache_eviction: str = "lrfu"         # VRAM : "lrfu" ou "stale" (rejeté le 02/10 : hobbit 51,5 -> 43,1 tok/s)
    prefetch_mass: float | None = None   # prefetch selon le score : experts prédits jusqu'à cette masse de probabilité
                                         # (au plus 2k) ; None = les k meilleurs (0,9 rejeté le 02/10 : routeur trop plat)
    # A5 (02/10) : token entier en UN graph, décisions sur le GPU, quand tout est en VRAM (voir fullgraph.py)
    full_graph: bool = True              # mesuré le 02/10 : lo 117,5 -> 132,1 tok/s ; hobbit neutre (rarement tenté)
    # Prefill (03/10) : le temps avant le premier token. P1 est exact ; P2 à P5 sont à mesurer
    # (scripts/profile_prefill.py) avant d'être activés par défaut.
    kv_reuse: bool = True                # P1 : en conversation, le cache du tour précédent est repris (chat, app)
    prefill_order: str = "fills"         # P2 : "completed" = experts manquants calculés dans l'ordre d'arrivée du disque
    prefill_prefetch: bool = False       # P3 : pendant le prefill, lectures disque -> RAM des experts prédits de L+1
    prefill_cache: str = "vram"          # P4 : "ram" = experts lus pour le prefill gardés hors du cache VRAM
    prefill_skip: float | None = None    # P5 : prefill, experts absents de la VRAM et de la RAM dont le plus grand
                                         # poids du routeur est sous ce seuil : sautés (qualité à mesurer)
    # Mesures du 03/10 (16 h) : le prefill n'attend presque pas le disque (0,1 s sur 5,8) mais le
    # fil principal ralentit quand les lectures tournent (phase 4 : 5 ms par expert déjà en VRAM).
    prefill_grouped: int = 192           # P6 : prefill jusqu'à ce nombre de tokens : experts calculés token par token
                                         # avec les noyaux groupés de la génération (2 lancements par token et par
                                         # couche au lieu d'~30 par expert) ; 0 = non. Mesuré le 05/10 (Qwen3-Next) :
                                         # 5,95 -> 3,76 s (17 tokens), 14,3 -> 10,5 s (93) ; mais 407 tokens :
                                         # 22,5 -> 27,9 s (chaque token relit ses experts). Bascule estimée ~220 : 192
    prefill_split: bool = True           # P7 : False = au prefill, un expert = une lecture. Rejeté le 05/10 :
                                         # 5,95 -> 6,31 s (le disque veut les lectures découpées, même au prefill)
    gil_switch_ms: float | None = None   # P8 : intervalle de bascule du GIL (Python : 5 ms). Rejeté le 05/10 : 0,5 ms
                                         # double le prefill (12,2 s) et divise la génération par 2


@dataclass
class RunStats:
    forwards: int = 0
    tokens: int = 0
    layer_calls: int = 0
    pairs: int = 0          # couples (token, expert) choisis par le routeur
    skipped: int = 0        # couples sautés (seuil t2)
    requests: int = 0       # experts distincts à calculer (par couche et par passe)
    need_hi: int = 0        # ... demandés en haute précision
    miss_downgrades: int = 0  # E3 : experts absents de la VRAM ramenés à une précision plus basse
    used_hi: int = 0        # ... calculés en haute précision (un fp16 en cache sert aussi une demande int4)
    fast_hits: int = 0      # déjà en VRAM
    prefetch_hits: int = 0  # ... dont chargés par le prefetch
    fills: int = 0          # transférés RAM -> VRAM à la demande
    cpu_execs: int = 0      # calculés sur CPU (branche q*, ou tout si pas de GPU)
    host_hits: int = 0      # trouvés en RAM
    disk_reads: int = 0     # lus depuis le store
    bytes_h2d: int = 0
    bytes_disk: int = 0
    prefetch_issued: int = 0
    pred_total: int = 0     # experts prédits par le prefetch
    pred_hits: int = 0      # ... réellement utilisés à la couche suivante
    pred_actual: int = 0    # experts réellement utilisés dans les couches prédites
    t_moe: float = 0.0      # secondes passées dans les couches MoE
    t_cpu: float = 0.0      # ... dans la branche CPU
    t_io_wait: float = 0.0  # ... à attendre le disque
    disk_ahead: int = 0     # lectures disque lancées à l'avance pour L+2.. (prefetch_disk_ahead)
    full_tokens: int = 0    # A5 : tokens calculés par le graph complet (tout était en VRAM)
    full_fallbacks: int = 0  # ... tentés mais refaits par le chemin normal (un expert manquait)
    prefill_skipped: int = 0  # P5 : experts sautés pendant le prefill (absents, poids faible)

    def as_dict(self) -> dict:
        d = asdict(self)
        r = max(self.requests, 1)
        d["fast_hit_rate"] = self.fast_hits / r
        d["hi_share"] = self.used_hi / r
        d["need_hi_share"] = self.need_hi / r
        d["skip_rate"] = self.skipped / max(self.pairs, 1)
        d["cpu_share"] = self.cpu_execs / r
        d["disk_per_token"] = self.disk_reads / max(self.tokens, 1)
        d["prefetch_recall"] = self.pred_hits / self.pred_actual if self.pred_actual else None
        d["prefetch_precision"] = self.pred_hits / self.pred_total if self.pred_total else None
        return d


@dataclass
class _Req:
    expert: int
    prec: str
    sl: slice
    hi: bool
    w: float = 0.0  # poids du routeur (génération : un seul token)
    level: int = 0  # niveau de précision demandé (0 = le plus précis)
    s: float = 0.0  # score d'inimportance HOBBIT (génération, pour E3)


def _is_silu(act_fn) -> bool:
    """Les noyaux groupés calculent silu(gate) * up : vrai pour OLMoE, Qwen3-MoE et Qwen3-Next."""
    return act_fn is F.silu or type(act_fn).__name__ in ("SiLU", "SiLUActivation")


def expert_ffn(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, act_fn) -> torch.Tensor:
    """Même calcul que OlmoeExperts/Qwen3NextExperts : down( act(gate(x)) * up(x) )."""
    gate, up = F.linear(x, gate_up).chunk(2, dim=-1)
    return F.linear(act_fn(gate) * up, down)


class MoERuntime:
    def __init__(self, store: ExpertStore, cfg: RuntimeConfig | None = None):
        self.store = store
        self.cfg = cfg or RuntimeConfig()
        self.policy = self.cfg.policy
        rank = store.rank
        self.fast: TierCache | None = None
        if self.cfg.fast_device and self.cfg.fast_budget_bytes > 0:
            name = "VRAM" if self.cfg.fast_device.startswith("cuda") else "rapide (simulé)"
            self.fast = TierCache(name, self.cfg.fast_device, self.cfg.fast_budget_bytes, rank,
                                  self.cfg.half_life, self.cfg.pin_fraction, eviction=self.cfg.cache_eviction)
        self.host = TierCache("RAM", "cpu", self.cfg.host_budget_bytes, rank, self.cfg.half_life, self.cfg.pin_fraction,
                              on_evict=self._to_release_later)
        self.cuda = self.fast is not None and self.fast.device.type == "cuda"
        self.pin = self.cuda if self.cfg.pin_memory is None else (self.cfg.pin_memory and torch.cuda.is_available())
        self.copy_stream = torch.cuda.Stream(device=self.fast.device) if self.cuda else None
        # Tous les tampons RAM viennent d'un pool borné (voir hostpool.py) : budget du cache
        # + une marge pour les lectures en cours et les tampons temporaires.
        rbs = {p: store.record_bytes(p) for p in store.precisions}
        margin = max(256 * 2**20, 4 * (self.cfg.io_workers + 8) * max(rbs.values()))
        self.pool = HostPool(rbs, self.cfg.host_budget_bytes + margin, self.pin)
        self._releases: list[torch.Tensor] = []
        self.io = ThreadPoolExecutor(max_workers=self.cfg.io_workers, thread_name_prefix="moelab-io")
        store.direct = self.cfg.direct_io
        store.read_parts = max(1, int(self.cfg.read_parts))
        self.set_unpacked(self.cfg.fast_int4_unpacked)
        self.triton_error: str | None = None
        self._triton = bool(self.cfg.triton_int4 and self.fast is not None and triton_usable(self.fast.device))
        self._grouped = bool(self.cfg.grouped_experts and self.fast is not None and triton_usable(self.fast.device))
        if self.cfg.hot_store:
            store.attach_hot(self.cfg.hot_store)
        self.pending: dict[tuple[int, int, str], Future] = {}
        self.usage = UsageLog(self.cfg.usage_path, store.store_id)
        self.stats = RunStats()
        self.gates: dict[int, nn.Module] = {}
        self.act_fns: dict[int, nn.Module] = {}
        self.preload_lo = False
        self.predicted: dict[int, set[int]] = {}
        self._inflight: list[tuple[object, torch.Tensor]] = []
        self._calm_layers = 0  # couches consécutives sans expert manquant (prefetch_adaptive)
        self.prof = None  # PhaseTimer (moelab/profiling.py) pour mesurer chaque phase de run_layer
        self.capture_hook = None  # A6 : pendant la capture des CUDA graphs (graphs.py), remplace les couches MoE
        self._ggraphs: dict[tuple, GroupedGraph | None] = {}  # A6 bis : un graph de calcul groupé par couche
        self._pred_rows: dict[int, list] = {}  # A9 : prédiction du routeur de la couche suivante, lue avec le routage
        self._pred_far: dict[int, list] = {}   # ... et des couches L+2.. (prefetch_disk_ahead), même lecture
        self.calib_hook = None  # Q2 (requant.py) : reçoit (couche, x, experts, poids) de chaque couche MoE
        # REAP (08/10, reap.py) : reçoit (couche, expert, poids du routeur, sortie de l'expert avant pondération)
        # pour chaque expert calculé. Actif, il coupe les noyaux groupés (qui ne rendent que la somme).
        self.expert_hook = None
        self._layer = -1
        self.pruned: set[tuple[int, int]] = set()  # REAP, élagage virtuel : experts que le routeur ne choisit plus
        self._warm_precs: set[str] = set()  # A5 : précisions dont les noyaux groupés ont déjà tourné (autotune fait)
        self._dev_table = None  # A5 : table GPU des experts résidents (fullgraph.DeviceTable)
        self._bulk = False      # P7 : couche en cours de prefill (plusieurs tokens)
        if self.cfg.gil_switch_ms:  # P8
            import sys

            sys.setswitchinterval(self.cfg.gil_switch_ms / 1000)

    def set_unpacked(self, on: bool) -> None:
        """B2 : layout des records dans le niveau rapide (int4 déballé, sinon celui du store).
        À appeler caches vides : les entrées déjà en VRAM gardent leur format."""
        self.cfg.fast_int4_unpacked = on
        self.fast_layouts = dict(self.store.layouts)
        self._unpack: dict[str, dict] = {}
        if on:
            for p, lay in self.store.layouts.items():
                l8 = unpacked_layout(lay)
                if l8 is not None:
                    self.fast_layouts[p] = l8
                    self._unpack[p] = l8

    def _is_hi(self, prec: str) -> bool:
        """Haute précision au sens HOBBIT (sans objet si le store n'a qu'une précision)."""
        return prec == self.store.hi and self.store.hi != self.store.lo

    # ------------------------------------------------------------------
    # Mémoire hôte (RAM) et disque
    # ------------------------------------------------------------------
    def _disk_job(self, layer: int, expert: int, prec: str, parts: int | None = None) -> torch.Tensor:
        return self.store.read(layer, expert, prec, out=self.pool.acquire(prec), parts=parts)

    def _to_release_later(self, blob: torch.Tensor) -> None:
        """Un tampon RAM n'est plus en cache : il sera rendu au pool au prochain point sûr."""
        self._releases.append(blob)

    def _flush_releases(self) -> None:
        """
        Point sûr : toutes les copies RAM -> VRAM qui lisent ces tampons sont déjà
        lancées. On les rend au pool avec un événement enregistré APRÈS elles : le
        pool attendra qu'il soit passé avant de réutiliser un slot.
        """
        if not self._releases:
            return
        ev = None
        if self.cuda:
            ev = torch.cuda.Event()
            ev.record(self.copy_stream)
        for b in self._releases:
            self.pool.release(b, ev)
        self._releases.clear()

    def _host_insert(self, key, blob: torch.Tensor, pinned: bool = False):
        """Range un tampon en RAM ; s'il n'y a pas de place (ou déjà présent), il redevient temporaire."""
        ent = self.host.insert(key, blob, pinned=pinned)
        if ent is None or ent.blob is not blob:
            self._releases.append(blob)
        return ent

    def _host_nowait(self, layer: int, expert: int, need: str):
        """
        (précision, tampon, True) si l'expert est déjà en RAM ;
        (précision, Future, False) sinon, après avoir lancé la lecture disque en arrière-plan.
        """
        ent = self.host.lookup(layer, expert, need, exact=not self.policy.allow_upgrade)
        if ent is not None:
            self.host.touch(ent)
            self.stats.host_hits += 1
            return ent.key[2], ent.blob, True
        key = (layer, expert, need)
        fut = self.pending.get(key)
        if fut is None:
            fut = self.io.submit(self._disk_job, layer, expert, need, 1 if self._bulk and not self.cfg.prefill_split else None)
            self.pending[key] = fut
            self.stats.disk_reads += 1
            self.stats.bytes_disk += self.store.record_bytes(need)
        return need, fut, False

    def _resolve(self, layer: int, expert: int, prec: str, fut: Future) -> torch.Tensor:
        t0 = time.perf_counter()
        blob = fut.result()
        self.stats.t_io_wait += time.perf_counter() - t0
        self.pending.pop((layer, expert, prec), None)
        self._host_insert((layer, expert, prec), blob)  # si la RAM est pleine d'épinglés, le tampon reste temporaire
        return blob

    # ------------------------------------------------------------------
    # Transfert vers le niveau rapide (VRAM)
    # ------------------------------------------------------------------
    def _to_fast(self, blob: torch.Tensor, prec: str):
        """Copie un record vers la VRAM sur un stream dédié. Retourne (tampon, événement)."""
        n = blob.numel()
        self.stats.bytes_h2d += n
        l8 = self._unpack.get(prec)
        if not self.cuda:
            dst = blob.clone()
            return (unpack_int4_record(dst, self.store.layouts[prec], l8) if l8 else dst), None
        with torch.cuda.stream(self.copy_stream):
            dst = torch.empty(n, dtype=torch.uint8, device=self.fast.device)
            dst.copy_(blob, non_blocking=self.pin)
            if l8:  # B2 : déballé une fois pour toutes, sur le stream de copie
                dst = unpack_int4_record(dst, self.store.layouts[prec], l8)
            ev = torch.cuda.Event()
            ev.record(self.copy_stream)
        self._inflight.append((ev, blob))  # garde le tampon hôte vivant jusqu'à la fin de la copie
        return dst, ev

    def _ready(self, blob: torch.Tensor, ev) -> None:
        """Avant d'utiliser sur le stream principal un tampon copié sur le stream de copie."""
        if ev is not None:
            cur = torch.cuda.current_stream(blob.device)
            cur.wait_event(ev)
            blob.record_stream(cur)

    def _accumulate(self, out, x, r, tok_dev, w_dev, layout, blob, act_fn) -> None:
        """out += poids x expert(x), pour les tokens qui ont choisi cet expert."""
        hook = self.expert_hook
        if tok_dev is None:  # génération (fast_decode) : 1 token, pas d'indexation
            if self._triton and layout["precision"] == "int4":  # B1 : l'int4 lu directement
                try:
                    y = expert_ffn_int4(x, views(layout, blob), layout["group_sizes"], act_fn)
                    out.addcmul_(y, w_dev[r.sl].to(out.dtype))
                    if hook is not None:
                        hook(self._layer, r.expert, w_dev[r.sl], y)
                    return
                except Exception as e:  # noyau indisponible sur cette machine : chemin PyTorch
                    self._triton = False
                    self.triton_error = f"{e.__class__.__name__}: {e}"
            y = expert_ffn(x, *materialize(layout, blob, x.dtype), act_fn)
            out.addcmul_(y, w_dev[r.sl].to(out.dtype))
        else:
            y = expert_ffn(x[tok_dev[r.sl]], *materialize(layout, blob, x.dtype), act_fn)
            out.index_add_(0, tok_dev[r.sl], (y * w_dev[r.sl, None]).to(out.dtype))
        if hook is not None:
            hook(self._layer, r.expert, w_dev[r.sl], y)

    def _int4_version(self) -> int:
        """Version des noyaux groupés int4 (0 = choisie une fois par mesure sur la carte)."""
        v = self.cfg.grouped_kernels
        if v != 0:
            return v
        if "int4" not in self.store.precisions or self.fast is None:
            return 4
        if getattr(self, "_auto_int4", None) is None:
            lay = self.fast_layouts["int4"]
            self._auto_int4 = pick_int4_version(lay, self.fast.device) if _pow2_groups(lay["group_sizes"]) else 1
        return self._auto_int4

    def _grouped_graph(self, layer: int, x: torch.Tensor, out: torch.Tensor, k: int) -> GroupedGraph | None:
        """Graph du calcul groupé de cette couche (créé au 1er usage, pour ces tampons fixes x / out)."""
        key = (layer, x.data_ptr())  # un graph par couche ET par jeu de tampons (paliers de longueur)
        gg = self._ggraphs.get(key)
        ver = self._int4_version()
        if (gg is not None and gg.out is out and gg.x.data_ptr() == x.data_ptr() and gg.k == k
                and gg.version == ver):
            return gg
        layouts = {}
        for p in self.store.precisions:
            lay = self.fast_layouts[p]
            if p == "int4" and _pow2_groups(lay["group_sizes"]):
                layouts[p] = lay
            elif p == "int8" and _pow2_groups8(lay["group_sizes"]):
                layouts[p] = lay
            elif PRECISIONS[p]["kind"] == "float":
                layouts[p] = lay
        gg = GroupedGraph(layouts, x, out, k, version=ver, capture=self.cuda) if layouts else None
        self._ggraphs[key] = gg
        return gg

    def _run_grouped(self, out, x, batch, tok_dev, w_dev, act_fn) -> None:
        """Calcule les experts réunis par précision : int4 et flottants en noyaux groupés, le reste un par un."""
        by_prec: dict[str, list] = {}
        for layout, blob, r in batch:
            by_prec.setdefault(layout["precision"], []).append((layout, blob, r))
        for prec, items in by_prec.items():
            layout = items[0][0]
            kind = PRECISIONS[prec]["kind"]
            try:
                entries = [(views(lay, blob), r.w) for lay, blob, r in items]
                if prec == "int4":
                    grouped_ffn_int4(x, entries, layout["group_sizes"], out, version=self._int4_version())
                    self._warm_precs.add(prec)
                    continue
                if prec == "int8" and _pow2_groups8(layout["group_sizes"]):
                    grouped_ffn_int8(x, entries, layout, out)
                    self._warm_precs.add(prec)
                    continue
                if kind == "float":
                    grouped_ffn_float(x, entries, out)
                    self._warm_precs.add(prec)
                    continue
            except Exception as e:  # noyau indisponible : chemin PyTorch, une fois pour toutes
                self._grouped = False
                self.triton_error = f"{e.__class__.__name__}: {e}"
            for lay, blob, r in items:
                self._accumulate(out, x, r, tok_dev, w_dev, lay, blob, act_fn)

    def _run_grouped_tokens(self, out, x, batch, tok_cpu, w_sorted_cpu, act_fn) -> None:
        """
        P6 : prefill d'un prompt court. Les noyaux groupés de la génération calculent la somme
        des experts pour UN vecteur x : on les appelle pour chaque token, avec ses experts et
        ses poids. 2 lancements par token et par précision, au lieu d'~30 par expert (déquan-
        tification, produits, index_add) ; chaque expert est relu une fois par token qui l'a
        choisi, ce qui reste négligeable tant que le prompt est court.
        """
        per_tok: dict[int, list] = {}
        weights: list[float] = []
        for layout, blob, r in batch:
            for t, w in zip(tok_cpu[r.sl].tolist(), w_sorted_cpu[r.sl].tolist()):
                j = len(weights)
                weights.append(w)
                per_tok.setdefault(t, []).append(
                    (layout, blob, _Req(r.expert, r.prec, slice(j, j + 1), r.hi, w, r.level)))
        w_dev = torch.tensor(weights, dtype=torch.float32, device=x.device)  # pour le repli (noyau indisponible)
        for t, items in per_tok.items():
            self._run_grouped(out[t:t + 1], x[t:t + 1], items, None, w_dev, act_fn)

    def _arrival_order(self, fill_jobs):
        """P2 : d'abord les experts déjà en RAM, puis les lectures disque dans l'ordre où elles finissent."""
        from concurrent.futures import as_completed

        waiting = {}
        for job in fill_jobs:
            if job[2] is not None:
                yield job
            else:
                waiting[job[4]] = job
        if waiting:
            it = as_completed(list(waiting))
            while True:
                t0 = time.perf_counter()
                try:
                    fut = next(it)
                except StopIteration:
                    return
                self.stats.t_io_wait += time.perf_counter() - t0
                yield waiting[fut]

    def _decide_general(self, layer, top_k_index, top_k_weights, T, k, dev, prof):
        """Décisions pour un nombre quelconque de tokens (prefill), avec vote par expert."""
        st, store = self.stats, self.store
        idx_cpu = top_k_index.to("cpu")
        w_cpu = top_k_weights.to("cpu", torch.float32)
        if prof:
            prof.mark("1 routage lu sur CPU (synchro)")
        lvl, skip = self.policy.levels(layer, w_cpu)
        st.layer_calls += 1
        st.pairs += T * k
        st.skipped += int(skip.sum())

        keep = (~skip).reshape(-1)
        flat_e = idx_cpu.reshape(-1)[keep]
        flat_pos = torch.arange(T * k)[keep]
        flat_lvl = lvl.reshape(-1)[keep]
        order = torch.argsort(flat_e, stable=True)
        flat_e, flat_pos, flat_lvl = flat_e[order], flat_pos[order], flat_lvl[order]
        uniq, counts = torch.unique_consecutive(flat_e, return_counts=True)
        tok_cpu = flat_pos // k
        pos_dev = flat_pos.to(dev)
        tok_dev = pos_dev // k
        w_dev = top_k_weights.reshape(-1)[pos_dev]
        w_sorted_cpu = w_cpu.reshape(-1)[flat_pos]

        reqs: list[_Req] = []
        start = 0
        for e, c in zip(uniq.tolist(), counts.tolist()):
            sl = slice(start, start + c)
            start += c
            level = self.policy.expert_level(flat_lvl[sl])
            reqs.append(_Req(e, level_to_precision(level, store.precisions), sl, level == 0, level=level))
            self.usage.record(layer, e, level == 0)
        return reqs, tok_cpu, tok_dev, w_dev, w_sorted_cpu

    def _decide_decode(self, layer, top_k_index, top_k_weights, k, prof, row=None):
        """
        Décisions pour UN token (génération), optimisation A2 du doc 18 : une seule copie
        GPU -> CPU (indices et poids ensemble), puis du Python pur sur k nombres.
        """
        st, store = self.stats, self.store
        if row is None:
            row = torch.cat((top_k_index.reshape(-1).float(), top_k_weights.reshape(-1).float())).tolist()
        if prof:
            prof.mark("1 routage lu sur CPU (synchro)")
        experts, weights = [int(e) for e in row[:k]], row[k:2 * k]
        levels, skip, s = self.policy.levels_one(layer, weights)
        st.layer_calls += 1
        st.pairs += k
        reqs: list[_Req] = []
        for j in range(k):
            if skip[j]:
                st.skipped += 1
                continue
            level = levels[j]
            reqs.append(_Req(experts[j], level_to_precision(level, store.precisions), slice(j, j + 1), level == 0,
                             weights[j], level, s[j]))
            self.usage.record(layer, experts[j], level == 0)
        return reqs

    # ------------------------------------------------------------------
    # Une couche MoE
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_layer(self, layer: int, x: torch.Tensor, top_k_index: torch.Tensor,
                  top_k_weights: torch.Tensor, act_fn, out: torch.Tensor | None = None,
                  packed: torch.Tensor | None = None) -> torch.Tensor:
        """
        `out` : tampon où écrire le résultat (A6 : adresse fixe lue par le CUDA graph suivant).
        `packed` (A9, génération en CUDA graphs) : indices et poids du routage, suivis de la
        prédiction du routeur de la couche suivante, déjà rassemblés par le graph en un seul
        tenseur : une seule lecture GPU -> CPU sert à la fois aux décisions et au prefetch.
        """
        t_start = time.perf_counter()
        if self.calib_hook is not None:
            self.calib_hook(layer, x, top_k_index, top_k_weights)
        prof = self.prof
        if prof:
            prof.start()
        st, store, cfg = self.stats, self.store, self.cfg
        T, H = x.shape
        k = top_k_index.shape[-1]
        dev = x.device
        self._bulk = T > 1
        self._layer = layer
        hooked = self.expert_hook is not None
        self.pool.advance()
        if self._inflight:
            self._inflight = [(e, b) for e, b in self._inflight if not e.query()]

        # -- 1. décisions de précision (HOBBIT) ----------------------------
        decode = T == 1 and cfg.fast_decode
        if decode:
            row = packed.tolist() if packed is not None else None
            if row is not None and len(row) > 2 * k:
                rest = row[2 * k:]
                P = self.prediction_len(k)
                if len(rest) > P and len(rest) % P == 0:  # prédictions de L+1, L+2, ... rassemblées par le graph
                    for i in range(len(rest) // P):
                        (self._pred_rows if i == 0 else self._pred_far)[layer + 1 + i] = rest[i * P:(i + 1) * P]
                else:
                    self._pred_rows[layer + 1] = rest
            reqs = self._decide_decode(layer, top_k_index, top_k_weights, k, prof, row)
            w_dev = top_k_weights.reshape(-1)
            tok_dev = tok_cpu = None
        else:
            reqs, tok_cpu, tok_dev, w_dev, w_sorted_cpu = self._decide_general(layer, top_k_index, top_k_weights, T, k, dev, prof)
        st.requests += len(reqs)
        st.need_hi += sum(self._is_hi(r.prec) for r in reqs)

        pred = self.predicted.pop(layer, None)
        if pred is not None:
            actual = {r.expert for r in reqs}
            st.pred_total += len(pred)
            st.pred_hits += len(pred & actual)
            st.pred_actual += len(actual)

        fast_ok = self.fast is not None and self.fast.device.type == dev.type
        # A6 bis : tampons fixes (CUDA graphs) + noyaux groupés -> calcul groupé rejoué en un graph
        use_gg = (out is not None and decode and fast_ok and self._grouped and cfg.moe_graphs and not self._unpack
                  and cfg.grouped_kernels != 1 and _is_silu(act_fn) and not hooked)
        if out is None:
            out = torch.zeros_like(x)
        elif not use_gg:
            out.zero_()
        if prof:
            prof.mark("2 décisions HOBBIT + regroupement")

        # -- 2. qui est déjà en VRAM ? ---------------------------------------
        hits, misses = [], []
        exact = not self.policy.allow_upgrade
        for r in reqs:
            ent = self.fast.lookup(layer, r.expert, r.prec, exact=exact) if fast_ok else None
            if ent is None:
                misses.append(r)
            else:
                self.fast.touch(ent)
                if ent.prefetched:
                    st.prefetch_hits += 1
                    ent.prefetched = False
                hits.append((r, ent))
        # E3 : un expert demandé en haute précision mais absent de la VRAM, et pas
        # assez important (s > miss_t1), descend d'un niveau ; s'il est alors en VRAM, c'est un hit.
        if decode and fast_ok and self.policy.miss_t1 is not None and misses:
            keep_miss = []
            for r in misses:
                if r.level == 0 and r.s > self.policy.miss_t1 + 1e-6:
                    r.level, r.prec, r.hi = 1, level_to_precision(1, store.precisions), False
                    st.miss_downgrades += 1
                    ent = self.fast.lookup(layer, r.expert, r.prec, exact=exact)
                    if ent is not None:
                        self.fast.touch(ent)
                        hits.append((r, ent))
                        continue
                keep_miss.append(r)
            misses = keep_miss
        # P5 : prefill, un expert qu'il faudrait lire sur le disque et dont aucun token n'a besoin
        # beaucoup (plus grand poids < prefill_skip) est sauté, comme le seuil t2 de HOBBIT.
        if T > 1 and cfg.prefill_skip is not None and misses:
            keep_miss = []
            for r in misses:
                if ((layer, r.expert, r.prec) not in self.pending
                        and self.host.lookup(layer, r.expert, r.prec, exact=exact, count=False) is None
                        and float(w_sorted_cpu[r.sl].max()) < cfg.prefill_skip):
                    st.prefill_skipped += 1
                    continue
                keep_miss.append(r)
            misses = keep_miss
        st.fast_hits += len(hits)
        self._calm_layers = 0 if misses else self._calm_layers + 1

        # -- 3. q* : transfert vers la VRAM ou calcul sur CPU ----------------
        if fast_ok and misses:
            q = qstar(len(misses), cfg.b_pcie, cfg.b_host) if cfg.use_qstar else len(misses)
            misses.sort(key=lambda r: (-(r.sl.stop - r.sl.start), -self.usage.score(layer, r.expert)))
            fills, cpu_set = misses[:q], misses[q:]
        else:
            fills, cpu_set = [], misses

        # -- 4. lancer tout de suite les lectures disque (recouvrement I/O) ---
        todo = fills + cpu_set
        if cfg.sort_reads:  # D3 : dans l'ordre du fichier (le HDD n'a pas à faire d'allers-retours)
            todo = sorted(todo, key=lambda r: (r.prec, r.expert))
        src = {r.expert: self._host_nowait(layer, r.expert, r.prec) for r in todo}
        if prof:
            prof.mark("3 recherche caches + q* + lectures lancées")

        x_cpu = None
        if cpu_set:
            x_cpu = x.to("cpu", cfg.cpu_dtype)

        # A4+B1 : en génération, tous les experts GPU de la couche en quelques lancements
        # P6 : au prefill (prompt court), même chose token par token
        grouped_tokens = (not decode and 1 < T <= cfg.prefill_grouped and self._grouped and fast_ok
                          and _is_silu(act_fn) and not self._unpack and not hooked)
        batch = [] if ((decode or grouped_tokens) and self._grouped and fast_ok and _is_silu(act_fn)
                       and not hooked) else None

        # Experts déjà en VRAM : calcul GPU (les noyaux partent en asynchrone)
        protect = {ent.key for _, ent in hits}
        for r, ent in hits:
            if ent.event is not None:
                self._ready(ent.blob, ent.event)
                ent.event = None
            if batch is not None:
                batch.append((self.fast_layouts[ent.key[2]], ent.blob, r))
            else:
                self._accumulate(out, x, r, tok_dev, w_dev, self.fast_layouts[ent.key[2]], ent.blob, act_fn)
            st.used_hi += self._is_hi(ent.key[2])
        if prof:
            prof.mark("4 calcul des experts déjà en VRAM")

        # Transferts dont la source est déjà en RAM : ils partent maintenant
        fill_jobs = []
        for r in fills:
            prec, s, ready = src[r.expert]
            if ready:
                dst, ev = self._to_fast(s, prec)
                fill_jobs.append((r, prec, dst, ev, None))
            else:
                fill_jobs.append((r, prec, None, None, s))

        # -- 5. branche CPU (pendant que le GPU calcule et que les copies avancent)
        if cpu_set:
            t0 = time.perf_counter()
            out_cpu = torch.zeros(T, H, dtype=cfg.cpu_dtype)
            for r in cpu_set:
                prec, s, ready = src[r.expert]
                blob = s if ready else self._resolve(layer, r.expert, prec, s)
                weights = materialize(store.layouts[prec], blob, cfg.cpu_dtype)
                if decode:
                    y = expert_ffn(x_cpu, *weights, act_fn)
                    out_cpu.add_(y, alpha=r.w)
                    if hooked:
                        self.expert_hook(layer, r.expert, torch.tensor([r.w]), y)
                else:
                    tok = tok_cpu[r.sl]
                    y = expert_ffn(x_cpu[tok], *weights, act_fn)
                    out_cpu.index_add_(0, tok, y * w_sorted_cpu[r.sl, None])
                    if hooked:
                        self.expert_hook(layer, r.expert, w_sorted_cpu[r.sl], y)
                st.used_hi += self._is_hi(prec)
            st.cpu_execs += len(cpu_set)
            st.t_cpu += time.perf_counter() - t0
        if prof:
            prof.mark("5 copies lancées + branche CPU")

        # -- 6. transferts restants, insertion en VRAM, calcul GPU ----------
        keep_vram = T == 1 or cfg.prefill_cache != "ram"
        for r, prec, dst, ev, fut in (fill_jobs if T == 1 or cfg.prefill_order != "completed"
                                      else self._arrival_order(fill_jobs)):
            if dst is None:
                dst, ev = self._to_fast(self._resolve(layer, r.expert, prec, fut), prec)
            key = (layer, r.expert, prec)
            # (protect est lu, pas modifié, par insert : plus de copie en frozenset à chaque expert,
            # qui coûtait O(n²) au prefill, quelques centaines d'experts par couche)
            ent = self.fast.insert(key, dst, protect=protect, event=ev) if keep_vram else None
            protect.add(key)
            self._ready(dst, ev)
            if ent is not None:
                ent.event = None
            if batch is not None:
                batch.append((self.fast_layouts[prec], dst, r))
            else:
                self._accumulate(out, x, r, tok_dev, w_dev, self.fast_layouts[prec], dst, act_fn)
            st.used_hi += self._is_hi(prec)
        st.fills += len(fill_jobs)
        cpu_done = False
        if use_gg:
            gg = self._grouped_graph(layer, x, out, k)
            if gg is not None and all(gg.supports(lay["precision"]) for lay, _, _ in batch):
                items: dict[str, list] = {}
                for lay, blob, r in batch:
                    items.setdefault(lay["precision"], []).append((blob, r.w))
                try:
                    gg.run(items, out_cpu.to(dev, out.dtype) if cpu_set else None)
                    self._warm_precs.update(items)
                    batch, cpu_done = [], True
                except Exception as e:  # graph impossible : chemin normal, une fois pour toutes
                    cfg.moe_graphs = False
                    self._ggraphs.clear()
                    self.triton_error = f"graph du calcul groupé : {e.__class__.__name__}: {e}"
                    out.zero_()
            else:
                out.zero_()
        if batch and grouped_tokens:
            self._run_grouped_tokens(out, x, batch, tok_cpu, w_sorted_cpu, act_fn)
        elif batch:
            self._run_grouped(out, x, batch, tok_dev, w_dev, act_fn)

        if cpu_set and not cpu_done:
            out.add_(out_cpu.to(dev, out.dtype))
        if prof:
            prof.mark("6 attente disque/copies + calcul des manquants")

        # -- 7. prefetch de la couche suivante -----------------------------
        calm = cfg.prefetch_adaptive and self._calm_layers >= len(store.layers)
        if cfg.prefetch and T > cfg.prefetch_max_tokens and cfg.prefill_prefetch and (layer + 1) in self.gates:
            self._prefetch(layer + 1, x, fast_ok, None, ram_only=True)  # P3 : disque -> RAM, couche suivante
        elif cfg.prefetch and not calm and T <= cfg.prefetch_max_tokens:
            for d in range(1, max(1, cfg.prefetch_depth) + 1):  # C2 : L+1, puis L+2 si demandé
                if (layer + d) in self.gates:
                    self._prefetch(layer + d, x, fast_ok, self._pred_rows.pop(layer + d, None) if d == 1 else None)
            for d in range(2, 2 + max(0, cfg.prefetch_disk_ahead)):  # disque -> RAM seulement, plus loin
                if (layer + d) in self.gates and T == 1:
                    self._prefetch(layer + d, x, fast_ok, self._pred_far.pop(layer + d, None), ram_only=True)
        if prof:
            prof.mark("7 prefetch")

        self._flush_releases()
        st.t_moe += time.perf_counter() - t_start
        return out

    # ------------------------------------------------------------------
    # A5 : token entier en un graph (fullgraph.py)
    # ------------------------------------------------------------------
    def device_table(self):
        """Table GPU des experts résidents, créée au premier usage puis tenue à jour par le cache VRAM."""
        if self._dev_table is None:
            from .fullgraph import DeviceTable

            self._dev_table = DeviceTable(self)
            self.fast.listener = self._dev_table
        return self._dev_table

    def full_possible(self) -> str | None:
        """None si le chemin A5 peut servir avec ces réglages, sinon la raison."""
        cfg = self.cfg
        if not cfg.full_graph:
            return "désactivé"
        if self.expert_hook is not None:
            return "crochet REAP actif (il faut la sortie de chaque expert)"
        if self.fast is None:
            return "pas de VRAM pour les experts"
        if not (self._grouped and cfg.moe_graphs and cfg.fast_decode) or self._unpack or cfg.grouped_kernels == 1:
            return "il faut les noyaux groupés en graph (A4, A6 bis)"
        if self.policy.miss_t1 is not None:
            return "E3 (miss_t1) dépend du contenu des caches"
        if not all(_is_silu(a) for a in self.act_fns.values()):
            return "activation autre que SiLU"
        if not self._warm_precs:
            return "noyaux pas encore échauffés"
        return None

    def full_ready(self) -> bool:
        """Tenter le graph complet pour ce token : possible, et le token précédent n'a eu aucun défaut de cache."""
        return self.full_possible() is None and self._calm_layers >= len(self.store.layers)

    def full_signature(self) -> tuple:
        """Ce qui est figé dans le graph complet : s'il change, on le recapture."""
        p = self.policy
        return (p.mode, p.t1, tuple(sorted((p.t1_layers or {}).items())), p.t_mid, p.t2, tuple(sorted(p.protect_layers)),
                p.allow_upgrade, frozenset(self._warm_precs), self._int4_version())

    def new_full_moe(self, k: int, x: torch.Tensor):
        from .fullgraph import FullMoE

        return FullMoE(self, k, x, set(self._warm_precs), self._int4_version())

    def after_full_token(self, vals: list[float], k: int) -> None:
        """Compteurs, scores des caches et historique d'après le routage lu après un token A5 réussi."""
        st, store, fast = self.stats, self.store, self.fast
        precs = store.precisions
        L = len(store.layers)
        st.full_tokens += 1
        for l in range(L):
            b = 1 + 3 * k * l
            ids, ch, lv = vals[b:b + k], vals[b + k:b + 2 * k], vals[b + 2 * k:b + 3 * k]
            st.layer_calls += 1
            st.pairs += k
            for j in range(k):
                if ch[j] < -1.5:
                    st.skipped += 1
                    continue
                e, p, lvl = int(ids[j]), precs[int(ch[j])], int(lv[j])
                st.requests += 1
                st.fast_hits += 1
                st.need_hi += self._is_hi(level_to_precision(lvl, precs))
                st.used_hi += self._is_hi(p)
                ent = fast.entries.get((l, e, p))
                if ent is not None:
                    fast.stats.hits += 1
                    fast.touch(ent)
                    if ent.prefetched:
                        st.prefetch_hits += 1
                        ent.prefetched = False
                self.usage.record(l, e, lvl == 0)
        self._calm_layers += L

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------
    def prediction_tensor(self, logits: torch.Tensor, scores: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Prédiction du prefetch pour UN token, en un tenseur (calculable dans un CUDA graph, A9).
          - prefetch_mass = None : [experts (k), poids (k)] du routeur ;
          - prefetch_mass = m (S3, prefetch selon le score, SpecMD) : les experts les plus
            probables jusqu'à une masse m de la distribution complète du routeur, au plus 2k :
            [experts (kp), poids (kp, 0 au-delà des k du routeur), choisi (kp, 1/0)].
            Peu d'experts quand le routeur est sûr de lui, plus quand il hésite.
        """
        m = self.cfg.prefetch_mass
        if m is None:
            return torch.cat((idx.reshape(-1).float(), scores.reshape(-1).float()))
        k = idx.shape[-1]
        kp = min(logits.shape[-1], 2 * k)
        probs = torch.softmax(logits.reshape(-1, logits.shape[-1])[-1].float(), dim=-1)
        pv, pi = probs.topk(kp)
        chosen = ((pv.cumsum(0) - pv) < m).float()
        w = torch.cat((scores.reshape(-1).float(), torch.zeros(kp - k, device=pv.device)))
        return torch.cat((pi.float(), w, chosen))

    def _prediction_wants(self, layer: int, row: list) -> dict[int, str]:
        """Liste lue (prediction_tensor) -> {expert: précision à précharger}."""
        precs = self.store.precisions
        if self.cfg.prefetch_mass is None:
            k = len(row) // 2
            levels, skip, _ = self.policy.levels_one(layer, row[k:2 * k])
            return {int(row[j]): level_to_precision(levels[j], precs) for j in range(k) if not skip[j]}
        kp = len(row) // 3
        gate = self.gates.get(layer)
        k = min(int(getattr(gate, "top_k", kp)), kp)
        levels, skip, _ = self.policy.levels_one(layer, row[kp:kp + k])
        want = {}
        for j in range(kp):
            if row[2 * kp + j] < 0.5:
                continue
            if j < k:
                if not skip[j]:
                    want[int(row[j])] = level_to_precision(levels[j], precs)
            else:  # au-delà des k du routeur : peu probable, basse précision
                want.setdefault(int(row[j]), precs[-1])
        return want

    def prediction_len(self, k: int) -> int:
        """Longueur d'une prédiction de prediction_tensor (pour découper la lecture rassemblée)."""
        if self.cfg.prefetch_mass is None:
            return 2 * k
        return 3 * min(self.store.num_experts, 2 * k)

    @torch.no_grad()
    def _prefetch(self, layer: int, x: torch.Tensor, fast_ok: bool, row: list | None = None,
                  ram_only: bool = False) -> None:
        """
        Applique le routeur de `layer` à l'état de la couche précédente et précharge sa prédiction.
        `row` (A9) : prédiction déjà calculée dans le CUDA graph et lue avec le routage courant.
        `ram_only` : seulement les lectures disque -> RAM (prefetch_disk_ahead), rien vers la VRAM.
        """
        want: dict[int, str] = {}
        if row is None:
            logits, scores, idx = self.gates[layer](x)
        if row is not None or (self.cfg.fast_decode and idx.shape[0] == 1):  # une seule synchro, Python pur (A2)
            if row is None:
                row = self.prediction_tensor(logits, scores, idx).tolist()
            want = self._prediction_wants(layer, row)
        else:
            lvl, skip = self.policy.levels(layer, scores.to("cpu", torch.float32))
            idx = idx.to("cpu")
            best: dict[int, int] = {}
            for t in range(idx.shape[0]):
                for j in range(idx.shape[1]):
                    if not skip[t, j]:
                        e = int(idx[t, j])
                        best[e] = min(best.get(e, 2), int(lvl[t, j]))
            want = {e: level_to_precision(v, self.store.precisions) for e, v in best.items()}
        exact = not self.policy.allow_upgrade
        if ram_only:
            for e, prec in sorted(want.items(), key=lambda kv: (kv[1], kv[0])):
                if fast_ok and self.fast.lookup(layer, e, prec, exact=exact, count=False) is not None:
                    continue
                if self.host.lookup(layer, e, prec, exact=exact, count=False) is None:
                    self._host_nowait(layer, e, prec)
                    self.stats.disk_ahead += 1
            return
        self.predicted[layer] = set(want) | (self.predicted.get(layer, set()) if self.cfg.prefetch_depth > 1 else set())
        items = sorted(want.items(), key=lambda kv: (kv[1], kv[0])) if self.cfg.sort_reads else want.items()
        for e, prec in items:
            if fast_ok:
                if self.fast.lookup(layer, e, prec, exact=exact, count=False) is not None:
                    continue
                ent = self.host.lookup(layer, e, prec, exact=exact, count=False)
                if ent is not None:
                    dst, ev = self._to_fast(ent.blob, ent.key[2])
                    new = self.fast.insert((layer, e, ent.key[2]), dst, event=ev)
                    if new is not None:
                        new.prefetched = True
                    self.stats.prefetch_issued += 1
                else:
                    self._host_nowait(layer, e, prec)  # lecture disque -> RAM en arrière-plan
                    self.stats.prefetch_issued += 1
            elif self.host.lookup(layer, e, prec, exact=exact, count=False) is None:
                self._host_nowait(layer, e, prec)
                self.stats.prefetch_issued += 1

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------
    def on_forward(self, n_tokens: int) -> None:
        self.stats.forwards += 1
        self.stats.tokens += n_tokens
        self.host.tick()
        if self.fast is not None:
            self.fast.tick()
        self._drain_pending()

    def _drain_pending(self) -> None:
        """Range en RAM les lectures disque terminées que personne n'a encore réclamées (prefetch)."""
        for key, fut in list(self.pending.items()):
            if fut.done():
                del self.pending[key]
                self._host_insert(key, fut.result())
        self._flush_releases()

    def warm_start(self, preload_lo: bool = False, log=print) -> None:
        """
        Remplissage initial :
          - preload_lo : tout le "pool" basse précision en RAM, épinglé (le pool hôte
            de FreeToken) — seulement si ça tient dans le budget RAM ;
          - puis les experts les plus utilisés d'après usage.json, en haute précision,
            en RAM puis en VRAM, épinglés (le cache d'apprentissage de Colibrì).
        """
        store = self.store
        self.preload_lo = preload_lo
        if self.usage.ignored_reason:
            log(f"  usage.json : {self.usage.ignored_reason}")
        jobs: list[tuple[int, int, str, bool]] = []
        reserved = 0
        if preload_lo:
            total = store.total_bytes(store.lo)
            if total <= self.host.budget * 0.95:
                self.host.pin_budget = max(self.host.pin_budget, total + int(self.host.budget * 0.05))
                reserved = total
                jobs += [(l, e, store.lo, True) for l in range(len(store.layers)) for e in range(store.num_experts)
                         if (l, e) not in self.pruned]
            else:
                log(f"  pool {store.lo} ({total / 2**30:.2f} Gio) plus grand que le budget RAM, pas de préchargement complet")
        # Experts chauds (historique) : seulement ce qui tient dans la part épinglable restante.
        n_hot = max(0, (self.host.pin_budget - reserved) // store.record_bytes(store.hi))
        hot = [(l, e) for (l, e) in self.usage.top(n_hot + len(self.pruned))
               if l < len(store.layers) and e < store.num_experts and (l, e) not in self.pruned][:n_hot]
        if self.policy.mode == "lo":
            hot = []  # la haute précision ne servira jamais
        if not (reserved and store.hi == store.lo):  # store à une seule précision déjà entièrement préchargé
            jobs += [(l, e, store.hi, True) for (l, e) in hot]
        if not jobs:
            self.reset_stats()  # sinon les compteurs du réglage précédent s'additionnent (bug du 30/09)
            return
        t0 = time.time()
        loaded, n_bytes = 0, 0
        futures = [(j, self.io.submit(self._disk_job, *j[:3])) for j in jobs]
        for (l, e, prec, pin), fut in futures:
            blob = fut.result()
            if self._host_insert((l, e, prec), blob, pinned=pin) is None:
                continue
            loaded += 1
            n_bytes += blob.numel()
        self.stats.bytes_disk += n_bytes
        self._flush_releases()
        log(f"  préchargement RAM : {loaded} experts, {n_bytes / 2**30:.2f} Gio en {time.time() - t0:.1f} s")
        if self.fast is not None and hot:
            n = 0
            for (l, e) in hot:
                ent = self.host.lookup(l, e, store.hi, exact=True, count=False)
                if ent is None:
                    continue
                dst, ev = self._to_fast(ent.blob, store.hi)
                if self.fast.insert((l, e, store.hi), dst, pinned=True, event=ev) is None:
                    break
                n += 1
            if self.cuda:
                torch.cuda.synchronize()
            log(f"  préchargement {self.fast.name} : {n} experts chauds")
        self.reset_stats()

    def end_of_request(self) -> dict:
        """Point sûr entre deux requêtes : ré-épinglage (anti-thrashing) et sauvegarde de l'historique."""
        swaps = {"host": self.host.repin()}
        if self.fast is not None:
            swaps["fast"] = self.fast.repin()
        self.usage.save()
        return swaps

    def reset_stats(self) -> None:
        self.stats = RunStats()
        self.store.bytes_read = 0
        self.store.read_seconds = 0.0

    def set_budgets(self, fast_bytes: int | None = None, host_bytes: int | None = None) -> None:
        """
        Budgets VRAM / RAM changés à chaud (05/10, pour l'autopilot) : caches vidés, nouveaux
        budgets ; le pool de RAM épinglée peut grandir (jamais rétrécir : ses plaques restent).
        À suivre d'un warm_start.
        """
        self.reset_caches()
        if fast_bytes is not None:
            if self.fast is None:
                raise ValueError("pas de cache VRAM sur cette machine (ou au chargement)")
            self.fast.budget = int(fast_bytes)
            self.fast.pin_budget = int(fast_bytes * self.cfg.pin_fraction)
            self.cfg.fast_budget_bytes = int(fast_bytes)
        if host_bytes is not None:
            self.host.budget = int(host_bytes)
            self.host.pin_budget = int(host_bytes * self.cfg.pin_fraction)
            self.cfg.host_budget_bytes = int(host_bytes)
            rbs = [self.store.record_bytes(p) for p in self.store.precisions]
            margin = max(256 * 2**20, 4 * (self.cfg.io_workers + 8) * max(rbs))
            self.pool.cap = max(self.pool.cap, int(host_bytes) + margin)

    def reset_caches(self) -> None:
        for fut in self.pending.values():
            self._releases.append(fut.result())
        self.pending.clear()
        self.predicted.clear()
        if self.cuda:
            torch.cuda.synchronize()
        self._inflight.clear()
        self.host.clear()
        if self.fast is not None:
            self.fast.clear()
        self._flush_releases()

    def describe(self) -> str:
        parts = [self.host.describe(), self.pool.describe()]
        if self.fast is not None:
            parts.insert(0, self.fast.describe())
        return " | ".join(parts)

    def close(self) -> None:
        self.io.shutdown(wait=True)
        if self.cuda:
            torch.cuda.synchronize()
        self.store.close()
        self.pool.close()


class OffloadedExperts(nn.Module):
    """
    Remplace le module `experts` d'une couche MoE de transformers (OlmoeExperts,
    Qwen3NextExperts...). Même signature d'appel : (hidden_states, top_k_index,
    top_k_weights). Pas de poids : ils vivent dans le store et les caches.
    """

    def __init__(self, runtime: MoERuntime, layer: int, act_fn: nn.Module, num_experts: int):
        super().__init__()
        self.runtime = runtime
        self.layer = layer
        self.act_fn = act_fn
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        hook = self.runtime.capture_hook
        if hook is not None:  # capture des CUDA graphs : la couche MoE est la frontière entre deux graphs
            return hook(self, hidden_states, top_k_index, top_k_weights)
        return self.runtime.run_layer(self.layer, hidden_states, top_k_index, top_k_weights, self.act_fn)

    def extra_repr(self) -> str:
        return f"couche_moe={self.layer}, experts={self.num_experts}, store={self.runtime.store.path.name}"
