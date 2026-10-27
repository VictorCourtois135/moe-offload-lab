"""
A6 (doc 18) : la partie dense de la génération rejouée en CUDA graphs.

Le constat du profil du 01/10 : avec les noyaux groupés, les couches MoE ne coûtent
plus que ~14 ms par token sur OLMoE, mais la partie dense (attention, normes,
RoPE, routeur) en coûte ~35. Elle ne calcule presque rien (0,9 Gio de poids, ~3 ms
de lecture VRAM) : c'est le coût de lancement de ~1 000 petits noyaux par token,
chacun lancé par du Python.

Un CUDA graph enregistre une suite de noyaux une fois, puis la relance en UN appel.
Le problème : entre deux morceaux denses, la couche MoE prend ses décisions sur CPU
(lecture du routage, caches, disque). Elle ne peut pas être dans un graph.

La solution, générique (aucun code propre au modèle) :
  - on capture UNE passe du modèle transformers, avec un cache KV statique ;
  - à chaque appel d'un module d'experts, on FERME le graph en cours et on en ouvre
    un nouveau : la passe est découpée en N+1 segments (N = couches MoE) ;
  - en génération, on rejoue : segment 0, couche MoE 0 (notre runtime, en Python,
    comme avant), segment 1, couche MoE 1, ..., segment N, puis la tête de sortie.

Les entrées et sorties de chaque couche MoE sont des tenseurs à adresse fixe (ceux
de la capture) : le segment précédent y écrit, le runtime les lit, et écrit son
résultat dans un tampon fixe que lit le segment suivant.

Ce qui rend ça possible : le cache statique de transformers (StaticCache) ne fait
que des opérations de tenseurs (position courante = tenseur sur GPU, écriture par
index_copy_). Aucune valeur n'est lue par Python pendant la passe, donc la rejouer
sans Python donne le même résultat.

Pour tester sans GPU, un émulateur (_EmuGraph) enregistre les opérations aten de la
capture et les rejoue sur les mêmes tenseurs : mêmes règles qu'un vrai CUDA graph
(rien n'est recalculé par Python, une lecture de valeur pendant la capture est une
erreur). tests/test_graphs.py compare ainsi la génération rejouée à la génération
normale, sur CPU.
"""

from __future__ import annotations

import os

from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from torch.utils._pytree import tree_flatten


class GraphCaptureError(RuntimeError):
    """La passe du modèle lit une valeur de tenseur depuis Python : elle ne peut pas être rejouée."""


# Opérations qui imposent une lecture de valeur par le CPU (donc une synchronisation
# et une décision Python figée dans le graph).
_FORBIDDEN = {"_local_scalar_dense", "nonzero", "unique", "_unique", "_unique2", "unique_consecutive",
              "unique_dim", "masked_select", "is_nonzero", "equal", "item"}


def _sid(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


class _EmuPool:
    """Équivalent du pool mémoire des graphs : les stockages créés pendant la capture."""

    def __init__(self):
        self.created: set[int] = set()


class _Recorder(TorchDispatchMode):
    def __init__(self, graph: "_EmuGraph"):
        super().__init__()
        self.g = graph

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func.overloadpacket.__name__
        if name in _FORBIDDEN:
            raise GraphCaptureError(f"aten::{name} pendant la capture (valeur lue par Python)")
        # Un vrai CUDA graph n'EXÉCUTE rien pendant la capture. L'émulateur exécute (il faut
        # des sorties), donc il note l'état des tenseurs préexistants qu'il modifie, pour
        # le restaurer en fin de capture.
        for i, a in enumerate(func._schema.arguments):
            if a.alias_info is not None and a.alias_info.is_write:
                t = args[i] if i < len(args) else kwargs.get(a.name)
                if isinstance(t, torch.Tensor) and _sid(t) not in self.g.pool.created:
                    self.g.undo.append((t, t.clone()))
        out = func(*args, **kwargs)
        outs = out if isinstance(out, (tuple, list)) else (out,)
        for r, t in zip(func._schema.returns, outs):
            if r.alias_info is None:  # nouveau stockage (une vue partage celui de son tenseur d'origine)
                for u in tree_flatten(t)[0]:
                    if isinstance(u, torch.Tensor):
                        self.g.pool.created.add(_sid(u))
        self.g.ops.append((func, args, kwargs, out))
        return out


class _EmuGraph:
    """Émulation CPU d'un CUDA graph (tests) : capture = enregistrement, replay = mêmes opérations, mêmes tenseurs."""

    def __init__(self, pool: _EmuPool):
        self.pool = pool
        self.ops: list = []
        self.undo: list = []
        self._mode: _Recorder | None = None

    def begin(self) -> None:
        self._mode = _Recorder(self)
        self._mode.__enter__()

    def end(self) -> None:
        if self._mode is not None:
            self._mode.__exit__(None, None, None)
            self._mode = None
        for t, saved in reversed(self.undo):
            t.copy_(saved)
        self.undo.clear()

    def abort(self) -> None:
        self.end()

    def call(self, fn) -> None:
        """Lancement opaque (noyaux Triton, A5) : exécuté hors enregistrement, rejoué tel quel."""
        with _disable_current_modes():
            fn()
        self.ops.append((fn, None, None, None))

    def replay(self) -> None:
        for func, args, kwargs, out in self.ops:
            if args is None:
                func()
                continue
            res = func(*args, **kwargs)
            rets = func._schema.returns
            outs = out if isinstance(out, (tuple, list)) else (out,)
            news = res if isinstance(res, (tuple, list)) else (res,)
            for r, o, n in zip(rets, outs, news):
                if r.alias_info is None and isinstance(o, torch.Tensor) and o is not n:
                    o.copy_(n)


class _CudaGraph:
    def __init__(self, pool):
        self.g = torch.cuda.CUDAGraph()
        self.pool = pool
        self.open = False

    def begin(self) -> None:
        # thread_local : les threads d'I/O du runtime ne cassent pas la capture
        self.g.capture_begin(pool=self.pool, capture_error_mode="thread_local")
        self.open = True

    def end(self) -> None:
        self.g.capture_end()
        self.open = False

    def call(self, fn) -> None:
        fn()  # les noyaux lancés pendant la capture font partie du graph

    def abort(self) -> None:
        if self.open:
            try:
                self.g.capture_end()
            except Exception:
                pass
            self.open = False

    def replay(self) -> None:
        self.g.replay()


class GraphDecoder:
    """
    Génération token par token avec la partie dense en CUDA graphs (voir l'en-tête).

        dec = GraphDecoder(model, max_len=2048)
        logits = dec.prefill(ids)            # passe normale, remplit le cache statique
        logits = dec.step(next_token)        # (1, 1) sur le GPU -> logits (1, vocab)

    Le 1er step est une passe normale (échauffement) ; les suivants rejouent des graphs.
    Si la capture échoue, `failed` contient la raison et la génération continue
    normalement (même cache, même résultat, sans le gain). `backend="emulate"` :
    émulateur CPU, pour les tests.

    Longueurs par paliers (01/10) : avec un cache statique, l'attention lit les
    `max_len` positions à chaque token, même quand la conversation n'en compte que 100
    (16 Mo par couche à 2 048 pour OLMoE, ~0,9 ms par token). Un jeu de graphs est
    donc capturé par palier de longueur (128, 256, 512... max_len) : pendant la
    capture, le cache est remplacé par une VUE de ses L premières positions (mêmes
    données, aucune copie). Chaque token rejoue le plus petit palier qui le contient.
    """

    def __init__(self, model, max_len: int = 2048, backend: str = "auto", min_bucket: int = 128):
        from transformers import StaticCache

        self.model = model
        self.rt = model.moelab_runtime
        self.dev = model.moelab_device
        if backend == "auto":
            backend = "cuda" if self.dev.type == "cuda" else "emulate"
        self.backend = backend
        self.max_len = max_len
        self.buckets = sorted({min_bucket * 2**i for i in range(24) if min_bucket * 2**i < max_len} | {max_len})
        self.base = model.base_model
        self.embed = model.get_input_embeddings()
        self.head = model.get_output_embeddings()
        self.failed: str | None = None
        if getattr(model.config, "final_logit_softcapping", None):
            self.failed = "final_logit_softcapping non géré"
        self.cache = StaticCache(config=model.config, max_cache_len=max_len)
        dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
        self.emb_buf = torch.zeros(1, 1, model.config.hidden_size, dtype=dtype, device=self.dev)
        self.sets: dict[int, dict] = {}   # palier -> {"segs", "moe", "final"}
        self.cur: dict | None = None
        self.warm = False
        self.length = 0
        self.captures = 0
        self.pack_routing = True  # A9 (désactivable pour comparer : RuntimeConfig.graph_pack_routing)
        self.stream = torch.cuda.Stream(device=self.dev) if backend == "cuda" else None
        self._mass = (self.rt.cfg.prefetch_mass, self.rt.cfg.prefetch_disk_ahead)
        # A5 : graphs du token entier (fullgraph.py), par palier et par signature des réglages
        self.full: dict = {}
        self.full_failed: str | None = None
        self._states: list[torch.Tensor] = []  # états modifiés à chaque token (compteurs de position, attention linéaire)
        # P1 (03/10) : réutilisation du cache d'un tour de conversation au suivant
        self.tokens: list[int] | None = None   # tokens déjà dans le cache, dans l'ordre (None = inconnus)
        self._snap: tuple | None = None        # (longueur, états) à la fin du dernier prefill
        self.reused = 0                        # tokens repris du cache au dernier prefill
        self.prefix: dict | None = None        # P9 : début fixe gardé (tokens, états, K/V sur CPU)
        self.prefix_dir: str | None = None     # ... et son dossier sur disque
        self.prefix_key = ""                   # ... identité du contexte (store, politique) dans le nom du fichier
        self.prefix_hit = False

    # Compatibilité : le jeu de graphs du palier courant
    @property
    def segs(self):
        return self.cur["segs"] if self.cur else None

    @property
    def moe(self):
        return self.cur["moe"] if self.cur else []

    # ------------------------------------------------------------------
    def _new_graph(self):
        return _CudaGraph(self._pool) if self.backend == "cuda" else _EmuGraph(self._pool)

    def _hook(self, mod, x, idx, w):
        """Appelé à la place d'une couche MoE pendant la capture : on coupe le graph ici."""
        # A9 : avant de couper, le graph prépare aussi la prédiction du prefetch (routeur de la
        # couche suivante appliqué au même état) et rassemble tout en un tenseur : le runtime
        # n'aura qu'une lecture GPU -> CPU à faire, et plus aucun noyau à lancer pour prédire.
        parts = [idx.reshape(-1).float(), w.reshape(-1).float()]
        nxt = self.rt.gates.get(mod.layer + 1)
        if nxt is not None and x.shape[0] == 1:
            lg, ps, pi = nxt(x)
            parts.append(self.rt.prediction_tensor(lg, ps, pi))  # S3 : selon prefetch_mass
            for d in range(2, 2 + max(0, self.rt.cfg.prefetch_disk_ahead)):  # disque -> RAM, plus loin
                far = self.rt.gates.get(mod.layer + d)
                if far is None:
                    break
                lg, ps, pi = far(x)
                parts.append(self.rt.prediction_tensor(lg, ps, pi))
        packed = torch.cat(parts)
        self._cur.end()
        self._segs.append(self._cur)
        out = torch.empty_like(x)  # tampon fixe : le runtime y écrira, le segment suivant le lira
        self._moe.append((mod, x, idx, w, out, packed))
        self._cur = self._new_graph()
        self._cur.begin()
        return out

    def _bucket(self, n: int) -> int:
        return next(b for b in self.buckets if b >= n)

    @torch.no_grad()
    def _capture(self, L: int) -> dict | None:
        self._pool = torch.cuda.graph_pool_handle() if self.backend == "cuda" else _EmuPool()
        self._segs, self._moe = [], []
        self._cur = None
        self.rt.capture_hook = self._hook
        # le cache vu par les graphs de ce palier : ses L premières positions
        saved = []
        for lay in self.cache.layers:
            if getattr(lay, "is_initialized", False) and hasattr(lay, "keys") and hasattr(lay, "max_cache_len"):
                saved.append((lay, lay.keys, lay.values, lay.max_cache_len))
                lay.keys, lay.values, lay.max_cache_len = lay.keys[:, :, :L], lay.values[:, :, :L], L
        stream = None
        try:
            if self.backend == "cuda":
                torch.cuda.synchronize()
                stream = self.stream
                stream.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(stream) if stream is not None else _null():
                self._cur = self._new_graph()
                self._cur.begin()
                h = self.base(inputs_embeds=self.emb_buf, past_key_values=self.cache, use_cache=True).last_hidden_state
                self._cur.end()
                self._segs.append(self._cur)
                self._cur = None
            if stream is not None:
                torch.cuda.current_stream(self.dev).wait_stream(stream)
                torch.cuda.synchronize()
            st = {"segs": self._segs, "moe": self._moe, "final": h}
            self.sets[L] = st
            self.captures += 1
            return st
        except Exception as e:  # capture impossible : on continue sans graphs
            if self._cur is not None:
                self._cur.abort()
            self.failed = f"{e.__class__.__name__}: {e}"
            if self.backend == "cuda":
                torch.cuda.synchronize()
            return None
        finally:
            self.rt.capture_hook = None
            for lay, k, v, m in saved:
                lay.keys, lay.values, lay.max_cache_len = k, v, m

    # ------------------------------------------------------------------
    # A5 : le token entier en un graph (fullgraph.py)
    # ------------------------------------------------------------------
    def _states_ready(self) -> bool:
        """
        Tout ce que le cache modifie à chaque token, hors des positions écrites (K/V, que le
        chemin normal réécrit aux mêmes positions) : compteur de position de chaque couche,
        états de l'attention linéaire (Qwen3-Next). Sauvegardé au début du graph complet,
        restauré si le token doit être refait. False si un état n'existe pas encore.
        """
        states = []
        for lay in self.cache.layers:
            found = False
            for name, v in vars(lay).items():
                if name in ("keys", "values"):
                    continue
                vals = v.values() if isinstance(v, dict) else (v,)
                for u in vals:
                    if isinstance(u, torch.Tensor):
                        states.append(u)
                        found = True
            if not found and not hasattr(lay, "keys"):
                return False
        self._states = states
        return True

    def _full_hook(self, mod, x, idx, w):
        if idx.shape[-1] != self._fm.k:
            raise RuntimeError(f"top-k {idx.shape[-1]} différent de celui du routeur ({self._fm.k})")
        self._fm.used = True
        return self._fm.layer(mod.layer, x, idx, w, self._cur.call)

    @torch.no_grad()
    def _capture_full(self, L: int, sig) -> dict | None:
        self._pool = torch.cuda.graph_pool_handle() if self.backend == "cuda" else _EmuPool()
        self._cur = None
        # tampons et constantes créés AVANT la capture (une copie hôte -> GPU est interdite pendant)
        gate = next(iter(self.rt.gates.values()), None)
        k = int(getattr(gate, "top_k", 0) or 0)
        if not k:
            self.full_failed = "top-k du routeur inconnu"
            return None
        self._fm = self.rt.new_full_moe(k, self.emb_buf.reshape(1, -1))
        self._fm.used = False
        self.rt.capture_hook = self._full_hook
        saved = []
        for lay in self.cache.layers:
            if getattr(lay, "is_initialized", False) and hasattr(lay, "keys") and hasattr(lay, "max_cache_len"):
                saved.append((lay, lay.keys, lay.values, lay.max_cache_len))
                lay.keys, lay.values, lay.max_cache_len = lay.keys[:, :, :L], lay.values[:, :, :L], L
        backups = [torch.empty_like(v) for v in self._states]
        stream = None
        try:
            if self.backend == "cuda":
                torch.cuda.synchronize()
                stream = self.stream
                stream.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(stream) if stream is not None else _null():
                self._cur = self._new_graph()
                self._cur.begin()
                # remis à zéro À CHAQUE replay (dans le graph) : sans ça, un seul défaut de cache
                # faisait échouer tous les tokens A5 suivants (mesure du 02/10 : 0 % de tokens en A5)
                self._fm.flag.zero_()
                for b, v in zip(backups, self._states):
                    b.copy_(v)
                h = self.base(inputs_embeds=self.emb_buf, past_key_values=self.cache, use_cache=True).last_hidden_state
                self._cur.end()
            if stream is not None:
                torch.cuda.current_stream(self.dev).wait_stream(stream)
                torch.cuda.synchronize()
            if not self._fm.used:
                raise RuntimeError("aucune couche MoE rencontrée")
            fs = {"graph": self._cur, "final": h, "fm": self._fm, "backups": backups, "states": list(self._states),
                  "sig": sig}
            self.full[(L, sig)] = fs
            self.captures += 1
            return fs
        except Exception as e:  # A5 impossible : on garde le chemin par segments
            if self._cur is not None:
                self._cur.abort()
            self.full_failed = f"{e.__class__.__name__}: {e}"
            if self.backend == "cuda":
                torch.cuda.synchronize()
            return None
        finally:
            self.rt.capture_hook = None
            self._cur = None
            for lay, k, v, m in saved:
                lay.keys, lay.values, lay.max_cache_len = k, v, m

    @torch.no_grad()
    def _replay_full(self, fs: dict, tok: torch.Tensor) -> torch.Tensor | None:
        """Le token entier en un replay ; None si un expert manquait (à refaire par le chemin normal)."""
        rt = self.rt
        rt.on_forward(1)
        self.emb_buf.copy_(self.embed(tok))
        rt.device_table().upload()
        fs["graph"].replay()
        fm = fs["fm"]
        vals = fm.readout()
        if vals[0] > 0.5:
            for v, b in zip(fs["states"], fs["backups"]):
                v.copy_(b)
            rt.stats.full_fallbacks += 1
            rt._calm_layers = 0
            return None
        rt.after_full_token(vals, fm.k)
        return self.head(fs["final"])[:, -1]

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # P1 (03/10) : reprise du cache d'un tour de conversation au suivant
    # ------------------------------------------------------------------
    def _layer_kinds(self) -> set[str]:
        from transformers.cache_utils import LinearAttentionLayer, StaticLayer

        kinds = set()
        for lay in self.cache.layers:
            kinds.add("attn" if type(lay) is StaticLayer else "linear" if type(lay) is LinearAttentionLayer else "other")
        return kinds

    def _snapshot(self) -> tuple:
        """États du cache hors K/V (compteurs de position, états de l'attention linéaire), copiés sur CPU."""
        snap = []
        for lay in self.cache.layers:
            d = {}
            for name, v in vars(lay).items():
                if name in ("keys", "values"):
                    continue
                if isinstance(v, torch.Tensor):
                    d[name] = v.detach().to("cpu", copy=True)
                elif isinstance(v, dict) and any(isinstance(u, (torch.Tensor, bool)) for u in v.values()):
                    d[name] = {i: (u.detach().to("cpu", copy=True) if isinstance(u, torch.Tensor) else u)
                               for i, u in v.items()}
            snap.append(d)
        return self.length, snap

    def _restore(self, snap: tuple) -> None:
        """Remet les états sauvegardés EN PLACE (les CUDA graphs gardent les mêmes adresses).
        Un état pas encore créé (cache neuf, P9) est d'abord créé à la bonne forme sur le device."""
        length, layers = snap
        for lay, d in zip(self.cache.layers, layers):
            for name, v in d.items():
                cur = getattr(lay, name)
                if isinstance(v, torch.Tensor):
                    cur.copy_(v)
                    continue
                for i, u in v.items():
                    if not isinstance(u, torch.Tensor):
                        cur[i] = u
                        continue
                    if cur.get(i) is None and name in ("conv_states", "recurrent_states"):
                        t = u.to(self.dev)
                        if name == "conv_states":
                            lay.lazy_initialization(conv_states=t, state_idx=i, conv_kernel_size=t.shape[-1])
                        else:
                            lay.lazy_initialization(recurrent_states=t, state_idx=i)
                        cur = getattr(lay, name)
                    if isinstance(cur.get(i), torch.Tensor):
                        cur[i].copy_(u)
        self.length = length

    # ------------------------------------------------------------------
    # P9 (05/10) : le début fixe d'un prompt (message système + outils d'un agent) calculé une
    # fois, gardé en mémoire et sur disque, rechargé au lieu d'être recalculé
    # ------------------------------------------------------------------
    def _prefix_file(self, toks: list[int]):
        import hashlib

        if not self.prefix_dir:
            return None
        h = hashlib.sha1((self.prefix_key + "|" + ",".join(map(str, toks))).encode()).hexdigest()[:20]
        return Path(self.prefix_dir) / f"prefix_{h}.pt"

    def _capture_state(self, toks: list[int]) -> dict:
        """Le cache contient exactement `toks` : copie sur CPU de ses K/V et de ses états."""
        from transformers.cache_utils import StaticLayer

        n = len(toks)
        kv = []
        for lay in self.cache.layers:
            if type(lay) is StaticLayer and getattr(lay, "is_initialized", False):
                kv.append((lay.keys[:, :, :n].to("cpu", copy=True), lay.values[:, :, :n].to("cpu", copy=True)))
            else:
                kv.append(None)
        return {"tokens": list(toks), "snap": self._snapshot(), "kv": kv}

    def export_state(self) -> dict | None:
        """
        Toute la conversation en cours (K/V, états, tokens), copiée sur CPU : à remettre avec
        import_state après avoir utilisé le décodeur pour autre chose (l'autopilot mesure entre
        deux tours de réflexion). None si la conversation n'est pas suivie.
        """
        if self.tokens is None or self.cache is None or len(self.tokens) != self.length \
                or not self._layer_kinds() <= {"attn", "linear"}:
            return None
        return self._capture_state(self.tokens)

    def import_state(self, entry: dict | None) -> bool:
        if entry is None or self.cache is None:
            return False
        self._load_prefix(entry)
        self._snap = entry["snap"]
        return True

    def _capture_prefix(self, toks: list[int]) -> None:
        """Le cache contient exactement `toks` : on garde K/V et états (CPU), et sur disque si demandé."""
        self.prefix = self._capture_state(toks)
        path = self._prefix_file(toks)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.prefix, path)
                self._prune_prefixes(path.parent)
            except OSError:
                pass  # disque plein ou protégé : le préfixe reste en mémoire

    MAX_PREFIX_FILES = 24

    @classmethod
    def _prune_prefixes(cls, folder) -> None:
        """Garde les 24 préfixes utilisés le plus récemment (06/10 : la date du jour dans les consignes en crée
        un nouveau chaque jour, par agent)."""
        files = sorted(Path(folder).glob("prefix_*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)
        for f in files[cls.MAX_PREFIX_FILES:]:
            try:
                f.unlink()
            except OSError:
                pass

    def _find_prefix(self, toks: list[int]) -> dict | None:
        if self.prefix is not None and self.prefix["tokens"] == toks:
            return self.prefix
        path = self._prefix_file(toks)
        if path is not None and path.exists():
            try:
                entry = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                return None
            if entry.get("tokens") == toks:
                self.prefix = entry
                try:
                    os.utime(path)  # utilisé : gardé par _prune_prefixes
                except OSError:
                    pass
                return entry
        return None

    def _load_prefix(self, entry: dict) -> int:
        n = len(entry["tokens"])
        for lay, kvp in zip(self.cache.layers, entry["kv"]):
            if kvp is None:
                continue
            k, v = (t.to(self.dev) for t in kvp)
            if not getattr(lay, "is_initialized", False):
                lay.lazy_initialization(k, v)
            lay.keys[:, :, :n].copy_(k)
            lay.values[:, :, :n].copy_(v)
        self._restore(entry["snap"])
        self.length = n
        self.tokens = list(entry["tokens"])
        return n

    def _set_length(self, n: int) -> None:
        """Attention seule : revenir à n tokens = remettre les compteurs (les positions au-delà sont masquées)."""
        for lay in self.cache.layers:
            lay.cumulative_length.fill_(n)
        self.length = n

    def _reuse_start(self, new: list[int]) -> int:
        """Nombre de tokens du prompt déjà dans le cache (0 = tout recalculer), après remise en état."""
        old = self.tokens
        if not old or self.cache is None:
            return 0
        p = 0
        for a, b in zip(new, old):
            if a != b:
                break
            p += 1
        p = min(p, len(new) - 1)  # au moins un token à calculer, pour avoir les logits
        if p <= 0:
            return 0
        kinds = self._layer_kinds()
        if p == len(old) == self.length:
            return p  # suite exacte : le cache contient déjà tout ce qui précède
        if kinds == {"attn"}:
            self._set_length(p)  # attention seule : on coupe où les tokens divergent
            return p
        if kinds <= {"attn", "linear"} and self._snap is not None and self._snap[0] <= p:
            self._restore(self._snap)  # attention linéaire : retour à la fin du prompt précédent
            return self._snap[0]
        return 0

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, reuse: bool = False, prefix_len: int | None = None) -> torch.Tensor:
        """
        reuse (P1) : si le début du prompt est déjà dans le cache (conversation : tout l'historique
        du tour précédent), seuls les nouveaux tokens sont calculés. Exact : mêmes K/V, mêmes états.
        prefix_len (P9) : les prefix_len premiers tokens sont un début fixe (message système et
        outils d'un agent) : calculés une fois puis gardés (mémoire, et disque si prefix_dir).
        """
        if ids.shape[0] != 1:
            raise ValueError("GraphDecoder : un seul prompt à la fois")
        n = ids.shape[1]
        if n >= self.max_len:
            raise ValueError(f"prompt de {n} tokens >= cache statique de {self.max_len} (--graph-max-len)")
        new = ids[0].tolist()
        start = self._reuse_start(new) if reuse else 0
        use_prefix = (reuse and prefix_len is not None and 0 < prefix_len < n
                      and self._layer_kinds() <= {"attn", "linear"})
        self.prefix_hit = False
        if use_prefix and start < prefix_len:
            entry = self._find_prefix(new[:prefix_len])
            if entry is not None:
                start = self._load_prefix(entry)
                self.prefix_hit = True
        if start == 0:
            self.cache.reset()
            self.length = 0
        self.reused = start
        if use_prefix and start < prefix_len:  # début fixe calculé à part, puis gardé
            self.model(input_ids=ids[:, start:prefix_len], past_key_values=self.cache, use_cache=True, logits_to_keep=1)
            self.length = prefix_len
            self._capture_prefix(new[:prefix_len])
            start = prefix_len
        out = self.model(input_ids=ids[:, start:], past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        self.length = n
        self.tokens = new
        self._snap = self._snapshot()
        return out.logits[:, -1]

    @torch.no_grad()
    def _eager_step(self, tok: torch.Tensor) -> torch.Tensor:
        self.rt.on_forward(1)
        self.emb_buf.copy_(self.embed(tok))
        h = self.base(inputs_embeds=self.emb_buf, past_key_values=self.cache, use_cache=True).last_hidden_state
        return self.head(h)[:, -1]

    @torch.no_grad()
    def _replay(self, st: dict, tok: torch.Tensor, tick: bool = True) -> torch.Tensor:
        if tick:
            self.rt.on_forward(1)
        self.emb_buf.copy_(self.embed(tok))
        rt = self.rt
        segs, moe = st["segs"], st["moe"]
        n_moe = len(moe)
        for i, seg in enumerate(segs):
            seg.replay()
            if i < n_moe:
                mod, x, idx, w, out, packed = moe[i]
                rt.run_layer(mod.layer, x, idx, w, mod.act_fn, out=out, packed=packed if self.pack_routing else None)
        return self.head(st["final"])[:, -1]

    @torch.no_grad()
    def step(self, tok: torch.Tensor, tok_id: int | None = None) -> torch.Tensor:
        """tok : (1, 1) entier, de préférence déjà sur le GPU (pas de copie hôte).
        tok_id : sa valeur, si l'appelant la connaît (P1 : suivi des tokens du cache)."""
        if self.length >= self.max_len:
            raise RuntimeError(f"cache statique plein ({self.max_len} tokens) : augmente --graph-max-len")
        if self.tokens is not None:
            if tok_id is None:
                self.tokens = None
            else:
                self.tokens.append(int(tok_id))
        self.length += 1
        if self.failed is not None:
            return self._eager_step(tok)
        if not self.warm:
            # Échauffement sur le stream de capture (initialisations paresseuses de cuBLAS,
            # Triton... faites AVANT la capture, comme le recommande PyTorch).
            self.warm = True
            if self.backend != "cuda":
                return self._eager_step(tok)
            cur = torch.cuda.current_stream(self.dev)
            self.stream.wait_stream(cur)
            with torch.cuda.stream(self.stream):
                logits = self._eager_step(tok)
            cur.wait_stream(self.stream)
            return logits
        key = (self.rt.cfg.prefetch_mass, self.rt.cfg.prefetch_disk_ahead)
        if self._mass != key:  # la prédiction capturée dépend de prefetch_mass et de prefetch_disk_ahead
            self.sets, self.cur, self._mass = {}, None, key
        L = self._bucket(self.length)
        st = self.sets.get(L) or self._capture(L)
        if st is None:
            return self._eager_step(tok)
        self.cur = st
        tick = True
        if self.full_failed is None and self.rt.full_ready() and self._states_ready():  # A5
            sig = self.rt.full_signature()
            fs = self.full.get((L, sig)) or self._capture_full(L, sig)
            if fs is not None:
                logits = self._replay_full(fs, tok)
                if logits is not None:
                    return logits
                tick = False  # même token, refait par le chemin normal
        return self._replay(st, tok, tick=tick)

    @property
    def active(self) -> bool:
        return bool(self.sets) and self.failed is None

    def describe(self) -> str:
        if self.failed:
            return f"CUDA graphs désactivés ({self.failed})"
        if not self.sets:
            return f"CUDA graphs : pas encore capturés (cache statique {self.max_len} tokens)"
        st = next(iter(self.sets.values()))
        d = (f"CUDA graphs : {len(st['segs'])} segments, {len(st['moe'])} couches MoE entre eux, "
             f"paliers capturés {sorted(self.sets)} sur {self.buckets}, cache statique {self.max_len} tokens")
        why = self.rt.full_possible()
        if self.full_failed:
            d += f" ; token entier (A5) : échec ({self.full_failed})"
        elif why is None or self.full:
            s = self.rt.stats
            d += f" ; token entier (A5) : {s.full_tokens} tokens, {s.full_fallbacks} refaits"
        elif why != "désactivé":
            d += f" ; token entier (A5) : pas utilisé ({why})"
        return d

    def release(self) -> None:
        """Libère les graphs et le cache statique (VRAM)."""
        self.sets, self.cur, self.full = {}, None, {}
        self.cache = None
        self.tokens, self._snap = None, None


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
