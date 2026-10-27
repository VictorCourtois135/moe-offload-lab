"""
REAP (08/10) : score d'importance de chaque expert routé, pour l'élagage.

REAP (Router-weighted Expert Activation Pruning, Cerebras, arXiv 2510.13999) : un expert
compte d'autant plus qu'il est choisi avec un poids fort ET que sa sortie est grande.
Pour la couche l et l'expert j, sur les tokens x routés vers j :

    S[l, j] = moyenne des  g_j(x) * ||f_j(x)||_2

g_j(x) : poids du routeur après top-k et normalisation (ce que reçoit le module experts) ;
f_j(x) : sortie de l'expert AVANT pondération. Un expert jamais choisi a un score de 0.
L'expert partagé de Qwen3-Next (toujours actif, hors du module experts) n'est jamais compté.

Calcul : le crochet `expert_hook` du runtime reçoit chaque sortie d'expert pendant des passes
de prefill normales. Coût : chaque passe lit presque tout le store (512 tokens choisissent
déjà ~tous les experts de chaque couche) ; on passe donc plusieurs échantillons par passe
(`batch`) pour qu'une lecture serve à batch x 512 tokens.

Accumulé par passe dans des tampons temporaires, ajouté aux totaux seulement si la passe va
au bout : une passe interrompue (mémoire GPU pleine, Ctrl+C) ne compte jamais à moitié.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from pathlib import Path

import torch

FIELDS = ("sum_gnorm", "sum_norm", "sum_g")  # g * ||f||, ||f|| (EAN), g (masse du routeur)


def ids_hash(ids: torch.Tensor) -> str:
    """Empreinte d'un jeu d'échantillons : la reprise n'a de sens que sur les mêmes."""
    return hashlib.sha1(ids.to(torch.int64).contiguous().numpy().tobytes()).hexdigest()[:16]


class ReapCollector:
    """Accumulateurs REAP d'un store : (couches x experts), en float64 sur CPU."""

    def __init__(self, n_layers: int, n_experts: int, store_id: str = "", data_hash: str = ""):
        self.L, self.E = n_layers, n_experts
        self.store_id, self.data_hash = store_id, data_hash
        self.totals = {f: torch.zeros(n_layers, n_experts, dtype=torch.float64) for f in FIELDS}
        self.count = torch.zeros(n_layers, n_experts, dtype=torch.int64)
        self.samples_done = 0
        self.tokens = 0
        self.seconds = 0.0
        self._tmp: dict = {}
        self._tmp_count = None

    # -- une passe ------------------------------------------------------------
    def begin(self) -> None:
        self._tmp = {}
        self._tmp_count = torch.zeros(self.L, self.E, dtype=torch.int64)

    def _buf(self, device: torch.device) -> dict:
        key = str(device)
        if key not in self._tmp:
            self._tmp[key] = {f: torch.zeros(self.L, self.E, dtype=torch.float64, device=device) for f in FIELDS}
        return self._tmp[key]

    def hook(self, layer: int, expert: int, w: torch.Tensor, y: torch.Tensor) -> None:
        """Appelé par le runtime : w (n,) poids du routeur, y (n, H) sorties de l'expert."""
        n = int(y.shape[0])
        if n == 0:
            return
        norm = y.float().norm(dim=-1)
        w = w.reshape(-1).to(norm.device, torch.float32)
        b = self._buf(norm.device)
        b["sum_gnorm"][layer, expert] += (w * norm).sum().double()
        b["sum_norm"][layer, expert] += norm.sum().double()
        b["sum_g"][layer, expert] += w.sum().double()
        self._tmp_count[layer, expert] += n

    def commit(self, samples: int, tokens: int) -> None:
        for bufs in self._tmp.values():
            for f in FIELDS:
                self.totals[f] += bufs[f].cpu()
        self.count += self._tmp_count
        self.samples_done += samples
        self.tokens += tokens
        self.begin()

    def discard(self) -> None:
        self.begin()

    # -- résultats ----------------------------------------------------------
    def scores(self) -> torch.Tensor:
        """S[l, j] = moyenne de g * ||f|| ; 0 pour un expert jamais choisi."""
        c = self.count.double()
        return torch.where(c > 0, self.totals["sum_gnorm"] / c.clamp(min=1), torch.zeros_like(c))

    def ean(self) -> torch.Tensor:
        """Variante sans le routeur (moyenne de ||f||), pour comparaison."""
        c = self.count.double()
        return torch.where(c > 0, self.totals["sum_norm"] / c.clamp(min=1), torch.zeros_like(c))

    def frequency(self) -> torch.Tensor:
        """Part des choix (token, expert) de la couche qui vont à chaque expert."""
        c = self.count.double()
        return c / c.sum(dim=1, keepdim=True).clamp(min=1)

    # -- disque ---------------------------------------------------------------
    def state(self) -> dict:
        return {"version": 1, "store_id": self.store_id, "data_hash": self.data_hash, "layers": self.L,
                "experts": self.E, "samples_done": self.samples_done, "tokens": self.tokens,
                "seconds": self.seconds, "count": self.count, **{f: self.totals[f] for f in FIELDS}}

    def save(self, path) -> None:
        """Écriture atomique : un fichier à moitié écrit ne remplace jamais le précédent."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state(), tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path) -> "ReapCollector":
        st = torch.load(path, map_location="cpu", weights_only=True)
        c = cls(st["layers"], st["experts"], st.get("store_id", ""), st.get("data_hash", ""))
        c.count = st["count"]
        for f in FIELDS:
            c.totals[f] = st[f]
        c.samples_done, c.tokens, c.seconds = st["samples_done"], st["tokens"], st.get("seconds", 0.0)
        return c


# ----------------------------------------------------------------------------
# Passes de prefill par lots (servent aussi à la perplexité, prune_eval.py)
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def bulk_mode(rt):
    """
    Réglages des passes en masse : q* coupé (tous les experts manquants vont sur le GPU au lieu
    d'être calculés sur CPU en float32 : plus rapide pour des milliers de tokens, et le même
    calcul pour tous les échantillons), aucun saut d'expert (P5, t2). Remis comme avant à la fin.
    """
    cfg, pol = rt.cfg, rt.policy
    saved = (cfg.use_qstar, cfg.prefill_skip, pol.t2)
    cfg.use_qstar = cfg.use_qstar and rt.fast is None  # sans VRAM pour les experts, q* n'a pas d'objet
    cfg.prefill_skip, pol.t2 = None, None
    try:
        yield
    finally:
        cfg.use_qstar, cfg.prefill_skip, pol.t2 = saved


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


@torch.no_grad()
def hidden_states(model, ids: torch.Tensor) -> torch.Tensor:
    """Dernier état caché (B, T, H) sans la tête de sortie : pas de logits B x T x 150k inutiles."""
    rt = model.moelab_runtime
    rt.on_forward(int(ids.numel()))
    base = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    return base(input_ids=ids.to(model.moelab_device), use_cache=False).last_hidden_state


def run_batches(ids: torch.Tensor, batch: int, fn, log=print, start: int = 0, on_batch=None):
    """
    Appelle fn(ids[i:j]) par lots ; si la mémoire GPU manque, le lot est coupé en deux et
    la taille réduite pour la suite. on_batch(j, taille) après chaque lot réussi.
    Retourne la taille de lot finalement utilisée.
    """
    i, n = start, ids.shape[0]
    while i < n:
        b = min(batch, n - i)
        try:
            fn(ids[i:i + b])
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if not _is_oom(e) or b == 1:
                raise
            batch = max(1, batch // 2)
            if b <= batch:
                batch = max(1, b // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(f"  mémoire GPU pleine : lots de {batch} échantillon(s)")
            continue
        i += b
        if on_batch is not None:
            on_batch(i, b)
    return batch


def calibrate(model, ids: torch.Tensor, out_path, batch: int = 8, save_every: int = 4, log=print,
              resume: bool = True) -> ReapCollector:
    """
    Calibration REAP sur ids (N x T), reprise automatique depuis out_path si le fichier
    correspond au même store et aux mêmes échantillons. Sauvegarde tous les `save_every` lots.
    """
    rt = model.moelab_runtime
    store = rt.store
    h = ids_hash(ids)
    out_path = Path(out_path)
    col = None
    if resume and out_path.exists():
        try:
            old = ReapCollector.load(out_path)
        except Exception as e:  # fichier illisible : on repart de zéro
            log(f"  {out_path.name} illisible ({e.__class__.__name__}) : calibration reprise de zéro")
            old = None
        if old is not None and old.store_id == store.store_id and old.data_hash == h:
            col = old
            if col.samples_done:
                log(f"  reprise : {col.samples_done}/{ids.shape[0]} échantillons déjà comptés")
        elif old is not None:
            log(f"  {out_path.name} vient d'un autre store ou d'autres échantillons : ignoré")
    if col is None:
        col = ReapCollector(len(store.layers), store.num_experts, store.store_id, h)
    n = ids.shape[0]
    if col.samples_done >= n:
        return col

    state = {"batches": 0, "t": time.time()}
    t_run = time.time()
    done_at_start = col.samples_done

    def one(chunk):
        col.begin()
        try:
            hidden_states(model, chunk)
        except BaseException:
            col.discard()
            raise
        col.commit(chunk.shape[0], int(chunk.numel()))

    def after(j, b):
        state["batches"] += 1
        col.seconds += time.time() - state["t"]
        state["t"] = time.time()
        if state["batches"] % save_every == 0 or j >= n:
            col.save(out_path)
        el = time.time() - t_run
        per = el / max(j - done_at_start, 1)
        log(f"  calibration REAP : {j}/{n} échantillons — {el / 60:.1f} min, reste ~{per * (n - j) / 60:.0f} min")

    rt.expert_hook = col.hook
    try:
        with bulk_mode(rt):
            run_batches(ids, batch, one, log=log, start=col.samples_done, on_batch=after)
    finally:
        rt.expert_hook = None
        if col.samples_done > done_at_start:
            col.save(out_path)
    return col


# ----------------------------------------------------------------------------
# Lecture des scores : ce que coûterait chaque taux d'élagage
# ----------------------------------------------------------------------------
def prune_mask(scores: torch.Tensor, ratio: float, top_k: int, layers=None) -> torch.Tensor:
    """
    Masque booléen (L, E) des experts élagués : dans chaque couche (de `layers`, toutes sinon),
    les round(ratio x E) plus bas scores, en gardant toujours au moins top_k experts.
    """
    L, E = scores.shape
    n = min(int(round(ratio * E)), E - top_k)
    mask = torch.zeros(L, E, dtype=torch.bool)
    if n <= 0:
        return mask
    for l in range(L) if layers is None else layers:
        order = torch.argsort(scores[l], stable=True)  # à score égal, l'indice le plus bas d'abord : déterministe
        mask[l, order[:n]] = True
    return mask


def summary(col: ReapCollector, ratios=(0.25, 0.4, 0.5), top_k: int | None = None, record_bytes: int | None = None,
            layer_names: list[str] | None = None) -> dict:
    """Résumé lisible : experts jamais choisis, part du score et des choix perdue à chaque taux."""
    S, cnt = col.scores(), col.count.double()
    L, E = S.shape
    per_layer = []
    for l in range(L):
        order = torch.argsort(S[l], stable=True)
        row = {"couche": l, "jamais_choisis": int((col.count[l] == 0).sum()),
               "score_max": float(S[l].max()), "score_median": float(S[l].median())}
        if layer_names:
            row["module"] = layer_names[l]
        tot_s, tot_c = float(S[l].sum()) or 1.0, float(cnt[l].sum()) or 1.0
        for r in ratios:
            k = int(round(r * E))
            row[f"score_perdu_{int(r * 100)}"] = float(S[l][order[:k]].sum()) / tot_s
            row[f"choix_perdus_{int(r * 100)}"] = float(cnt[l][order[:k]].sum()) / tot_c
        per_layer.append(row)
    out = {"store_id": col.store_id, "echantillons": col.samples_done, "tokens": col.tokens,
           "minutes": round(col.seconds / 60, 1), "couches": L, "experts": E,
           "jamais_choisis": int((col.count == 0).sum()), "par_couche": per_layer, "taux": {}}
    for r in ratios:
        key = int(r * 100)
        k = int(round(r * E))
        if top_k is not None:
            k = min(k, E - top_k)
        t = {"experts_retires_par_couche": k,
             "choix_perdus_moyen": sum(p[f"choix_perdus_{key}"] for p in per_layer) / L,
             "choix_perdus_max": max(p[f"choix_perdus_{key}"] for p in per_layer)}
        if record_bytes:
            t["store_gio"] = round(L * (E - k) * record_bytes / 2**30, 2)
        out["taux"][str(r)] = t
    if record_bytes:
        out["store_gio_complet"] = round(L * E * record_bytes / 2**30, 2)
    return out


def write_summary(col: ReapCollector, path, **kw) -> dict:
    s = summary(col, **kw)
    Path(path).write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    return s


# ----------------------------------------------------------------------------
# Étape 3 : élagage virtuel par masque (aucune écriture dans le store)
# ----------------------------------------------------------------------------
def _masked_forward(self, hidden_states):
    """Même calcul que les routeurs TopK de transformers (OLMoE, Qwen3-MoE, Qwen3-Next), avec -inf sur les
    logits des experts élagués AVANT le softmax et le top-k : le routeur choisit parmi les experts restants,
    la normalisation du top-k reste la même. Le prefetch et les CUDA graphs appellent ce même module."""
    import torch.nn.functional as F

    h = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(h, self.weight) + self.moelab_bias
    probs = torch.softmax(logits, dtype=torch.float, dim=-1)
    top, idx = torch.topk(probs, self.top_k, dim=-1)
    if getattr(self, "norm_topk_prob", False):
        top = top / top.sum(dim=-1, keepdim=True)
    return logits, top.to(logits.dtype), idx


def apply_prune_mask(model, mask: torch.Tensor | None, log=print) -> int:
    """
    Masque (couches, experts) -> routeurs modifiés en place ; None retire le masque. Les caches sont vidés puis
    remplis sans les experts masqués (ils ne seront jamais chargés : ni demandés, ni prédits, ni préchargés).
    À appeler juste après le chargement, avant le premier message (les CUDA graphs capturent le routeur).
    """
    import types

    rt = model.moelab_runtime
    rt.pruned = set()
    for li, gate in rt.gates.items():
        if not all(hasattr(gate, a) for a in ("weight", "top_k", "hidden_dim")):
            raise NotImplementedError(f"routeur de la couche {li} non géré : {type(gate).__name__}")
        if mask is None:
            if "forward" in gate.__dict__:
                del gate.forward
            continue
        row = mask[li]
        keep = int((~row).sum())
        if keep < gate.top_k:
            raise ValueError(f"couche {li} : {keep} experts gardés, moins que top_k={gate.top_k}")
        bias = torch.zeros(row.shape[0], dtype=gate.weight.dtype, device=gate.weight.device)
        bias[row.to(bias.device)] = float("-inf")
        gate.moelab_bias = bias
        gate.forward = types.MethodType(_masked_forward, gate)
        rt.pruned |= {(li, int(e)) for e in row.nonzero().flatten().tolist()}
    rt.reset_caches()
    rt.warm_start(preload_lo=rt.preload_lo, log=log)
    return len(rt.pruned)


def parse_layers(spec: str | None, n_layers: int) -> list[int] | None:
    """« 10-20 », « 0,5,7 » ou None (toutes)."""
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        out |= set(range(int(a), int(b) + 1)) if b else {int(a)}
    bad = [x for x in out if not 0 <= x < n_layers]
    if bad:
        raise ValueError(f"couches hors limites : {bad} (0 à {n_layers - 1})")
    return sorted(out)
