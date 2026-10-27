"""
moelab autopilot (05/10) : le modèle local optimise son propre moteur (doc 19).

Principe : le code garantit, le modèle propose.

  - Le code mesure (`Bench`), applique et défait les réglages, refuse ce qui dégrade la
    qualité (KL et accord top-1 contre une référence), ne croit à un gain qu'au-delà du
    bruit et après une 2e mesure, tient le journal et écrit le profil final.
  - Le modèle choisit les expériences et explique, avec 5 outils : machine, espace,
    mesurer, journal, terminer.

Une seule carte : le cerveau et le sujet sont le même moteur. Chaque expérience applique
ses réglages, mesure (caches vidés), puis remet la meilleure configuration connue avant
que le modèle ne reprenne la parole.

Score d'une configuration : la durée d'une réponse type, le premier token plus 200 tokens
de génération (secondes, plus petit = mieux). Il réunit les deux chiffres qu'on a
optimisés à la main : tok/s et temps avant le premier token.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from . import options
from .agent import Agent, Tool, moelab_generate_fn, tool
from .cache import UsageLog
from .generate import make_decoder

ANSWER_TOKENS = 200  # longueur de la « réponse type » du score
PROFILE_FILE = "autopilot_profile.json"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------
# Mesure
# ----------------------------------------------------------------------------
class Bench:
    """
    Une mesure = caches vidés et préchargés comme au démarrage, un prompt d'échauffement,
    puis pour chaque prompt : lecture du prompt chronométrée, et `gen` tokens imposés
    (ceux de la référence) chronométrés, dont on compare les probabilités à la référence.
    """

    def __init__(self, model, tokenizer, prompts: list[str], gen: int = 24, warm: str | None = None):
        from .bench import chat_ids

        self.model, self.tok, self.rt = model, tokenizer, model.moelab_runtime
        dev = model.moelab_device
        self.ids = [chat_ids(tokenizer, p, dev) for p in prompts]
        self.warm_ids = chat_ids(tokenizer, warm or "Bonjour, présente-toi en une phrase.", dev)
        self.gen = gen
        self.ref: list[dict] | None = None

    def _fresh(self):
        rt = self.rt
        rt.reset_caches()
        rt.usage = UsageLog(None, rt.store.store_id)
        rt.warm_start(preload_lo=rt.preload_lo, log=lambda *a: None)

    @torch.no_grad()
    def _one(self, dec, ids, ref: dict | None) -> dict:
        _sync()
        t0 = time.perf_counter()
        logits = dec.prefill(ids)
        _sync()
        ttft = time.perf_counter() - t0
        logps = [F.log_softmax(logits.float(), -1).cpu()]
        toks = ref["tokens"] if ref else []
        nxt = torch.tensor([[toks[0]]], device=ids.device) if ref else logits.argmax(-1, keepdim=True).view(1, 1)
        out = [int(nxt)]
        _sync()
        t0 = time.perf_counter()
        for i in range(self.gen - 1):
            lg = dec.step(nxt)
            logps.append(F.log_softmax(lg.float(), -1).cpu())
            nxt = torch.tensor([[toks[i + 1]]], device=ids.device) if ref else lg.argmax(-1, keepdim=True).view(1, 1)
            out.append(int(nxt))
        _sync()
        dt = time.perf_counter() - t0
        return {"ttft": ttft, "decode_s": dt, "tokens": out, "logp": torch.cat(logps)}

    def run(self) -> dict:
        rt = self.rt
        self._fresh()
        dec = make_decoder(self.model)
        self._one(dec, self.warm_ids, None)  # échauffement : graphs, noyaux, caches
        rt.reset_stats()
        rows = [self._one(dec, ids, self.ref[i] if self.ref else None) for i, ids in enumerate(self.ids)]
        st = rt.stats.as_dict()
        ttft = sum(r["ttft"] for r in rows) / len(rows)
        tok_s = len(rows) * (self.gen - 1) / max(sum(r["decode_s"] for r in rows), 1e-9)
        res = {"premier_token_s": round(ttft, 2), "tok_s": round(tok_s, 2),
               "reponse_type_s": round(ttft + ANSWER_TOKENS / tok_s, 1),
               "hit_vram_pourcent": round(100 * st["fast_hit_rate"], 1),
               "disque_par_token": round(st["disk_per_token"], 1)}
        if self.ref is None:
            self.ref = [{"tokens": r["tokens"], "logp": r["logp"]} for r in rows]
            res["kl"], res["top1_pourcent"] = 0.0, 100.0
        else:
            kls, tops = [], []
            for r, ref in zip(rows, self.ref):
                rp, lp = ref["logp"], r["logp"]
                kls.append(float((rp.exp() * (rp - lp)).sum(-1).mean()))
                tops.append(float((rp.argmax(-1) == lp.argmax(-1)).float().mean()))
            res["kl"] = round(sum(kls) / len(kls), 4)
            res["top1_pourcent"] = round(100 * sum(tops) / len(tops), 1)
        return res


# ----------------------------------------------------------------------------
# Espace de recherche
# ----------------------------------------------------------------------------
def search_space(rt) -> dict:
    """Réglages que l'autopilot peut changer à chaud, avec leurs valeurs permises."""
    sp = {
        "read_parts": {"valeurs": [1, 2, 4, 8]},
        "prefill_grouped": {"valeurs": [0, 64, 128, 192, 256, 512]},
        "cache_eviction": {"valeurs": ["lrfu", "stale"]},
        "prefetch": {"valeurs": [True, False]},
        "prefetch_depth": {"valeurs": [1, 2]},
        "prefetch_mass": {"valeurs": [None, 0.8, 0.9]},
        "prefetch_disk_ahead": {"valeurs": [0, 1, 2]},
        "full_graph": {"valeurs": [True, False]},
        "prefill_skip": {"valeurs": [None, 0.02, 0.05, 0.1]},
        "ram_gb": {"min": 1.0, "max": round(_ram_max(rt), 2)},
    }
    if rt.fast is not None and rt.fast.device.type == "cuda":
        sp["vram_gb"] = {"min": 0.5, "max": round(_vram_max(rt), 2)}
    if len(rt.store.precisions) > 1:
        sp["policy"] = {"valeurs": ["lo", "hobbit", "hi"]}
        sp["t1"] = {"min": 0.3, "max": 0.95}
    return sp


def _vram_max(rt) -> float:
    """
    Budget VRAM au plus = VRAM totale - ce qui n'est pas à PyTorch (contexte CUDA, autres programmes)
    - ce que PyTorch utilise hors experts (partie dense, cache KV, graphs) - les activations (0,55 Gio).

    Deux erreurs corrigées (épreuve du 05/10) : la VRAM gardée par l'allocateur de PyTorch après une
    baisse de budget était oubliée (max 1,74 Gio pour un budget de 1,84) ; puis, comptée en plus du
    budget alors que les caches venaient d'être vidés, elle l'était deux fois (3,0 Gio accepté sur
    une carte de 6 Go : sous Windows, le pilote déborde alors en silence dans la RAM, « mémoire GPU
    partagée », et les mesures ne veulent plus dire grand-chose).
    """
    if not torch.cuda.is_available():
        return 0.5
    from .plan import ACTIVATIONS_RESERVE

    free, total = torch.cuda.mem_get_info()
    reserved, allocated = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
    not_torch = (total - free) - reserved
    others = allocated - (rt.fast.used if rt.fast is not None else 0)
    cap = (total - not_torch - others - ACTIVATIONS_RESERVE) / 2**30
    # jamais sous le budget choisi au chargement (05/10 : plafond 1,96 Gio pour un budget de 2,11 qui
    # tourne très bien, d'où un modèle perplexe), ni sous le budget en cours
    floor = max(getattr(rt, "load_fast_budget", 0), rt.fast.budget if rt.fast is not None else 0) / 2**30
    return max(0.5, cap, floor)


def _ram_max(rt) -> float:
    """RAM au plus : 40 % de la RAM totale (au-delà, Windows pagine la mémoire épinglée)."""
    import psutil

    return max(1.0, 0.40 * psutil.virtual_memory().total / 2**30)


def check_change(space: dict, key: str, value):
    if key not in space:
        raise ValueError(f"« {key} » n'est pas dans l'espace de recherche ({', '.join(space)})")
    sp = space[key]
    if "valeurs" in sp:
        if isinstance(value, str) and value.lower() in ("none", "null", "aucun"):
            value = None
        if value not in sp["valeurs"]:
            raise ValueError(f"{key} = {value!r} non permis ; valeurs : {sp['valeurs']}")
        return value
    v = float(value)
    if not sp["min"] - 1e-9 <= v <= sp["max"] + 1e-9:
        raise ValueError(f"{key} = {v} hors de [{sp['min']}, {sp['max']}]")
    return round(v, 2)


# ----------------------------------------------------------------------------
# La boucle
# ----------------------------------------------------------------------------
@dataclass
class Trial:
    n: int
    changes: dict
    config: dict
    result: dict
    verdict: str
    seconds: float


@dataclass
class Autopilot:
    """
    model, tokenizer : le moteur chargé (cerveau et sujet). bench : un Bench (ou un objet
    avec run() -> dict, pour les tests). Le reste : les garde-fous.
    """
    model: object
    tokenizer: object
    bench: object
    max_trials: int = 12
    minutes: float = 60.0
    noise: float = 0.05          # gain minimal cru (5 %), confirmé par une 2e mesure
    kl_max: float = 0.005        # au-delà : qualité dégradée (bruit des variantes exactes : 0,0001 à 0,003)
    top1_min: float = 90.0
    blind: bool = False          # épreuve : l'effet mesuré des réglages n'est pas montré au modèle
    log: object = print
    trials: list = field(default_factory=list)
    best: dict = field(default_factory=dict)
    best_score: float = float("inf")
    best_res: dict = field(default_factory=dict)
    start: dict = field(default_factory=dict)
    start_result: dict = field(default_factory=dict)
    report: str = ""
    done: bool = False

    def __post_init__(self):
        self.rt = self.model.moelab_runtime
        self.space = search_space(self.rt)
        self.t0 = time.time()

    # -- configuration --------------------------------------------------
    def current_config(self) -> dict:
        cur = options.current(self.model)
        return {k: cur.get(k) for k in self.space}

    def apply_config(self, cfg: dict) -> None:
        cur = self.current_config()
        for k, v in cfg.items():
            if cur.get(k) != v:
                options.apply(self.model, k, v)

    # -- mesures ----------------------------------------------------------
    def baseline(self) -> dict:
        self.start = self.current_config()
        res = self.bench.run()
        self.best, self.best_score, self.start_result = dict(self.start), res["reponse_type_s"], res
        self.best_res = {k: res[k] for k in ("reponse_type_s", "premier_token_s", "tok_s")}
        self.trials.append(Trial(0, {}, dict(self.start), res, "départ", 0.0))
        self.log(f"  départ : réponse type {res['reponse_type_s']} s ({res['tok_s']} tok/s, 1er token "
                 f"{res['premier_token_s']} s)")
        return res

    def _deltas(self, res: dict, ref: dict | None = None) -> dict:
        """Écarts à la meilleure configuration (ou à `ref`), en % (positif = mieux), montrés au modèle."""
        b = ref or self.best_res
        return {"ecart_reponse_type_pourcent": round(100 * (1 - res["reponse_type_s"] / b["reponse_type_s"]), 1),
                "ecart_premier_token_pourcent": round(100 * (1 - res["premier_token_s"] / b["premier_token_s"]), 1),
                "ecart_tok_s_pourcent": round(100 * (res["tok_s"] / b["tok_s"] - 1), 1)}

    def _better(self, res: dict, confirm: bool = False, ref: dict | None = None) -> str | None:
        """
        Mieux que la meilleure configuration ? (1er essai de l'épreuve, 05/10 : P6 a fait passer le
        1er token de 6,0 à 3,9 s, mais la réponse type n'a gagné que 4,7 % et le changement a été
        refusé. D'où deux façons de gagner.)
          - la réponse type gagne au moins `noise` (5 %) ;
          - OU un des deux chiffres gagne nettement (1er token >= 10 %, tok/s >= noise) sans que
            l'autre perde plus que le bruit.
        À la 2e mesure (confirm), il suffit de la moitié de ces seuils.
        """
        d = self._deltas(res, ref)
        k = 0.5 if confirm else 1.0
        g, gf, gt = d["ecart_reponse_type_pourcent"] / 100, d["ecart_premier_token_pourcent"] / 100, d["ecart_tok_s_pourcent"] / 100
        if g >= k * self.noise:
            return f"réponse type {100 * g:+.1f} %"
        if gf >= k * 2 * self.noise and gt >= -self.noise:
            return f"1er token {100 * gf:+.1f} %, tok/s {100 * gt:+.1f} %"
        if gt >= k * self.noise and gf >= -2 * self.noise:
            return f"tok/s {100 * gt:+.1f} %, 1er token {100 * gf:+.1f} %"
        return None

    def _quality_ok(self, res: dict) -> str | None:
        if res["kl"] > self.kl_max:
            return f"qualité dégradée : KL {res['kl']} > {self.kl_max}"
        if res["top1_pourcent"] < self.top1_min:
            return f"qualité dégradée : top-1 {res['top1_pourcent']} % < {self.top1_min} %"
        return None

    def experiment(self, changes: dict) -> dict:
        if self.done:
            return {"erreur": "l'autopilot est terminé"}
        if len(self.trials) - 1 >= self.max_trials:
            return {"erreur": f"budget de {self.max_trials} essais épuisé : appelle terminer"}
        if (time.time() - self.t0) / 60 > self.minutes:
            return {"erreur": f"budget de {self.minutes:.0f} min épuisé : appelle terminer"}
        if not changes:
            return {"erreur": "aucun changement demandé"}
        try:
            changes = {k: check_change(self.space, k, v) for k, v in changes.items()}
        except (ValueError, TypeError) as e:
            return {"erreur": str(e)}
        cfg = {**self.best, **changes}
        if cfg == self.best:
            return {"erreur": "aucun changement : ces valeurs sont déjà celles de la meilleure configuration"}
        if any(t.config == cfg for t in self.trials):
            return {"erreur": "configuration déjà mesurée (voir journal)"}
        n = len(self.trials)
        t0 = time.time()
        self.log(f"  essai {n} : {changes} ...")
        dec = getattr(self.model, "moelab_graph_decoder", None)
        saved = dec.export_state() if dec is not None and hasattr(dec, "export_state") else None  # la conversation du cerveau
        try:
            self.apply_config(cfg)
            res = self.bench.run()
            res.update(self._deltas(res))
            verdict = self._quality_ok(res)
            if verdict is None:
                d = res["ecart_reponse_type_pourcent"]
                if self._better(res) is not None or abs(d) >= 100 * self.noise:
                    # Résultat net, dans un sens ou dans l'autre : la meilleure config est remesurée tout
                    # de suite (2e essai de l'épreuve, 05/10 : d'un essai à l'autre, la même config a varié
                    # de 4,9 à 7,7 tok/s ; comparer à une mesure d'il y a 10 min ne veut rien dire).
                    self.apply_config(self.best)
                    ref = self.bench.run()
                    drift = 1 - ref["reponse_type_s"] / self.best_res["reponse_type_s"]
                    if abs(drift) >= self.noise:
                        res["derive_pourcent"] = round(100 * drift, 1)
                        self.log(f"    dérive : la meilleure config mesure maintenant {ref['reponse_type_s']} s "
                                 f"(au lieu de {self.best_res['reponse_type_s']:.1f})")
                    ref = {k: ref[k] for k in ("reponse_type_s", "premier_token_s", "tok_s")}
                    self.best_res = ref  # la référence suit les conditions du moment
                    self.best_score = ref["reponse_type_s"]
                    res.update(self._deltas(res, ref))
                    self.apply_config(cfg)
                why = self._better(res)
                if why is None:
                    verdict = (f"pas mieux (réponse type {res['ecart_reponse_type_pourcent']:+.1f} %, 1er token "
                               f"{res['ecart_premier_token_pourcent']:+.1f} %, tok/s {res['ecart_tok_s_pourcent']:+.1f} %"
                               + (", meilleure config remesurée à côté" if "derive_pourcent" in res or abs(d) >= 100 * self.noise
                                  else "") + ")")
                else:
                    res2 = self.bench.run()  # 2e mesure : un gain doit se répéter
                    why2 = self._better(res2, confirm=True)
                    if why2 is not None and self._quality_ok(res2) is None:
                        avg = {k: (res[k] + res2[k]) / 2 for k in ("reponse_type_s", "premier_token_s", "tok_s")}
                        verdict = f"GARDÉ : {why} (confirmé : {why2})"
                        res["confirmation_reponse_type_s"] = res2["reponse_type_s"]
                        self.best, self.best_score, self.best_res = dict(cfg), avg["reponse_type_s"], avg
                    else:
                        verdict = f"non confirmé ({why}, puis rien de net) : bruit"
        except Exception as e:  # mémoire GPU épuisée, réglage impossible...
            res, verdict = {}, f"échec : {e.__class__.__name__}: {str(e)[:200]}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            try:
                self.apply_config(self.best)  # le cerveau reprend avec la meilleure config connue
            except Exception as e:
                self.log(f"  (remise de la meilleure config incomplète : {e})")
            dec = getattr(self.model, "moelab_graph_decoder", None)
            if saved is not None and dec is not None:
                dec.import_state(saved)  # sa conversation est reprise telle quelle : seul le résultat sera calculé
        tr = Trial(n, changes, cfg, res, verdict, time.time() - t0)
        self.trials.append(tr)
        self.log(f"    -> {verdict} ; {res}")
        return {"essai": n, "changements": changes, **{k: v for k, v in res.items() if k != "confirmation_reponse_type_s"},
                "verdict": verdict, "meilleur": {k: round(v, 2) for k, v in self.best_res.items()},
                "essais_restants": self.max_trials - n}

    def journal_rows(self) -> list[dict]:
        return [{"essai": t.n, "changements": t.changes or "départ",
                 "reponse_type_s": t.result.get("reponse_type_s"), "tok_s": t.result.get("tok_s"),
                 "premier_token_s": t.result.get("premier_token_s"), "kl": t.result.get("kl"), "verdict": t.verdict}
                for t in self.trials]

    # -- outils du modèle --------------------------------------------------
    def tools(self) -> list[Tool]:
        ap = self

        @tool
        def machine() -> dict:
            """La machine et le modèle : GPU, VRAM, RAM, débit disque mesuré, taille des experts."""
            import platform

            import psutil

            from .hw import load_bandwidth

            rt = ap.rt
            bw = load_bandwidth(rt.store.path) or {}
            d = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "vram_totale_gio": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
                 if torch.cuda.is_available() else 0,
                 "ram_totale_gio": round(psutil.virtual_memory().total / 2**30, 1),
                 "systeme": platform.system(),
                 "couches_moe": len(rt.store.layers), "experts_par_couche": rt.store.num_experts,
                 "precisions_du_store": list(rt.store.precisions),
                 "mio_par_expert": round(rt.store.record_bytes(rt.store.lo) / 2**20, 2),
                 "store_total_gio": round(rt.store.total_bytes(rt.store.lo) / 2**30, 1)}
            if bw.get("disk_parallel"):
                d["disque_go_s_au_mieux"] = round(max(bw["disk_parallel"].values()) / 1e9, 2)
            return d

        @tool
        def espace() -> dict:
            """Les réglages que tu peux changer, leurs valeurs permises et la valeur de la meilleure configuration actuelle."""
            out = {}
            for k, sp in ap.space.items():
                d = {**sp, "actuel": ap.best.get(k)}
                o = options.option(k)
                d["role"] = o.help[:160]
                if not ap.blind and o.measured:
                    d["deja_mesure"] = o.measured
                out[k] = d
            return out

        @tool
        def mesurer(changements: dict) -> dict:
            """Mesure la meilleure configuration actuelle avec ces changements (~1 min). Le code décide si on la garde.

            Args:
                changements: réglages à changer, ex. {"read_parts": 4} ; un ou deux à la fois
            """
            return ap.experiment(changements)

        @tool
        def journal() -> list:
            """Toutes les expériences déjà faites, avec leur score et leur verdict."""
            return ap.journal_rows()

        @tool
        def terminer(rapport: str) -> str:
            """Termine l'optimisation et enregistre la meilleure configuration.

            Args:
                rapport: ce que tu as trouvé et pourquoi, en quelques phrases, en français
            """
            ap.report, ap.done = rapport, True
            return "terminé : la meilleure configuration est enregistrée"

        return [machine, espace, mesurer, journal, terminer]

    def system_prompt(self) -> str:
        return (
            "Tu es l'autopilot de moelab, le moteur sur lequel tu tournes toi-même. Ton but : réduire la durée d'une "
            f"réponse type (premier token + {ANSWER_TOKENS} tokens) sur ce PC, sans dégrader la qualité. "
            "Méthode : 1) regarde la machine et l'espace des réglages ; 2) formule une hypothèse ; 3) mesure UN ou deux "
            "changements à la fois avec mesurer ; 4) lis le résultat (tok/s, premier token, hit VRAM, lectures disque "
            "par token, écarts en % à la meilleure configuration) avant le choix suivant. Le code garde un changement "
            "s'il gagne plus que le bruit sur la réponse type, ou nettement sur le premier token ou les tok/s sans "
            "perdre sur l'autre, et refuse ceux qui dégradent la qualité : ne réessaie pas une configuration refusée. "
            "Pour un réglage numérique, si une valeur aide, essaie aussi la suivante. "
            f"Tu as {self.max_trials} essais. Quand tu n'as plus d'idée utile ou plus d'essais, appelle terminer "
            "avec un court rapport en français. Sois bref entre deux outils. RÈGLE : chacun de tes messages se termine "
            "par un appel d'outil (mesurer, journal ou terminer) ; n'annonce jamais une mesure sans l'appeler."
        )

    def run(self, generate_fn=None, on_step=None) -> dict:
        """Mesure le départ, puis laisse le modèle mener les essais. Retourne le résumé."""
        self.log("Mesure de départ ...")
        self.baseline()
        gen = generate_fn or moelab_generate_fn(self.model, self.tokenizer, max_new_tokens=400)
        agent = Agent(self.tools(), gen, max_steps=3 * self.max_trials + 8, on_step=on_step,
                      stop_when=lambda: self.done)
        messages = [{"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": "Optimise moelab pour ce PC. Commence."}]
        self.nudges = 0
        while not self.done and self.nudges < 5:
            last = agent.run(messages)
            if self.done:
                break
            # le modèle s'est arrêté sans appeler d'outil (vu le 05/10 : « Mesurons. » sans l'appel) : on le relance
            self.nudges += 1
            self.log(f"\n  (relance {self.nudges}/5 : réponse sans appel d'outil"
                     + (", vide" if not last.strip() else "") + ")")
            messages.append({"role": "user", "content":
                             "Tu n'as appelé aucun outil. Si tu as annoncé une mesure, appelle mesurer maintenant avec "
                             "ces changements ; sinon appelle journal ou terminer."})
        if not self.done:
            self.report = self.report or f"(le modèle s'est arrêté sans appeler terminer, après {self.nudges} relances)"
        self.final_check()
        self.apply_config(self.best)
        return self.summary()

    def final_check(self) -> None:
        """Vérification finale : départ et meilleure config mesurés l'un après l'autre, dans les mêmes conditions."""
        self.final = {}
        if self.best == self.start:
            return
        self.log("Vérification finale : départ puis meilleure configuration, l'un après l'autre ...")
        try:
            for name, cfg in (("depart", self.start), ("meilleur", self.best)):
                self.apply_config(cfg)
                r = self.bench.run()
                self.final[name] = {k: r[k] for k in ("reponse_type_s", "premier_token_s", "tok_s", "kl")}
            a, b = self.final["depart"], self.final["meilleur"]
            self.final["gain_pourcent"] = round(100 * (1 - b["reponse_type_s"] / a["reponse_type_s"]), 1)
            self.log(f"  départ {a['reponse_type_s']} s ({a['tok_s']} tok/s, 1er token {a['premier_token_s']} s) -> "
                     f"meilleur {b['reponse_type_s']} s ({b['tok_s']} tok/s, 1er token {b['premier_token_s']} s) : "
                     f"{self.final['gain_pourcent']:+.1f} %")
        except Exception as e:
            self.final = {"erreur": f"{e.__class__.__name__}: {e}"}

    def summary(self) -> dict:
        kept = [t for t in self.trials if t.verdict.startswith("GARDÉ")]
        return {"depart": self.start, "depart_reponse_type_s": self.start_result.get("reponse_type_s"),
                "meilleur": self.best, "meilleur_reponse_type_s": round(self.best_score, 1),
                "gain_pourcent": round(100 * (1 - self.best_score / self.start_result["reponse_type_s"]), 1)
                if self.start_result else 0.0,
                "essais": len(self.trials) - 1, "gardes": [t.changes for t in kept],
                "relances": getattr(self, "nudges", 0), "verification_finale": getattr(self, "final", {}),
                "minutes": round((time.time() - self.t0) / 60, 1), "rapport": self.report}

    # -- sorties ------------------------------------------------------------
    def save(self, out_dir, model_name: str, write_profile: bool = True) -> tuple[Path | None, Path]:
        """Profil machine (<store>/autopilot_profile.json) + rapport (results/autopilot_<modèle>_<date>.md).
        Le profil n'est écrit que si write_profile et qu'un changement a été gardé."""
        s = self.summary()
        changed = {k: v for k, v in self.best.items() if self.start.get(k) != v}
        prof = {"modele": model_name, "store_id": self.rt.store.store_id, "date": datetime.now().isoformat(timespec="minutes"),
                "reglages": self.best, "changes_par_rapport_au_depart": changed,
                "reponse_type_s": s["meilleur_reponse_type_s"], "depart_reponse_type_s": s["depart_reponse_type_s"]}
        ppath = Path(self.rt.store.path) / PROFILE_FILE if write_profile and changed else None
        if ppath is not None:
            ppath.write_text(json.dumps(prof, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        md = [f"# Autopilot {model_name} — {stamp}", "",
              f"Réponse type (1er token + {ANSWER_TOKENS} tokens) : **{s['depart_reponse_type_s']} → "
              f"{s['meilleur_reponse_type_s']} s** ({s['gain_pourcent']:+.1f} %), {s['essais']} essais, {s['minutes']} min"
              + (" ; épreuve à l'aveugle" if self.blind else ""), "",
              *([f"Vérification finale (mesurés l'un après l'autre) : départ {s['verification_finale']['depart']} → "
                 f"meilleur {s['verification_finale']['meilleur']} : **{s['verification_finale']['gain_pourcent']:+.1f} %**", ""]
                if s.get("verification_finale", {}).get("depart") else []),
              "## Rapport du modèle", "", self.report or "(pas de rapport)", "",
              "## Journal", "", "| essai | changements | réponse type (s) | tok/s | 1er token (s) | KL | verdict |",
              "|---|---|---|---|---|---|---|"]
        for r in self.journal_rows():
            md.append(f"| {r['essai']} | {r['changements']} | {r['reponse_type_s']} | {r['tok_s']} | "
                      f"{r['premier_token_s']} | {r['kl']} | {r['verdict']} |")
        md += ["", "## Meilleure configuration", "", "```json", json.dumps(self.best, indent=1, default=str), "```", ""]
        rpath = out / f"autopilot_{model_name}_{stamp}.md"
        rpath.write_text("\n".join(md), encoding="utf-8")
        (out / f"autopilot_{model_name}_{stamp}.json").write_text(
            json.dumps({"summary": s, "journal": self.journal_rows(), "profil": prof}, indent=1, ensure_ascii=False,
                       default=str), encoding="utf-8")
        return ppath, rpath


def apply_profile(model, store_dir, skip: tuple = (), log=print) -> dict:
    """Applique <store>/autopilot_profile.json (réglages trouvés par l'autopilot). `skip` : clés à laisser."""
    path = Path(store_dir) / PROFILE_FILE
    if not path.exists():
        return {}
    prof = json.loads(path.read_text(encoding="utf-8"))
    if prof.get("store_id") not in (None, model.moelab_runtime.store.store_id):
        log("  profil autopilot ignoré : fait pour un autre store")
        return {}
    done = {}
    cur = options.current(model)
    for k, v in prof.get("reglages", {}).items():
        if k in skip or cur.get(k) == v:
            continue
        try:
            options.apply(model, k, v)
            done[k] = v
        except Exception as e:
            log(f"  profil autopilot : {k} = {v} non appliqué ({e})")
    if done:
        log(f"  profil autopilot appliqué ({prof.get('date', '?')}) : {done}")
    return done


# Point de départ de l'épreuve : les réglages du 02/10, avant l'optimisation de Qwen3-Next
# à la main (lectures découpées, réserve VRAM, P6). L'éviction en tas, un changement de
# code, reste active : le départ sera donc un peu plus rapide qu'au 02/10.
def start_0210(rt) -> dict:
    out = {"read_parts": 1, "prefill_grouped": 0}
    if rt.fast is not None and rt.fast.device.type == "cuda":
        out["vram_gb"] = round(max(0.5, rt.fast.budget / 2**30 - 0.27), 2)  # réserve 1,0 Gio au lieu de 0,74
    return out
