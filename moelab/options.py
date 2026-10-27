"""
Catalogue de tous les réglages de moelab, pour l'app (et pour chat.py dès maintenant).

Chaque réglage dit :
  - ce qu'il fait, en une phrase, et son effet MESURÉ sur la RTX 2060 (doc 18) ;
  - ses valeurs possibles et sa valeur par défaut ;
  - comment il s'applique :
      "live"    : à chaud, entre deux messages, sans rien recharger ;
      "caches"  : à chaud, mais les caches VRAM/RAM sont vidés puis re-préchargés ;
      "reload"  : il faut recharger le modèle (quelques secondes à une minute) ;
      "store"   : il faut un autre store sur le disque (build_store.py, plusieurs minutes).

Utilisation :
    from moelab.options import catalog, current, apply, PRESETS
    for opt in catalog(): print(opt.key, opt.label, opt.apply)
    apply(model, "t1", 0.5)            # réglage à chaud
    apply(model, "grouped_kernels", 2) # idem, avec les effets de bord nécessaires
    current(model)                      # {clé: valeur actuelle}
    describe_json()                     # le catalogue complet, prêt pour une interface
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

GROUPS = {
    "precision": "Précision (qualité contre vitesse)",
    "prefill": "Lecture du prompt (temps avant le premier token)",
    "memory": "Mémoire (budgets et placement)",
    "gpu": "Calcul GPU (noyaux et CUDA graphs)",
    "io": "Transferts et disque",
    "store": "Store (format des experts sur le disque)",
}


@dataclass(frozen=True)
class Option:
    key: str
    label: str
    group: str
    kind: str                       # "bool", "int", "float", "choice", "path", "layers"
    default: object
    apply: str                      # "live", "caches", "reload", "store"
    target: str                     # "policy.<champ>", "cfg.<champ>", "load.<argument>", "store"
    help: str
    measured: str = ""              # effet mesuré (doc 18), vide si non mesuré
    choices: tuple = ()
    min: float | None = None
    max: float | None = None
    unit: str = ""
    advanced: bool = False          # caché par défaut dans l'interface
    tags: tuple = field(default_factory=tuple)

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["group_label"] = GROUPS[self.group]
        return d


_OPTIONS: list[Option] = [
    # ------------------------------------------------------------------ précision
    Option("policy", "Politique de précision", "precision", "choice", "hobbit", "live", "policy.mode",
           "hi : tout en haute précision (référence) ; lo : tout en int4 (le plus rapide) ; hobbit : la précision "
           "de chaque expert selon son importance pour le token.",
           "OLMoE, store int8+int4 : lo 74 tok/s (KL 0,072), hobbit@0.8 22,6 tok/s (KL 0,0066).",
           choices=("hi", "lo", "hobbit")),
    Option("t1", "Seuil HOBBIT t1", "precision", "float", 0.6, "live", "policy.t1",
           "Plus t1 est haut, plus d'experts passent en haute précision : meilleure qualité, plus de transferts.",
           "OLMoE int8+int4 : t1=0,5 → 21,1 tok/s, KL 0,0205 ; t1=0,8 → 22,6 tok/s, KL 0,0066.",
           min=0.0, max=1.0),
    Option("t1_layers", "Seuils t1 par couche", "precision", "path", None, "live", "policy.t1_layers",
           "E1 : un seuil par couche calibré par measure_routing.py ('auto' = <store>/t1_layers.json).",
           "calibré à ~0,82-0,86 selon la couche sur OLMoE.", advanced=True),
    Option("t_mid", "Seuil du palier intermédiaire", "precision", "float", None, "live", "policy.t_mid",
           "E2 : avec un store à trois précisions (fp16, int8, int4), les experts entre t1 et t_mid passent en int8.",
           min=0.0, max=1.0, advanced=True),
    Option("t2", "Seuil de saut", "precision", "float", None, "live", "policy.t2",
           "Au-delà de ce score, l'expert n'est pas calculé du tout. Change les sorties : à mesurer avant usage.",
           min=0.0, max=1.0, advanced=True),
    Option("miss_t1", "Seuil sur défaut de cache (E3)", "precision", "float", None, "live", "policy.miss_t1",
           "Un expert absent de la VRAM et peu important est calculé en basse précision au lieu d'être copié.",
           "rejeté sur OLMoE : +6 % de vitesse, KL ×2,4.", min=0.0, max=1.0, advanced=True),
    Option("protect", "Couches protégées", "precision", "layers", "", "live", "policy.protect_layers",
           "Couches toujours en haute précision : '0,1,15' ou 'edges' (première, deuxième et dernière).", advanced=True),
    Option("hi_vote", "Vote haute précision (prefill)", "precision", "float", 0.5, "live", "policy.hi_vote",
           "En prefill, part des tokens d'un expert qui doivent demander la haute précision pour qu'il la reçoive.",
           min=0.0, max=1.0, advanced=True),
    Option("upgrade", "Réutiliser la haute précision en cache", "precision", "bool", True, "live", "policy.upgrade",
           "Un expert déjà en VRAM en haute précision sert aussi une demande basse précision (qualité gratuite).",
           advanced=True),
    # ------------------------------------------------------------------ store
    Option("store", "Store d'experts", "store", "path", None, "store", "store",
           "Le dossier des experts convertis. Sa haute précision (fp16 ou int8) se choisit à la construction : "
           "build_store.py --hi int8 --lo int4.",
           "OLMoE, hobbit@0.8 : fp16+int4 5,6 tok/s, int8+int4 26,2 tok/s (mêmes budgets), même qualité "
           "(KL 0,0074 contre 0,0079) ; int8 seul : KL ~0,002 contre fp16.", tags=("int8", "fp16")),
    Option("int4_quant", "Quantification de l'int4", "store", "choice", "mse", "store", "store.int4_scale",
           "Q1/Q2 : comment les nombres int4 du store sont choisis. absmax = échelle max|w|/7 (stores d'avant le "
           "02/10) ; mse = meilleure échelle par groupe, sans calibration ; gptq = erreur compensée d'après de vraies "
           "entrées (calibration). Même format, mêmes noyaux, même vitesse : requant_store.py réécrit l'int4 en place.",
           "OLMoE (KL contre fp16) : lo 0,070 → 0,050 (mse) → 0,049 (gptq), perplexité +10,2 → +5,0 → +3,4 % ; "
           "hobbit@0.8 0,0089 → 0,0065 → 0,0061 ; vitesse inchangée.", choices=("absmax", "mse", "gptq"),
           tags=("Q1", "Q2")),
    # ------------------------------------------------------------------ mémoire
    Option("vram_gb", "Budget VRAM des experts", "memory", "float", None, "caches", "load.vram_gb",
           "VRAM réservée au cache d'experts (vide = automatique : ce qui reste après la partie dense et la réserve). "
           "Change à chaud depuis le 05/10 (caches vidés et re-préchargés).",
           "Qwen3-Next : 1,84 → 2,24 Gio = 5,93 → 6,81 tok/s (hit 58 → 61 %).", unit="Gio", min=0.0),
    Option("ram_gb", "Budget RAM des experts", "memory", "float", None, "caches", "load.ram_gb",
           "RAM réservée au cache d'experts (vide = automatique). Au-delà de ~35 % de la RAM totale épinglée, "
           "Windows se met à paginer. Change à chaud depuis le 05/10 (caches vidés et re-préchargés).",
           "Qwen3-Next : 4 → 6 Gio = lectures disque 140 → 109 par token, vitesse dans le bruit.", unit="Gio", min=0.0),
    Option("vram_reserve_gb", "Réserve VRAM", "memory", "float", None, "reload", "load.vram_reserve_gb",
           "VRAM laissée aux activations et au cache KV (vide = automatique : 0,55 Gio d'activations + le cache KV "
           "statique à la longueur max ; OLMoE 0,80 Gio, Qwen3-Next 0,74 Gio à 8 192 tokens). Avant le 03/10 : 1 Gio fixe.",
           "Qwen3-Next, lo, RAM 5 Gio : réserve 1,0 → 0,6 Gio, VRAM d'experts 1,84 → 2,24 Gio, hit 58 → 61 %, "
           "5,93 → 6,81 tok/s.", unit="Gio", min=0.0, advanced=True),
    Option("ram_reserve_gb", "Réserve RAM", "memory", "float", 3.0, "reload", "load.ram_reserve_gb",
           "RAM laissée à Windows et aux autres programmes.", unit="Gio", min=0.0, advanced=True),
    Option("dense", "Placement de la partie dense", "memory", "choice", "auto", "reload", "load.dense",
           "gpu : tout en VRAM ; split : embeddings et tête de sortie sur CPU (pour les gros modèles) ; cpu : tout en RAM.",
           choices=("auto", "gpu", "split", "cpu")),
    Option("dense_int8", "Partie dense en int8", "memory", "bool", True, "reload", "load.dense_int8",
           "D5 : attention, experts partagés et tête de sortie en int8 par groupes de 64 : 2 fois moins de VRAM "
           "(laissée au cache d'experts) et d'octets lus par token. Indispensable pour Qwen3-Next-80B.",
           "OLMoE : lo 104,7 → 118,9 tok/s (profil) ; hobbit@0.8 24,8 → 28,9 tok/s au benchmark, KL 0,0069 → 0,0058 "
           "(qualité inchangée, dans le bruit).",
           tags=("D5",)),
    Option("preload_lo", "Précharger tout le pool basse précision", "memory", "choice", "auto", "reload",
           "load.preload_lo", "auto : si le pool int4 tient dans 60 % du budget RAM.", choices=("auto", "yes", "no"),
           advanced=True),
    # ------------------------------------------------------------------ GPU
    Option("cuda_graphs", "CUDA graphs (partie dense)", "gpu", "bool", True, "live", "cfg.cuda_graphs",
           "A6 : la partie dense de chaque token rejouée en 17 graphs au lieu de ~1 000 lancements.",
           "lo : 20,9 → 41,9 tok/s.", tags=("A6",)),
    Option("moe_graphs", "Calcul groupé en graph", "gpu", "bool", True, "live", "cfg.moe_graphs",
           "A6 bis : le calcul des experts de chaque couche rejoué en un graph (avec les CUDA graphs).",
           "lo : 47,0 → 89,6 tok/s.", tags=("A6 bis",)),
    Option("graph_buckets", "Paliers de longueur", "gpu", "bool", True, "live", "cfg.graph_buckets",
           "A8 : l'attention ne lit que les positions utilisées du cache (paliers 128, 256... 2 048).",
           "lo : 89,4 → 105,9 tok/s.", tags=("A8",)),
    Option("graph_pack_routing", "Routage et prédiction lus en une fois", "gpu", "bool", True, "live",
           "cfg.graph_pack_routing",
           "A9 : le graph calcule aussi la prédiction du prefetch et la rassemble avec le routage ; une seule "
           "lecture GPU -> CPU par couche, plus de routeur lancé en Python.", advanced=True, tags=("A9",)),
    Option("graph_max_len", "Longueur max (cache statique)", "gpu", "int", None, "live", "cfg.graph_max_len",
           "Prompt + réponse maximum avec les CUDA graphs (toute la conversation dans chat). Vide = automatique : la "
           "plus grande de 8 192 / 4 096 / 2 048 dont le cache KV tient dans 256 Mio — OLMoE 2 048 (128 Ko/token), "
           "Qwen3-Next 8 192 (24 Ko/token : 12 couches d'attention sur 48). La réserve VRAM automatique est calculée "
           "au chargement pour cette longueur : l'augmenter à chaud peut manquer de VRAM.",
           unit="tokens", min=128, max=32768),
    Option("full_graph", "Token entier en un graph", "gpu", "bool", True, "live", "cfg.full_graph",
           "A5 : quand tous les experts demandés sont en VRAM, tout le token (couches MoE comprises) est rejoué en "
           "un seul graph, décisions HOBBIT prises sur le GPU ; une seule synchronisation par token au lieu d'une "
           "par couche. Un expert absent : le token est refait par le chemin normal.",
           "lo : 117,5 → 132,1 tok/s (100 % des tokens en un graph) ; hobbit : neutre (rarement tenté).", tags=("A5",)),
    Option("grouped_experts", "Noyaux groupés", "gpu", "bool", True, "live", "cfg.grouped_experts",
           "A4 : tous les experts d'une couche en 2 noyaux Triton par précision.", "lo : 12,4 → 20,2 tok/s.",
           tags=("A4",)),
    Option("grouped_kernels", "Version des noyaux int4", "gpu", "choice", 0, "live", "cfg.grouped_kernels",
           "0 = la plus rapide sur la carte (mesurée au démarrage) ; 1 à 4 = forcée.",
           "en graph : v3 176 Go/s, v2 167, v4 121.", choices=(0, 1, 2, 3, 4), advanced=True),
    Option("triton_int4", "Noyau Triton int4", "gpu", "bool", True, "live", "cfg.triton_int4",
           "B1 : l'int4 lu directement par le noyau, sans décompression.", "lo : 6,7 → 11,8 tok/s.", advanced=True),
    Option("fast_decode", "Décisions de génération allégées", "gpu", "bool", True, "live", "cfg.fast_decode",
           "A2 : en génération, les décisions HOBBIT en Python pur avec une seule synchronisation.", advanced=True),
    Option("fast_int4_unpacked", "int4 déballé en VRAM", "gpu", "bool", False, "caches", "cfg.fast_int4_unpacked",
           "B2 : l'int4 rangé en int8 dans la VRAM (2 fois plus de place).", "rejeté : hit VRAM 100 → 93 %.",
           advanced=True),
    # ------------------------------------------------------------------ transferts et disque
    Option("prefetch", "Prefetch de la couche suivante", "io", "bool", True, "live", "cfg.prefetch",
           "Le routeur de la couche suivante, appliqué à l'état courant, prédit ses experts : on les charge à l'avance.",
           "précision et rappel du prefetch : 84 % sur OLMoE."),
    Option("prefetch_adaptive", "Prefetch adaptatif", "io", "bool", True, "live", "cfg.prefetch_adaptive",
           "Saute le prefetch tant qu'aucun expert ne manque.", "lo : prefetch 0,45 → 0,007 ms par couche.",
           advanced=True),
    Option("prefetch_mass", "Prefetch selon le score", "io", "float", None, "live", "cfg.prefetch_mass",
           "S3 (SpecMD) : précharger les experts prédits jusqu'à cette part de la probabilité du routeur (au plus "
           "2k), au lieu des k meilleurs : moins d'experts quand le routeur est sûr, plus quand il hésite. "
           "Vide = les k meilleurs.", "rejeté sur OLMoE à 0,9 : hobbit 49,8 → 20,7 tok/s (49,8 préchargés par token, "
           "précision 49 %) ; le routeur d'OLMoE est trop plat.", min=0.1, max=0.999, advanced=True, tags=("S3",)),
    Option("cache_eviction", "Éviction du cache VRAM", "io", "choice", "lrfu", "live", "cfg.cache_eviction",
           "lrfu : score récence + fréquence ; stale (SpecMD) : les experts de la passe en cours (utilisés ou "
           "préchargés) ne sont évincés qu'en dernier.", "rejeté sur OLMoE : hobbit 51,5 → 43,1 tok/s (plus de "
           "prefetch évincés puis rechargés) ; neutre en lo.", choices=("lrfu", "stale"), advanced=True, tags=("S3",)),
    Option("prefetch_disk_ahead", "Lectures disque à l'avance", "io", "int", 0, "live", "cfg.prefetch_disk_ahead",
           "Pour les modèles qui ne tiennent pas en RAM (Qwen3-Next) : les experts prédits des couches L+2, L+3... "
           "sont lus du disque vers la RAM à l'avance (jamais copiés en VRAM). Prédiction calculée dans le graph.",
           "rejeté sur Qwen3-Next : 3,35 → 3,01 (1) → 2,83 tok/s (2) ; l'attente vient des experts non prévus.",
           min=0, max=4, advanced=True, tags=("disque",)),
    Option("prefetch_depth", "Profondeur du prefetch", "io", "int", 1, "live", "cfg.prefetch_depth",
           "C2 : précharger aussi la couche L+2.", "rejeté : hit VRAM 78 → 72 %.", min=1, max=2, advanced=True),
    Option("qstar", "q* (calcul sur CPU)", "io", "bool", True, "live", "cfg.use_qstar",
           "Répartit les experts manquants entre copie vers la VRAM et calcul direct sur CPU.",
           "sur la RTX 2060, q* copie tout (B_Host < B_PCIe).", advanced=True),
    Option("direct_io", "Lecture disque sans cache", "io", "bool", True, "live", "cfg.direct_io",
           "D1 : lectures non tamponnées (Windows : FILE_FLAG_NO_BUFFERING), repli automatique.",
           "hobbit : +9 %."),
    Option("sort_reads", "Lectures triées", "io", "bool", True, "live", "cfg.sort_reads",
           "D3 : les lectures d'une couche dans l'ordre du fichier (utile sur HDD).", advanced=True),
    Option("read_parts", "Morceaux par lecture d'expert", "io", "int", 1, "live", "cfg.read_parts",
           "Un expert lu sur disque en N lectures simultanées : un SSD ne donne son débit qu'avec plusieurs lectures "
           "en vol (mesuré par measure_bandwidth.py).", "Qwen3-Next : 1 → 4 morceaux = 6,82 → 10,85 tok/s (attente 1,82 → 0,65 ms par couche).",
           min=1, max=16, advanced=True, tags=("disque",)),
    Option("io_workers", "Lectures en parallèle", "io", "int", 4, "reload", "cfg.io_workers",
           "D2 : nombre de lectures disque simultanées (mesuré par measure_bandwidth.py).", min=1, max=32,
           advanced=True),
    Option("hot_store", "Store chaud (NVMe)", "io", "path", None, "reload", "cfg.hot_store",
           "D4 : les experts les plus utilisés copiés sur le disque rapide (build_hot_store.py).", advanced=True),
    # ------------------------------------------------------------------ prefill (03/10)
    Option("kv_reuse", "Reprendre la conversation du cache", "prefill", "bool", True, "live", "cfg.kv_reuse",
           "P1 : à chaque message, seuls les nouveaux tokens sont calculés ; l'historique déjà vu reste dans le "
           "cache (avant : toute la conversation recalculée à chaque message). Exact. Qwen3-Next (attention "
           "linéaire) : reprise exacte de la suite, sinon retour à la fin du prompt précédent.", tags=("P1",)),
    Option("prefill_order", "Ordre de calcul au prefill", "prefill", "choice", "fills", "live", "cfg.prefill_order",
           "P2 : completed = les experts manquants sont calculés dans l'ordre où le disque les rend, au lieu "
           "d'attendre le plus demandé.", "Rejeté : aucun effet (5,83 → 5,88 s).", choices=("fills", "completed"), advanced=True, tags=("P2",)),
    Option("prefill_prefetch", "Prefetch au prefill", "prefill", "bool", False, "live", "cfg.prefill_prefetch",
           "P3 : pendant le prefill, les experts prédits de la couche suivante sont lus du disque vers la RAM "
           "pendant le calcul de la couche courante.", "Rejeté : 5,83 → 6,25 s (plus de lectures en concurrence).",
           advanced=True, tags=("P3",)),
    Option("prefill_cache", "Experts du prefill en VRAM", "prefill", "choice", "vram", "live", "cfg.prefill_cache",
           "P4 : ram = les experts lus pour le prompt ne remplacent pas ceux de la génération dans le cache VRAM.",
           "Rejeté : prefill égal, génération qui suit plus lente (les experts du prompt servent à la réponse).",
           choices=("vram", "ram"), advanced=True, tags=("P4",)),
    Option("prefill_grouped", "Noyaux groupés au prefill", "prefill", "int", 192, "live", "cfg.prefill_grouped",
           "P6 : prompts jusqu'à ce nombre de tokens : experts calculés token par token avec les noyaux groupés de "
           "la génération (2 lancements par token au lieu d'~30 par expert). 0 = non.",
           "Qwen3-Next : 1er token 5,95 → 3,76 s (17 tokens), 11,66 → 7,90 s (64), 14,3 → 10,5 s (93) ; mais "
           "22,5 → 27,9 s à 407 tokens (chaque token relit ses experts) : seuil 192. Exact.", unit="tokens", min=0, max=512, advanced=True, tags=("P6",)),
    Option("prefill_split", "Lectures découpées au prefill", "prefill", "bool", True, "live", "cfg.prefill_split",
           "P7 : off = au prefill, un expert = une seule lecture.", "Rejeté : 5,95 → 6,31 s.",
           advanced=True, tags=("P7",)),
    Option("prefill_skip", "Saut au prefill", "prefill", "float", None, "live", "cfg.prefill_skip",
           "P5 : au prefill, un expert à lire sur le disque dont le plus grand poids du routeur est sous ce seuil "
           "est sauté (comme t2). Moins de lectures, qualité à vérifier (profile_prefill.py la mesure).",
           "Avec P6 : 0,05 → 1er token 3,76 → ~3,4 s, KL dans le bruit ; 0,10 → 2,5 s, KL ×4 (0,012 sur 64 tokens).",
           min=0.0, max=1.0, advanced=True, tags=("P5",)),
]

_BY_KEY = {o.key: o for o in _OPTIONS}

# Réglages tout faits pour l'interface (clés du catalogue -> valeurs)
PRESETS = {
    "vitesse": {"policy": "lo"},
    "équilibre": {"policy": "hobbit", "t1": 0.5},
    "qualité": {"policy": "hobbit", "t1": 0.8},
    "référence": {"policy": "hi"},
}


def catalog(advanced: bool = True) -> list[Option]:
    return [o for o in _OPTIONS if advanced or not o.advanced]


def option(key: str) -> Option:
    if key not in _BY_KEY:
        raise KeyError(f"réglage inconnu : {key} (choix : {', '.join(_BY_KEY)})")
    return _BY_KEY[key]


def describe_json(advanced: bool = True) -> str:
    return json.dumps({"groups": GROUPS, "options": [o.as_dict() for o in catalog(advanced)], "presets": PRESETS},
                      ensure_ascii=False, indent=1, default=str)


# ----------------------------------------------------------------------
# Lecture et application
# ----------------------------------------------------------------------

def parse(key: str, text: str):
    """Valeur tapée (texte) -> valeur typée, vérifiée."""
    o = option(key)
    t = text.strip()
    if t.lower() in ("none", "", "-", "auto") and o.kind in ("float", "path", "int") and o.default is None:
        return "auto" if (o.kind == "path" and t.lower() == "auto") else None
    if o.kind == "bool":
        if t.lower() in ("1", "true", "on", "oui", "yes"):
            return True
        if t.lower() in ("0", "false", "off", "non", "no"):
            return False
        raise ValueError(f"{key} : on/off attendu")
    if o.kind == "int":
        v = int(t)
    elif o.kind == "float":
        v = float(t)
    elif o.kind == "choice":
        v = type(o.choices[0])(t)
        if v not in o.choices:
            raise ValueError(f"{key} : choix possibles {o.choices}")
        return v
    else:
        return t
    if o.min is not None and v < o.min or o.max is not None and v > o.max:
        raise ValueError(f"{key} : entre {o.min} et {o.max}")
    return v


def _runtime(model):
    return model.moelab_runtime if hasattr(model, "moelab_runtime") else model


def current(model) -> dict:
    """Valeurs actuelles des réglages lisibles à chaud (policy.* et cfg.*)."""
    rt = _runtime(model)
    out = {}
    for o in _OPTIONS:
        kind, _, name = o.target.partition(".")
        if kind == "policy":
            v = getattr(rt.policy, name)
            if name == "protect_layers":
                v = ",".join(str(x) for x in sorted(v))
            elif name == "t1_layers":
                v = "calibré" if v else None
            out[o.key] = v
        elif kind == "cfg":
            out[o.key] = getattr(rt.cfg, name)
    out["vram_gb"] = round((rt.fast.budget if rt.fast is not None else 0) / 2**30, 3)
    out["ram_gb"] = round(rt.host.budget / 2**30, 3)
    out["store"] = str(rt.store.path)
    if "int4" in rt.store.layouts:
        out["int4_quant"] = rt.store.layouts["int4"].get("scale") or "absmax"
    out["store_precisions"] = list(rt.store.precisions)
    return out


def apply(model, key: str, value) -> str:
    """
    Applique un réglage « live » ou « caches » au modèle chargé. Retourne une phrase
    décrivant ce qui a été fait. Les réglages « reload » et « store » lèvent une erreur
    explicite : c'est à l'app de recharger le modèle avec la nouvelle valeur.
    """
    rt = _runtime(model)
    o = option(key)
    if isinstance(value, str) and o.kind not in ("path", "layers") and not (o.kind == "choice" and isinstance(o.choices[0], str)):
        value = parse(key, value)
    if o.kind == "choice" and value not in o.choices:
        raise ValueError(f"{key} : choix possibles {o.choices}")
    if o.apply in ("reload", "store"):
        raise ValueError(f"{key} ({o.label}) : demande de {'recharger le modèle' if o.apply == 'reload' else 'changer de store'}")
    kind, _, name = o.target.partition(".")
    if key in ("vram_gb", "ram_gb"):  # budgets à chaud (05/10) : caches vidés puis re-préchargés
        if value is None:
            raise ValueError(f"{key} : une valeur en Gio est attendue à chaud (l'automatique se calcule au chargement)")
        b = int(float(value) * 2**30)
        rt.set_budgets(fast_bytes=b if key == "vram_gb" else None, host_bytes=b if key == "ram_gb" else None)
        rt.warm_start(preload_lo=rt.preload_lo, log=lambda *a: None)
        return f"{o.label} : {float(value):.2f} Gio (caches vidés et re-préchargés)"
    if kind == "policy":
        if name == "protect_layers":
            from .policy import PrecisionPolicy  # noqa: F401  (validation par le dataclass)

            n = len(rt.store.layers)
            value = (frozenset({0, 1, n - 1}) if value == "edges"
                     else frozenset(int(x) for x in str(value).split(",") if x.strip()))
        elif name == "t1_layers" and value is not None:
            path = Path(rt.store.path) / "t1_layers.json" if value == "auto" else Path(value)
            value = {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8"))["t1_layers"].items()}
        pol = dataclasses.replace(rt.policy, **{name: value})
        rt.policy = rt.cfg.policy = pol
        shown = sorted(value) if isinstance(value, frozenset) else ("par couche" if isinstance(value, dict) else value)
        return f"{o.label} : {shown} (politique : {pol.describe()})"

    if name == "graph_max_len" and value is None:
        value = getattr(rt, "auto_graph_max_len", 2048)
    setattr(rt.cfg, name, value)
    if name == "direct_io":
        rt.store.direct = value
    elif name == "read_parts":
        rt.store.read_parts = max(1, int(value))
    elif name == "cache_eviction" and rt.fast is not None:
        rt.fast.eviction = value
    elif name in ("triton_int4", "grouped_experts"):
        from .kernels import triton_usable

        ok = rt.fast is not None and triton_usable(rt.fast.device)
        rt._triton = bool(rt.cfg.triton_int4 and ok)
        rt._grouped = bool(rt.cfg.grouped_experts and ok)
        rt._ggraphs.clear()
    elif name in ("grouped_kernels", "moe_graphs"):
        rt._auto_int4 = None
        rt._ggraphs.clear()
    elif name == "fast_int4_unpacked":
        rt.set_unpacked(value)
        rt.reset_caches()
        rt._ggraphs.clear()
        rt.warm_start(preload_lo=rt.preload_lo, log=lambda *a: None)
        return f"{o.label} : {value} (caches vidés et re-préchargés)"
    # cuda_graphs, graph_buckets, graph_max_len : lus par make_decoder à la prochaine génération
    return f"{o.label} : {value}"


def apply_preset(model, name: str) -> list[str]:
    if name not in PRESETS:
        raise KeyError(f"préréglage inconnu : {name} (choix : {', '.join(PRESETS)})")
    return [apply(model, k, v) for k, v in PRESETS[name].items()]


def table(model=None, advanced: bool = False) -> str:
    """Tableau texte du catalogue (chat.py /options), avec les valeurs actuelles si un modèle est donné."""
    cur = current(model) if model is not None else {}
    lines = []
    for g, label in GROUPS.items():
        opts = [o for o in catalog(advanced) if o.group == g]
        if not opts:
            continue
        lines.append(f"\n{label}")
        for o in opts:
            val = cur.get(o.key, o.default)
            how = {"live": "à chaud", "caches": "vide les caches", "reload": "recharger", "store": "autre store"}[o.apply]
            lines.append(f"  {o.key:18s} = {str(val):10s} [{how}]  {o.label}")
            if o.measured:
                lines.append(f"  {'':18s}   mesuré : {o.measured}")
    return "\n".join(lines)
