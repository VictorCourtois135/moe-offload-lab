"""
moelab — offloading intelligent d'experts MoE sur matériel grand public.

Briques (une par fichier) :
    quant       int4 / int8 par groupes de 64, déquantification FP8 des checkpoints
    checkpoint  lecture directe des .safetensors, tenseur par tenseur
    store       experts convertis sur disque, un record contigu par expert
    cache       caches VRAM / RAM (score LRFU, épinglage, anti-thrashing) + historique d'usage
    policy      précision par expert (HOBBIT) et répartition GPU/CPU (q* de FreeToken)
    runtime     exécution d'une couche MoE à travers les niveaux de mémoire + prefetch
    loader      chargement du modèle sans ses experts
    plan        auto-dimensionnement des budgets mémoire
    hw          mesure des bandes passantes
    bench       métriques de qualité et de vitesse, règle d'acceptation A/B
    options     catalogue de tous les réglages (pour l'app et chat.py /options, /set)
"""

from . import options
from .cache import TierCache, UsageLog
from .loader import load_offloaded_model
from .policy import PrecisionPolicy, qstar
from .runtime import MoERuntime, RuntimeConfig
from .store import ExpertStore, build_store

__version__ = "1.0.0"

__all__ = [
    "ExpertStore",
    "MoERuntime",
    "PrecisionPolicy",
    "RuntimeConfig",
    "TierCache",
    "UsageLog",
    "build_store",
    "load_offloaded_model",
    "options",
    "qstar",
]
