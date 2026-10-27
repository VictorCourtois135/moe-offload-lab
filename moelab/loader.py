"""
Chargement d'un modèle MoE "déchargé".

Au lieu de `AutoModelForCausalLM.from_pretrained` (qui charge tout, 12 minutes et
13 Go pour OLMoE) :

  1. on construit le squelette du modèle sur le device "meta" : la structure
     existe, mais aucun poids n'est alloué (instantané, 0 octet) ;
  2. on remplace chaque module `experts` par un `OffloadedExperts` (sans poids) ;
  3. on ne lit depuis le checkpoint QUE les poids restants (attention, normes,
     routeurs, embeddings) : ~1 Go pour OLMoE, ~4,6 Go pour Qwen3-Next-80B ;
  4. les experts restent dans le store et arrivent à la demande.

Placement de la partie dense (`dense`) :
  - "gpu"   : tout sur GPU
  - "cpu"   : tout sur CPU
  - "split" : les couches sur GPU, mais la table d'embeddings et la tête de sortie
              (lm_head) restent en RAM et sont calculées sur CPU — elles pèsent
              1,2 Go à elles deux sur Qwen3-Next et ne coûtent qu'une ligne / un
              produit matrice-vecteur par token
  - "auto"  : "gpu" si ça tient, sinon "split", sinon "cpu"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

from .checkpoint import Checkpoint
from .runtime import MoERuntime, OffloadedExperts, RuntimeConfig
from .store import ExpertStore


def load_config(model_dir):
    config = AutoConfig.from_pretrained(model_dir)
    if getattr(config, "quantization_config", None) is not None:
        # Les poids FP8 sont déquantifiés par moelab à la lecture : on construit un modèle "normal".
        try:
            delattr(config, "quantization_config")
        except AttributeError:
            config.quantization_config = None
    return config


def build_skeleton(config, dtype: torch.dtype):
    """Le modèle complet, sur le device 'meta' : la structure sans les poids."""
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config, dtype=dtype)


def iter_experts_modules(model) -> list[tuple[str, nn.Module]]:
    """Les modules qui contiennent les poids des experts d'une couche (tenseurs 3D gate_up_proj / down_proj)."""
    found = []
    for name, m in model.named_modules():
        gu, dn = getattr(m, "gate_up_proj", None), getattr(m, "down_proj", None)
        if isinstance(gu, nn.Parameter) and isinstance(dn, nn.Parameter) and gu.dim() == 3 and dn.dim() == 3:
            if getattr(m, "is_transposed", False) or not getattr(m, "has_gate", True) or getattr(m, "has_bias", False):
                raise NotImplementedError(
                    f"{name} : variante d'experts non gérée (transposée, sans gate ou avec biais). "
                    "Familles testées : OLMoE, Qwen3-MoE, Qwen3-Next."
                )
            found.append((name, m))
    return found


def discover_experts(model_dir) -> dict:
    config = load_config(model_dir)
    model = build_skeleton(config, torch.float16)
    layers, shapes = [], set()
    for name, m in iter_experts_modules(model):
        E, two_i, H = m.gate_up_proj.shape
        _, H2, I = m.down_proj.shape
        if two_i != 2 * I or H2 != H:
            raise ValueError(f"{name} : formes inattendues {tuple(m.gate_up_proj.shape)} / {tuple(m.down_proj.shape)}")
        layers.append(name)
        shapes.add((E, H, I))
    if not layers:
        raise ValueError("aucun module d'experts trouvé : ce n'est pas un modèle MoE géré")
    if len(shapes) != 1:
        raise NotImplementedError(f"experts de tailles différentes selon les couches : {shapes}")
    E, H, I = shapes.pop()
    return {"model_type": config.model_type, "layers": layers, "num_experts": E, "hidden": H, "inter": I}


# Placement "split" : les poids de ces deux modules sont de simples attributs tenseurs,
# volontairement PAS des nn.Parameter. transformers déduit `model.device` du premier
# paramètre du modèle, qui serait alors cette table sur CPU : generate() déplacerait
# les entrées sur CPU et les couches (sur GPU) recevraient des tenseurs CPU.
# Sans paramètre sur CPU, `model.device` vaut bien cuda.


class CpuEmbedding(nn.Module):
    """Table d'embeddings gardée en RAM : seules les lignes demandées sont envoyées sur `out_device`."""

    def __init__(self, weight: torch.Tensor, padding_idx: int | None, out_device: torch.device | str | None = None):
        super().__init__()
        self.weight = weight.detach()
        self.padding_idx = padding_idx
        self.out_device = torch.device(out_device) if out_device is not None else None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = F.embedding(input_ids.to("cpu"), self.weight, self.padding_idx)
        return out.to(self.out_device if self.out_device is not None else input_ids.device)


class CpuLinear(nn.Module):
    """Tête de sortie calculée sur CPU (poids en RAM, calcul en float32 par blocs), résultat renvoyé sur le device d'entrée."""

    def __init__(self, weight: torch.Tensor, chunk: int = 16384):
        super().__init__()
        self.weight = weight.detach()
        self.chunk = chunk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.to("cpu", torch.float32)
        parts = [F.linear(xc, self.weight[i : i + self.chunk].float()) for i in range(0, self.weight.shape[0], self.chunk)]
        return torch.cat(parts, dim=-1).to(x.device, x.dtype)


def _set_param(model: nn.Module, name: str, value: torch.Tensor) -> None:
    mod_name, _, attr = name.rpartition(".")
    mod = model.get_submodule(mod_name) if mod_name else model
    mod._parameters[attr] = nn.Parameter(value, requires_grad=False)


def _param_bytes(params) -> int:
    return sum(p.numel() * p.element_size() for p in params)


SPLIT_EMB_BYTES = 256 * 2**20  # au-delà, des embeddings sur CPU libèrent assez de VRAM pour compter


def choose_dense_placement(dense_bytes: int, emb_head_bytes: int, vram_reserve: int, emb_bytes: int = 0,
                           head_on_gpu: bool = False) -> str:
    """
    gpu / split / cpu. `head_on_gpu` (D5) : la tête de sortie est en int8 et reste sur le GPU même
    en split ; si les embeddings sont gros (Qwen3-Next : 0,58 Gio), split est alors préférable à
    gpu même quand tout tient : une ligne d'embedding par token se lit sans coût sur CPU, et la
    VRAM ainsi libérée va au cache d'experts (02/10).
    """
    if not torch.cuda.is_available():
        return "cpu"
    free = torch.cuda.mem_get_info()[0]
    if dense_bytes + vram_reserve <= free:
        return "split" if head_on_gpu and emb_bytes >= SPLIT_EMB_BYTES else "gpu"
    if dense_bytes - emb_head_bytes + vram_reserve <= free:
        return "split"
    return "cpu"


def _setup_generation(model, model_dir) -> None:
    """
    Réglages de generate() : ceux du modèle (generation_config.json : tokens de fin,
    etc.), plus le cache KV forcé. Certains checkpoints, dont OLMoE, ont
    "use_cache": false dans config.json : sans cache KV, generate() recalcule TOUTE
    la séquence à chaque token, donc redemande presque tous les experts à chaque
    pas (caches inutiles, vitesse qui s'effondre avec la longueur).
    """
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_dir)
    except (OSError, ValueError):
        pass  # pas de generation_config.json : on garde celui dérivé de config.json
    model.config.use_cache = True
    model.generation_config.use_cache = True


def load_offloaded_model(
    model_dir,
    store_dir,
    cfg: RuntimeConfig | None = None,
    dense: str = "auto",
    dtype: torch.dtype = torch.float16,
    vram_reserve_gb: float | None = None,
    ram_reserve_gb: float = 3.0,
    log=print,
    dense_int8: bool = False,
):
    """
    Retourne (model, runtime). `model` s'utilise comme un modèle transformers normal
    (model(...), model.generate(...)). Les entrées doivent être sur `model.moelab_device`.

    Si cfg.fast_budget_bytes / cfg.host_budget_bytes valent None, ils sont calculés
    d'après la mémoire réellement libre après le chargement de la partie dense
    (auto-dimensionnement, voir plan.py). cfg.graph_max_len = None et
    vram_reserve_gb = None : longueur max et réserve VRAM d'après le cache KV du
    modèle (plan.auto_graph_max_len / auto_vram_reserve).
    """
    from .plan import auto_budgets, auto_graph_max_len, auto_vram_reserve, kv_bytes_per_token

    cfg = cfg or RuntimeConfig(fast_budget_bytes=None, host_budget_bytes=None)
    config = load_config(model_dir)
    graph_len_auto = auto_graph_max_len(config)
    if cfg.graph_max_len is None:
        cfg.graph_max_len = graph_len_auto
    if vram_reserve_gb is None:
        vram_reserve_gb = auto_vram_reserve(config, cfg.graph_max_len) / 2**30
        log(f"  réserve VRAM auto : {vram_reserve_gb:.2f} Gio (activations + cache KV de {cfg.graph_max_len} tokens, "
            f"{kv_bytes_per_token(config) // 1024} Ko/token)")
    store = ExpertStore(store_dir)
    model = build_skeleton(config, dtype)

    experts = iter_experts_modules(model)
    names = [n for n, _ in experts]
    if names != store.layers:
        raise ValueError(
            f"le store {store.path} ne correspond pas à ce modèle ({len(store.layers)} couches MoE dans le store, "
            f"{len(names)} dans le modèle). Reconstruis-le avec scripts/build_store.py."
        )
    placeholders: list[OffloadedExperts] = []
    gates: dict[int, nn.Module] = {}
    act_fns: dict[int, nn.Module] = {}
    for name, m in experts:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        li = store.layer_index(name)
        new = OffloadedExperts(None, li, m.act_fn, m.num_experts)
        setattr(parent, attr, new)
        placeholders.append(new)
        act_fns[li] = m.act_fn
        if isinstance(getattr(parent, "gate", None), nn.Module):
            gates[li] = parent.gate

    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    emb_ids = {id(p) for p in emb.parameters()}
    head_ids = {id(p) for p in head.parameters()} if head is not None else set()
    params = list(model.named_parameters())
    dense_bytes = _param_bytes(p for _, p in params)
    emb_head_bytes = _param_bytes(p for _, p in params if id(p) in emb_ids | head_ids)
    from .denseq import quantizable, quantize_dense

    qnames = {f"{n}.weight" for n, _ in quantizable(model)} if dense_int8 else set()
    if dense_int8:  # D5 : place estimée après passage en int8 (~0,53 octet par octet fp16)
        lin_b = _param_bytes(p for n, p in params if n in qnames)
        head_b = _param_bytes(p for n, p in params if n in qnames and id(p) in head_ids)
        dense_bytes_eff, emb_head_eff = dense_bytes - int(0.47 * lin_b), emb_head_bytes - int(0.47 * head_b)
    else:
        dense_bytes_eff, emb_head_eff = dense_bytes, emb_head_bytes
    if dense == "auto":
        emb_b = _param_bytes(emb.parameters())
        head_gpu = dense_int8 and head is not None and not getattr(config, "tie_word_embeddings", False) and any(n in qnames for n, p in params if id(p) in head_ids)
        dense = choose_dense_placement(dense_bytes_eff, emb_head_eff, int(vram_reserve_gb * 2**30), emb_b, head_gpu)
    if dense not in ("gpu", "cpu", "split"):
        raise ValueError(f"placement dense inconnu : {dense}")
    if dense in ("gpu", "split") and not torch.cuda.is_available():
        raise RuntimeError(f"placement '{dense}' demandé mais CUDA n'est pas disponible")
    main = torch.device("cuda" if dense in ("gpu", "split") else "cpu")
    log(f"  partie dense : {dense_bytes / 2**30:.2f} Gio, placement '{dense}' (couches sur {main})")

    ckpt = Checkpoint(model_dir)
    tied = bool(getattr(config, "tie_word_embeddings", False))
    missing = []
    if dense_int8 and main.type != "cuda":
        log("  dense_int8 avec la partie dense sur CPU : exact mais plus lent (déquantification à chaque passe)")
    for pname, p in params:
        # les poids qui passeront en int8 restent d'abord en RAM : ils sont quantifiés un par un
        on_cpu = (dense == "split" and id(p) in emb_ids | head_ids) or pname in qnames
        dev = torch.device("cpu") if on_cpu else main
        if not ckpt.has(pname):
            if id(p) in head_ids and tied:
                continue
            missing.append(pname)
            continue
        _set_param(model, pname, ckpt.get(pname).to(dtype=p.dtype).to(dev))
    if missing:
        raise KeyError(f"{len(missing)} poids absents du checkpoint, par ex. {missing[:5]}")
    if tied and head is not None:
        head.weight = emb.weight

    for name, mod in list(model.named_modules()):
        if any(b.is_meta for b in mod.buffers(recurse=False)):
            if not type(mod).__name__.endswith("RotaryEmbedding"):
                raise NotImplementedError(f"tampon non initialisé dans {name} ({type(mod).__name__})")
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, type(mod)(mod.config).to(main))

    if dense_int8:  # D5 : y compris la tête de sortie, qui reste alors sur le GPU même en placement split
        quantize_dense(model, main, log=log)
    if dense == "split":
        model.set_input_embeddings(CpuEmbedding(emb.weight.data, getattr(emb, "padding_idx", None), out_device=main))
        if head is not None and isinstance(model.get_output_embeddings(), nn.Linear):
            model.set_output_embeddings(CpuLinear(model.get_input_embeddings().weight.data if tied else head.weight.data))

    known = {n for n, _ in params}
    extra = [k for k in ckpt.keys() if ".experts." not in k and not k.endswith("_scale_inv") and k not in known]
    if extra:
        log(f"  {len(extra)} tenseurs du checkpoint non utilisés (ex. {extra[0]}), c'est normal pour les têtes MTP")
    ckpt.close()

    if cfg.fast_budget_bytes is None or cfg.host_budget_bytes is None:
        auto_budgets(cfg, main, vram_reserve_gb, ram_reserve_gb)
    if cfg.fast_device is None and main.type == "cuda" and cfg.fast_budget_bytes:
        cfg.fast_device = "cuda"
    runtime = MoERuntime(store, cfg)
    runtime.auto_graph_max_len = graph_len_auto
    runtime.load_fast_budget = cfg.fast_budget_bytes or 0  # budget VRAM au chargement (plancher du plafond de l'autopilot)
    runtime.gates = gates
    runtime.act_fns = act_fns
    for ph in placeholders:
        ph.runtime = runtime

    def _tick(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            ids = kwargs.get("inputs_embeds")
        runtime.on_forward(int(ids.shape[1]) if ids is not None and ids.dim() > 1 else 1)

    model.register_forward_pre_hook(_tick, with_kwargs=True)
    _setup_generation(model, model_dir)
    model.eval()
    model.moelab_runtime = runtime
    model.moelab_device = main
    model.moelab_dense = dense
    model.moelab_dense_int8 = dense_int8
    return model, runtime
