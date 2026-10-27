"""
Boucle de génération du projet (remplace model.generate pour chat, benchmark et profil).

Deux décodeurs, même interface :
  - EagerDecoder : la passe transformers normale, cache KV dynamique ;
  - GraphDecoder (graphs.py) : partie dense en CUDA graphs, cache KV statique (A6).

make_decoder(model) choisit selon RuntimeConfig.cuda_graphs. Le token choisi reste sur
le GPU d'un pas à l'autre (une seule synchronisation par token : la lecture du token
pour l'affichage).
"""

from __future__ import annotations

import torch

from .graphs import GraphDecoder


class EagerDecoder:
    max_len = None
    active = False
    failed = None

    def __init__(self, model):
        self.model = model
        self.past = None

    reused = 0

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, reuse: bool = False, prefix_len: int | None = None) -> torch.Tensor:
        out = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
        self.past = out.past_key_values
        return out.logits[:, -1]

    @torch.no_grad()
    def step(self, tok: torch.Tensor, tok_id: int | None = None) -> torch.Tensor:
        out = self.model(input_ids=tok, past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        return out.logits[:, -1]

    def describe(self) -> str:
        return "CUDA graphs : non (passe normale)"


def make_decoder(model, log=None):
    """GraphDecoder si cuda_graphs est activé et que le modèle est sur GPU, sinon EagerDecoder."""
    rt = model.moelab_runtime
    if rt.cfg.cuda_graphs and model.moelab_device.type == "cuda":
        dec = getattr(model, "moelab_graph_decoder", None)
        min_bucket = 128 if rt.cfg.graph_buckets else rt.cfg.graph_max_len
        if dec is not None and (dec.max_len != rt.cfg.graph_max_len or dec.buckets[0] != min(min_bucket, rt.cfg.graph_max_len)):
            dec.release()
            dec = None
        if dec is None:
            dec = GraphDecoder(model, rt.cfg.graph_max_len, min_bucket=min_bucket)
            model.moelab_graph_decoder = dec
        if dec.failed is None:
            dec.pack_routing = rt.cfg.graph_pack_routing
            return dec
        if dec.cache is not None:
            if log:
                log(dec.describe())
            dec.release()
    return EagerDecoder(model)


def pick(logits: torch.Tensor, temperature: float = 0.0, top_p: float = 1.0) -> torch.Tensor:
    """Token suivant, (1, 1) sur le device des logits : glouton si temperature = 0, sinon échantillonnage top-p."""
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True).view(1, 1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sp, si = probs.sort(dim=-1, descending=True)
        keep = sp.cumsum(-1) - sp < top_p
        sp = sp * keep
        choice = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1)
        return si.gather(-1, choice).view(1, 1)
    return torch.multinomial(probs, 1).view(1, 1)


@torch.no_grad()
def generate(model, ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.0, top_p: float = 1.0,
             eos_token_id=None, streamer=None, decoder=None, log=None, reuse: bool = False,
             timings: dict | None = None, prefix_len: int | None = None, prefix_dir: str | None = None) -> list[int]:
    """Génère jusqu'à max_new_tokens tokens (arrêt au token de fin). Retourne les nouveaux tokens.
    reuse (P1, conversation) : reprendre du cache le début du prompt déjà calculé au tour précédent.
    timings : rempli avec prompt_tokens, reused (tokens repris du cache) et ttft_s (premier token prêt).
    prefix_len, prefix_dir (P9) : début fixe du prompt gardé en mémoire et sur disque (agents)."""
    import time

    t0 = time.perf_counter()
    dec = decoder or make_decoder(model, log=log)
    if eos_token_id is None:
        eos_token_id = getattr(model.generation_config, "eos_token_id", None)
    eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]) - {None}
    if dec.max_len is not None and ids.shape[1] >= dec.max_len - 1:
        if log:
            log(f"(prompt de {ids.shape[1]} tokens trop long pour le cache statique de {dec.max_len} : passe normale)")
        dec = EagerDecoder(model)
    if dec.max_len is not None:
        room = dec.max_len - ids.shape[1]
        if room < max_new_tokens and log:
            log(f"(réponse limitée à {room} tokens par le cache statique de {dec.max_len} tokens)")
        max_new_tokens = min(max_new_tokens, room)
    if streamer is not None:
        streamer.put(ids.cpu())
    if prefix_dir is not None and hasattr(dec, "prefix_dir"):
        rt = model.moelab_runtime
        dec.prefix_dir = prefix_dir
        dec.prefix_key = f"{rt.store.store_id}|{rt.policy.describe()}|{getattr(model.config, '_name_or_path', '')}"
    logits = dec.prefill(ids, reuse=reuse, prefix_len=prefix_len)
    tokens: list[int] = []
    for i in range(max_new_tokens):
        nxt = pick(logits, temperature, top_p)
        t = int(nxt)
        if i == 0 and timings is not None:
            timings.update(prompt_tokens=int(ids.shape[1]), reused=int(getattr(dec, "reused", 0)),
                           prefix_hit=bool(getattr(dec, "prefix_hit", False)), ttft_s=time.perf_counter() - t0)
        tokens.append(t)
        if streamer is not None:
            streamer.put(torch.tensor([t]))
        if t in eos or i == max_new_tokens - 1:
            break
        logits = dec.step(nxt, t)
    if streamer is not None:
        streamer.end()
    return tokens


@torch.no_grad()
def warmup(model, tokenizer, log=None) -> float:
    """
    Échauffement au chargement (05/10) : un petit prompt et 2 tokens, pour que la compilation
    des noyaux Triton, l'initialisation de cuBLAS et la capture des CUDA graphs se fassent
    pendant le chargement plutôt qu'au premier message. Retourne la durée (s).
    """
    import time

    rt = model.moelab_runtime
    t0 = time.perf_counter()
    text = "Bonjour, peux-tu te présenter en une phrase ? Merci beaucoup, c'est pour un test."
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                            return_tensors="pt", return_dict=False)
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
    else:
        ids = tokenizer(text, return_tensors="pt").input_ids
    generate(model, ids.to(model.moelab_device), 3, eos_token_id=-1)
    dec = getattr(model, "moelab_graph_decoder", None)
    if dec is not None:
        dec.tokens = None  # rien à reprendre de ce prompt-là
    rt.reset_stats()
    dt = time.perf_counter() - t0
    if log:
        log(f"  échauffement (noyaux, CUDA graphs) : {dt:.1f} s")
    return dt
