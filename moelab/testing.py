"""
Modèles miniatures aléatoires (quelques centaines de Ko) : même code transformers
que les vrais OLMoE / Qwen3-Next, même format de checkpoint sur disque, dimensions
minuscules. Servent aux tests et à scripts/check_setup.py (qui vérifie le chemin
CUDA sur la vraie machine sans rien télécharger).
"""

from __future__ import annotations

from pathlib import Path

import torch


def tiny_olmoe_config(**kw):
    from transformers import OlmoeConfig

    base = dict(vocab_size=128, hidden_size=64, intermediate_size=32, num_hidden_layers=3,
                num_attention_heads=4, num_key_value_heads=4, num_experts=8, num_experts_per_tok=3,
                max_position_embeddings=128, eos_token_id=1, pad_token_id=0, bos_token_id=None)
    base.update(kw)
    return OlmoeConfig(**base)


def tiny_qwen3next_config(**kw):
    from transformers import Qwen3NextConfig

    base = dict(vocab_size=128, hidden_size=64, intermediate_size=64, moe_intermediate_size=32,
                shared_expert_intermediate_size=32, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, num_experts=16, num_experts_per_tok=4,
                linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
                linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4,
                max_position_embeddings=128, decoder_sparse_step=1, mlp_only_layers=[],
                eos_token_id=1, pad_token_id=0, bos_token_id=None)
    base.update(kw)
    return Qwen3NextConfig(**base)


def tiny_qwen3moe_config(**kw):
    """Famille de Qwen3-30B-A3B (qwen3_moe) : attention complète, experts sans expert partagé."""
    from transformers import Qwen3MoeConfig

    base = dict(vocab_size=128, hidden_size=64, intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=3,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=16, num_experts_per_tok=4,
                max_position_embeddings=128, decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True,
                eos_token_id=1, pad_token_id=0, bos_token_id=None)
    base.update(kw)
    return Qwen3MoeConfig(**base)


def make_model_dir(path: str | Path, config, seed: int = 0) -> Path:
    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.save_pretrained(path)
    return Path(path)


def reference_logits(model_dir: str | Path, ids: torch.Tensor) -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        return ref(ids).logits


CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def make_tiny_tokenizer(path: str | Path, vocab_size: int = 128):
    """Tokenizer caractère par caractère (aucun téléchargement), avec un modèle de chat minimal."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<eos>", "<unk>"]
    chars = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?'\"()-_\n/*+=<>|%éèêàùçôî")
    vocab = {t: i for i, t in enumerate(specials + chars[: vocab_size - len(specials)])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tk.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(path)
    return tok


def make_demo_model(path: str | Path, arch: str = "olmoe", seed: int = 0) -> Path:
    """Modèle miniature + tokenizer : de quoi faire tourner tous les scripts sans rien télécharger."""
    cfg = {"olmoe": tiny_olmoe_config, "qwen3moe": tiny_qwen3moe_config}.get(arch, tiny_qwen3next_config)()
    make_model_dir(path, cfg, seed)
    make_tiny_tokenizer(path, cfg.vocab_size)
    return Path(path)
