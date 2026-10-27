"""
Données de l'élagage (08/10) : jeu de calibration REAP et jeu d'évaluation, figés sur disque.

Les deux jeux sont construits une fois, puis enregistrés en tokens (`results/reap/*.pt`) :
toutes les mesures (référence, chaque taux d'élagage) utilisent exactement les mêmes
échantillons, même si les jeux de données en ligne changent entre-temps.

Évaluation (disjointe de la calibration) :
  - WikiText-2, partie test : 50 extraits de 512 tokens répartis sur tout le texte ;
  - Wikipédia en français : 20 extraits de 512 tokens, pris dans les FR_EVAL_ARTICLES
    premiers articles ; la calibration commence après FR_CALIB_SKIP articles.

Calibration (graine fixe), mélange de domaines comme dans REAP (le domaine compte beaucoup) :
  texte général (WikiText-2 train), français (Wikipédia), code (HumanEval, MBPP),
  maths (GSM8K), conversation (UltraChat), et TES données : conversations et runs d'agents
  de l'app (app_data), rendus avec le modèle de chat, appels d'outils compris. Une source
  indisponible (pas de réseau, pas de données locales) : sa part passe aux suivantes.

Les jeux en ligne passent par la bibliothèque `datasets` (pip install datasets), en
streaming : seuls les morceaux lus sont téléchargés.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch

SEQ_LEN = 512
FR_EVAL_ARTICLES = 400
FR_CALIB_SKIP = 2000

# part de chaque source ; l'ordre compte : une part manquante passe aux sources suivantes,
# donc les plus sûres (Wikipédia) viennent en dernier
MIX = {"local": 0.10, "chat": 0.15, "code": 0.20, "math": 0.15, "wiki_fr": 0.20, "wiki_en": 0.20}


def _need_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError as e:
        raise SystemExit("Il faut la bibliothèque datasets : pip install datasets") from e
    from datasets import load_dataset

    return load_dataset


# ----------------------------------------------------------------------------
# Sources de texte (générateurs de chaînes)
# ----------------------------------------------------------------------------
def wikitext_text(split: str) -> str:
    load = _need_datasets()
    ds = load("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return "".join(ds["text"])


def frwiki_texts(start: int, seed: int | None = None) -> Iterator[str]:
    load = _need_datasets()
    ds = load("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    ds = ds.skip(start)
    if seed is not None:
        ds = ds.shuffle(seed=seed, buffer_size=2000)
    for row in ds:
        t = row.get("text") or ""
        if len(t) > 500:
            yield t


def code_texts(seed: int) -> Iterator[str]:
    load = _need_datasets()
    rows = []
    for r in load("openai/openai_humaneval", split="test"):
        rows.append(r["prompt"] + r["canonical_solution"])
    for split in ("train", "test", "validation"):
        for r in load("google-research-datasets/mbpp", "full", split=split):
            rows.append(f'"""{r["text"]}"""\n{r["code"]}\n')
    random.Random(seed).shuffle(rows)
    yield from rows


def math_texts(seed: int) -> Iterator[str]:
    load = _need_datasets()
    ds = load("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    for r in ds:
        yield f"Question : {r['question']}\nRéponse : {r['answer']}"


def chat_texts(tok, seed: int) -> Iterator[str]:
    load = _need_datasets()
    ds = load("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).shuffle(seed=seed, buffer_size=2000)
    for r in ds:
        yield render_chat(tok, r["messages"])


def render_chat(tok, messages: list[dict]) -> str:
    msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in messages
            if m.get("role") in ("system", "user", "assistant", "tool")]
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(msgs, tokenize=False)
        except Exception:
            pass
    return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)


def _strings(obj, out: list[str], min_len: int = 40) -> None:
    if isinstance(obj, str):
        if len(obj) >= min_len:
            out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _strings(v, out, min_len)
    elif isinstance(obj, list):
        for v in obj:
            _strings(v, out, min_len)


def local_texts(tok, data_dir) -> list[str]:
    """Conversations (rendues avec le modèle de chat) et runs d'agents de l'app."""
    data_dir = Path(data_dir)
    out: list[str] = []
    for f in sorted((data_dir / "conversations").glob("*.json")):
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        msgs = list(c.get("messages") or [])
        if c.get("systeme"):
            msgs.insert(0, {"role": "system", "content": c["systeme"]})
        if len(msgs) >= 2:
            out.append(render_chat(tok, msgs))
    for f in sorted((data_dir / "agent_runs").glob("*.json")):
        try:
            run = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        parts: list[str] = []
        _strings(run, parts)
        if parts:
            out.append("\n\n".join(parts))
    return out


# ----------------------------------------------------------------------------
# Découpage en échantillons de SEQ_LEN tokens
# ----------------------------------------------------------------------------
def pack(texts: Iterable[str], tok, n: int, seq_len: int = SEQ_LEN) -> list[list[int]]:
    """Textes mis bout à bout (séparés par le token de fin), coupés en n échantillons de seq_len."""
    eos = tok.eos_token_id
    buf: list[int] = []
    out: list[list[int]] = []
    if n <= 0:
        return out
    for t in texts:
        if not t:
            continue
        buf += tok(t, add_special_tokens=False).input_ids
        if eos is not None:
            buf.append(eos)
        while len(buf) >= seq_len and len(out) < n:
            out.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(out) >= n:
            break
    return out


def spread_chunks(ids: list[int], n: int, seq_len: int = SEQ_LEN) -> list[list[int]]:
    """n extraits sans recouvrement, répartis régulièrement sur un long texte."""
    total = len(ids) // seq_len
    if total == 0:
        return []
    picks = sorted({int(i * total / n) for i in range(n)}) if n < total else range(total)
    return [ids[i * seq_len:(i + 1) * seq_len] for i in picks]


def build_calibration(tok, n: int, seq_len: int = SEQ_LEN, seed: int = 0, data_dir=None,
                      sources: dict[str, Callable[[], Iterable[str]]] | None = None, mix: dict | None = None,
                      log=print) -> dict:
    """
    {"ids": tenseur (n, seq_len) int32, "sources": [nom de la source de chaque échantillon], ...}.
    `sources` remplace les sources par défaut (tests : aucun téléchargement).
    """
    mix = dict(mix or MIX)
    if sources is None:
        sources = {
            "local": (lambda: local_texts(tok, data_dir)) if data_dir else (lambda: []),
            "chat": lambda: chat_texts(tok, seed),
            "code": lambda: code_texts(seed),
            "math": lambda: math_texts(seed),
            "wiki_fr": lambda: frwiki_texts(FR_CALIB_SKIP, seed),
            "wiki_en": lambda: [wikitext_text("train")],
        }
    names = [s for s in mix if s in sources]
    total_w = sum(mix[s] for s in names)
    quotas = {s: int(round(n * mix[s] / total_w)) for s in names}
    quotas[names[-1]] += n - sum(quotas.values())
    samples: list[list[int]] = []
    labels: list[str] = []
    counts: dict[str, int] = {}
    carry = 0
    for s in names:
        want = quotas[s] + carry
        try:
            got = pack(sources[s](), tok, want, seq_len)
        except SystemExit:
            raise
        except Exception as e:  # réseau, jeu renommé... : la part passe aux sources suivantes
            log(f"  source {s} indisponible ({e.__class__.__name__}: {str(e)[:120]}) : sa part passe aux suivantes")
            got = []
        samples += got
        labels += [s] * len(got)
        counts[s] = len(got)
        carry = want - len(got)
        log(f"  {s:8s} : {len(got)} échantillon(s)")
    if carry > 0:
        raise RuntimeError(f"pas assez de texte : {len(samples)}/{n} échantillons")
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)  # les lots mélangent les domaines
    ids = torch.tensor([samples[i] for i in order], dtype=torch.int32)
    return {"ids": ids, "sources": [labels[i] for i in order], "counts": counts, "seed": seed, "seq_len": seq_len}


def build_eval(tok, n_en: int = 50, n_fr: int = 20, seq_len: int = SEQ_LEN,
               sources: dict[str, Callable[[], Iterable[str]]] | None = None) -> dict:
    """{"wikitext": (n_en, seq_len), "fr": (n_fr, seq_len)} en int32."""
    if sources is None:
        sources = {"wikitext": lambda: [wikitext_text("test")],
                   "fr": lambda: _head(frwiki_texts(0), FR_EVAL_ARTICLES)}
    out = {}
    en_ids = []
    for t in sources["wikitext"]():
        en_ids += tok(t, add_special_tokens=False).input_ids
    out["wikitext"] = torch.tensor(spread_chunks(en_ids, n_en, seq_len), dtype=torch.int32)
    out["fr"] = torch.tensor(pack(sources["fr"](), tok, n_fr, seq_len), dtype=torch.int32)
    for k, v in out.items():
        if v.numel() == 0:
            raise RuntimeError(f"jeu d'évaluation {k} vide")
    out["seq_len"] = seq_len
    return out


def _head(it: Iterable[str], n: int) -> Iterator[str]:
    for i, t in enumerate(it):
        if i >= n:
            return
        yield t


def load_or_build(path, build: Callable[[], dict], log=print, rebuild: bool = False) -> dict:
    """Jeu figé sur disque : construit une fois, relu ensuite (mêmes échantillons pour toutes les mesures)."""
    path = Path(path)
    if path.exists() and not rebuild:
        return torch.load(path, map_location="cpu", weights_only=False)
    log(f"Construction de {path.name} (une seule fois) ...")
    data = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data
