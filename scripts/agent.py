"""
Agent local, étape 1 du projet « moelab autopilot » : le modèle chargé par moelab
appelle des outils, sans internet et sans service payant.

    python scripts\\agent.py --model D:\\models\\qwen3-next-80b-fp8 --store C:\\moe-store\\qwen3-next --ram-gb 5

Outils de cette première version (moelab se décrit et se règle lui-même) :
    etat()                     budgets, caches, dernières statistiques
    reglages(groupe)           réglages, valeur actuelle, effet mesuré
    changer_reglage(cle, val)  réglage « à chaud » (pas ceux qui demandent un rechargement)
    calculer(expression)       arithmétique exacte (le modèle compte mal de tête)

Exemples : « Dans quel état est le moteur ? », « Quels réglages touchent au prefill ? »,
« Passe en politique hobbit avec t1 = 0,7 puis dis-moi ce que ça change ».
Commandes : /reset (oublier la conversation), /quit.
"""

from __future__ import annotations

import argparse
import ast
import operator
import time

import _common
from transformers import TextStreamer

from moelab import options
from moelab.agent import Agent, moelab_generate_fn, native_tools, tool

SYSTEM = (
    "Tu es l'assistant de moelab, un moteur qui fait tourner de gros modèles MoE sur un PC modeste "
    "(experts répartis entre VRAM, RAM et disque). Tu tournes toi-même sur ce moteur. Utilise les outils "
    "pour connaître l'état réel du moteur et ses réglages au lieu de deviner, et pour tout calcul. "
    "Réponds en français, brièvement."
)

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
        ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("seuls les nombres et + - * / // % ** sont permis")

    return ev(ast.parse(expr.replace(",", ".").replace("^", "**"), mode="eval").body)


def make_tools(model, rt):
    @tool
    def etat() -> dict:
        """État du moteur : budgets et remplissage des caches VRAM/RAM, politique, statistiques de la question en cours."""
        s = rt.stats.as_dict()
        fast = rt.fast
        out = {"politique": rt.policy.describe(),
               "vram_experts": {"utilise_mio": round(fast.used / 2**20) if fast else 0,
                                "budget_mio": round(fast.budget / 2**20) if fast else 0,
                                "experts": len(fast.entries) if fast else 0},
               "ram_experts": {"utilise_mio": round(rt.host.used / 2**20), "budget_mio": round(rt.host.budget / 2**20),
                               "experts": len(rt.host.entries)},
               "question_en_cours": {
                   "_note": "depuis le début de la question, lecture du prompt comprise",
                   "tokens_traites": s["tokens"],
                   "experts_trouves_en_vram_pourcent": round(100 * s["fast_hit_rate"], 1),
                   "experts_lus_sur_disque_par_token": round(s["disk_per_token"], 1),
                   "experts_calcules_sur_cpu_pourcent": round(100 * s["cpu_share"], 1)}}
        if rt.pool.fallbacks:
            out["pool_ram"] = f"{rt.pool.fallbacks} tampons pris hors du pool épinglé (copies vers la VRAM plus lentes)"
        return out

    @tool
    def reglages(groupe: str = "") -> list:
        """Réglages de moelab : valeur actuelle, application (live = à chaud, reload = rechargement), effet mesuré.

        Args:
            groupe: precision, prefill, memory, gpu, io ou store ; vide = tous
        """
        cur = options.current(model)
        out = []
        for o in options.catalog():
            if groupe and o.group != groupe:
                continue
            d = {"cle": o.key, "valeur": cur.get(o.key, o.default), "application": o.apply}
            if o.measured:
                d["effet_mesure"] = o.measured
            out.append(d)
        return out

    @tool
    def changer_reglage(cle: str, valeur: str) -> str:
        """Change un réglage à chaud (application « live » ou « caches » seulement).

        Args:
            cle: la clé du réglage, ex. policy, t1, prefill_skip
            valeur: la nouvelle valeur, en texte, ex. hobbit, 0.7, on, off
        """
        return options.apply(model, cle, valeur)

    @tool
    def calculer(expression: str) -> float:
        """Calcule une expression arithmétique exacte.

        Args:
            expression: ex. (6.8 - 3.35) / 3.35 * 100
        """
        return safe_eval(expression)

    return [etat, reglages, changer_reglage, calculer]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common.add_model_args(ap)
    _common.add_policy_args(ap)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--max-steps", type=int, default=6, help="tours d'outils au plus par question")
    ap.add_argument("--no-profile", action="store_true",
                    help="ne pas appliquer <store>/autopilot_profile.json (réglages trouvés par l'autopilot)")
    ap.add_argument("--no-warmup", action="store_true",
                    help="pas d'échauffement au chargement (compilation des noyaux au 1er message)")
    args = ap.parse_args()

    print("Chargement ...")
    model, rt, tok = _common.load(args)
    if not args.no_profile:  # réglages trouvés par l'autopilot (doc 19) ; ceux donnés en ligne de commande priment
        from moelab.autopilot import apply_profile

        apply_profile(model, args.store, skip=tuple(k for k in ("vram_gb", "ram_gb", "read_parts")
                                                    if getattr(args, k, None) is not None), log=print)
    if not args.no_warmup:
        from moelab.generate import warmup

        warmup(model, tok, log=print)
    print(f"Outils décrits par le modèle de chat : {'oui' if native_tools(tok) else 'non (moelab les décrit lui-même)'}")
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    timings: list = []
    gen = moelab_generate_fn(model, tok, args.max_new_tokens, streamer=streamer, timings=timings)

    def on_step(step):
        for c, r in zip(step.calls, step.results):
            shown = r if len(r) < 400 else r[:400] + " ..."
            print(f"\n  [outil] {c.get('name')}({c.get('arguments', {})}) -> {shown}")

    agent = Agent(make_tools(model, rt), gen, max_steps=args.max_steps, on_step=on_step)
    messages = [{"role": "system", "content": SYSTEM}]
    print("Pose ta question (/reset, /quit).")
    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            messages = messages[:1]
            print("(conversation oubliée)")
            continue
        messages.append({"role": "user", "content": user})
        timings.clear()
        t0 = time.perf_counter()
        agent.run(messages)
        dt = time.perf_counter() - t0
        toks = sum(t["new_tokens"] for t in timings)
        firsts = ", ".join(f"{t['ttft_s']:.1f} s ({t['reused']}/{t['prompt_tokens']} repris"
                           + (", début fixe rechargé" if t.get("prefix_hit") else "") + ")" for t in timings)
        print(f"\n[{len(agent.steps)} tour(s), {toks} tokens générés, {dt:.0f} s au total ; 1er token par tour : {firsts}]")
    rt.end_of_request()
    rt.close()


if __name__ == "__main__":
    main()
