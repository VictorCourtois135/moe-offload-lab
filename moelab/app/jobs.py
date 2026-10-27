"""
Les jobs longs de l'app (05/10) : autopilot et tâches de fond.

Autopilot : le moteur lui est réservé le temps du run (une demande de chat attend la fin).
Tâches de fond : une instruction appliquée à chaque fichier d'un dossier (« résume »,
« extrais les montants »...), résultats dans un fichier Markdown. La tâche se découpe : un
fichier par passage dans la file, pour qu'un message de chat passe entre deux fichiers.
"""

from __future__ import annotations

import fnmatch
import time
from pathlib import Path

from .engine import CONTINUE, Cancelled, TextSink


def autopilot_job(trials: int = 12, minutes: float = 60, start: str = "actuel", blind: bool = False,
                  prompts: int = 3, gen: int = 24):
    def run(job, engine):
        from .. import options
        from ..agent import moelab_generate_fn
        from ..autopilot import Autopilot, Bench, start_0210
        from ..bench import load_prompts

        model, tok, rt = engine.model, engine.tok, engine.rt
        if start == "02-10":
            for k, v in start_0210(rt).items():
                job.emit("log", texte="départ 02/10 : " + options.apply(model, k, v))
        texts = load_prompts()
        bench = Bench(model, tok, texts[1:1 + prompts], gen=gen, warm=texts[0])

        def log(*a):
            msg = " ".join(str(x) for x in a)
            job.emit("log", texte=msg)
            if job.cancel_requested:
                raise Cancelled()

        pilot = Autopilot(model, tok, bench, max_trials=trials, minutes=minutes, blind=blind, log=log)
        orig_experiment, orig_baseline = pilot.experiment, pilot.baseline

        def experiment(changes):
            r = orig_experiment(changes)
            if "essai" in r:
                job.progress = min(1.0, r["essai"] / trials)
                job.emit("trial", ligne=pilot.journal_rows()[-1], meilleur=pilot.best_res, config=pilot.best)
            return r

        def baseline():
            r = orig_baseline()
            job.emit("trial", ligne=pilot.journal_rows()[-1], meilleur=pilot.best_res, config=pilot.best)
            return r

        pilot.experiment, pilot.baseline = experiment, baseline
        sink = TextSink(tok, job)
        gen_fn = moelab_generate_fn(model, tok, max_new_tokens=400, streamer=sink)

        def on_step(step):
            for c, r in zip(step.calls, step.results):
                job.emit("tool", nom=c.get("name"), arguments=c.get("arguments", {}), resultat=r[:2000])
            job.emit("turn")

        summary = pilot.run(generate_fn=gen_fn, on_step=on_step)
        out = getattr(engine, "results_dir", None) or Path(engine.data_dir) / "results"
        ppath, rpath = pilot.save(out, engine.model_info.get("name", "modele"), write_profile=start == "actuel")
        summary["profil"] = str(ppath) if ppath else None
        summary["rapport_fichier"] = str(rpath)
        return summary

    return run


TEXT_EXT = (".txt", ".md", ".py", ".json", ".csv", ".html", ".xml", ".yaml", ".yml", ".log", ".pdf", ".rst", ".tex")


def batch_job(folder: str, pattern: str, instruction: str, output: str, max_chars: int = 12000,
              max_new_tokens: int = 500):
    """Tâche de fond : `instruction` appliquée à chaque fichier de `folder` qui correspond à `pattern`."""

    def run(job, engine):
        from ..agent import moelab_generate_fn
        from .tools import read_text_file

        st = job.state
        if "files" not in st:
            pats = [p.strip() for p in pattern.split(";") if p.strip()] or ["*"]
            files = [str(f) for f in sorted(Path(folder).expanduser().rglob("*"))
                     if f.is_file() and f.suffix.lower() in TEXT_EXT and any(fnmatch.fnmatch(f.name, p) for p in pats)]
            st.update(files=files, i=0, t0=time.time())
            out = Path(output).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f"# {instruction}\n\nDossier : {folder} ({len(files)} fichiers)\n\n", encoding="utf-8")
            job.emit("log", texte=f"{len(files)} fichiers à traiter -> {out}")
            if not files:
                return {"fichiers": 0, "sortie": str(out)}
        files, i = st["files"], st["i"]
        f = Path(files[i])
        job.emit("step", texte=f"{i + 1}/{len(files)} : {f.name}")
        try:
            text = read_text_file(f)[:max_chars]
            sink = TextSink(engine.tok, job)
            gen = moelab_generate_fn(engine.model, engine.tok, max_new_tokens, temperature=0.3, streamer=sink,
                                     prefix_cache=False)
            answer = gen([{"role": "user", "content": f"{instruction}\n\nFichier : {f.name}\n\n---\n{text}"}], [])
        except Cancelled:
            raise
        except Exception as e:
            answer = f"(erreur : {e})"
        with open(Path(output).expanduser(), "a", encoding="utf-8") as fh:
            fh.write(f"## {f.name}\n\n{answer}\n\n")
        engine.rt.end_of_request()
        st["i"] = i + 1
        job.progress = st["i"] / len(files)
        job.emit("progress", fait=st["i"], total=len(files))
        if st["i"] < len(files):
            return CONTINUE
        return {"fichiers": len(files), "sortie": str(Path(output).expanduser()), "minutes": round((time.time() - st["t0"]) / 60, 1)}

    return run
