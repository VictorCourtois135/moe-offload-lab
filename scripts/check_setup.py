"""
Étape 0 : vérifie la machine ET le code, sans rien télécharger.

1. Versions, GPU, VRAM, RAM, espace disque.
2. Tests matériels : mémoire épinglée, copie RAM -> VRAM, support bfloat16.
3. Auto-test : un OLMoE miniature aléatoire passe par tout le pipeline (store,
   caches, q*, prefetch) sur CPU puis sur GPU, et ses sorties sont comparées au
   modèle transformers d'origine. Si tout est "OK", le code marche sur ta machine.

    python scripts\\check_setup.py --paths C:\\ D:\\
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import _common  # noqa: F401  (ajoute la racine du repo au chemin d'import)
import psutil
import torch

from moelab import PrecisionPolicy, RuntimeConfig, build_store, load_offloaded_model
from moelab.plan import disk_free
from moelab.testing import make_model_dir, reference_logits, tiny_olmoe_config

GIB = 2**30
MB = 2**20
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'OK' if ok else 'ÉCHEC'}] {name}" + (f" — {detail}" if detail else ""))


def environment(paths) -> None:
    import transformers

    print("Environnement")
    print(f"  torch {torch.__version__}, transformers {transformers.__version__}")
    vm = psutil.virtual_memory()
    print(f"  RAM : {vm.total / GIB:.1f} Gio au total, {vm.available / GIB:.1f} Gio disponibles")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU : {torch.cuda.get_device_name(0)} (capacité {'.'.join(map(str, torch.cuda.get_device_capability(0)))}), "
              f"VRAM {free / GIB:.1f}/{total / GIB:.1f} Gio libres")
    else:
        print("  GPU : CUDA indisponible (vérifie que torch vient de l'index cu130)")
    for p in paths:
        try:
            print(f"  disque {p} : {disk_free(p) / GIB:.1f} Gio libres")
        except OSError as e:
            print(f"  disque {p} : illisible ({e})")


def hardware_tests() -> None:
    print("Matériel")
    if not torch.cuda.is_available():
        check("CUDA", False, "tests GPU sautés")
        return
    try:
        buf = torch.empty(512 * MB, dtype=torch.uint8, pin_memory=True)
        dst = torch.empty_like(buf, device="cuda")
        dst.copy_(buf, non_blocking=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            dst.copy_(buf, non_blocking=True)
        torch.cuda.synchronize()
        bw = 3 * buf.numel() / (time.perf_counter() - t0)
        check("mémoire épinglée + copie asynchrone", True, f"{bw / 1e9:.1f} Go/s RAM -> VRAM")
        del buf, dst
    except RuntimeError as e:
        check("mémoire épinglée + copie asynchrone", False, str(e))
    try:
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        ok = torch.isfinite(a @ a).all().item()
        check("bfloat16 sur GPU", ok, "natif" if torch.cuda.is_bf16_supported(including_emulation=False) else "émulé (plus lent)")
    except RuntimeError as e:
        check("bfloat16 sur GPU", False, f"{e} -> utilise --dtype float16 ou --dense cpu --dtype bfloat16")
    from moelab.kernels import HAVE_TRITON

    if not HAVE_TRITON:
        print("  [--] Triton absent (optionnel, pour --triton-int4) : pip install triton-windows")
    else:
        try:
            from moelab.kernels import gemv_int4
            from moelab.quant import dequantize, quantize

            w = torch.randn(2048, 2048)
            q, s = quantize(w, 4, 64)
            q, s, x = q.cuda(), s.cuda(), torch.randn(2048, device="cuda", dtype=torch.float16)
            ref = dequantize(q, s, 4, torch.float32) @ x.float()
            err = ((gemv_int4(q, s, x, 64).float() - ref).abs().max() / ref.abs().max()).item()
            check("noyau Triton int4", err < 1e-2, f"écart relatif {err:.1e}")
        except Exception as e:
            check("noyau Triton int4", False, f"{e.__class__.__name__}: {str(e)[:200]}")
        try:  # noyaux groupés (A4+B1) : 8 experts int4 et 8 experts fp16 d'un coup
            import torch.nn.functional as F

            from moelab.kernels import grouped_ffn_float, grouped_ffn_int4, grouped_ffn_int8
            from moelab.runtime import expert_ffn
            from moelab.store import encode_record, materialize, record_layout, views

            x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
            for prec in ("int4", "int8", "fp16"):
                lay = record_layout(prec, 2048, 1024)
                ents, ref = [], torch.zeros(1, 2048, device="cuda")
                for _ in range(8):
                    blob = torch.from_numpy(encode_record(lay, torch.randn(2048, 2048) * 0.02, torch.randn(2048, 1024) * 0.02)).cuda()
                    ents.append((views(lay, blob), 0.125))
                    ref += 0.125 * expert_ffn(x.float(), *materialize(lay, blob, torch.float32), F.silu)
                for ver in ((1, 2, 3, 4) if prec == "int4" else (2,)):
                    def call():
                        if prec == "int4":
                            grouped_ffn_int4(x, ents, lay["group_sizes"], out, version=ver)
                        elif prec == "int8":
                            grouped_ffn_int8(x, ents, lay, out)
                        else:
                            grouped_ffn_float(x, ents, out)

                    out = torch.zeros(1, 2048, device="cuda", dtype=torch.float16)
                    call()  # 1er appel : autotune (v2), quelques secondes
                    err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(50):
                        call()
                    torch.cuda.synchronize()
                    ms = 1000 * (time.perf_counter() - t0) / 50
                    check(f"noyaux groupés v{ver} {prec}", err < 2e-2, f"écart relatif {err:.1e}, {ms:.3f} ms pour 8 experts")
        except Exception as e:
            check("noyaux groupés", False, f"{e.__class__.__name__}: {str(e)[:200]}")
        try:  # D5 : partie dense en int8, sur une matrice de la taille de la tête de sortie d'OLMoE
            from moelab.denseq import QLinear

            lin = torch.nn.Linear(2048, 50304, bias=False, device="cuda", dtype=torch.float16)
            ql = QLinear.from_linear(lin, "cuda")
            x = torch.randn(1, 1, 2048, device="cuda", dtype=torch.float16)
            ref = lin(x).float()
            err = ((ql(x).float() - ref).abs().max() / ref.abs().max()).item()
            times = []
            for f in (lin, ql):
                for _ in range(3):
                    f(x)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(50):
                    f(x)
                torch.cuda.synchronize()
                times.append(1000 * (time.perf_counter() - t0) / 50)
            check("partie dense int8 (--dense-int8)", err < 2e-2,
                  f"écart relatif {err:.1e} ; tête 50304x2048 : fp16 {times[0]:.3f} ms, int8 {times[1]:.3f} ms")
            del lin, ql
        except Exception as e:
            check("partie dense int8 (--dense-int8)", False, f"{e.__class__.__name__}: {str(e)[:200]}")


def self_test() -> None:
    print("Auto-test du pipeline (OLMoE miniature aléatoire)")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        md = make_model_dir(tmp / "model", tiny_olmoe_config())
        store = build_store(md, tmp / "store", ("fp32", "int4"), log=lambda *a: None)
        ids = torch.randint(2, 128, (1, 20), generator=torch.Generator().manual_seed(0))
        ref = reference_logits(md, ids)
        tiny = 64 * 32 * 3 * 4  # un record fp32 de ce modèle fait ~24 Kio

        def run(label, dense, **cfg):
            cfg.setdefault("host_budget_bytes", 32 * MB)
            try:
                model, rt = load_offloaded_model(md, store, RuntimeConfig(policy=PrecisionPolicy("hi"), **cfg),
                                                 dense=dense, dtype=torch.float32, log=lambda *a: None)
                with torch.no_grad():
                    out = model(ids.to(model.moelab_device)).logits.cpu()
                    gen = model.generate(ids[:, :6].to(model.moelab_device), max_new_tokens=8, do_sample=False).cpu()
                ref_model = __import__("transformers").AutoModelForCausalLM.from_pretrained(md, dtype=torch.float32).eval()
                with torch.no_grad():
                    ref_gen = ref_model.generate(ids[:, :6], max_new_tokens=8, do_sample=False)
                diff = (out - ref).abs().max().item()
                s = rt.stats
                check(label, diff < 1e-3 and torch.equal(gen, ref_gen),
                      f"écart max {diff:.1e}, {s.fills} transferts, {s.cpu_execs} experts sur CPU, "
                      f"{s.fast_hits} hits rapides, {s.prefetch_issued} prefetch")
                rt.close()
            except Exception as e:  # noqa: BLE001
                check(label, False, f"{type(e).__name__}: {e}")

        def graphs(label, dense, policy="hi", **cfg):
            """A6 : génération rejouée en CUDA graphs = génération normale, et plus rapide."""
            from moelab.generate import EagerDecoder
            from moelab.graphs import GraphDecoder

            try:
                model, rt = load_offloaded_model(md, store, RuntimeConfig(policy=PrecisionPolicy(policy, upgrade=False), host_budget_bytes=32 * MB,
                                                                          **cfg), dense=dense, dtype=torch.float32, log=lambda *a: None)
                x = ids[:, :6].to(model.moelab_device)

                def go(dec, n=24):
                    with torch.no_grad():
                        lg = [dec.prefill(x)]
                        nxt = lg[-1].argmax(-1, keepdim=True).view(1, 1)
                        for _ in range(2):  # échauffement : capture des graphs, autotune des noyaux
                            lg.append(dec.step(nxt))
                            nxt = lg[-1].argmax(-1, keepdim=True).view(1, 1)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for _ in range(n):
                            lg.append(dec.step(nxt))
                            nxt = lg[-1].argmax(-1, keepdim=True).view(1, 1)
                        torch.cuda.synchronize()
                        return torch.cat(lg).cpu(), 1000 * (time.perf_counter() - t0) / n

                ref, t_ref = go(EagerDecoder(model))
                dec = GraphDecoder(model, max_len=64)
                out, t_cg = go(dec)
                if dec.failed:
                    check(label, False, dec.failed[:300])
                else:
                    diff = (out - ref).abs().max().item()
                    same = torch.equal(out.argmax(-1), ref.argmax(-1))
                    n_gg = sum(1 for g in rt._ggraphs.values() if g is not None and g.replays)
                    a5 = ""
                    ok_a5 = True
                    if cfg.get("full_graph"):  # A5 : le token entier a bien servi
                        ok_a5 = dec.full_failed is None and rt.stats.full_tokens > 0
                        a5 = (f" ; A5 : {rt.stats.full_tokens} tokens en un graph, {rt.stats.full_fallbacks} refaits"
                              + (f" ({dec.full_failed[:200]})" if dec.full_failed else ""))
                    check(label, dec.active and same and diff < 1e-3 and ok_a5,
                          f"écart max {diff:.1e}, {len(dec.segs)} graphs + {n_gg} graphs de calcul groupé ; "
                          f"{t_ref:.1f} -> {t_cg:.1f} ms/token sur ce mini-modèle" + a5)
                rt.close()
            except Exception as e:  # noqa: BLE001
                check(label, False, f"{type(e).__name__}: {e}")

        run("CPU seul (RAM + disque)", "cpu")
        if torch.cuda.is_available():
            run("GPU : VRAM minuscule + q* + prefetch", "gpu", fast_device="cuda", fast_budget_bytes=6 * tiny,
                b_pcie=1.0, b_host=3.0)
            run("GPU : placement split (embeddings et tête sur CPU)", "split", fast_device="cuda",
                fast_budget_bytes=100 * tiny)
            graphs("CUDA graphs (--cuda-graphs) : même génération", "gpu", fast_device="cuda", fast_budget_bytes=100 * tiny)
            graphs("CUDA graphs + VRAM minuscule + q*", "gpu", fast_device="cuda", fast_budget_bytes=6 * tiny,
                   b_pcie=1.0, b_host=3.0)
            graphs("CUDA graphs, int4 (lo)", "gpu", policy="lo", fast_device="cuda", fast_budget_bytes=100 * tiny)
            graphs("CUDA graphs, int4 + fp32 (hobbit)", "gpu", policy="hobbit", fast_device="cuda", fast_budget_bytes=100 * tiny)
            graphs("A5 : token entier en un graph (lo)", "gpu", policy="lo", fast_device="cuda",
                   fast_budget_bytes=100 * tiny, full_graph=True, prefetch_adaptive=False)
            graphs("A5 : token entier en un graph (hobbit)", "gpu", policy="hobbit", fast_device="cuda",
                   fast_budget_bytes=100 * tiny, full_graph=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", nargs="*", default=[], help="disques ou dossiers dont afficher l'espace libre")
    ap.add_argument("--cpu-only", action="store_true", help="machine sans GPU : ne pas compter l'absence de CUDA comme une erreur")
    args = ap.parse_args()
    environment(args.paths)
    hardware_tests()
    self_test()

    problems = [f"{name} : {detail}" for name, ok, detail in results if not ok and name != "CUDA"]
    if not torch.cuda.is_available() and not args.cpu_only:
        problems.append(
            f"CUDA indisponible avec torch {torch.__version__}"
            + (" (version CPU : le GPU ne sera jamais utilisé)" if "+cpu" in torch.__version__ else "")
            + ". Correctif : pip uninstall -y torch puis "
            "pip install torch --index-url https://download.pytorch.org/whl/cu130"
        )
    avail = psutil.virtual_memory().available
    if avail < 6 * GIB:
        problems.append(
            f"seulement {avail / GIB:.1f} Gio de RAM disponibles : ferme les programmes gourmands "
            "(un noyau Jupyter qui garde un modèle chargé, des onglets du navigateur) avant de lancer le modèle"
        )
    print()
    if not problems:
        print("Tout est bon.")
        return
    print("À corriger avant de continuer :")
    for p in problems:
        print(f"  - {p}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
