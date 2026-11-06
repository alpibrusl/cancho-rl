"""R3 (docs/design.md section 34): what the weight-reading products of a batched decode step cost on Metal. A decode step of Qwen2.5-1.5B for `m` sequences multiplies m rows of
activations by the same weights, so the step is bound by reading the 3.1 GB of f16 weights once: 28 layers of qkv (2048 x 1536), o (1536 x 1536), gate/up (17920 x 1536) and down
(1536 x 8960), and the output product (152064 x 1536). This session has exactly those programs (random data; no attention, norms or sampling) and times a request of N steps.

  GPU=<cancho-gpu checkout> METAL=<cancho-gpu-metal-session> python3 scripts/r3_products.py WORKDIR [--m 32] [--kernel rows|mma|frag] [--steps 20] [--sched "threads warps bn bk"]"""
import os, subprocess, sys, time
import numpy as np

GPU, METAL = os.environ["GPU"], os.environ["METAL"]
arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
work = os.path.abspath(sys.argv[1]); os.makedirs(work, exist_ok=True)
m, kind, steps = int(arg("--m", 32)), arg("--kernel", "rows"), int(arg("--steps", 20))
sched = arg("--sched", "128 1,4 64 32").split()
rng = np.random.default_rng(0)
LAYERS = 28
shapes = {"qkv": (2048, 1536), "o": (1536, 1536), "gu": (17920, 1536), "dn": (1536, 8960), "head": (152064, 1536)}
L = []
for nm, (n, k) in shapes.items():
    (rng.normal(size=(n, k)) * 0.02).astype("<f2").tofile(os.path.join(work, f"w_{nm}"))
    x = rng.normal(size=(m, k))
    x.astype("<f4").tofile(os.path.join(work, f"x32_{nm}")); x.astype("<f2").tofile(os.path.join(work, f"x16_{nm}"))
src = open(os.path.join(GPU, "tests", "chx", "ported", "gemm_mma.chx")).read()
mma_path = os.path.join(work, "gemm_mma_s.chx")
open(mma_path, "w").write("\n".join(("schedule gemm_mma for apple-m-series { threads %s warps %s bm %d bn %s bk %s }" % (sched[0], sched[1].replace(",", " "), m, sched[2], sched[3])) if ln.startswith("schedule gemm_mma for apple-m-series") else ln for ln in src.split("\n")))
rows_path = os.path.join(GPU, "tests", "chx", "matvec_rows_f16_m.chx")
if kind == "frag":
    # the register-fragment lowering (cancho-gpu src/matmul.cho frag_rows_f16): --sched "threads bn frag"
    fs = arg("--fsched", "256 128 2").split()
    rows_path = os.path.join(work, "matvec_rows_f16_frag.chx")
    src2 = open(os.path.join(GPU, "tests", "chx", "matvec_rows_f16_frag.chx")).read()
    open(rows_path, "w").write("\n".join(("schedule matvec_rows_f16_frag for apple-m-series { threads %s bn %s frag %s }" % tuple(fs)) if ln.startswith("schedule matvec_rows_f16_frag for apple-m-series") else ln for ln in src2.split("\n")))
def product(name, nm):
    n, k = shapes[nm]
    if kind in ("rows", "frag"):
        L.append(f"program {name} {rows_path} m={m} n={n} k={k}")
        L.append(f"resident {name}.x {work}/x32_{nm} f32")
    else:
        L.append(f"program {name} {mma_path} m={m} n={n} k={k}")
        L.append(f"resident {name}.a {work}/x16_{nm} f16")
    L.append(f"resident {name}.w {work}/w_{nm} f16")
for l in range(LAYERS):
    for nm in ("qkv", "o", "gu", "dn"):
        product(f"{nm}_{l}", nm)
product("head", "head")
L.append("steps")
L.append("answer head.y")
open(os.path.join(work, "session"), "w").write("\n".join(L) + "\n")
p = subprocess.Popen(["/usr/bin/time", "-l", METAL, os.path.join(work, "session")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
def ask(s):
    t0 = time.time(); p.stdin.write(s + "\n"); p.stdin.flush(); o = p.stdout.readline(); return o, time.time() - t0
first = ask("1")[1]; ask("1")
runs = [ask(str(steps))[1] / steps * 1000 for _ in range(3)]
p.stdin.close(); p.wait(); err = p.stderr.read()
peak = [int(l.split()[0]) / 1e9 for l in err.split("\n") if "peak memory footprint" in l]
gb = sum(n * k * 2 for n, k in shapes.values() if True) / 1e9
tot = (sum(n * k * 2 for nm, (n, k) in shapes.items() if nm != "head") * LAYERS + shapes["head"][0] * shapes["head"][1] * 2) / 1e9
med = sorted(runs)[1]
print(f"m={m} kernel={kind} sched={sched}: {med:.2f} ms a step (runs {', '.join(f'{r:.2f}' for r in runs)}; first request {first:.1f} s); weights read {tot:.2f} GB -> {tot / med * 1000:.0f} GB/s; peak footprint {peak[0] if peak else float('nan'):.1f} GB; rc {p.returncode}")
if p.returncode:
    print(err[-500:])
