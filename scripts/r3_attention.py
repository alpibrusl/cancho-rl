"""R3 (docs/design.md section 36): what the attention of a batched decode step costs on Metal. For `nb` sequences with caches of `tmax` positions, 28 layers of
attention_scores_b + attention_pv_b (Qwen2.5-1.5B: 2 key/value heads, 6 query heads each, heads of 128), random data, no other kernel, f32 caches. A request of N steps; the answer
is a tiny product's so no time goes to printing.

  GPU=<cancho-gpu checkout> METAL=<cancho-gpu-metal-session> python3 scripts/r3_attention.py WORKDIR [--nb 32] [--t 1152] [--ts 64] [--dw 16] [--steps 20]"""
import os, subprocess, sys, time
import numpy as np

GPU, METAL = os.environ["GPU"], os.environ["METAL"]
arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
work = os.path.abspath(sys.argv[1]); os.makedirs(work, exist_ok=True)
nb, t, ts, dw, steps = int(arg("--nb", 32)), int(arg("--t", 1152)), int(arg("--ts", 64)), int(arg("--dw", 16)), int(arg("--steps", 20))
nkv, g, d, LAYERS = 2, 6, 128, 28
rng = np.random.default_rng(0)
rng.normal(size=(nb * nkv * g, d)).astype("<f4").tofile(os.path.join(work, "q"))
rng.normal(size=(nb * nkv * t, d)).astype("<f4").tofile(os.path.join(work, "k"))
rng.normal(size=(nb * nkv * t, d)).astype("<f4").tofile(os.path.join(work, "v"))
np.zeros((nb, t), dtype="<f4").tofile(os.path.join(work, "mask"))
np.zeros((1, 32), dtype="<f4").tofile(os.path.join(work, "x_tiny")); np.zeros((8, 32), dtype="<f2").tofile(os.path.join(work, "w_tiny"))
ks = os.path.join(GPU, "tests", "chx", "attention_scores_b.chx"); kp = os.path.join(GPU, "tests", "chx", "attention_pv_b.chx"); km = os.path.join(GPU, "tests", "chx", "matvec_rows_f16_m.chx")
L = []
for l in range(LAYERS):
    L += [f"program s{l} {ks} nb={nb} nkv={nkv} g={g} t={t} ts={ts} d={d} tmax={t}",
          f"resident s{l}.q {work}/q f32", f"resident s{l}.k {work}/k f32", f"resident s{l}.mask {work}/mask f32",
          f"program p{l} {kp} nb={nb} nkv={nkv} g={g} t={t} dw={dw} d={d} tmax={t}",
          f"link s{l}.sc -> p{l}.sc", f"resident p{l}.v {work}/v f32"]
L += [f"program tiny {km} m=1 n=8 k=32", f"resident tiny.x {work}/x_tiny f32", f"resident tiny.w {work}/w_tiny f16", "steps", "answer tiny.y"]
open(os.path.join(work, "session"), "w").write("\n".join(L) + "\n")
p = subprocess.Popen(["/usr/bin/time", "-l", METAL, os.path.join(work, "session")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
def ask(s):
    t0 = time.time(); p.stdin.write(s + "\n"); p.stdin.flush(); o = p.stdout.readline(); return o, time.time() - t0
first = ask("1")[1]; ask("1")
runs = [ask(str(steps))[1] / steps * 1000 for _ in range(3)]
p.stdin.close(); p.wait(); err = p.stderr.read()
peak = [int(l.split()[0]) / 1e9 for l in err.split("\n") if "peak memory footprint" in l]
kv = 2 * nb * nkv * t * d * 4 * LAYERS / 1e9
med = sorted(runs)[1]
print(f"nb={nb} t={t} ts={ts} dw={dw}: {med:.2f} ms a step of 28 layers (runs {', '.join(f'{r:.2f}' for r in runs)}; first request {first:.1f} s); keys and values {kv:.2f} GB -> {kv / med * 1000:.0f} GB/s; peak footprint {peak[0] if peak else float('nan'):.1f} GB; rc {p.returncode}")
if p.returncode:
    print(err[-400:])
