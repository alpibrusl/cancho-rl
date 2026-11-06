"""R2 (docs/design.md section 32): can one optimiser update be made from several micro-batches? Two micro-batches in one session, each with its own weights.

A session holds `wscale` (the per-position loss weight) as a resident file, one vector for all its steps. Here the generated session's line `resident cg.wt .../wscale f32` is
changed to `table cg.wt .../wscale S f32`, a file of S blocks of which repetition j reads block j mod S (cancho-gpu docs/resident.md section 17), and the gradient of each step is
read after its own request with `checkpoint`. Test: micro-batch 2 is micro-batch 1 with its rows reversed (tokens and weights), so, per-step weights working, its gradient equals
micro-batch 1's (the sum over the same sequences); with the resident file (the control) it does not.

  GPU=... QSW=... QWEN_WEIGHTS=... METAL=... python3 scripts/r2_accum.py PACKED_PREFIX WORKDIR --nb 4 --t 1152"""
import os, subprocess, sys
import numpy as np

GPU = os.environ["GPU"]
os.environ["QWEN_MODEL"] = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct")
sys.path.insert(0, os.path.join(GPU, "examples", "qwen"))
import qwen_ref as R, qwen_session as S, qwen_train as T  # noqa: E402

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
prefix, work = sys.argv[1], sys.argv[2]
nb, t = int(arg("--nb", 4)), int(arg("--t", 1152)); ls = 16.0
W, METAL, QSW = os.environ["QWEN_WEIGHTS"], os.environ["METAL"], os.environ["QSW"]
c = R.config_of(W, t, nb)
tok1 = np.fromfile(prefix + ".tokens.i32", dtype="<i4").reshape(nb, t)
w1 = np.fromfile(prefix + ".weights.f64", dtype="<f8").reshape(nb, t)
tok2, w2 = tok1[::-1].copy(), w1[::-1].copy()
AB = R.init_lora(c, 16, 7, b_std=0.02)

def session(name, per_step):
    d = os.path.abspath(os.path.join(work, name)); os.makedirs(d, exist_ok=True)
    fl = {"layers": c.layers, "d": c.d, "h": c.h, "hkv": c.hkv, "dh": c.dh, "ff": c.ff, "v": c.v, "t": c.t, "nb": nb, "tied": int(c.tied), "r": 16, "steps": 2}
    open(os.path.join(d, "cfg.txt"), "w").write("".join(f"{k} {v}\n" for k, v in fl.items()))
    np.concatenate([tok1, tok2]).astype("<i4").tofile(os.path.join(d, "tokens.i32"))
    with open(os.path.join(d, "adapters.f64"), "wb") as f:
        for nm in R.lora_names(c):
            f.write(np.asarray(AB[nm], dtype="<f8").reshape(-1).tobytes())
    w1.astype("<f8").tofile(os.path.join(d, "wt.f64"))
    out = os.path.join(d, "out"); os.makedirs(out, exist_ok=True)
    done = subprocess.run([QSW, "write", "--config", os.path.join(d, "cfg.txt"), "--wdir", os.path.abspath(W), "--out", out, "--root", GPU, "--tokens", os.path.join(d, "tokens.i32"),
                           "--adapters", os.path.join(d, "adapters.f64"), "--weights", os.path.join(d, "wt.f64"), "--no-train", "--grads", os.path.join(out, "grads.bin")], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    sess = os.path.join(out, "session")
    if per_step:
        np.concatenate([w1.reshape(-1), w2.reshape(-1)]).__mul__(ls).astype("<f4").tofile(os.path.join(out, "wscale"))
        text = open(sess).read()
        old = f"resident cg.wt {out}/wscale f32"
        assert old in text
        open(sess, "w").write(text.replace(old, f"table cg.wt {out}/wscale 2 f32"))
    ss = S.Session(METAL, sess)
    got = []
    for _ in range(2):
        ss.ask("1"); ss.checkpoint()
        raw = T.read_fused(c, 16, os.path.join(out, "grads.bin"))
        got.append(T.split_grads(c, 16, raw, nb * (t - 1), ls))
    ss.close()
    return got

def cosine(a, b):
    x = np.concatenate([a[k].ravel() for k in sorted(a)]); y = np.concatenate([b[k].ravel() for k in sorted(b)])
    return float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)), float(np.linalg.norm(x) / np.linalg.norm(y))

for name, per_step in (("table", True), ("control", False)):
    g1, g2 = session(name, per_step)
    cs, ratio = cosine(g2, g1)
    print(f"{name}: step 2 against step 1 (the same sequences, rows reversed): cosine {cs:.6f}, norm ratio {ratio:.5f}", flush=True)
