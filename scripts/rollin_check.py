#!/usr/bin/env python3
"""`rl rollin` against numpy: the files it writes for 4 prompts x 8 samples (docs/design.md section 42): shapes, padding, the last token and position of every row, uniforms in [0, 1)
and roughly uniform, the same seed giving the same file and another a different one, and the refusals.

    python3 scripts/rollin_check.py BUILD/rl CANCHO-GPU TOKENIZER.json"""
import os, subprocess, sys, tempfile
import numpy as np

RL, GPU, TOKJ = sys.argv[1:4]
bad = 0
def say(ok, text):
    global bad
    print(("ok   " if ok else "FAIL ") + text); bad += 0 if ok else 1

d = tempfile.mkdtemp(prefix="rollin_")
tasks = sorted(os.listdir("tasks"))[:4]
texts = [subprocess.run([RL, "chat", "prompts/primer.md", f"tasks/{t}"], capture_output=True, text=True, check=True).stdout for t in tasks]
open(f"{d}/c.txt", "w").write("\n%%\n".join(texts))
tk = subprocess.run([GPU, "tokenize", TOKJ, f"{d}/c.txt"], capture_output=True, text=True, check=True).stdout
open(f"{d}/p.txt", "w").write(tk)
ids = [list(map(int, ln.split(":", 1)[1].split())) for ln in tk.splitlines() if ln.startswith("ids ")]
T, STEPS, G, PAD = 1152, 400, 8, 151643
def run(seed, prefix, group=G, t=T, steps=STEPS, file=f"{d}/p.txt"):
    return subprocess.run([RL, "rollin", file, str(group), str(t), str(steps), str(PAD), str(seed), prefix], capture_output=True, text=True)
r = run(7, f"{d}/a")
say(r.returncode == 0 and '"prompts":4' in r.stdout.replace(" ", "") and '"sequences":32' in r.stdout.replace(" ", ""), f"4 prompts x 8: {r.stdout.strip()} {r.stderr.strip()[:100]}")
pre = np.fromfile(f"{d}/a.prefill.i32", dtype="<i4").reshape(4, T)
say(all(pre[i, :len(ids[i])].tolist() == ids[i] and (pre[i, len(ids[i]):] == PAD).all() for i in range(4)), "prefill: each prompt, then the pad, to T")
tok = np.fromfile(f"{d}/a.tokens.i32", dtype="<i4").reshape(32, STEPS)
pos = np.fromfile(f"{d}/a.pos0.i32", dtype="<i4")
lens = np.fromfile(f"{d}/a.lens.i32", dtype="<i4")
say(all(tok[r_, 0] == ids[r_ // G][-1] and (tok[r_, 1:] == 0).all() and pos[r_] == len(ids[r_ // G]) - 1 and lens[r_] == 1 for r_ in range(32)), "decode rows: the last token of prompt r // 8, position length - 1, length 1")
u = np.fromfile(f"{d}/a.uniforms.f64", dtype="<f8")
say(u.size == 32 * STEPS and u.min() >= 0 and u.max() < 1 and abs(u.mean() - 0.5) < 0.01 and abs(u.std() - 0.2887) < 0.01, f"uniforms: {u.size} of them in [0, 1), mean {u.mean():.4f} std {u.std():.4f}")
say(float(np.abs(np.corrcoef(u[:-1], u[1:])[0, 1])) < 0.03 and len(np.unique(u)) > 0.999 * u.size, "uniforms: no serial correlation, (almost) all distinct")
chi = np.histogram(u, bins=20, range=(0, 1))[0]; e = u.size / 20
say(((chi - e) ** 2 / e).sum() < 43.8, f"uniforms: chi-square over 20 bins {((chi - e) ** 2 / e).sum():.1f} (limit 43.8, 19 dof, p 0.001)")
run(7, f"{d}/b"); run(8, f"{d}/c")
say(open(f"{d}/a.uniforms.f64", "rb").read() == open(f"{d}/b.uniforms.f64", "rb").read() and open(f"{d}/a.uniforms.f64", "rb").read() != open(f"{d}/c.uniforms.f64", "rb").read(), "the same seed gives the same file, another a different one")
pl = [int(x) for x in open(f"{d}/a.plens.txt").read().split()]
say(pl == [len(x) for x in ids], f"plens.txt: {pl}")
for name, kw, rule in (("more than 32 sequences", dict(group=9), "rl.rollin.shape"), ("a prompt and the steps past T", dict(t=1000), "rl.rollin.long"), ("a missing file", dict(file=f"{d}/none"), "rl.rollin.input")):
    r = run(1, f"{d}/x", **kw)
    say(r.returncode == 2 and rule in r.stderr, f"refusal {rule}: {name}")
open(f"{d}/empty.txt", "w").write("tokenizer ids=1\n")
r = run(1, f"{d}/x", file=f"{d}/empty.txt")
say(r.returncode == 2 and "rl.rollin.prompts" in r.stderr, "refusal rl.rollin.prompts: no ids line")
print("bad" if bad else "all ok"); sys.exit(1 if bad else 0)
