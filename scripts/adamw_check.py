"""The check of `rl adamw` (docs/design.md section 33): the host optimiser step against torch.optim.AdamW after torch.nn.utils.clip_grad_norm_, in float64 (a comparison helper).

  python3 scripts/adamw_check.py build/rl

Several updates in a row (so the moments and the bias corrections are exercised), the gradient of each the sum of three files, clipping active in some and not in others,
weight decay on and off, the first update with "-" for the moments. Parameters, moments, the reported norm and the factor must agree to 1e-10 relative (the arithmetic is
the same but not in the same order, so not to the bit)."""
import json, os, subprocess, sys, tempfile
import numpy as np
import torch

RL = sys.argv[1]
bad = 0

def say(ok, text):
    global bad
    print(("ok   " if ok else "FAIL ") + text)
    bad += 0 if ok else 1

def rl(args):
    return subprocess.run([RL, "adamw"] + args, capture_output=True, text=True)

def rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)))

d = tempfile.mkdtemp(prefix="adamw_")
rng = np.random.default_rng(2)
n = 5000
for wd, clip, lr, name in ((0.0, 1.0, 1e-4, "R1's settings (lr 1e-4, clip 1, no weight decay)"), (0.01, 0.0, 3e-3, "weight decay 0.01, no clipping"), (0.1, 500.0, 1e-2, "weight decay and a clip that only the later, larger gradients reach")):
    p0 = rng.normal(size=n)
    param = torch.nn.Parameter(torch.tensor(p0.copy(), dtype=torch.float64))
    opt = torch.optim.AdamW([param], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
    pcur, m, v = os.path.join(d, "p0"), "-", "-"
    p0.astype("<f8").tofile(pcur)
    worst, worst_norm, clipped = 0.0, 0.0, 0
    for step in range(1, 7):
        gs = [rng.normal(size=n) * s for s in (0.7, 1.3, 2.0 * step)]
        files = []
        for k, g in enumerate(gs):
            f = os.path.join(d, f"g{k}"); g.astype("<f8").tofile(f); files.append(f)
        cfg = os.path.join(d, "cfg.json")
        json.dump({"step": step, "lr": lr, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": wd, "clip": clip}, open(cfg, "w"))
        out = os.path.join(d, f"o{step}")
        done = rl([cfg, pcur, m, v, out] + files)
        if done.returncode != 0:
            say(False, f"{name}, step {step}: the tool failed: {done.stderr.strip()[:200]}"); break
        info = json.loads(done.stdout)
        gsum = sum(gs)
        param.grad = torch.tensor(gsum, dtype=torch.float64)
        total = float(torch.nn.utils.clip_grad_norm_([param], clip)) if clip > 0 else float(torch.linalg.vector_norm(param.grad))
        opt.step()
        ref_p = param.detach().numpy(); st = opt.state[param]
        ours_p = np.fromfile(out + ".params.f64", dtype="<f8"); ours_m = np.fromfile(out + ".m.f64", dtype="<f8"); ours_v = np.fromfile(out + ".v.f64", dtype="<f8")
        worst = max(worst, rel(ours_p, ref_p), rel(ours_m, st["exp_avg"].numpy()), rel(ours_v, st["exp_avg_sq"].numpy()))
        worst_norm = max(worst_norm, abs(info["grad_norm"] - total) / total)
        clipped += 1 if clip > 0 and total > clip else 0
        pcur, m, v = out + ".params.f64", out + ".m.f64", out + ".v.f64"
    else:
        say(worst < 1e-10 and worst_norm < 1e-12, f"{name}: 6 updates, parameters and moments agree with torch to {worst:.1e}, the norm to {worst_norm:.1e}; clipped in {clipped} of 6")
        if clip >= 100:
            say(clipped > 0 and clipped < 6, f"{name}: clipping was active in some updates and not in others")

# refusals
def refuses(name, rule, args):
    done = rl(args)
    say(done.returncode == 2 and f'"{rule}"' in done.stderr, f"refusal {rule}: {name} (exit {done.returncode})")

cfg = os.path.join(d, "cfg.json"); json.dump({"step": 1, "lr": 1e-3}, open(cfg, "w"))
pf = os.path.join(d, "p"); np.zeros(8).astype("<f8").tofile(pf)
gf = os.path.join(d, "g"); np.ones(8).astype("<f8").tofile(gf)
g2 = os.path.join(d, "g2"); np.ones(16).astype("<f8").tofile(g2)
refuses("a gradient file that differs in length", "rl.adamw.shape", [cfg, pf, "-", "-", os.path.join(d, "r"), gf, g2])
refuses("a missing file", "rl.adamw.input", [cfg, os.path.join(d, "none"), "-", "-", os.path.join(d, "r"), gf])
bc = os.path.join(d, "bad.json"); json.dump({"step": 0, "lr": 1e-3}, open(bc, "w"))
refuses("step 0", "rl.adamw.config", [bc, pf, "-", "-", os.path.join(d, "r"), gf])
refuses("M a file and V not", "rl.usage", [cfg, pf, pf, "-", os.path.join(d, "r"), gf])
refuses("too few arguments", "rl.usage", [cfg, pf])
refuses("an output directory that does not exist", "rl.adamw.write", [cfg, pf, "-", "-", os.path.join(d, "nodir", "r"), gf])
np.zeros(3).astype("<f8").tofile(os.path.join(d, "odd")); open(os.path.join(d, "odd"), "ab").write(b"x")
refuses("an array that is not whole words", "rl.adamw.shape", [cfg, os.path.join(d, "odd"), "-", "-", os.path.join(d, "r"), os.path.join(d, "odd")])
print("bad" if bad else "all ok")
sys.exit(1 if bad else 0)
