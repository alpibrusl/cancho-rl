"""The check of `rl pack` (docs/design.md section 31): group-normalised advantages and the weight vector, against an independent numpy computation (a comparison helper).

  python3 scripts/pack_check.py build/rl

Samples are random token lists in several groups (mixed rewards, an all-equal group, a group of one, a group split in two runs); the expected tokens must match byte for byte,
the weights to 1e-12 relative (the same arithmetic in a different order). The refusals are checked by rule tag."""
import json, os, subprocess, sys, tempfile
import numpy as np

RL = sys.argv[1]
bad = 0

def say(ok, text):
    global bad
    print(("ok   " if ok else "FAIL ") + text)
    bad += 0 if ok else 1

def run(args):
    return subprocess.run([RL, "pack"] + args, capture_output=True, text=True)

def expected(samples, nb, t, pad):
    groups, i = [], 0
    while i < len(samples):
        j = i
        while j < len(samples) and samples[j]["group"] == samples[i]["group"]:
            j += 1
        groups.append((i, j)); i = j
    adv = np.zeros(len(samples))
    for lo, hi in groups:
        r = np.array([s["reward"] for s in samples[lo:hi]])
        if len(r) > 1 and r.std(ddof=1) > 0:
            adv[lo:hi] = (r - r.mean()) / (r.std(ddof=1) + 1e-4)
    keep = [k for k in range(len(samples)) if adv[k] != 0]
    n = sum(len(samples[k]["completion"]) for k in keep)
    tok = np.full((nb, t), pad, dtype="<i4"); w = np.zeros((nb, t))
    for row, k in enumerate(keep):
        p, c = samples[k]["prompt"], samples[k]["completion"]
        tok[row, :len(p) + len(c)] = p + c
        w[row, len(p) - 1:len(p) + len(c) - 1] = adv[k] * nb * (t - 1) / n
    return tok, w, len(keep), n

def sample(rng, group, reward, t):
    p = int(rng.integers(3, 20)); c = int(rng.integers(2, 25))
    return {"group": group, "reward": reward, "prompt": rng.integers(0, 150000, p).tolist(), "completion": rng.integers(0, 150000, c).tolist()}

d = tempfile.mkdtemp(prefix="pack_")
rng = np.random.default_rng(4)
t, nb, pad = 64, 12, 151643
S = []
for g, rs in (("a", [0, 1, 0, 0, 0.5]), ("b", [1, 1, 1]), ("c", [0.25]), ("d", [0, 0, 1, 0]), ("a", [1, 0, 0])):
    S += [sample(rng, g, r, t) for r in rs]
path = os.path.join(d, "s.jsonl")
open(path, "w").write("\n".join(json.dumps(s) for s in S) + "\n")
done = run([path, os.path.join(d, "out"), str(nb), str(t), str(pad)])
tok, w, kept, n = expected(S, nb, t, pad)
say(done.returncode == 0 and json.loads(done.stdout) == {"samples": len(S), "kept": kept, "completion_tokens": n}, f"the summary: {done.stdout.strip()} (expected kept {kept}, tokens {n})")
got_t = np.fromfile(os.path.join(d, "out.tokens.i32"), dtype="<i4").reshape(nb, t)
got_w = np.fromfile(os.path.join(d, "out.weights.f64"), dtype="<f8").reshape(nb, t)
say(np.array_equal(got_t, tok), "the tokens are the kept samples' prompt + completion, padded, rows after them all pad")
say(np.allclose(got_w, w, rtol=1e-12, atol=0) and np.array_equal(got_w == 0, w == 0), f"the weights equal advantage x nb (t - 1) / N on completion targets, else 0 (largest relative difference {np.max(np.abs(got_w - w) / np.maximum(np.abs(w), 1e-300)):.1e})")
say(float(w[w != 0].max()) > 0 and float(w[w != 0].min()) < 0, "both signs occur (a failed and a solved completion)")

# nothing kept: no files, kept 0
d2 = os.path.join(d, "none"); os.makedirs(d2)
open(os.path.join(d2, "s"), "w").write("\n".join(json.dumps(sample(rng, "g", 1.0, t)) for _ in range(4)) + "\n")
done = run([os.path.join(d2, "s"), os.path.join(d2, "o"), "4", str(t), "0"])
say(done.returncode == 0 and json.loads(done.stdout)["kept"] == 0 and not os.path.exists(os.path.join(d2, "o.tokens.i32")), "all rewards equal: kept 0, nothing written")

# long lines (a trap once: the arrays were in a 64 KiB region) and many lines
def big(name, groups, per, plen, clen, nb_, t_):
    rng2 = np.random.default_rng(9)
    S2 = []
    for g in range(groups):
        for k in range(per):
            S2.append({"group": f"g{g}", "reward": float(rng2.integers(0, 2)), "prompt": rng2.integers(0, 150000, plen).tolist(), "completion": rng2.integers(0, 150000, clen).tolist()})
    q = os.path.join(d, name + ".jsonl"); open(q, "w").write("".join(json.dumps(x) + "\n" for x in S2))
    r = run([q, os.path.join(d, name), str(nb_), str(t_), str(pad)])
    tk, ww, kp, nn = expected(S2, nb_, t_, pad)
    ok = r.returncode == 0 and json.loads(r.stdout)["kept"] == kp
    if ok and kp:
        ok = np.array_equal(np.fromfile(os.path.join(d, name + ".tokens.i32"), dtype="<i4").reshape(nb_, t_), tk) and np.allclose(np.fromfile(os.path.join(d, name + ".weights.f64")).reshape(nb_, t_), ww, rtol=1e-12, atol=0)
    say(ok, f"{name}: {len(S2)} samples of {plen}+{clen} tokens, kept {kp} (exit {r.returncode}{' ' + r.stderr.strip()[:80] if r.returncode else ''})")

big("long lines", 1, 4, 700, 400, 4, 1152)
big("many lines", 60, 8, 5, 6, 512, 64)

def refuses(name, rule, samples_text, nb_=nb, t_=t, args=None):
    p = os.path.join(d, "r_" + name); open(p, "w").write(samples_text)
    r = run(args or [p, os.path.join(d, "r_out"), str(nb_), str(t_), "0"])
    say(r.returncode == 2 and f'"{rule}"' in r.stderr, f"refusal {rule}: {name} (exit {r.returncode})")

line = lambda s: json.dumps(s) + "\n"
refuses("more kept than rows", "rl.pack.overflow", "".join(line(s) for s in S), nb_=2)
refuses("a sample longer than t", "rl.pack.long", "".join(line(sample(rng, "g", r, t)) for r in (0, 1)), t_=8)
refuses("not json", "rl.pack.sample", "not json\n")
refuses("a token that is not an integer", "rl.pack.sample", line({"group": "g", "reward": 1, "prompt": [1.5], "completion": [2]}))
refuses("a missing field", "rl.pack.sample", line({"group": "g", "reward": 1, "prompt": [1]}))
r = run([os.path.join(d, "no_such"), os.path.join(d, "o"), "4", "64", "0"])
say(r.returncode == 2 and '"rl.pack.input"' in r.stderr, "refusal rl.pack.input: no samples file")
r = run([path, os.path.join(d, "no_dir", "o"), str(nb), str(t), "0"])
say(r.returncode == 2 and '"rl.pack.write"' in r.stderr, "refusal rl.pack.write: an output directory that does not exist")
r = run([path])
say(r.returncode == 2 and '"rl.usage"' in r.stderr, "refusal rl.usage: missing arguments")
print("bad" if bad else "all ok")
sys.exit(1 if bad else 0)
