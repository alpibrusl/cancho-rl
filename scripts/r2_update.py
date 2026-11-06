"""R2 (docs/design.md section 32): one update from several micro-batches, end to end. `rl pack` splits the kept samples into K batches of NB rows with N over all of them; `qsw write`
makes one session of K steps whose `wscale` is a table of K blocks; the Metal runner does one request and one `checkpoint` per step, and the K gradients are summed on the host; the
reference is peft float32 on the CPU minimising `sum A CE / N` over every kept sample (micro-batches of NB rows, gradients accumulated, which is the same loss).

  GPU=... QSW=... QWEN_WEIGHTS=... METAL=... python3 scripts/r2_update.py PACKED_PREFIX WORKDIR --nb 4 --t 1152 [--step]

With --step (and RL=build/rl) the update is finished on the host the way the loop will: each step's fused gradients go through `qsw split` (per-projection binary64), `rl adamw` sums them,
clips the norm to 1 and applies AdamW (lr 1e-4, no weight decay: r1_grpo.py's), and the new adapters are compared with torch's: the peft gradient, clip_grad_norm_ and torch.optim.AdamW
on the same adapters. The first AdamW step moves every element by about lr whatever its gradient, so the comparison is of the moves, not only of the gradients."""
import os, subprocess, sys, time
import numpy as np

GPU = os.environ["GPU"]
os.environ["QWEN_MODEL"] = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct")
sys.path.insert(0, os.path.join(GPU, "examples", "qwen"))
import torch  # noqa: E402
import qwen_ref as R, qwen_session as S, qwen_train as T, peft_reference as P  # noqa: E402

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
prefix, work = sys.argv[1], sys.argv[2]
nb, t = int(arg("--nb", 4)), int(arg("--t", 1152)); ls = 16.0
W, METAL, QSW = os.environ["QWEN_WEIGHTS"], os.environ["METAL"], os.environ["QSW"]
os.makedirs(work, exist_ok=True)
tokens = np.fromfile(prefix + ".tokens.i32", dtype="<i4").reshape(-1, nb, t)
wsess = np.fromfile(prefix + ".weights.f64", dtype="<f8").reshape(-1, nb, t)
K = tokens.shape[0]
N = float(np.count_nonzero(wsess)); ncount = float(nb * (t - 1))
wt = wsess * N / ncount                      # the unscaled advantages on completion targets
c = R.config_of(W, t, nb)
AB = R.init_lora(c, 16, 7, b_std=0.02)
print(f"{K} batches of {nb} x {t}; completion tokens {int(N)}; distinct advantages {sorted(set(np.round(wt[wt != 0], 3).tolist()))[:6]}...", flush=True)

# --- the reference: peft float32, all samples, gradients accumulated over the micro-batches
ref_path = os.path.join(work, "ref.npz")
if not os.path.exists(ref_path):
    torch.set_num_threads(os.cpu_count())
    model, handles = P.build(c, AB, base=None)
    model.zero_grad(); total = 0.0; t0 = time.time()
    for k in range(K):
        ids = torch.tensor(tokens[k].astype(np.int64)); w = torch.tensor(wt[k], dtype=torch.float32)
        logits = model(input_ids=ids).logits[:, :-1].float()
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="none").reshape(nb, t - 1)
        loss = (ce * w[:, :-1]).sum() / N
        loss.backward(); total += loss.item()
        del logits, ce, loss
    grads = {k_: p.grad.detach().cpu().numpy().astype(np.float64) for k_, p in handles.items()}
    print(f"peft: weighted loss {total:.6f}, {time.time() - t0:.0f} s", flush=True)
    np.savez(ref_path, loss=np.array([total]), **grads)
    del model, handles
ref = np.load(ref_path)

# --- the session: K steps, a table of K blocks of weights, one checkpoint a step
d = os.path.abspath(os.path.join(work, "s")); out = os.path.join(d, "out"); os.makedirs(out, exist_ok=True)
fl = {"layers": c.layers, "d": c.d, "h": c.h, "hkv": c.hkv, "dh": c.dh, "ff": c.ff, "v": c.v, "t": c.t, "nb": nb, "tied": int(c.tied), "r": 16, "steps": K}
open(os.path.join(d, "cfg.txt"), "w").write("".join(f"{k_} {v}\n" for k_, v in fl.items()))
with open(os.path.join(d, "adapters.f64"), "wb") as f:
    for nm in R.lora_names(c):
        f.write(np.asarray(AB[nm], dtype="<f8").reshape(-1).tobytes())
done = subprocess.run([QSW, "write", "--config", os.path.join(d, "cfg.txt"), "--wdir", os.path.abspath(W), "--out", out, "--root", GPU, "--tokens", prefix + ".tokens.i32", "--adapters", os.path.join(d, "adapters.f64"),
                       "--weights", prefix + ".weights.f64", "--no-train", "--grads", os.path.join(out, "grads.bin")], capture_output=True, text=True)
assert done.returncode == 0, done.stderr
ss = S.Session(METAL, os.path.join(out, "session"))
acc = None; t0 = time.time(); per = []
for k in range(K):
    ts = time.time(); ss.ask("1"); ss.checkpoint()
    if "--step" in sys.argv:
        subprocess.run([QSW, "split", "--config", os.path.join(d, "cfg.txt"), "--grads", os.path.join(out, "grads.bin"), "--out", os.path.join(d, f"g{k}.f64")], check=True)
    g = T.split_grads(c, 16, T.read_fused(c, 16, os.path.join(out, "grads.bin")), ncount, ls)
    acc = g if acc is None else {n: acc[n] + g[n] for n in g}
    per.append(time.time() - ts)
ss.close()
print(f"session: {K} steps with a checkpoint each, {sum(per):.1f} s in all ({', '.join(f'{x:.1f}' for x in per)}); the first includes the compile", flush=True)
cos, rel, dot, na, nr = [], [], 0.0, 0.0, 0.0
for k_ in sorted(acc):
    a_, b_ = acc[k_].ravel(), ref[k_].ravel()
    cos.append(float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_)))); rel.append(float(np.linalg.norm(a_ - b_) / np.linalg.norm(b_)))
    dot += a_ @ b_; na += a_ @ a_; nr += b_ @ b_
print(f"{len(cos)} matrices: smallest cosine {min(cos):.6f}, largest relative norm error {max(rel):.2e}; whole-gradient cosine {dot / np.sqrt(na * nr):.6f}, norm ratio {np.sqrt(na / nr):.5f}")
print("sign:", "positive" if dot > 0 else "NEGATIVE")

if "--step" in sys.argv:
    import json
    RL = os.environ["RL"]; LR = 1e-4
    names = R.lora_names(c)
    json.dump({"step": 1, "lr": LR, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.0, "clip": 1.0}, open(os.path.join(d, "adamw.json"), "w"))
    t0 = time.time()
    done = subprocess.run([RL, "adamw", os.path.join(d, "adamw.json"), os.path.join(d, "adapters.f64"), "-", "-", os.path.join(d, "new")] + [os.path.join(d, f"g{k}.f64") for k in range(K)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    info = json.loads(done.stdout)
    print(f"rl adamw: {info['n']} values, gradient norm {info['grad_norm']:.4f}, clip factor {info['coef']:.5f}, {time.time() - t0:.1f} s", flush=True)
    old = np.fromfile(os.path.join(d, "adapters.f64"), dtype="<f8"); new = np.fromfile(os.path.join(d, "new.params.f64"), dtype="<f8")
    flat_ref = np.concatenate([ref[nm].ravel() for nm in names])
    prm = torch.nn.Parameter(torch.tensor(old.copy(), dtype=torch.float64)); prm.grad = torch.tensor(flat_ref, dtype=torch.float64)
    tn = float(torch.nn.utils.clip_grad_norm_([prm], 1.0)); torch.optim.AdamW([prm], lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0).step()
    want = prm.detach().numpy()
    dn, dw = new - old, want - old
    print(f"torch: gradient norm {tn:.4f}")
    print(f"the move of the adapters: cosine {float(dn @ dw / np.linalg.norm(dn) / np.linalg.norm(dw)):.6f}, norm ratio {float(np.linalg.norm(dn) / np.linalg.norm(dw)):.5f}, "
          f"elements whose move differs in sign {float(np.mean(np.sign(dn) != np.sign(dw))) * 100:.3f}%, largest difference {float(np.max(np.abs(dn - dw))) / LR:.2f} lr")
