"""R2 (docs/design.md section 30): the policy-gradient (advantage-weighted) loss gradient of the LoRA adapters of Qwen2.5-1.5B-Instruct on a real batch, in a cancho-gpu
Metal session with the loss weight `wt[i] = advantage x completion mask x loss_scale` (the resident file `wscale`), against peft's float32 autograd on the CPU. A comparison helper.

  GPU=<cancho-gpu checkout> QWEN_WEIGHTS=<convert.py output of the Instruct model> METAL=<cancho-gpu-metal-session> python3 scripts/r2_grad.py BATCH.npz WORKDIR [--b-std 0.02] [--loss-scale 16]

The loss both sides compute is  sum over completion tokens of  A_seq x CE(token) / N,  N the number of completion tokens (the quantity r1_grpo.py minimises, one update a batch).
The session writes sum_i wt[i] (softmax - onehot) f16, and `split_grads` divides the loss scale and N out."""
import glob, os, sys, time
import numpy as np

GPU = os.environ["GPU"]
HF = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct")
os.environ["QWEN_MODEL"] = HF
sys.path.insert(0, os.path.join(GPU, "examples", "qwen"))
import torch  # noqa: E402
import data, qwen_ref as R, qwen_session as S, qwen_train as T, peft_reference as P  # noqa: E402

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
batch, work = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
W = os.environ["QWEN_WEIGHTS"]
METAL = os.environ["METAL"]
ls = float(arg("--loss-scale", 16.0)); bstd = float(arg("--b-std", 0.02))
B = np.load(batch)
tok = B["tokens"].astype(np.int64); adv = B["adv"]; plen = B["plen"]; clen = B["clen"]
nb, t = tok.shape
c = R.config_of(W, t, nb)
# the weight of the row i, whose target is token i + 1: the advantage of its sequence when that token is a completion token
wt = np.zeros((nb, t))
for b in range(nb):
    wt[b, plen[b] - 1: plen[b] + clen[b] - 1] = adv[b]
N = float(clen.sum())
print(f"nb {nb} t {t}; completion tokens {int(N)} of {nb * t}; advantages {np.round(adv, 3).tolist()}; weights x loss scale in [{(wt * ls).min():.3f}, {(wt * ls).max():.3f}]", flush=True)
AB = R.init_lora(c, 16, 7, b_std=bstd)

# --- peft, float32, CPU
ref_path = os.path.join(work, "ref.npz")
if not os.path.exists(ref_path):
    torch.set_num_threads(os.cpu_count())
    model, handles = P.build(c, AB, base=None)
    ids = torch.tensor(tok); w = torch.tensor(wt, dtype=torch.float32)
    t0 = time.time()
    model.zero_grad()
    logits = model(input_ids=ids).logits[:, :-1].float()
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="none").reshape(nb, t - 1)
    loss = (ce * w[:, :-1]).sum() / N
    loss.backward()
    grads = {k: p.grad.detach().cpu().numpy().astype(np.float64) for k, p in handles.items()}
    # the f16 range of the session's dl = wt x loss_scale x (softmax - onehot), from these logits
    with torch.no_grad():
        pr = torch.softmax(logits, -1); pr.scatter_add_(2, ids[:, 1:, None], -torch.ones(nb, t - 1, 1))
        dl = (pr.abs().amax(-1) * w[:, :-1].abs() * ls)
        nz = pr.abs()[w[:, :-1] != 0] * (w[:, :-1].abs()[w[:, :-1] != 0] * ls)[:, None]
        rng = dict(dl_max=float(dl.max()), dl_nonzero_frac_f16_subnormal=float(((nz > 0) & (nz < 2 ** -14)).float().mean()), dl_nonzero_frac_below_f16_min=float(((nz > 0) & (nz < 2 ** -24)).float().mean()))
    print(f"peft: weighted loss {loss.item():.6f}, {time.time() - t0:.0f} s; range {rng}", flush=True)
    np.savez(ref_path, loss=np.array([loss.item()]), **grads, **{f"_{k}": np.array([v]) for k, v in rng.items()})
    del model, handles
ref = np.load(ref_path)

# --- the session, with the weights written into wscale
d = os.path.join(work, "s")
os.makedirs(d, exist_ok=True)
sess = T.write(d, c, W, tok.astype(np.int32), AB, r=16, loss_scale=ls, train=False, grads=os.path.join(d, "grads.bin"))
(wf,) = glob.glob(os.path.join(d, "wscale*"))
default = np.fromfile(wf, dtype="<f4")
assert default.size == nb * t
(wt * ls).reshape(-1).astype("<f4").tofile(wf)
ss = S.Session(METAL, sess)
out, first = ss.ask("1")
ss.checkpoint(); ss.close()
got = T.split_grads(c, 16, T.read_fused(c, 16, os.path.join(d, "grads.bin")), N, ls)
cos, rel = [], []
dot = na = nr = 0.0
for k in sorted(got):
    a_, b_ = got[k].ravel(), ref[k].ravel()
    cos.append(float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_)))); rel.append(float(np.linalg.norm(a_ - b_) / np.linalg.norm(b_)))
    dot += a_ @ b_; na += a_ @ a_; nr += b_ @ b_
print(f"session first request {first:.1f} s; {len(cos)} matrices")
print(f"smallest cosine {min(cos):.6f}, largest relative norm error {max(rel):.2e}; whole-gradient cosine {dot / np.sqrt(na * nr):.6f}, norm ratio {np.sqrt(na / nr):.5f}")
print("sign: whole-gradient dot product is", "positive" if dot > 0 else "NEGATIVE")
