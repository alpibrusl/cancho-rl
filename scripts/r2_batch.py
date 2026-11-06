"""R2 (docs/design.md section 30): one real policy-gradient batch for the gradient comparison. The base Qwen2.5-1.5B-Instruct samples a group of completions for
tasks of the R1 pool, `rl grade` gives the rewards, advantages are group-normalised as in r1_grpo.py, and the first NB sequences with a non-zero advantage are kept.
A comparison helper, not part of the package.

  CANCHO=... python3 scripts/r2_batch.py OUT.npz [--nb 4] [--t 1152] [--seed 1] [--samples FILE.jsonl --want 8]

With --samples, every sample of the visited groups is written as `rl pack` reads them ({group, reward, prompt, completion}) and the loop stops when at least --want samples have
a non-zero advantage; the .npz is then not written.

OUT.npz: tokens [nb, t] int64 (the prompt, the completion up to and including <|im_end|>, then the pad token), adv [nb], plen [nb], clen [nb]."""
import argparse, json, os, random, subprocess, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--nb", type=int, default=4); ap.add_argument("--t", type=int, default=1152)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--group", type=int, default=8)
ap.add_argument("--samples", default=""); ap.add_argument("--want", type=int, default=8)
ap.add_argument("--max-new", type=int, default=400); ap.add_argument("--temp", type=float, default=0.8)
a = ap.parse_args()
CANCHO = os.environ.get("CANCHO", "cancho")
random.seed(a.seed); torch.manual_seed(a.seed)
M = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16).to("mps").eval()
rl = lambda *x: subprocess.run(["./build/rl", *x], capture_output=True, text=True).stdout
pool = [t.strip() for t in open("work/r1_pool.txt") if t.strip()]
random.shuffle(pool)
END = tok.convert_tokens_to_ids("<|im_end|>")

def extract(text):
    out, on = [], False
    for ln in text.split("\n"):
        if ln.startswith("```cancho"): on = True; continue
        if ln.startswith("```"): on = False; continue
        if on: out.append(ln)
    return "\n".join(out) + "\n"

keep = []
samples_out = []
live = 0
for t in pool:
    if (a.samples and live >= a.want) or (not a.samples and len(keep) >= a.nb): break
    text = rl("prompt", "prompts/primer.md", f"tasks/{t}")
    p = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    pid = tok([p], return_tensors="pt").input_ids.to("mps")
    with torch.no_grad():
        out = model.generate(input_ids=pid, do_sample=True, temperature=a.temp, top_p=1.0, top_k=0, max_new_tokens=a.max_new, num_return_sequences=a.group, pad_token_id=tok.pad_token_id)
    L = pid.shape[1]
    rs = []
    for k, o in enumerate(out):
        txt = tok.decode(o[L:], skip_special_tokens=True)
        c = f"work/r2_cand_{t}_{k}.cho"; open(c, "w").write(extract(txt))
        r = json.loads(rl("grade", CANCHO, f"tasks/{t}", c, f"work/r2_work_{t}_{k}.cho") or "{}")
        rs.append(r.get("reward", 0.0))
    rs = torch.tensor(rs); print(t, rs.tolist(), flush=True)
    if a.samples:
        for k, o in enumerate(out):
            comp = o[L:].cpu(); e = (comp == END).nonzero(); n = int(e[0]) + 1 if len(e) else len(comp)
            samples_out.append({"group": t, "reward": float(rs[k]), "prompt": pid[0].cpu().tolist(), "completion": comp[:n].tolist()})
        if rs.std() > 0: live += a.group
        continue
    if rs.std() > 0:
        adv = (rs - rs.mean()) / (rs.std() + 1e-4)
        for k in torch.randperm(a.group).tolist():
            if len(keep) < a.nb and abs(adv[k]) > 0:
                comp = out[k][L:].cpu()
                e = (comp == END).nonzero()
                n = int(e[0]) + 1 if len(e) else len(comp)
                keep.append((pid[0].cpu(), comp[:n], float(adv[k])))
if a.samples:
    open(a.samples, "w").write("".join(json.dumps(x) + "\n" for x in samples_out)); print("saved", len(samples_out), "samples,", live, "with a non-zero advantage"); sys.exit(0)
assert len(keep) == a.nb, "not enough mixed groups"
tokens = np.full((a.nb, a.t), tok.pad_token_id, dtype=np.int64)
plen, clen, adv = [], [], []
for i, (p, c, A) in enumerate(keep):
    s = torch.cat([p, c]).numpy(); assert len(s) <= a.t, len(s)
    tokens[i, :len(s)] = s; plen.append(len(p)); clen.append(len(c)); adv.append(A)
np.savez(a.out, tokens=tokens, adv=np.array(adv), plen=np.array(plen), clen=np.array(clen))
print("saved", a.out, "plen", plen, "clen", clen, "adv", [round(x, 3) for x in adv])
