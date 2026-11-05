"""R1 (docs/design.md section 26): GRPO with peft LoRA on torch/MPS, the established-library reference. A comparison helper, not part of the package:
rewards come from `rl grade` (cancho), tasks and the split from `rl`.

  CANCHO=... python3 scripts/r1_grpo.py --run NAME --seed 1 --iters 40 --eval-every 10 [--smoke]
"""
import argparse, json, os, random, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--iters", type=int, default=40)
ap.add_argument("--eval-every", type=int, default=10)
ap.add_argument("--tasks-per-iter", type=int, default=4)
ap.add_argument("--group", type=int, default=8)
ap.add_argument("--eval-samples", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--max-new", type=int, default=400)
ap.add_argument("--temp", type=float, default=0.8)
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--pool", default="work/r1_pool.txt")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()

CANCHO = os.environ.get("CANCHO", "cancho")
OUT = f"work/r1/{a.run}"
os.makedirs(OUT, exist_ok=True)
random.seed(a.seed); torch.manual_seed(a.seed)
dev = "mps"

def rl(*args):
    return subprocess.run(["./build/rl", *args], capture_output=True, text=True).stdout

tasks = sorted(os.listdir("tasks"))
side = {t: rl("split", f"tasks/{t}").strip() for t in tasks}
heldout = [t for t in tasks if side[t] == "heldout"]
pool = [t.strip() for t in open(a.pool) if t.strip()]
assert all(side[t] == "train" for t in pool), "pool must be train-side tasks"
if a.smoke:
    heldout = heldout[:3]; pool = pool[:4]

tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)

prompt_cache = {}
def prompt_ids(t):
    if t not in prompt_cache:
        text = rl("prompt", "prompts/primer.md", f"tasks/{t}")
        prompt_cache[t] = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return prompt_cache[t]

def extract(text):
    out, on = [], False
    for ln in text.split("\n"):
        if ln.startswith("```cancho"): on = True; continue
        if ln.startswith("```"): on = False; continue
        if on: out.append(ln)
    return "\n".join(out) + "\n"

def grade(args):
    t, k, text = args
    cand = f"{OUT}/cand_{t}_{k}.cho"
    open(cand, "w").write(extract(text))
    r = rl("grade", CANCHO, f"tasks/{t}", cand, f"{OUT}/work_{t}_{k}.cho")
    try:
        j = json.loads(r)
        return j["reward"], bool(j["solved"]), j["status"]
    except Exception:
        return 0.0, False, "grade_error"

def generate(t, n):
    enc = tok([prompt_ids(t)], return_tensors="pt").to(dev)
    model.eval()
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=a.temp, top_p=1.0, top_k=0, max_new_tokens=a.max_new, use_cache=True, num_return_sequences=n, pad_token_id=tok.pad_token_id)
    L = enc["input_ids"].shape[1]
    return enc["input_ids"], out, L

def decode(out, L):
    return [tok.decode(o[L:], skip_special_tokens=True) for o in out]

def evaluate(step):
    t0 = time.time(); rows = []
    jobs = []
    for t in heldout:
        _, out, L = generate(t, a.eval_samples)
        for k, txt in enumerate(decode(out, L)): jobs.append((t, f"e{step}_{k}", txt))
    with ThreadPoolExecutor(6) as ex: res = list(ex.map(grade, jobs))
    per = {}
    for (t, _, _), (r, s, st) in zip(jobs, res): per.setdefault(t, []).append(1.0 if s else 0.0)
    rate = sum(sum(v) / len(v) for v in per.values()) / len(per)
    rec = {"kind": "eval", "step": step, "heldout_tasks": len(per), "samples": len(jobs), "rate": rate, "per_task": {t: sum(v) / len(v) for t, v in per.items()}, "secs": round(time.time() - t0, 1)}
    print(json.dumps({k: v for k, v in rec.items() if k != "per_task"}), flush=True)
    open(f"{OUT}/log.jsonl", "a").write(json.dumps(rec) + "\n")

def train_step(step):
    t0 = time.time()
    chosen = random.sample(pool, min(a.tasks_per_iter, len(pool)))
    seqs, advs, masks_info = [], [], []
    stats = {"reward": [], "solved": [], "status": {}}
    for t in chosen:
        pids, out, L = generate(t, a.group)
        texts = decode(out, L)
        with ThreadPoolExecutor(6) as ex: res = list(ex.map(grade, [(t, f"t{step}_{k}", x) for k, x in enumerate(texts)]))
        rs = torch.tensor([r for r, _, _ in res])
        stats["reward"] += rs.tolist(); stats["solved"] += [1 if s else 0 for _, s, _ in res]
        for _, _, st in res: stats["status"][st] = stats["status"].get(st, 0) + 1
        adv = (rs - rs.mean()) / (rs.std() + 1e-4) if rs.std() > 0 else torch.zeros_like(rs)
        for k in range(a.group):
            if adv[k] != 0:
                seqs.append((out[k].cpu(), L)); advs.append(float(adv[k]))
    gen_secs = time.time() - t0
    loss_val = 0.0
    if seqs:
        model.train(); opt.zero_grad()
        # token-mean over the whole batch
        ntok = 0
        parts = []
        for ids, L in seqs:
            comp = ids[L:]
            end = (comp == tok.convert_tokens_to_ids("<|im_end|>")).nonzero()
            n = (int(end[0]) + 1) if len(end) else len(comp)
            parts.append(n); ntok += n
        mb = 4
        for i in range(0, len(seqs), mb):
            batch = seqs[i:i + mb]
            W = max(len(s) for s, _ in batch)
            ids = torch.full((len(batch), W), tok.pad_token_id); att = torch.zeros((len(batch), W), dtype=torch.long); cmask = torch.zeros((len(batch), W))
            for j, (s, L) in enumerate(batch):
                ids[j, :len(s)] = s  # right-padded: the left pad of the prompt is part of s already
                att[j, :len(s)] = (s != tok.pad_token_id).long()
                att[j, :L] = (s[:L] != tok.pad_token_id).long()
                n = parts[i + j]; cmask[j, L:L + n] = 1
            pos = (att.cumsum(1) - 1).clamp(min=0)
            ids, att, cmask, pos = ids.to(dev), att.to(dev), cmask.to(dev), pos.to(dev)
            logits = model(input_ids=ids, attention_mask=att, position_ids=pos).logits[:, :-1].float()
            logp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]
            w = torch.tensor(advs[i:i + len(batch)], device=dev)[:, None]
            loss = -(w * logp * cmask[:, 1:]).sum() / ntok
            loss.backward(); loss_val += float(loss.detach())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
    rec = {"kind": "train", "step": step, "tasks": chosen, "mean_reward": sum(stats["reward"]) / len(stats["reward"]), "solve_rate": sum(stats["solved"]) / len(stats["solved"]),
           "used_samples": len(seqs), "loss": loss_val, "status": stats["status"], "gen_secs": round(gen_secs, 1), "secs": round(time.time() - t0, 1)}
    print(json.dumps(rec), flush=True)
    open(f"{OUT}/log.jsonl", "a").write(json.dumps(rec) + "\n")

print(json.dumps({"run": a.run, "seed": a.seed, "model": a.model, "pool": len(pool), "heldout": len(heldout), "args": vars(a), "torch": torch.__version__}), flush=True)
evaluate(0)
for step in range(1, a.iters + 1):
    train_step(step)
    if step % a.eval_every == 0 or step == a.iters: evaluate(step)
