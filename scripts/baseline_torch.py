#!/usr/bin/env python3
"""The step-0 baseline from torch (docs/design.md section 45): the base model (peft's LoRA with B = 0 changes nothing) sampled on the held-out tasks with R1's own prompt, generation settings, `extract` and
`rl grade`, in the number format and on the device asked for, to find out whether R1's step-0 held-out rate (0.071, 0.107) and ours (0.254) differ because of the sampler or the arithmetic.

    CANCHO=... python3 scripts/baseline_torch.py OUT --dtype bfloat16|float32 --device mps|cpu [--samples 8] [--seed 1] [--tasks N] [--greedy]"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--device", default="mps")
ap.add_argument("--samples", type=int, default=8); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--tasks", type=int, default=0)
ap.add_argument("--greedy", action="store_true"); ap.add_argument("--rep", type=float, default=None, help="repetition_penalty; unset keeps the model's generation_config (1.1 for Qwen2.5-Instruct), as R1 does"); ap.add_argument("--temp", type=float, default=0.8); ap.add_argument("--max-new", type=int, default=400); ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
a = ap.parse_args()
CANCHO = os.environ.get("CANCHO", "cancho"); OUT = a.out; os.makedirs(OUT, exist_ok=True)
torch.manual_seed(a.seed)
rl = lambda *args: subprocess.run(["./build/rl", *args], capture_output=True, text=True).stdout
tasks = sorted(os.listdir("tasks"))
heldout = [t for t in tasks if rl("split", f"tasks/{t}").strip() == "heldout"]
if a.tasks: heldout = heldout[:a.tasks]
tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(a.device).eval()

def extract(text):  # scripts/r1_grpo.py
    out, on = [], False
    for ln in text.split("\n"):
        if ln.startswith("```cancho"): on = True; continue
        if ln.startswith("```"): on = False; continue
        if on: out.append(ln)
    return "\n".join(out) + "\n"

def grade(args):
    t, k, text = args
    cand = f"{OUT}/cand_{t}_{k}.cho"; open(cand, "w").write(extract(text))
    r = rl("grade", CANCHO, f"tasks/{t}", cand, f"{OUT}/work_{t}_{k}.cho")
    try: j = json.loads(r); return bool(j["solved"]), j["status"]
    except Exception: return False, "grade_error"

t0 = time.time(); jobs = []; stops = []
for t in heldout:
    text = tok.apply_chat_template([{"role": "user", "content": rl("prompt", "prompts/primer.md", f"tasks/{t}")}], tokenize=False, add_generation_prompt=True)
    enc = tok([text], return_tensors="pt").to(a.device)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False, repetition_penalty=1.0, max_new_tokens=a.max_new, use_cache=True, pad_token_id=tok.pad_token_id) if a.greedy else model.generate(**enc, do_sample=True, temperature=a.temp, top_p=1.0, top_k=0, max_new_tokens=a.max_new, use_cache=True, num_return_sequences=a.samples, pad_token_id=tok.pad_token_id, **({"repetition_penalty": a.rep} if a.rep is not None else {}))
    L = enc["input_ids"].shape[1]
    for k, o in enumerate(out):
        ids = o[L:].tolist(); stops.append(any(i in (151645, 151643) for i in ids))
        jobs.append((t, k, tok.decode(ids, skip_special_tokens=True)))
    print(f"{t} generated ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
with ThreadPoolExecutor(6) as ex: res = list(ex.map(grade, jobs))
per = {}; status = {}
for (t, _, _), (s, st) in zip(jobs, res): per.setdefault(t, []).append(1.0 if s else 0.0); status[st] = status.get(st, 0) + 1
rate = sum(sum(v) / len(v) for v in per.values()) / len(per)
print(json.dumps({"dtype": a.dtype, "device": a.device, "seed": a.seed, "rep": a.rep, "tasks": len(per), "samples": len(jobs), "rate": rate, "stopped": sum(stops) / len(stops), "status": status, "secs": round(time.time() - t0)}))
json.dump({t: sum(v) / len(v) for t, v in per.items()}, open(f"{OUT}/per_task.json", "w"))
