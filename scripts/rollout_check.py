#!/usr/bin/env python3
"""The rollout glue after the decode (docs/design.md section 42) against transformers and r1_grpo.py's own functions, with no GPU: replies written by hand (a reference solution in a fenced block,
a wrong one, prose, a reply cut off at the step limit, a reply followed by more draws after its end-of-turn token) are encoded with Hugging Face, laid out as a decode session's answer, and
sent through `rl complete`, `cancho-gpu detokenize`, `rl extract`, `rl grade` and `rl samples`; then `rl pack` reads the samples.

    python3 scripts/rollout_check.py BUILD/rl CANCHO-GPU TOKENIZER.json MODEL_ID CANCHO"""
import json, os, random, subprocess, sys, tempfile
from transformers import AutoTokenizer

RL, GPU, TOKJ, MODEL, CANCHO = sys.argv[1:6]
bad = 0
def say(ok, text):
    global bad
    print(("ok   " if ok else "FAIL ") + text); bad += 0 if ok else 1

def extract(text):  # scripts/r1_grpo.py
    out, on = [], False
    for ln in text.split("\n"):
        if ln.startswith("```cancho"): on = True; continue
        if ln.startswith("```"): on = False; continue
        if on: out.append(ln)
    return "\n".join(out) + "\n"

tok = AutoTokenizer.from_pretrained(MODEL)
STOP, EOT = 151645, 151643
tasks = ["abs_val", "max4"]
G, STEPS = 4, 160
d = tempfile.mkdtemp(prefix="rollout_")
ref = {t: open(f"tasks/{t}/reference.cho").read() for t in tasks}
def block(code): return "```cancho\n" + code + "```"
replies = {  # per task: four replies and how each ends
    0: [(block(ref["abs_val"]), "stop"), ("Here is the code:\n" + block("edition 7;\nfn abs_val(n: int) -> [] int { return n; }\n") + "\nIt returns n.", "stop"), ("I think you should use a loop. 你好 🙂 \"quoted\" \\ back", "stop"), (block(ref["abs_val"]) + " and then it goes on and on", "cut")],
    1: [(block(ref["max4"]), "eot"), ("no code at all", "stop"), (block(ref["max4"]), "stop"), ("```cancho\nnot a program\n```", "stop")],
}
rng = random.Random(3)
rows, want = [], []
for t_i in (0, 1):
    for text, how in replies[t_i]:
        ids = tok(text)["input_ids"]
        if how == "cut":
            ids = (ids * 20)[:STEPS]                      # runs to the step limit with no stop
            comp, dec_ids = list(ids), list(ids)
        else:
            end = STOP if how == "stop" else EOT
            comp, dec_ids = ids + [end], list(ids)
            ids = comp + [rng.randrange(1000) for _ in range(STEPS - len(comp))]   # more draws after the stop: ignored
        rows.append([i + 1 for i in ids[:STEPS]]); want.append((comp, dec_ids, how))
open(f"{d}/draws", "w").write("\n".join(" ".join(map(str, r)) for r in rows) + "\n")
r = subprocess.run([RL, "complete", f"{d}/draws", str(G), str(STEPS), f"{d}/c"], capture_output=True, text=True)
say(r.returncode == 0 and json.loads(r.stdout) == {"sequences": 8, "stopped": 7}, f"rl complete: {r.stdout.strip()} {r.stderr.strip()[:80]}")
meta = open(f"{d}/c.meta").read().splitlines()
ok = True
for k, (comp, dec_ids, how) in enumerate(want):
    head, ids = meta[k].split(":")
    rr, task, stopped, n = head.split()
    ok &= (int(rr), int(task), int(stopped), int(n)) == (k, k // G, 0 if how == "cut" else 1, len(comp)) and list(map(int, ids.split())) == comp
say(ok, "rl complete: every completion is cut at its first end-of-turn or end-of-text token (stop kept) and a reply with none keeps all the steps")
dec = subprocess.run([GPU, "detokenize", TOKJ, f"{d}/c.ids"], capture_output=True, text=True)
say(dec.returncode == 0 and len(dec.stdout.splitlines()) == 8, "cancho-gpu detokenize decodes the eight replies")
open(f"{d}/dec", "w").write(dec.stdout)
r = subprocess.run([RL, "extract", f"{d}/dec", f"{d}/cand"], capture_output=True, text=True)
say(r.returncode == 0 and json.loads(r.stdout)["programs"] == 8, f"rl extract: {r.stdout.strip()}")
same = all(open(f"{d}/cand.{k}.cho").read() == extract(tok.decode(dec_ids, skip_special_tokens=True)) for k, (_, dec_ids, _) in enumerate(want))
say(same, "rl extract: all eight programs equal r1_grpo.py's extract of transformers' decode")
for k in range(8):
    t = tasks[k // G]
    out = subprocess.run([RL, "grade", CANCHO, f"tasks/{t}", f"{d}/cand.{k}.cho", f"{d}/work{k}.cho"], capture_output=True, text=True).stdout
    open(f"{d}/res.{k}.json", "w").write(out)
rewards = [json.loads(open(f"{d}/res.{k}.json").read())["reward"] for k in range(8)]
say(rewards[0] == 1.0 and rewards[2] == 0.0 and rewards[3] == 1.0 and rewards[4] == 1.0 and rewards[5] == 0.0 and rewards[6] == 1.0 and rewards[7] == 0.0, f"rl grade: rewards {rewards}")
os.remove(f"{d}/res.5.json")   # a missing result is a reward of 0, not a crash
ptxt = subprocess.run([GPU, "tokenize", TOKJ, subprocess.run(["sh", "-c", f"printf 'one\\n%%%%\\ntwo\\n' > {d}/p.txt; echo {d}/p.txt"], capture_output=True, text=True).stdout.strip()], capture_output=True, text=True).stdout
open(f"{d}/prompts", "w").write(ptxt)
open(f"{d}/tasks", "w").write("\n".join(tasks) + "\n")
r = subprocess.run([RL, "samples", f"{d}/c.meta", f"{d}/prompts", f"{d}/tasks", f"{d}/res", f"{d}/samples.jsonl"], capture_output=True, text=True)
say(r.returncode == 0 and json.loads(r.stdout)["samples"] == 8, f"rl samples: {r.stdout.strip()} {r.stderr.strip()[:80]}")
lines = [json.loads(x) for x in open(f"{d}/samples.jsonl")]
pids = [list(map(int, ln.split(":")[1].split())) for ln in ptxt.splitlines() if ln.startswith("ids ")]
good = len(lines) == 8
for k, s in enumerate(lines):
    good &= s["group"] == tasks[k // G] and s["completion"] == want[k][0] and s["prompt"] == pids[k // G] and s["reward"] == (0.0 if k == 5 else rewards[k])
say(good, "rl samples: group, prompt ids, completion ids (stop token kept) and reward of every line; the missing result scores 0")
r = subprocess.run([RL, "pack", f"{d}/samples.jsonl", f"{d}/pk", "8", "256", str(STOP)], capture_output=True, text=True)
say(r.returncode == 0, f"rl pack reads the samples file: {r.stdout.strip()} {r.stderr.strip()[:100]}")
for name, args, rule in (("a draws file that is not whole rows", [RL, "complete", f"{d}/draws", "4", "159", f"{d}/x"], "rl.complete.shape"), ("a missing draws file", [RL, "complete", f"{d}/none", "4", "160", f"{d}/x"], "rl.complete.input"),
                         ("a task index past the tasks file", [RL, "samples", f"{d}/c.meta", f"{d}/prompts", f"{d}/tasks1", f"{d}/res", f"{d}/x"], "rl.samples.task"), ("a missing meta file", [RL, "samples", f"{d}/none", f"{d}/prompts", f"{d}/tasks", f"{d}/res", f"{d}/x"], "rl.samples.input")):
    if rule == "rl.samples.task":
        open(f"{d}/tasks1", "w").write("abs_val\n")
    r = subprocess.run(args, capture_output=True, text=True)
    say(r.returncode == 2 and rule in r.stderr, f"refusal {rule}: {name}")
print("bad" if bad else "all ok"); sys.exit(1 if bad else 0)
