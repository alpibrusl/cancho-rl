#!/usr/bin/env python3
"""`rl chat` + `cancho-gpu tokenize` against transformers: for every task, the prompt ids cancho produces (the chat-templated prompt of `rl chat`, encoded by cancho-gpu's BPE) equal
`AutoTokenizer.apply_chat_template([{"role": "user", "content": <rl prompt>}], add_generation_prompt=True)` as r1_grpo.py builds them.

    python3 scripts/chat_check.py BUILD/rl CANCHO-GPU TOKENIZER.json MODEL_ID [TASKS_DIR]"""
import os, subprocess, sys, tempfile
from transformers import AutoTokenizer

RL, GPU, TOKJ, MODEL = sys.argv[1:5]
TASKS = sys.argv[5] if len(sys.argv) > 5 else "tasks"
tok = AutoTokenizer.from_pretrained(MODEL)
names = sorted(os.listdir(TASKS))
texts, want = [], []
for t in names:
    plain = subprocess.run([RL, "prompt", "prompts/primer.md", f"{TASKS}/{t}"], capture_output=True, text=True, check=True).stdout
    chat = subprocess.run([RL, "chat", "prompts/primer.md", f"{TASKS}/{t}"], capture_output=True, text=True, check=True).stdout
    templ = tok.apply_chat_template([{"role": "user", "content": plain}], tokenize=False, add_generation_prompt=True)
    assert chat == templ, f"{t}: rl chat text differs from the template's"
    assert "\n%%\n" not in chat
    texts.append(chat)
    want.append(tok(templ)["input_ids"])
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write("\n%%\n".join(texts))
out = subprocess.run([GPU, "tokenize", TOKJ, f.name], capture_output=True, text=True, check=True).stdout.splitlines()
ids = [list(map(int, ln.split(":", 1)[1].split())) for ln in out if ln.startswith("ids ")]
assert len(ids) == len(want), (len(ids), len(want))
bad = [t for t, a, b in zip(names, ids, want) if a != b]
print(f"{len(names)} tasks: the text of `rl chat` equals the template's for all; token ids differ for {len(bad)}: {bad[:5]}")
print("lengths", min(map(len, ids)), max(map(len, ids)))
sys.exit(1 if bad else 0)
