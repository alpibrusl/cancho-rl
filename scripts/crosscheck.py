#!/usr/bin/env python3
"""Compare the cancho oracles with the same functions written in Python (a comparison helper, docs/design.md section 14).

    python3 scripts/crosscheck.py [CANCHO] [TASK...]

For each task it turns the `case:` lines of tasks/<id>/task.spec into a cancho program that answers every case with the value PYTHON
computes, and has `build/rl grade` grade it against the cancho oracle. A task agrees when that program is solved (every case), so a
disagreement is a bug in the oracle, in the Python, or in the case list. Also checks the hand-computed anchors against the Python.
Exit 0 only if every task agrees. Python is not needed to build, grade or train."""
import os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import oracles_py

cancho = sys.argv[1] if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) else os.environ.get("CANCHO", "cancho")
want = [a for a in sys.argv[1:] if not os.path.exists(a)]
py = {t["name"]: t["py"] for t in oracles_py.TASKS}
bad = 0
for name, fn in sorted(py.items()):
    if want and name not in want:
        continue
    for a, v in oracles_py.HAND.get(name, []):
        a = a if isinstance(a, tuple) else (a,)
        if fn(*a) != v:
            print(f"{name}: anchor {a} -> {v}, python says {fn(*a)}"); bad += 1
    d = os.path.join(ROOT, "tasks", name)
    spec = open(os.path.join(d, "task.spec")).read().splitlines()
    ret = next(l[4:].strip() for l in spec if l.startswith("ret:"))
    params = next(l for l in spec if l.startswith("sig:"))[4:].strip()
    cases = [tuple(int(x) for x in l[5:].split()) for l in spec if l.startswith("case:")]
    def lit(v):
        if isinstance(v, bool): return "true" if v else "false"
        return f"(0 - {-v})" if v < 0 else str(v)
    args = [p.split(":")[0].strip() for p in params[params.index("(") + 1:params.index(")")].split(",")]
    body = "".join(f"    if {' && '.join(f'{p} == {lit(x)}' for p, x in zip(args, c))} {{ return {lit(fn(*c))}; }}\n" for c in cases)
    src = f"edition 7;\n\n{params} {{\n{body}    return {'false' if ret == 'bool' else '0'};\n}}\n"
    with tempfile.TemporaryDirectory() as t:
        cand = os.path.join(t, "table.cho"); open(cand, "w").write(src)
        r = subprocess.run([os.path.join(ROOT, "build", "rl"), "grade", cancho, d, cand, os.path.join(t, "scratch.cho")], capture_output=True, text=True)
    ok = '"solved":true' in r.stdout
    print(("agree   " if ok else "DISAGREE ") + name + ("" if ok else "  " + r.stdout.strip()))
    bad += 0 if ok else 1
sys.exit(1 if bad else 0)
