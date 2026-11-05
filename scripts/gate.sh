#!/bin/sh
# The gate (docs/design.md section 14): build `rl`, every reference solves its task, every probe scores what it should, and the Python
# cross-check (a comparison helper) agrees with the cancho oracles.   CANCHO=path/to/cancho scripts/gate.sh
set -u
cd "$(dirname "$0")/.." || exit 1
C="${CANCHO:-cancho}"
mkdir -p build work
"$C" build lib/rl/text.cho lib/rl/side.cho lib/rl/grade.cho lib/rl/report.cho tools/rl/main.cho --std -o build/rl 2>&1 | grep -v "ld: warning" ; [ -x build/rl ] || { echo "FAIL build"; exit 1; }
fail=0
for t in $(ls tasks); do
    out=$(./build/rl grade "$C" "tasks/$t" "tasks/$t/reference.cho" "work/$t.cho")
    case "$out" in *'"solved":true'*) ;; *) echo "FAIL reference $t: $out"; fail=1 ;; esac
done
expect() { # probe, status, passed
    out=$(./build/rl grade "$C" tasks/fib "probes/$1.cho" work/p.cho)
    case "$out" in *"\"passed\":$3,"*"\"status\":\"$2\""*) ;; *) echo "FAIL probe $1: wanted $2/$3, got $out"; fail=1 ;; esac
}
expect empty build_error 0
expect garbage build_error 0
expect div_zero ok 0
expect const_zero ok 1
expect early_exit ok 1
expect mutant_fib ok 1
[ "${GATE_SLOW:-0}" = 1 ] && expect loop_forever timeout 0
# the prompt never carries a hidden case, and the split rule is the documented one (FNV-1a 32, mod 4 == 0 held out)
for t in $(ls tasks); do
    ./build/rl prompt prompts/primer.md "tasks/$t" | grep -q 'case:' && { echo "FAIL prompt leaks a case: $t"; fail=1; }
done
heldout=$(for t in $(ls tasks); do [ "$(./build/rl split "tasks/$t")" = heldout ] && echo "$t"; done | tr '\n' ' ')
# the same rule written independently in Python (a comparison helper): FNV-1a 32 bit over the task name, mod 4 == 0
want=$(python3 -c 'import sys
for n in sys.argv[1:]:
    h = 2166136261
    for b in n.encode():
        h = ((h ^ b) * 16777619) % 4294967296
    print(n) if h % 4 == 0 else None' $(ls tasks) | tr '\n' ' ')
[ "$heldout" = "$want" ] || { echo "FAIL split: $heldout"; fail=1; }
# every task has a kind, and it is what the reference does: "loop" when the reference has a `while` or calls itself, else "branch" (a mechanical rule, not a judgement)
for t in $(ls tasks); do
    r=tasks/$t/reference.cho
    if grep -q 'while' "$r" || [ "$(grep -c "$t(" "$r")" -ge 2 ]; then want=loop; else want=branch; fi
    [ "$(grep '^kind:' tasks/$t/task.spec | sed 's/^kind: *//')" = "$want" ] || { echo "FAIL kind of $t is not $want"; fail=1; }
done
# `rl report` on a fixture with known counts (a detail field that contains an escaped "solved":true must not count), and its refusals
./build/rl report tasks probes/report_fixture.jsonl > work/report.out
want_report='{"group":"branch/train","tasks":1,"samples":2,"solved":1,"rate":0.5,"zero":0,"window":1,"high":0}
{"group":"branch/heldout","tasks":0,"samples":0,"solved":0,"rate":0.0,"zero":0,"window":0,"high":0}
{"group":"loop/train","tasks":1,"samples":3,"solved":0,"rate":0.0,"zero":1,"window":0,"high":0}
{"group":"loop/heldout","tasks":0,"samples":0,"solved":0,"rate":0.0,"zero":0,"window":0,"high":0}
{"group":"all","tasks":2,"samples":5,"solved":1,"rate":0.2,"zero":1,"window":1,"high":0}'
[ "$(cat work/report.out)" = "$want_report" ] || { echo "FAIL report fixture"; fail=1; }
./build/rl report tasks /nonexistent 2>&1 | grep -q '"rl.input"' || { echo "FAIL report refusal rl.input"; fail=1; }
./build/rl report 2>&1 | grep -q '"rl.usage"' || { echo "FAIL report refusal rl.usage"; fail=1; }
echo '{"id":"no_such_task","solved":true}' > work/report_bad.jsonl
./build/rl report tasks work/report_bad.jsonl 2>&1 | grep -q '"rl.task"' || { echo "FAIL report refusal rl.task"; fail=1; }
python3 scripts/crosscheck.py "$C" > work/crosscheck.txt 2>&1 || { echo "FAIL crosscheck"; grep DISAGREE work/crosscheck.txt; fail=1; }
[ $fail = 0 ] && echo "gate: ok"
exit $fail
