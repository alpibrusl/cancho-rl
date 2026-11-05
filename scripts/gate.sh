#!/bin/sh
# The gate (docs/design.md section 14): build `rl`, every reference solves its task, every probe scores what it should, and the Python
# cross-check (a comparison helper) agrees with the cancho oracles.   CANCHO=path/to/cancho scripts/gate.sh
set -u
cd "$(dirname "$0")/.." || exit 1
C="${CANCHO:-cancho}"
mkdir -p build work
"$C" build lib/rl/text.cho lib/rl/grade.cho tools/rl/main.cho --std -o build/rl 2>&1 | grep -v "ld: warning" ; [ -x build/rl ] || { echo "FAIL build"; exit 1; }
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
python3 scripts/crosscheck.py "$C" > work/crosscheck.txt 2>&1 || { echo "FAIL crosscheck"; grep DISAGREE work/crosscheck.txt; fail=1; }
[ $fail = 0 ] && echo "gate: ok"
exit $fail
