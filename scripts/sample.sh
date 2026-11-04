#!/bin/sh
# Sample K completions per task from a model with mlx_lm (the other framework: a runner, not part of the package) and grade each with `rl grade`.
#   CANCHO=... scripts/sample.sh MODEL K OUTDIR [TASK...]
# Refuses to start while another GPU job runs (workspace rule: one GPU job at a time). Writes OUTDIR/<task>/<k>.cho and OUTDIR/results.jsonl.
set -u
cd "$(dirname "$0")/.." || exit 1
model=$1; k=$2; out=$3; shift 3
if pgrep -f "m8_train|mlx_lm|train.py" >/dev/null; then echo "another GPU job is running; not starting" >&2; exit 3; fi
tasks=${*:-$(ls tasks)}
mkdir -p "$out"; : > "$out/results.jsonl"
for t in $tasks; do
    mkdir -p "$out/$t"
    prompt=$(./build/rl prompt prompts/primer.md "tasks/$t")
    i=1
    while [ $i -le "$k" ]; do
        mlx_lm.generate --model "$model" --prompt "$prompt" --temp 0.8 --max-tokens 600 --seed $i 2>/dev/null |
            awk '/^```cancho/{on=1;next} /^```/{on=0} on' > "$out/$t/$i.cho"
        ./build/rl grade "${CANCHO:-cancho}" "tasks/$t" "$out/$t/$i.cho" "work/$t.$i.cho" >> "$out/results.jsonl"
        i=$((i+1))
    done
done
