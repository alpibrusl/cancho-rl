#!/bin/bash
# R4's loop (docs/design.md section 44): ITERS iterations of scripts/iterate.sh from the starting adapters, the moments carried over, with a held-out evaluation before the first and every
# EVERY. R1's protocol: 4 train tasks an iteration from work/r1_pool.txt, 8 samples each, lr 1e-4; the evaluation is every held-out task (rl split) x 8 samples (R1: 4; the prefill saves 8 copies of a prompt's caches, so a batch is 4 tasks x 8), the rate the mean over tasks
# of the fraction solved. The environment of scripts/iterate.sh, and
#   scripts/loop.sh OUT ADAPTERS0.f64 [ITERS=40] [EVERY=10] [SEED=1]
set -u
OUT=$1; AD0=$2; ITERS=${3:-40}; EVERY=${4:-10}; SEED=${5:-1}
mkdir -p "$OUT"; : > "$OUT/log.jsonl"
HELD=$(for t in $(ls tasks); do [ "$("$RL" split "tasks/$t")" = heldout ] && echo "$t"; done)
POOL=($(cat work/r1_pool.txt))
solved_rate() { # dirs...: mean over tasks of the fraction solved, from res.k.json and tasks.txt
  for d in "$@"; do n=$(wc -l < "$d/tasks.txt" | tr -d ' '); g=$(( $(ls "$d"/res.*.json | wc -l) / n ))
    for ((k = 0; k < n * g; k++)); do echo "$(sed -n "$((k / g + 1))p" "$d/tasks.txt") $(grep -c '"solved":true' "$d/res.$k.json")"; done; done |
    awk '{s[$1]+=$2; c[$1]++} END{for(t in s){r+=s[t]/c[t]; n++}; printf "%.6f %d", r/n, n}'
}
evaluate() { # step adapters
  local step=$1 ad=$2 t0=$(date +%s) dirs=() b=0 batch
  set -- $HELD
  while [ $# -gt 0 ]; do
    batch="${*:1:4}"; shift $(( $# < 4 ? $# : 4 ))
    TASKS="$batch" GROUP=8 EVAL=1 SEED=$((1000 + step * 10 + b)) bash scripts/iterate.sh "$OUT/e${step}_$b" "$ad" > "$OUT/e${step}_$b.log" 2>&1 || { echo "FAIL eval $step batch $b"; exit 1; }
    dirs+=("$OUT/e${step}_$b"); b=$((b + 1))
  done
  read -r rate n <<< "$(solved_rate "${dirs[@]}")"
  echo "{\"kind\":\"eval\",\"step\":$step,\"heldout_tasks\":$n,\"rate\":$rate,\"secs\":$(( $(date +%s) - t0 ))}" | tee -a "$OUT/log.jsonl"
  rm -rf "${dirs[@]}"
}
AD=$AD0; M=-; V=-
evaluate 0 "$AD"
for ((i = 1; i <= ITERS; i++)); do
  tasks=$(awk -v s=$((SEED * 1000 + i)) -v p="${POOL[*]}" 'BEGIN{srand(s); n=split(p,a," "); for(k=0;k<4;k++){j=k+int(rand()*(n-k)); x=a[k+1]; a[k+1]=a[j+1]; a[j+1]=x; printf "%s ", a[k+1]}}')
  t0=$(date +%s)
  TASKS="$tasks" M=$M V=$V STEP=$i SEED=$((SEED * 1000 + i)) bash scripts/iterate.sh "$OUT/i$i" "$AD" > "$OUT/i$i.log" 2>&1 || { echo "FAIL iteration $i"; tail -5 "$OUT/i$i.log"; exit 1; }
  read -r rate n <<< "$(solved_rate "$OUT/i$i")"
  kept=$(sed -n 's/.*"kept":\([0-9]*\).*/\1/p' "$OUT/i$i/pack.json" | tail -1)
  gn=$(sed -n 's/.*"grad_norm":\([0-9.e-]*\).*/\1/p' "$OUT/i$i/adamw.out" | tail -1)
  echo "{\"kind\":\"train\",\"step\":$i,\"tasks\":\"${tasks% }\",\"rate\":$rate,\"kept\":${kept:-0},\"grad_norm\":${gn:-0},\"secs\":$(( $(date +%s) - t0 ))}" | tee -a "$OUT/log.jsonl"
  if [ -f "$OUT/i$i/new.params.f64" ]; then
    [ $i -gt 1 ] && [ $(( (i - 1) % EVERY )) -ne 0 ] && rm -f "$OUT/i$((i - 1))"/new.*.f64 "$OUT/i$((i - 1))"/new.params.f64
    AD="$OUT/i$i/new.params.f64"; M="$OUT/i$i/new.m.f64"; V="$OUT/i$i/new.v.f64"
  fi
  rm -rf "$OUT/i$i"/{kv,pre,dec,train,work,cand} "$OUT/i$i"/g*.f64
  if [ $((i % EVERY)) -eq 0 ] || [ $i -eq $ITERS ]; then evaluate "$i" "$AD"; fi
done
echo "done: adapters $AD"
