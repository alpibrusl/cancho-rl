#!/bin/bash
# One GRPO iteration, every stage a separate program, each timed (docs/design.md section 43). A shell script on purpose: it finds out what the order and the restarts cost before the driver is
# written in cancho. Needs a Metal GPU; one GPU job at a time.
#
#   RL=build/rl GPU=path/to/cancho-gpu QSW=.../build/qsw METAL=.../build/cancho-gpu-metal-session ROOT=path/to/cancho-gpu-checkout \
#   W=converted-weights-dir TOKJ=tokenizer.json CHO=path/to/cancho \
#   scripts/iterate.sh OUT ADAPTERS.f64 [TASKS="a b c d"] [STEP=n] [SEED=n]
#
# OUT gets one directory a stage. ADAPTERS.f64 is the current adapters (binary64, qwen_ref.lora_names order); the new ones are OUT/new.params.f64 (and the moments OUT/new.m.f64, new.v.f64;
# set M and V to the previous iteration's to continue the optimiser, and STEP to its step number).
set -u
OUT=$1; AD=$2
: "${TASKS:=abs_val clamp gcd max4}" "${STEP:=1}" "${SEED:=1}" "${M:=-}" "${V:=-}" "${STEPS:=400}" "${T:=1152}" "${NBT:=4}" "${GROUP:=8}" "${PAD:=151643}" "${STOP:=151645}"
: "${LR:=0.0001}" "${INVT:=1.25}"
mkdir -p "$OUT"/{kv,pre,dec,cand,train,work}
tm() { local n=$1; shift; local t0=$(date +%s.%N); "$@"; local rc=$?; printf '%-12s %7.1f s\n' "$n" "$(echo "$(date +%s.%N) - $t0" | bc)" | tee -a "$OUT/times.txt" >&2; return $rc; }
die() { echo "FAIL $*" >&2; exit 1; }
bits() { python3 -c 'import struct,sys;print(struct.unpack("<q",struct.pack("<d",float(sys.argv[1])))[0])' "$1"; }
cfg() { # path nb steps [extra...]
  local p=$1 nb=$2 st=$3; shift 3
  { echo "layers 28"; echo "d 1536"; echo "h 12"; echo "hkv 2"; echo "dh 128"; echo "ff 8960"; echo "v 151936"; echo "t $T"; echo "nb $nb"; echo "tied 1"; echo "r 16"; echo "steps $st"
    echo "eps $(bits 1e-6)"; echo "theta $(bits 1e6)"; for x in "$@"; do echo "$x"; done; } > "$p"
}
: > "$OUT/times.txt"
N=$(echo $TASKS | wc -w | tr -d " "); NB=$((N * GROUP))

# 1. prompts: the chat template, then the token ids
rm -f "$OUT/chat.txt"; first=1
for t in $TASKS; do
  [ $first = 1 ] || printf '\n%%%%\n' >> "$OUT/chat.txt"; first=0
  "$RL" chat prompts/primer.md "tasks/$t" >> "$OUT/chat.txt" || die chat
done
echo $TASKS | tr ' ' '\n' > "$OUT/tasks.txt"
tm tokenize "$GPU" tokenize "$TOKJ" "$OUT/chat.txt" > "$OUT/prompts.txt" || die tokenize
tm rollin "$RL" rollin "$OUT/prompts.txt" "$GROUP" "$T" "$STEPS" "$PAD" "$SEED" "$OUT/r" > "$OUT/rollin.json" || die rollin

# 2. prefill (the training forward over the distinct prompts, with the adapters, saving the caches)
cfg "$OUT/pre.cfg" "$N" 1
"$QSW" write --config "$OUT/pre.cfg" --wdir "$W" --out "$OUT/pre" --root "$ROOT" --tokens "$OUT/r.prefill.i32" --adapters "$AD" --no-train --kv-export "$OUT/kv" 2> "$OUT/pre.err" \
  || { cat "$OUT/pre.err"; die "qsw write prefill"; }
run1() { # session, request line(s)
  printf '%s\n' "$2" | tm "$3" "$METAL" "$1" > "$OUT/$4" 2> "$OUT/$4.err" || { cat "$OUT/$4.err"; die "$3"; }
}
run1 "$OUT/pre/session" $'1\ncheckpoint' prefill prefill.out

# 3. decode, from those caches and those adapters
cfg "$OUT/dec.cfg" "$NB" "$STEPS" "invt $(bits $INVT)"
"$QSW" decode --config "$OUT/dec.cfg" --wdir "$W" --out "$OUT/dec" --root "$ROOT" --tokens "$OUT/r.tokens.i32" --lens "$OUT/r.lens.i32" --uniforms "$OUT/r.uniforms.f64" --pos0 "$OUT/r.pos0.i32" \
  --kv-dir "$OUT/kv" --adapters "$OUT/pre" 2> "$OUT/dec.err" || { cat "$OUT/dec.err"; die "qsw decode"; }
run1 "$OUT/dec/session" "$STEPS" decode draws.txt

# 4. completions, text, programs, rewards, samples
tm complete "$RL" complete "$OUT/draws.txt" "$GROUP" "$STEPS" "$OUT/c" > "$OUT/complete.json" || die complete
tm detokenize "$GPU" detokenize "$TOKJ" "$OUT/c.ids" > "$OUT/dec.txt" || die detokenize
tm extract "$RL" extract "$OUT/dec.txt" "$OUT/cand/c" > "$OUT/extract.json" || die extract
grade_one() { k=$1; t=$(sed -n "$((k / GROUP + 1))p" "$OUT/tasks.txt"); "$RL" grade "$CHO" "tasks/$t" "$OUT/cand/c.$k.cho" "$OUT/work/w$k.cho" > "$OUT/res.$k.json" 2>/dev/null; }
export -f grade_one 2>/dev/null; export RL CHO OUT GROUP
gt0=$(date +%s.%N)
for ((k = 0; k < NB; k += 6)); do for ((j = k; j < k + 6 && j < NB; j++)); do grade_one $j & done; wait; done
printf '%-12s %7.1f s\n' grade "$(echo "$(date +%s.%N) - $gt0" | bc)" | tee -a "$OUT/times.txt" >&2
tm samples "$RL" samples "$OUT/c.meta" "$OUT/prompts.txt" "$OUT/tasks.txt" "$OUT/res" "$OUT/samples.jsonl" > "$OUT/samples.json" || die samples
[ "${EVAL:-0}" = 1 ] && exit 0   # an evaluation: sampled and graded, no update
tm pack "$RL" pack "$OUT/samples.jsonl" "$OUT/pk" "$NBT" "$T" "$PAD" > "$OUT/pack.json" || die pack
cat "$OUT/pack.json"
[ -f "$OUT/pk.tokens.i32" ] || { echo "nothing kept: no update (every group had equal rewards)"; exit 0; }
K=$(( $(stat -f %z "$OUT/pk.tokens.i32") / (4 * NBT * T) ))

# 5. the update: K steps of the training session, a checkpoint and a split each, then AdamW
cfg "$OUT/train.cfg" "$NBT" "$K"
"$QSW" write --config "$OUT/train.cfg" --wdir "$W" --out "$OUT/train" --root "$ROOT" --tokens "$OUT/pk.tokens.i32" --adapters "$AD" --weights "$OUT/pk.weights.f64" --no-train --grads "$OUT/train/grads.bin" \
  2> "$OUT/train.err" || { cat "$OUT/train.err"; die "qsw write train"; }
coproc S { "$METAL" "$OUT/train/session" 2> "$OUT/train.run.err"; }
tt0=$(date +%s.%N)
for ((k = 0; k < K; k++)); do
  echo 1 >&"${S[1]}"; read -r _ <&"${S[0]}" || die "training step $k"
  echo checkpoint >&"${S[1]}"; read -r _ <&"${S[0]}" || die "checkpoint $k"
  "$QSW" split --config "$OUT/train.cfg" --grads "$OUT/train/grads.bin" --out "$OUT/g$k.f64" || die split
done
eval "exec ${S[1]}>&-"; wait
printf '%-12s %7.1f s (%d steps)\n' train "$(echo "$(date +%s.%N) - $tt0" | bc)" "$K" | tee -a "$OUT/times.txt"
printf '{"step": %d, "lr": %s, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.0, "clip": 1.0}\n' "$STEP" "$LR" > "$OUT/adamw.json"
gs=(); for ((k = 0; k < K; k++)); do gs+=("$OUT/g$k.f64"); done
tm adamw "$RL" adamw "$OUT/adamw.json" "$AD" "$M" "$V" "$OUT/new" "${gs[@]}" > "$OUT/adamw.out" || die adamw
cat "$OUT/adamw.out"
echo "times:"; cat "$OUT/times.txt"
