#!/bin/bash
# `rl iterate` against scripts/iterate.sh (docs/design.md section 47): the same settings and the same adapters through both, every data file of the two output directories compared byte for byte.
#   (environment of iterate.sh: RL GPU QSW METAL ROOT W TOKJ CHO)  scripts/iterate_check.sh WORKDIR ADAPTERS.f64 [TASKS="abs_val clamp gcd max4"]
set -u
cd "$(dirname "$0")/.." || exit 1
D=$1; AD=$2; : "${TASKS:=abs_val clamp gcd max4}"
rm -rf "$D/sh" "$D/cho"; mkdir -p "$D/cho"/{kv,pre,dec,cand,train,work}
{ for k in RL GPU QSW METAL ROOT W TOKJ CHO; do printf '%s %s\n' "$(echo $k | tr A-Z a-z)" "${!k}"; done; echo "tasks $TASKS"; echo "step 1"; echo "seed 1"; } > "$D/settings.txt"
bash scripts/iterate.sh "$D/sh" "$AD" > "$D/sh.log" 2>&1 || { echo "FAIL iterate.sh"; tail -5 "$D/sh.log"; exit 1; }
"$RL" iterate "$D/settings.txt" "$D/cho" "$AD" > "$D/cho.log" 2>&1 || { echo "FAIL rl iterate"; tail -5 "$D/cho.log"; exit 1; }
bad=0; n=0
for f in chat.txt tasks.txt prompts.txt r.prefill.i32 r.tokens.i32 r.lens.i32 r.uniforms.f64 r.pos0.i32 pre.cfg dec.cfg prefill.out draws.txt c.ids c.meta dec.txt samples.jsonl pk.tokens.i32 pk.weights.f64 train.cfg adamw.json new.params.f64 new.m.f64 new.v.f64 $(cd "$D/sh" && ls cand/c.*.cho res.*.json g*.f64); do
  n=$((n + 1))
  case $f in
    res.*) # the grader's message names the scratch file, which is under the output directory
      [ "$(sed "s#$D/sh#OUT#g" "$D/sh/$f")" = "$(sed "s#$D/cho#OUT#g" "$D/cho/$f")" ] || { echo "DIFFERS $f"; bad=1; } ;;
    *) cmp -s "$D/sh/$f" "$D/cho/$f" || { echo "DIFFERS $f"; bad=1; } ;;
  esac
done
echo "compared $n files"; [ $bad = 0 ] && echo "iterate_check: identical" || echo "iterate_check: DIFFERENT"; exit $bad
