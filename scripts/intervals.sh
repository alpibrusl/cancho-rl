#!/bin/sh
# Rates with intervals from results files: the bootstrap is over TASKS (a task's samples are not independent draws of the task), 95%, seed 1, 4000 replicates.
#   scripts/intervals.sh RESULTS...      needs build/rl and work/tools/ (scripts/tools.sh)
# One line per group: tasks, the mean over tasks of the per-task solve rate, and its interval; then the status counts and the sha256 of the results.
set -eu
cd "$(dirname "$0")/.."
[ -x work/tools/stats ] || { echo "run scripts/tools.sh first" >&2; exit 2; }
mkdir -p work
./build/rl tasks tasks "$@" > work/tasks.csv
boot() { # label kind side
    awk -F, -v k="$2" -v s="$3" 'NR==1 || ((k=="all" || $2==k) && (s=="all" || $3==s))' work/tasks.csv > work/group.csv
    n=$(($(wc -l < work/group.csv) - 1))
    [ "$n" -gt 1 ] || { echo "$1: tasks=$n (no interval)"; return; }
    work/tools/stats ci boot work/group.csv --cols rate --statistic mean --seed 1 --replicates 4000 --level 0.95 |
        sed -n 's/.*"estimate":\([0-9.]*\),"low":\([0-9.]*\),"high":\([0-9.]*\).*/\1 \2 \3/p' | head -1 |
        { read e lo hi; echo "$1: tasks=$n mean_rate=$e interval=[$lo, $hi]"; }
}
boot all all all
boot branch/train branch train
boot branch/heldout branch heldout
boot loop/train loop train
boot loop/heldout loop heldout
echo "statuses:"; cat "$@" | work/tools/tally --field 16 --delim '"' --top 6 --format text
for f in "$@"; do work/tools/hash --root . "$f" | sed -n 's/.*"path":"\([^"]*\)".*"hex":"\([0-9a-f]*\)".*/sha256 \2  \1/p'; done
