#!/bin/sh
# Build the sibling tools cancho-rl uses, each at a pinned commit (docs/design.md section 22), into work/tools/:
#   stats  (cancho-stats)  `stats ci boot`, the bootstrap interval of a rate
#   tally, hash  (cancho-tools)  counts of statuses and errors; a digest of a results file
#   CANCHO=/path/to/cancho scripts/tools.sh
# They pin an older compiler than this repository's, so the build passes --ignore-compiler-rev (said in section 22; the builds are checked by their output, not by trust).
set -eu
cd "$(dirname "$0")/.."
C=${CANCHO:-cancho}
STATS_REV=fbb34c875551e9425c2149172894015142fb3904
TOOLS_REV=3399bd7a6c29d1bab3af7fac9021d2aa33644af1
mkdir -p work/tools/src
fetch() { # repo rev
    [ -d "work/tools/src/$1" ] || gh repo clone "alpibrusl/$1" "work/tools/src/$1" -- -q
    (cd "work/tools/src/$1" && git fetch -q origin "$2" 2>/dev/null; git checkout -q "$2")
}
fetch cancho-stats $STATS_REV
fetch cancho-tools $TOOLS_REV
(cd work/tools/src/cancho-stats && "$C" install --ignore-compiler-rev >/dev/null && "$C" build --bin stats --ignore-compiler-rev 2>&1 | grep -v 'ld: warning' | grep -v '^note:' ; cp build/stats ../../stats)
for b in tally hash; do
    (cd work/tools/src/cancho-tools && "$C" install --ignore-compiler-rev >/dev/null && "$C" build --bin $b --ignore-compiler-rev 2>&1 | grep -v 'ld: warning' | grep -v '^note:'; cp build/$b ../../$b)
done
echo "tools: stats@$STATS_REV tally,hash@$TOOLS_REV"
