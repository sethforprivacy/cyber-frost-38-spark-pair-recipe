#!/usr/bin/env bash
# Mixed-load soak: repeats cyber_bench parts until DURATION seconds pass (default 2700); logs every round.
# Run from bench/ (it calls ./cyber_bench.py) with KEYFILE / API set:  DURATION=2700 bash soak.sh out/soak
D=${DURATION:-2700}; OUT=$1; t0=$(date +%s); r=0
mkdir -p $OUT
while [ $(( $(date +%s) - t0 )) -lt $D ]; do
  r=$((r+1)); python3 cyber_bench.py $OUT/round$r decode16 multiturn hotturn prefill > $OUT/round$r.log 2>&1
  echo "round $r rc=$? $(date -u +%T) $(grep -cE "Traceback|HTTPError|URLError" $OUT/round$r.log) errors; $(grep -E "^DECODE16|^MULTITURN" $OUT/round$r.log | cut -c1-160 | tr "\n" " ")"
done
echo SOAK_DONE
