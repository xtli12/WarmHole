#!/bin/bash
# Progress of the Wormhole v2 runs: episodes done / planned per run directory, live workers, GPUs.
BH=${BH:-/data/blackhole}
for spec in $BH/specs/w2/*_part*.jsonl; do
  b=$(basename $spec .jsonl); bb=${b%%_*}; rest=${b#*_}; name=${rest%_part*}; k=${rest##*_part}
  d=$BH/runs/${bb}_${name}_p$k
  [ -d $d ] || continue
  done_n=$(cat $d/results_*.jsonl 2>/dev/null | wc -l)
  alive=$(pgrep -f "[b]h_client.py --spec $spec" | wc -l)
  echo "$bb $name p$k: $done_n/$(wc -l < $spec) done, $alive workers"
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -4
free -g | sed -n 2p
