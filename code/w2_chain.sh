#!/bin/bash
# Run several spec files of one backbone back to back on its servers.
# usage: w2_chain.sh BB "PORT:GPU PORT:GPU" NWORKERS NAME1 [NAME2 ...]
# Waits for each stage to finish (all episodes written, or no worker left), relaunches the missing
# episodes once if workers died, then starts the next stage. Log: $BH/logs/w2_chain_BB.log
BH=${BH:-/data/blackhole}
BB=$1; SERVERS=$2; NW=$3; shift 3
LOG=$BH/logs/w2_chain_$BB.log
log() { echo "[$(date '+%F %T')] $*" >> $LOG; }
count() { cat $BH/runs/${BB}_$1_p*/results_*.jsonl 2>/dev/null | wc -l; }
alive() { pgrep -f "[b]h_client.py --spec $BH/specs/w2/${BB}_$1_part" | wc -l; }
for NAME in "$@"; do
  total=$(wc -l < $BH/specs/w2/${BB}_${NAME}.jsonl)
  log "stage $NAME: $total episodes on $SERVERS"
  $BH/code/w2_launch.sh $NAME $BB "$SERVERS" $NW >> $LOG 2>&1
  for attempt in 1 2; do
    while true; do
      c=$(count $NAME); a=$(alive $NAME)
      [ "$c" -ge "$total" ] && break
      [ "$a" -eq 0 ] && break
      sleep 30
    done
    c=$(count $NAME)
    [ "$c" -ge "$total" ] && break
    log "stage $NAME: workers gone at $c/$total, relaunching the missing episodes (attempt $attempt)"
    $BH/code/w2_launch.sh $NAME $BB "$SERVERS" $NW >> $LOG 2>&1
  done
  log "stage $NAME done: $(count $NAME)/$total"
  touch $BH/logs/w2_${BB}_${NAME}.done
done
log "chain done"
