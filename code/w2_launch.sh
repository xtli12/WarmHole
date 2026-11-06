#!/bin/bash
# Launch one spec file of the Wormhole v2 experiments across several policy servers.
# usage: w2_launch.sh NAME BB "PORT:GPU PORT:GPU ..." NWORKERS_PER_SERVER
# The spec $BH/specs/w2/${BB}_${NAME}.jsonl is split round-robin; part k runs on server k and
# writes to $BH/runs/${BB}_${NAME}_p$k (each worker writes results_<w>.jsonl and traj/).
BH=${BH:-/data/blackhole}
NAME=$1; BB=$2; SERVERS=$3; NW=$4
CFG=${BB}_libero
CK=$BH/openpi_data/openpi-assets/checkpoints/$CFG
SPEC=$BH/specs/w2/${BB}_${NAME}.jsonl
[ -f $SPEC ] || { echo "no spec $SPEC"; exit 1; }
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
n=$(echo $SERVERS | wc -w); k=0
for sg in $SERVERS; do
  port=${sg%%:*}; gpu=${sg##*:}
  part=$BH/specs/w2/${BB}_${NAME}_part$k.jsonl
  awk -v n=$n -v k=$k '(NR-1)%n==k' $SPEC > $part
  echo "[$(date '+%F %T')] $NAME/$BB part $k: $(wc -l < $part) episodes -> port $port GPU $gpu"
  $BH/code/run_exp.sh $CFG $CK $part $BH/runs/${BB}_${NAME}_p$k $NW $port $gpu $gpu
  k=$((k+1))
done
