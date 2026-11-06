#!/bin/bash
# Server-side pipeline for one backbone.
# usage: driver.sh BB CONFIG PORT STAGES SERVER_GPU RENDER_GPU ANALYSIS_GPU NWORKERS
#   STAGES: comma list from e0,ana,e12,heads,e3,e1b
# GPUs are shared with other users' jobs: JAX does not preallocate, so each process
# only takes the memory it actually needs.
BB=$1; CFG=$2; PORT=$3; STAGES=$4; SGPU=$5; RGPU=$6; AGPU=$7; NW=${8:-12}; WAITFOR=$9
BH=${BH:-/data/blackhole}
PY=$BH/openpi/.venv/bin/python
CK=$BH/openpi_data/openpi-assets/checkpoints/$CFG
LOG=$BH/logs/driver_$BB.log
A="pick up the black bowl on the stove and place it on the plate"
B="pick up the black bowl on the wooden cabinet and place it on the plate"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 XLA_PYTHON_CLIENT_PREALLOCATE=false
log() { echo "[$(date '+%F %T')] $*" >> $LOG; }
has() { [[ ",$STAGES," == *",$1,"* ]]; }
wait_runs() {
  local spec=$1 out=$2 n c alive
  n=$(wc -l < "$spec")
  while true; do
    c=$(cat "$out"/results_*.jsonl 2>/dev/null | wc -l)
    alive=$(pgrep -f "bh_client.py --spec $spec" | wc -l)
    [ "$c" -ge "$n" ] && break
    if [ "$alive" -eq 0 ]; then log "workers gone at $c/$n for $out"; break; fi
    sleep 30
  done
  log "$out: $c/$n episodes"
}
run() {  # name spec_file nworkers
  $BH/code/run_exp.sh $CFG $CK $2 $BH/runs/${BB}_$1 $3 $PORT $SGPU $RGPU >> $LOG 2>&1
  wait_runs $2 $BH/runs/${BB}_$1
  $PY $BH/code/status.py $BH/runs/${BB}_$1 >> $LOG 2>&1
  touch $BH/logs/${BB}_$1.done
}
log "driver start ($CFG, port $PORT, stages $STAGES, server GPU $SGPU, render GPU $RGPU, analysis GPU $AGPU, $NW workers)"
if [ -n "$WAITFOR" ]; then until [ -f "$WAITFOR" ]; do sleep 60; done; log "marker $WAITFOR present"; fi
until [ -f $BH/logs/$CFG.complete ]; do sleep 30; done
log "checkpoint verified"
until [ -f $BH/logs/client.ok ]; do sleep 30; done
log "client venv ok"
cd $BH/openpi
if [ ! -f $BH/logs/selftest_$BB.ok ]; then
  CUDA_VISIBLE_DEVICES=$SGPU OPENPI_DATA_HOME=$BH/openpi_data PYTHONPATH=$BH/code/overlay timeout 1500 .venv/bin/python $BH/code/bh_server.py \
     --config $CFG --ckpt $CK --selftest > $BH/logs/selftest_$BB.log 2>&1 && touch $BH/logs/selftest_$BB.ok
  grep -E "selftest|infer s|multi|Error" $BH/logs/selftest_$BB.log | head -20 >> $LOG
fi
if has e0; then
  cat $BH/specs/${BB}_e0.jsonl $BH/specs/${BB}_lib.jsonl > $BH/specs/${BB}_e0lib.jsonl
  run e0 $BH/specs/${BB}_e0lib.jsonl $NW
  mkdir -p $BH/lib
  $PY $BH/code/build_library.py $BH/runs/${BB}_e0 "A->B:new" $BH/lib/${BB}_spAB.npz 10 >> $LOG 2>&1
  $PY $BH/code/build_library.py $BH/runs/${BB}_e0 "B->A:new" $BH/lib/${BB}_spBA.npz 10 >> $LOG 2>&1
fi
if has ana; then
  CUDA_VISIBLE_DEVICES=$AGPU OPENPI_DATA_HOME=$BH/openpi_data XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 ANA_MIN_FREE_MB=11000 PYTHONPATH=$BH/code/overlay \
    setsid nohup .venv/bin/python $BH/code/bh_analyze.py --config $CFG --ckpt $CK \
    --obs_glob "$BH/runs/${BB}_e0/obs/${BB}_e0_*.npz" --pairs $BH/specs/analysis_pairs.json \
    --out $BH/runs/${BB}_ana --nsamples 4 > $BH/logs/ana_$BB.log 2>&1 < /dev/null &
  log "analysis started"
fi
if has e12; then
  grep '"suite": "libero_spatial"' $BH/specs/${BB}_e1.jsonl > $BH/specs/${BB}_e1a.jsonl
  cat $BH/specs/${BB}_e1a.jsonl $BH/specs/${BB}_e2.jsonl > $BH/specs/${BB}_e12.jsonl
  run e12 $BH/specs/${BB}_e12.jsonl $NW
fi
if has heads; then
  cd $BH/code
  for d in AB BA; do
    if [ $d = AB ]; then OLD="$A"; NEW="$B"; LO=$BH/lib/${BB}_spBA.npz; LN=$BH/lib/${BB}_spAB.npz;
    else OLD="$B"; NEW="$A"; LO=$BH/lib/${BB}_spAB.npz; LN=$BH/lib/${BB}_spBA.npz; fi
    CUDA_VISIBLE_DEVICES=$AGPU OPENPI_DATA_HOME=$BH/openpi_data XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONPATH=$BH/code/overlay \
      $PY $BH/code/embed_instr.py $CFG $CK $BH/lib/${BB}_emb_$d.npz "$OLD" "$NEW" >> $LOG 2>&1
    CUDA_VISIBLE_DEVICES=$AGPU $PY $BH/code/train_bridge.py $LO $LN $BH/lib/${BB}_emb_$d.npz $BH/lib/${BB}_head_$d.npz --steps 20000 >> $LOG 2>&1
  done
  cd $BH/openpi
fi
if has e3; then run e3 $BH/specs/${BB}_e3.jsonl $NW; fi
if has e1b; then
  grep -v '"suite": "libero_spatial"' $BH/specs/${BB}_e1.jsonl > $BH/specs/${BB}_e1b.jsonl
  run e1b $BH/specs/${BB}_e1b.jsonl $NW
fi
pkill -f "bh_server.py --config $CFG --ckpt" && log "server stopped"
touch $BH/logs/driver_$BB.done
log "driver done"
