#!/bin/bash
# b1_start_awc.sh -- runs DETACHED on the server (log $A/chain/b1.log).  Retrain AW with correct placement (awc = the 10-28 AW,
# kind aw, seed 0, same data and hyper-parameters, + --host_place) on GPUs 4-7, as written in the handover 7.8 before any awc
# result (10-31 08:57).  Checked at step 0 against the single-card fingerprint of aw seed 0 (readout 1.1562, flow 0.0408)
# and at step 50 (no skipped update); stopped otherwise.
BH=${BH:-/data/blackhole}
L=$BH/lens
A=$L/aw
PY=${PY:-$BH/openpi/.venv/bin/python}
export OPENPI_DATA_HOME=$BH/openpi_data
CH=$A/chain
mkdir -p $CH
TR="--bs 32 --steps 6000 --lr 5e-5 --lr_new 5e-4 --lam_loc 0.1 --lam_coord 0.2 --log_every 50 --val_every 500 --save_every 1000 --seed 0 --skip_gn 1000 --host_place"
ts() { date '+%F %T'; }
fp_ok() {  # log, flow, coord: step-0 validation must match the single-card fingerprint; nothing skipped by step 50
  local v c sk
  v=$(grep -E "VAL step 0 " $1 | sed -E 's/.*flow=([0-9.eE+-]+|nan|inf).*/\1/')
  c=$(grep -E "VAL step 0 " $1 | sed -E 's/.*coord=([0-9.eE+-]+|nan|inf).*/\1/')
  sk=$(grep -E " step 50 .*skipped=" $1 | sed -E 's/.*skipped=([0-9.]+).*/\1/')
  echo "$1: step-0 validation flow $v (single card $2), readout $c (single card $3); skipped by step 50: $sk"
  awk -v v="$v" -v f="$2" -v c="$c" -v r="$3" -v sk="$sk" 'BEGIN{d=v-f; e=c-r; exit !(v != "" && d < 0.0015 && -d < 0.0015 && e < 0.005 && -e < 0.005 && sk != "" && sk+0 == 0)}'
}
echo "== $(ts) b1 start (pid $$)"
busy=$(timeout 60 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1>=4 && $2>2000' | wc -l)
[ "$busy" -eq 0 ] || { echo "GPUs 4-7 busy: awc not started"; echo "== $(ts) b1 END"; exit 0; }
[ "$(pgrep -fc "aw_train[.]py .*--out $A/run_awc( |$)")" -eq 0 ] || { echo "awc already running: nothing started"; echo "== $(ts) b1 END"; exit 0; }
rm -rf $A/run_awc
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=$A setsid nohup $PY $A/aw_train.py --data $A/data_pi05 --kind aw $TR \
  --out $A/run_awc > $A/run_awc.log 2>&1 < /dev/null &
for i in $(seq 1 60); do grep -qE " step 50 |Traceback" $A/run_awc.log && break; sleep 10; done
grep -E "trainable|data roots|frames [0-9]+: train|VAL step 0|Traceback" $A/run_awc.log | cut -c1-300 | head -n 6
if fp_ok $A/run_awc.log 0.0408 1.1562; then echo "awc healthy, kept"; else echo "awc NOT healthy: stopped"; pkill -f "aw_train[.]py .*--out $A/run_awc( |$)"; fi
echo "== $(ts) b1 END"
