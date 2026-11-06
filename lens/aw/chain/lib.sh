# lib.sh -- functions shared by the detached chain scripts in $A/chain (10-31).  Sourced, not run.
BH=${BH:-/data/blackhole}
L=$BH/lens
A=$L/aw
PY=${PY:-$BH/openpi/.venv/bin/python}
export OPENPI_DATA_HOME=$BH/openpi_data
CH=$A/chain
mkdir -p $CH $A/pack
TR="--bs 32 --steps 6000 --lr 5e-5 --lr_new 5e-4 --lam_loc 0.1 --lam_coord 0.2 --log_every 50 --val_every 500 --save_every 1000 --seed 0 --skip_gn 1000 --host_place"
DAG="$A/data_pi05,$A/data_pi05_dag"
ts() { date '+%F %T'; }
fp_ok() {  # log, flow, coord: step-0 validation must match the single-card fingerprint; nothing skipped by step 50
  local v c sk
  v=$(grep -E "VAL step 0 " $1 | sed -E 's/.*flow=([0-9.eE+-]+|nan|inf).*/\1/')
  c=$(grep -E "VAL step 0 " $1 | sed -E 's/.*coord=([0-9.eE+-]+|nan|inf).*/\1/')
  sk=$(grep -E " step 50 .*skipped=" $1 | sed -E 's/.*skipped=([0-9.]+).*/\1/')
  echo "$1: step-0 validation flow $v (single card $2), readout $c (single card $3); skipped by step 50: $sk"
  awk -v v="$v" -v f="$2" -v c="$c" -v r="$3" -v sk="$sk" 'BEGIN{d=v-f; e=c-r; exit !(v != "" && d < 0.0015 && -d < 0.0015 && e < 0.005 && -e < 0.005 && sk != "" && sk+0 == 0)}'
}
fp_bc() {  # log, flow: the same for BC (no readout)
  local v sk
  v=$(grep -E "VAL step 0 " $1 | sed -E 's/.*flow=([0-9.eE+-]+|nan|inf).*/\1/')
  sk=$(grep -E " step 50 .*skipped=" $1 | sed -E 's/.*skipped=([0-9.]+).*/\1/')
  echo "$1: step-0 validation flow $v (single card $2); skipped by step 50: $sk"
  awk -v v="$v" -v f="$2" -v sk="$sk" 'BEGIN{d=v-f; exit !(v != "" && d < 0.0015 && -d < 0.0015 && sk != "" && sk+0 == 0)}'
}
train_health() {  # run name: finished, last validation finite and flow < 0.03, fewer than 100 skipped updates (2.2 #11, 7.7)
  local vf sk
  vf=$(grep -E "VAL step " $A/run_$1.log | tail -n 1 | sed -E 's/.*flow=([0-9.eE+-]+|nan|inf).*/\1/')
  sk=$(grep -E " step [0-9]+ .*skipped=" $A/run_$1.log | tail -n 1 | sed -E 's/.*skipped=([0-9.]+).*/\1/')
  echo "$1: params_6000 $([ -f $A/run_$1/params_6000.npz ] && echo present || echo MISSING); last validation flow $vf; skipped $sk; spikes $(cat $A/run_$1/spikes.jsonl 2>/dev/null | wc -l)"
  [ -f $A/run_$1/params_6000.npz ] && awk -v v="$vf" -v sk="$sk" 'BEGIN{exit !(v != "" && v+0 == v && v+0 < 0.03 && sk != "" && sk+0 < 100)}'
}
anon() { awk '$1=="anon"{printf "%d", $2/1073741824}' /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.stat 2>/dev/null; }
mem_wait() {  # GiB about to be added: wait (max 60 min) until this user's anonymous memory + it <= 100 GiB (2.2 #18)
  local i a
  for i in $(seq 1 120); do
    a=$(anon); [ -z "$a" ] && return 0
    [ $((a + $1)) -le 100 ] && return 0
    [ $((i % 10)) -eq 1 ] && echo "$(ts) memory: anon $a GiB + $1 GiB planned > 100 GiB, waiting"
    sleep 30
  done
  echo "$(ts) memory wait timed out (anon $(anon) GiB): continuing"
}
lock_eval() { exec 9>$CH/eval.lock; echo "$(ts) waiting for the evaluation lock"; flock 9; echo "$(ts) evaluation lock acquired"; }
unlock_eval() { flock -u 9; exec 9>&-; echo "$(ts) evaluation lock released"; }
wait_train() {  # run name, max minutes: returns 0 once no training process writes to run_<name>
  local i
  for i in $(seq 1 $(($2 * 2))); do
    [ "$(pgrep -fc "aw_train[.]py .*--out $A/run_$1( |$)")" -eq 0 ] && return 0
    [ $((i % 20)) -eq 1 ] && echo "$(ts) $1 $(grep -E ' step [0-9]+ ' $A/run_$1.log | tail -n 1 | cut -c25-110)"
    sleep 30
  done
  return 1
}
start_train() {  # run name, gpus, training args...: detached; returns after step 50 or a traceback
  local k=$1 g=$2 i; shift 2
  rm -rf $A/run_$k
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$A setsid nohup $PY $A/aw_train.py "$@" $TR --out $A/run_$k > $A/run_$k.log 2>&1 < /dev/null &
  for i in $(seq 1 60); do grep -qE " step 50 |Traceback" $A/run_$k.log && break; sleep 10; done
}
stop_train() { pkill -f "aw_train[.]py .*--out $A/run_$1( |$)"; }
val0() {  # name, gpu, data, kind: single-card initial-model validation (fingerprint) into $A/val0_<name>.log
  CUDA_VISIBLE_DEVICES=$2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$A timeout 900 $PY $A/aw_train.py --data $3 --kind $4 \
    --bs 32 --lam_loc 0.1 --lam_coord 0.2 --seed 0 --val_only --out $A/val0_$1 > $A/val0_$1.log 2>&1 < /dev/null
  grep -E "VAL0ONLY|Traceback" $A/val0_$1.log | cut -c25-300
}
fpv() { grep -E "VAL0ONLY" $A/val0_$1.log | sed -E "s/.*$2=([0-9.eE+-]+).*/\1/"; }  # name, field (flow / coord)
gpus_free() { [ "$(timeout 60 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v lo=$1 -v hi=$2 '$1>=lo && $1<=hi && $2>2000' | wc -l)" -eq 0 ]; }
servers_up() {  # kind, params, tag, "port:gpu ...": start them; 0 only if all are healthy
  local sg ok=0 n=0
  PHASE=servers $A/aw_eval.sh $1 $2 $3 "$4" 2 | grep -E " ok$|DOWN|refus|not healthy"
  for sg in $4; do n=$((n+1)); curl -s -m 3 localhost:${sg%%:*}/healthz > /dev/null && ok=$((ok+1)); done
  echo "$(ts) servers $3 ($4): $ok/$n healthy"
  [ $ok -eq $n ]
}
servers_down() {  # kind, params, "port:gpu ...": stop exactly these servers
  local sg
  for sg in $3; do pkill -f "aw_server[.]py --kind $1 --bb pi05 --params $2 --port ${sg%%:*}( |$)"; done
  sleep 20
  for sg in $3; do pkill -9 -f "aw_server[.]py --kind $1 --bb pi05 --params $2 --port ${sg%%:*}( |$)"; done
  sleep 5
}
nres() { local n=0 s; for s in $1; do n=$((n + $(cat $BH/runs/lens/pi05_${s}_p*/results_*.jsonl 2>/dev/null | wc -l))); done; echo $n; }
nerr() { local n=0 s; for s in $1; do n=$((n + $(cat $BH/runs/lens/pi05_${s}_p*/results_*.jsonl 2>/dev/null | grep -c '"error": "'))); done; echo $n; }
nwk() { pgrep -fc "bh_client_lens[.]py --spec $BH/specs/lens/pi05_($(echo $1 | tr ' ' '|'))_part"; }
kill_workers() {
  pkill -f "bh_client_lens[.]py --spec $BH/specs/lens/pi05_($(echo $1 | tr ' ' '|'))_part"; sleep 15
  pkill -9 -f "bh_client_lens[.]py --spec $BH/specs/lens/pi05_($(echo $1 | tr ' ' '|'))_part"; sleep 5
}
wait_workers() {  # label, "spec names", episodes expected, max minutes
  local i
  for i in $(seq 1 $(($4 * 4))); do
    [ "$(nwk "$2")" -eq 0 ] && break
    [ $((i % 20)) -eq 1 ] && echo "$(ts) $1: workers $(nwk "$2"), results $(nres "$2")/$3, errors $(nerr "$2"); memory anon $(anon) GiB"
    sleep 15
  done
  echo "$(ts) $1: workers left $(nwk "$2"); results $(nres "$2")/$3; errors $(nerr "$2"); $(ps -eo stat=,wchan:40= | awk '$1 ~ /^D/ && $2 ~ /mem_cgroup/' | wc -l) processes throttled by the memory cgroup"
}
pack() {  # archive name, "spec names": results jsonl into $A/pack/aw_results_<name>.tgz
  local s l=""
  for s in $2; do l="$l $(cd $BH/runs/lens && ls -d pi05_${s}_p* 2>/dev/null | sed 's|$|/results_*.jsonl|' | tr '\n' ' ')"; done
  (cd $BH/runs/lens && tar czf $A/pack/aw_results_$1.tgz $l 2>&1 | tail -n 2)
  md5sum $A/pack/aw_results_$1.tgz
}
cnt() { awk -v t="$1" '$1=="dev2" && $2==t && $3=="committed"{split($4,a,"/"); if (a[2]==120) print a[1]}' $2; }  # tag, dev2 table file
