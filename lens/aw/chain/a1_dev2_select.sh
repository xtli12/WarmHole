#!/bin/bash
# a1_dev2_select.sh -- runs DETACHED on the server (setsid nohup, log $A/chain/a1.log), so a sleeping laptop / dropped
# bridge cannot cut it off (10-31: 158 died waiting for awx3 when the laptop slept 09:06-16:40).
# (1) awx3 and awx4 (both trained by 10:23) on dev2 only: development split t_s 12-42 + no-switch rollouts + late-switch
#     extension t_s 52/62, 180 episodes each; GPUs 0-3, two single-card servers per model (awx3 8634/8635 on GPUs 0/1,
#     awx4 8661/8662 on GPUs 2/3), workers render on the server's card (as in 156); 20 workers.  Training health rule
#     first (finished, last validation finite and flow < 0.03, fewer than 100 skipped updates).
# (2) dev2 table of awx2c, awx3, awx4, bcc + paired tests; results packed ($A/pack/aw_results_a1.tgz).
# (3) The selection rule of 7.7 (fixed 10-30 08:14 before any new result; baseline awx2c since 09:12), applied mechanically
#     to the committed dev2 successes (of 120):
#       awx3 >= awx2c and awx4 >= awx2c -> train awx5 (geometric augmentation + DAgger data);
#       else awx3 >= awx2c              -> awx3 is the candidate: train its matched BC bc3 (kind bc, same augmentation);
#       else awx4 >= awx2c              -> awx4 is the candidate: train its matched BC bc4 (kind bc, same two data roots);
#       else                            -> no new test split (negative result).
#     The training goes on GPUs 0-3 with --host_place, checked at step 0 against the single-card fingerprint.
# The decision is written to $A/chain/a1_decision.txt ("NEXT=...").
BH=${BH:-/data/blackhole}
L=$BH/lens
A=$L/aw
PY=${PY:-$BH/openpi/.venv/bin/python}
export OPENPI_DATA_HOME=$BH/openpi_data
CH=$A/chain
mkdir -p $CH $A/pack
TR="--bs 32 --steps 6000 --lr 5e-5 --lr_new 5e-4 --lam_loc 0.1 --lam_coord 0.2 --log_every 50 --val_every 500 --save_every 1000 --seed 0 --skip_gn 1000 --host_place"
DAG="$A/data_pi05,$A/data_pi05_dag"
S3="8634:0 8635:1"
S4="8661:2 8662:3"
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
train_health() {  # run name: finished, last validation finite and flow < 0.03, fewer than 100 skipped updates
  local vf sk
  vf=$(grep -E "VAL step " $A/run_$1.log | tail -n 1 | sed -E 's/.*flow=([0-9.eE+-]+|nan|inf).*/\1/')
  sk=$(grep -E " step [0-9]+ .*skipped=" $A/run_$1.log | tail -n 1 | sed -E 's/.*skipped=([0-9.]+).*/\1/')
  echo "$1: params_6000 $([ -f $A/run_$1/params_6000.npz ] && echo present || echo MISSING); last validation flow $vf; skipped $sk; spikes $(cat $A/run_$1/spikes.jsonl 2>/dev/null | wc -l)"
  [ -f $A/run_$1/params_6000.npz ] && awk -v v="$vf" -v sk="$sk" 'BEGIN{exit !(v != "" && v+0 == v && v+0 < 0.03 && sk != "" && sk+0 < 100)}'
}
NAMES="awdawx3 awpawx3 aweawx3 awdawx4 awpawx4 aweawx4"
nres() { local n=0 s; for s in $NAMES; do n=$((n + $(cat $BH/runs/lens/pi05_${s}_p*/results_*.jsonl 2>/dev/null | wc -l))); done; echo $n; }
nerr() { cat $BH/runs/lens/pi05_aw[dpe]awx[34]_p*/results_*.jsonl 2>/dev/null | grep -c '"error": "'; }
nwk() { pgrep -fc "bh_client_lens[.]py --spec $BH/specs/lens/pi05_aw[dpe]awx[34]_part"; }
anon() { awk '$1=="anon"{printf "%d", $2/1073741824}' /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.stat 2>/dev/null; }
echo "== $(ts) a1 start (pid $$)"
echo "NEXT=pending" > $CH/a1_decision.txt
H3=0; H4=0
train_health awx3 && H3=1
train_health awx4 && H4=1
echo "health: awx3 $H3, awx4 $H4"
[ $H3 -eq 1 ] && [ $H4 -eq 1 ] || { echo "a training does not pass the health rule: by 7.7 it is retrained once with the same settings (not automatic); nothing evaluated"; echo "NEXT=unhealthy" > $CH/a1_decision.txt; echo "== $(ts) a1 END"; exit 0; }
busy=$(timeout 60 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1<4 && $2>2000' | wc -l)
[ "$busy" -eq 0 ] || { echo "GPUs 0-3 busy: nothing started"; echo "NEXT=gpus_busy" > $CH/a1_decision.txt; echo "== $(ts) a1 END"; exit 0; }
echo "== $(ts) servers: awx3 ($S3), awx4 ($S4); memory anon $(anon) GiB"
PHASE=servers $A/aw_eval.sh awx $A/run_awx3/params_6000.npz awx3 "$S3" 2 | grep -E " ok$|DOWN|refus|not healthy"
PHASE=servers $A/aw_eval.sh awx $A/run_awx4/params_6000.npz awx4 "$S4" 2 | grep -E " ok$|DOWN|refus|not healthy"
nok=0; for p in 8634 8635 8661 8662; do curl -s -m 3 localhost:$p/healthz > /dev/null && nok=$((nok+1)); done
if [ "$nok" -ne 4 ]; then
  echo "only $nok/4 servers healthy: no rollouts"
  pkill -f "aw_server[.]py --kind awx --bb pi05 --params $A/run_awx3/params_6000.npz --port 863[45]( |$)"
  pkill -f "aw_server[.]py --kind awx --bb pi05 --params $A/run_awx4/params_6000.npz --port 866[12]( |$)"
  echo "NEXT=servers_down" > $CH/a1_decision.txt; echo "== $(ts) a1 END"; exit 0
fi
echo "== $(ts) dev2 rollouts: per model development split 8 workers, dev2 extension 2 workers; memory anon $(anon) GiB"
PHASE=workers $A/aw_eval.sh awx $A/run_awx3/params_6000.npz awx3 "$S3" 2 | grep -E "episodes ->|DOWN|not healthy"
PHASE=workers $A/aw_eval.sh awx $A/run_awx3/params_6000.npz awx3 "$S3" 1 dev2ext | grep -E "episodes ->|DOWN|not healthy"
PHASE=workers $A/aw_eval.sh awx $A/run_awx4/params_6000.npz awx4 "$S4" 2 | grep -E "episodes ->|DOWN|not healthy"
PHASE=workers $A/aw_eval.sh awx $A/run_awx4/params_6000.npz awx4 "$S4" 1 dev2ext | grep -E "episodes ->|DOWN|not healthy"
sleep 120
echo "$(ts) after 2 min: workers $(nwk), results $(nres)/360"
for i in $(seq 1 240); do
  [ "$(nwk)" -eq 0 ] && break
  [ $((i % 20)) -eq 1 ] && echo "$(ts) workers $(nwk), results $(nres)/360, errors $(nerr); memory anon $(anon) GiB, used $(free -g | awk 'NR==2{print $3}') GB"
  sleep 15
done
echo "$(ts) workers left $(nwk); results $(nres)/360; errors $(nerr); $(ps -eo stat=,wchan:40= | awk '$1 ~ /^D/ && $2 ~ /mem_cgroup/' | wc -l) processes throttled by the memory cgroup"
cd $A
$PY aw_summary.py awdawx2c awdawx3 awdawx4 awpawx2c awpawx3 awpawx4 2>/dev/null
$PY aw_dev2.py awx2c awx3 awx4 bcc --pairs awx3:awx2c,awx4:awx2c,awx4:awx3,awx3:bcc,awx4:bcc | tee $CH/dev2_a1.txt
(cd $BH/runs/lens && tar czf $A/pack/aw_results_a1.tgz $(ls -d pi05_aw[dpe]awx[34]_p* | sed 's|$|/results_*.jsonl|') 2>&1 | tail -n 2)
md5sum $A/pack/aw_results_a1.tgz
echo "== $(ts) stop the awx3 / awx4 servers"
pkill -f "aw_server[.]py --kind awx --bb pi05 --params $A/run_awx3/params_6000.npz --port 863[45]( |$)"
pkill -f "aw_server[.]py --kind awx --bb pi05 --params $A/run_awx4/params_6000.npz --port 866[12]( |$)"
sleep 20
pkill -9 -f "aw_server[.]py --kind awx --bb pi05 --params $A/run_awx3/params_6000.npz --port 863[45]( |$)"
pkill -9 -f "aw_server[.]py --kind awx --bb pi05 --params $A/run_awx4/params_6000.npz --port 866[12]( |$)"
sleep 10
echo "servers left: $(pgrep -fc 'aw_server[.]py')"
echo "== $(ts) selection rule (7.7)"
cnt() { awk -v t="$1" '$1=="dev2" && $2==t && $3=="committed"{split($4,a,"/"); if (a[2]==120) print a[1]}' $CH/dev2_a1.txt; }
n2c=$(cnt awx2c); n3=$(cnt awx3); n4=$(cnt awx4)
echo "committed dev2 successes of 120: awx2c ${n2c:-?}, awx3 ${n3:-?}, awx4 ${n4:-?}"
[ -n "$n2c" ] && [ -n "$n3" ] && [ -n "$n4" ] || { echo "a dev2 result is incomplete: no decision, nothing trained"; echo "NEXT=incomplete" > $CH/a1_decision.txt; echo "== $(ts) a1 END"; exit 0; }
if [ "$n3" -ge "$n2c" ] && [ "$n4" -ge "$n2c" ]; then NEXT=awx5
elif [ "$n3" -ge "$n2c" ]; then NEXT=bc3
elif [ "$n4" -ge "$n2c" ]; then NEXT=bc4
else NEXT=none; fi
echo "decision: $NEXT"
echo "NEXT=$NEXT n2c=$n2c n3=$n3 n4=$n4" > $CH/a1_decision.txt
[ "$NEXT" = "none" ] && { echo "neither awx3 nor awx4 reaches awx2c on dev2: no new test split (negative result)"; echo "== $(ts) a1 END"; exit 0; }
busy=$(timeout 60 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1<4 && $2>2000' | wc -l)
[ "$busy" -eq 0 ] || { echo "GPUs 0-3 busy: $NEXT not started"; echo "NEXT=$NEXT not_started_gpus_busy" > $CH/a1_decision.txt; echo "== $(ts) a1 END"; exit 0; }
F4=$(grep -E "VAL0ONLY" $A/val0_awx4.log | sed -E 's/.*flow=([0-9.eE+-]+).*/\1/')
C4=$(grep -E "VAL0ONLY" $A/val0_awx4.log | sed -E 's/.*coord=([0-9.eE+-]+).*/\1/')
rm -rf $A/run_$NEXT
case $NEXT in
  awx5)
    echo "== $(ts) awx5 (geometric augmentation + DAgger data) on GPUs 0-3; fingerprint = awx4's single-card values $F4 / $C4"
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$A setsid nohup $PY $A/aw_train.py --data $DAG --kind awx --geo_aug $TR \
      --out $A/run_awx5 > $A/run_awx5.log 2>&1 < /dev/null &
    for i in $(seq 1 60); do grep -qE " step 50 |Traceback" $A/run_awx5.log && break; sleep 10; done
    fp_ok $A/run_awx5.log $F4 $C4 && echo "awx5 healthy, kept" || { echo "awx5 NOT healthy: stopped"; pkill -f "aw_train[.]py .*--out $A/run_awx5( |$)"; echo "NEXT=awx5 stopped_fingerprint" > $CH/a1_decision.txt; } ;;
  bc3)
    echo "== $(ts) bc3 (matched BC of awx3: kind bc, same augmentation) on GPUs 0-3; fingerprint bc seed 0 = 0.0410"
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$A setsid nohup $PY $A/aw_train.py --data $A/data_pi05 --kind bc --geo_aug $TR \
      --out $A/run_bc3 > $A/run_bc3.log 2>&1 < /dev/null &
    for i in $(seq 1 60); do grep -qE " step 50 |Traceback" $A/run_bc3.log && break; sleep 10; done
    fp_bc $A/run_bc3.log 0.0410 && echo "bc3 healthy, kept" || { echo "bc3 NOT healthy: stopped"; pkill -f "aw_train[.]py .*--out $A/run_bc3( |$)"; echo "NEXT=bc3 stopped_fingerprint" > $CH/a1_decision.txt; } ;;
  bc4)
    echo "== $(ts) bc4 (matched BC of awx4: kind bc, same two data roots): single-card fingerprint on GPU 0, then GPUs 0-3"
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$A timeout 900 $PY $A/aw_train.py --data $DAG \
      --kind bc --bs 32 --lam_loc 0.1 --lam_coord 0.2 --seed 0 --val_only --out $A/val0_bc4 > $A/val0_bc4.log 2>&1 < /dev/null
    grep -E "data roots|frames [0-9]+: train|VAL0ONLY|Traceback" $A/val0_bc4.log | cut -c25-330
    FB=$(grep -E "VAL0ONLY" $A/val0_bc4.log | sed -E 's/.*flow=([0-9.eE+-]+).*/\1/')
    [ -n "$FB" ] || { echo "no fingerprint: bc4 not started"; echo "NEXT=bc4 no_fingerprint" > $CH/a1_decision.txt; echo "== $(ts) a1 END"; exit 0; }
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$A setsid nohup $PY $A/aw_train.py --data $DAG --kind bc $TR \
      --out $A/run_bc4 > $A/run_bc4.log 2>&1 < /dev/null &
    for i in $(seq 1 60); do grep -qE " step 50 |Traceback" $A/run_bc4.log && break; sleep 10; done
    fp_bc $A/run_bc4.log $FB && echo "bc4 healthy, kept" || { echo "bc4 NOT healthy: stopped"; pkill -f "aw_train[.]py .*--out $A/run_bc4( |$)"; echo "NEXT=bc4 stopped_fingerprint" > $CH/a1_decision.txt; } ;;
esac
timeout 60 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo
free -g | sed -n 2,3p
echo "== $(ts) a1 END"
