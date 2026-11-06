#!/bin/bash
# c_b.sh -- DETACHED on the server (log $A/chain/c_b.log).  Lane B (GPUs 4-7).
# (1) Mechanism measurement redone on the correctly trained models (same 100 switch frames and code as 10-29, 128/133):
#     [LOC] / last-query attention over the base-camera patches for awx2c, awx3, awc, awac (designated heads 6-13:0-3) and
#     bcc, bc3 (top heads), two at a time on GPUs 4-7; compact copies small_hr_<tag>.npz for the figure.
# (2) Seed replication (handover 7.9, written before any seed-1 result): awx3s1 = awx3 with --seed 1 (kind awx, same
#     augmentation and settings), fingerprint of awx seed 1 (single card 0.8999 / 0.0540; the augmentation does not change
#     the validation), training on GPUs 4-7, then development split (+ no-switch), dev2 extension and test split once each.
# (3) When lane A (c_a.sh, bc3s1) is also done: combined tables of both seeds.
source ${CHLIB:-${BH:-/data/blackhole}/lens/aw/chain/lib.sh}
S="8634:4 8635:5"; P=$A/run_awx3s1/params_6000.npz; N="awdawx3s1 awpawx3s1 aweawx3s1 awtawx3s1"
G="episodes ->|DOWN|not healthy"
ST="$BH/runs/lens/pi05_awd0hr_p0/steps/*_flush_*.npz"
end() { echo "== $(ts) c_b END"; exit 0; }
echo "== $(ts) c_b start (pid $$)"
gpus_free 4 7 || { echo "GPUs 4-7 busy: nothing started"; end; }
echo "== $(ts) head analyses ($(ls $ST 2>/dev/null | wc -l) step files)"
heads() {  # tag kind gpu designated
  mem_wait 25
  CUDA_VISIBLE_DEVICES=$3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 PYTHONPATH=$A setsid nohup $PY $A/pi_heads.py $A/heads_hr_$1 --kind $2 \
    --params $A/run_$1/params_6000.npz --frames "$ST" --max 100 > $A/heads_hr_$1.log 2>&1 < /dev/null &
}
for batch in "awx2c:awx:4 awx3:awx:5" "awc:aw:4 awac:awa:5" "bcc:bc:4 bc3:bc:5"; do
  for j in $batch; do IFS=: read t k g <<< "$j"; heads $t $k $g; sleep 20; done
  for i in $(seq 1 60); do [ "$(pgrep -fc 'pi_heads[.]py')" -eq 0 ] && break; sleep 15; done
done
cd $A
for t in awx2c awx3 awc awac; do $PY heads_small.py $A/heads_hr_$t $A/small_hr_$t.npz --frames "$ST" --k 5 --designated "6-13:0-3" > /dev/null 2>&1; done
for t in bcc bc3; do $PY heads_small.py $A/heads_hr_$t $A/small_hr_$t.npz --frames "$ST" --k 5 > /dev/null 2>&1; done
for t in awx2c awx3 awc awac bcc bc3; do
  echo "-- $t: $(grep -E 'Traceback|Error' $A/heads_hr_$t.log | head -n 2 | cut -c1-150)"
  $PY -c "
import json; j = json.load(open('$A/heads_hr_$t.json'))
print({k: v for k, v in j.items() if k not in ('top_heads', 'per_head')})
print('top heads', [(h['layer'], h['head'], round(h.get('score', h.get('acc_new', 0)), 3)) for h in j.get('top_heads', [])[:3]])" 2>&1 | cut -c1-400
  ls -la $A/small_hr_$t.npz 2>/dev/null | awk '{print "   small:", $5, "bytes"}'
done
(cd $A && tar czf $A/pack/heads_c_b.tgz heads_hr_{awx2c,awx3,awc,awac,bcc,bc3}.json small_hr_{awx2c,awx3,awc,awac,bcc,bc3}.npz 2>&1 | tail -n 2); md5sum $A/pack/heads_c_b.tgz
echo "== $(ts) awx3s1 (seed 1) on GPUs 4-7"
gpus_free 4 7 || { echo "GPUs 4-7 busy: awx3s1 not started"; end; }
TR=${TR/--seed 0/--seed 1}; echo "training flags: $TR"
mem_wait 40
start_train awx3s1 4,5,6,7 --data $A/data_pi05 --kind awx --geo_aug
if fp_ok $A/run_awx3s1.log 0.0540 0.8999; then echo "awx3s1 healthy, kept"; else echo "awx3s1 NOT healthy: stopped"; stop_train awx3s1; end; fi
$PY -c "import json; print('seed in args.json:', json.load(open('$A/run_awx3s1/args.json')).get('seed'))" 2>&1 | tail -n 1
wait_train awx3s1 150 || { echo "awx3s1 still running after 150 min: stopping here"; end; }
grep -E "VAL step|Traceback" $A/run_awx3s1.log | cut -c1-300 | tail -n 13
train_health awx3s1 || { echo "awx3s1 does not pass the health rule: not evaluated"; end; }
gpus_free 4 7 || { echo "GPUs 4-7 busy: not evaluated"; end; }
lock_eval
mem_wait 13
if ! servers_up awx $P awx3s1 "$S"; then echo "servers not healthy: no rollouts"; servers_down awx $P "$S"; unlock_eval; end; fi
mem_wait 20
echo "== $(ts) awx3s1: development split, dev2 extension, test split (16 workers)"
PHASE=workers $A/aw_eval.sh awx $P awx3s1 "$S" 2 | grep -E "$G"
PHASE=workers $A/aw_eval.sh awx $P awx3s1 "$S" 1 dev2ext | grep -E "$G"
PHASE=workers $A/aw_eval.sh awx $P awx3s1 "$S" 3 test | grep -E "$G"
sleep 120
wait_workers "awx3s1" "$N" 460 70
servers_down awx $P "$S"
unlock_eval
cd $A
$PY aw_summary.py awdawx3 awdawx3s1 awpawx3s1 awtawx3 awtawx3s1 2>/dev/null
pack c_b "$N"
echo "== $(ts) wait for lane A (c_a.sh) to finish, then combined tables"
for i in $(seq 1 240); do [ "$(pgrep -fc 'c_a_bc3s1[.]sh')" -eq 0 ] && break; sleep 30; done
$PY aw_dev2.py awx3 awx3s1 bc3 bc3s1 awx2c bcc --pairs awx3s1:bc3s1,awx3:bc3,awx3s1:awx3,bc3s1:bc3
for pr in "awtawx3s1 awtbc3s1" "awtawx3 awtbc3" "awtawx3s1 awtawx3" "awtbc3s1 awtbc3"; do set -- $pr; $PY aw_paired.py $1 $2; done
end
