#!/bin/bash
# c_a.sh -- DETACHED on the server (log $A/chain/c_a.log).  Lane A (GPUs 0-3).  Seed replication of the 7.7 result
# (rule written into the handover 7.9 before any seed-1 result, 11-01 05:41): bc3s1 = bc3 with --seed 1 (kind bc, same
# geometric augmentation and settings).  Single-card fingerprint of bc seed 1 first (GPU 0), training on GPUs 0-3
# (--host_place, checked at step 0 / 50), then -- if it passes the training health rule -- development split (+ no-switch),
# dev2 extension and test split once each, whatever the result (replication, not selection).
source ${CHLIB:-${BH:-/data/blackhole}/lens/aw/chain/lib.sh}
TR=${TR/--seed 0/--seed 1}
S="8653:0 8654:1"; P=$A/run_bc3s1/params_6000.npz; N="awdbc3s1 awpbc3s1 awebc3s1 awtbc3s1"
G="episodes ->|DOWN|not healthy"
end() { echo "== $(ts) c_a END"; exit 0; }
echo "== $(ts) c_a start (pid $$); training flags: $TR"
gpus_free 0 3 || { echo "GPUs 0-3 busy: nothing started"; end; }
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$A timeout 900 $PY $A/aw_train.py --data $A/data_pi05 --kind bc \
  --bs 32 --lam_loc 0.1 --lam_coord 0.2 --seed 1 --val_only --out $A/val0_bc_s1 > $A/val0_bc_s1.log 2>&1 < /dev/null
grep -E "VAL0ONLY|Traceback" $A/val0_bc_s1.log | cut -c25-300
FB=$(fpv bc_s1 flow); [ -n "$FB" ] || { echo "no fingerprint: bc3s1 not started"; end; }
mem_wait 40
start_train bc3s1 0,1,2,3 --data $A/data_pi05 --kind bc --geo_aug
if fp_bc $A/run_bc3s1.log $FB; then echo "bc3s1 healthy, kept"; else echo "bc3s1 NOT healthy: stopped"; stop_train bc3s1; end; fi
$PY -c "import json; print('seed in args.json:', json.load(open('$A/run_bc3s1/args.json')).get('seed'))" 2>&1 | tail -n 1
wait_train bc3s1 150 || { echo "bc3s1 still running after 150 min: stopping here"; end; }
grep -E "VAL step|Traceback" $A/run_bc3s1.log | cut -c1-300 | tail -n 13
train_health bc3s1 || { echo "bc3s1 does not pass the health rule: not evaluated"; end; }
gpus_free 0 3 || { echo "GPUs 0-3 busy: not evaluated"; end; }
lock_eval
mem_wait 13
if ! servers_up bc $P bc3s1 "$S"; then echo "servers not healthy: no rollouts"; servers_down bc $P "$S"; unlock_eval; end; fi
mem_wait 20
echo "== $(ts) bc3s1: development split, dev2 extension, test split (16 workers)"
PHASE=workers $A/aw_eval.sh bc $P bc3s1 "$S" 2 | grep -E "$G"
PHASE=workers $A/aw_eval.sh bc $P bc3s1 "$S" 1 dev2ext | grep -E "$G"
PHASE=workers $A/aw_eval.sh bc $P bc3s1 "$S" 3 test | grep -E "$G"
sleep 120
wait_workers "bc3s1" "$N" 460 70
servers_down bc $P "$S"
unlock_eval
cd $A
$PY aw_summary.py awdbc3 awdbc3s1 awpbc3s1 awtbc3 awtbc3s1 2>/dev/null
$PY aw_dev2.py bc3 bc3s1 bcc --pairs bc3s1:bc3
$PY aw_paired.py awtbc3s1 awtbc3
pack c_a "$N"
end
