#!/bin/bash
# d_b.sh c2 -- DETACHED on the server (log $A/chain/d_b.log), launched by d0_p0fix.sh after the pi0 server equivalence
# check (GPU 6).  Lane B (GPUs 4-7).  Round d (handover 7.10): awx3p0 = the selected configuration on the second backbone
# (kind awx, --bb pi0, data_pi0, geometric augmentation, seed 0, --host_place).  No selection on pi0.
# c2 (11-01, before any pi0 result): aw_train.py c10 (openpi's pi0_libero action convention, see d_a.sh c2); the same
# gates (single-card step-0 flow < 0.5; evaluation only with $CH/p0_equiv.ok) and the same revised health rule.
# Then development split (+ no-switch), dev2 extension, test split once each, whatever the result; when d_a.sh is also
# done: paired comparison with bc3p0.
source ${CHLIB:-${BH:-/data/blackhole}/lens/aw/chain/lib2.sh}
BBP=pi0
S="8634:4 8635:5"; P=$A/run_awx3p0/params_6000.npz; N="awdawx3p0 awpawx3p0 aweawx3p0 awtawx3p0"
G="episodes ->|DOWN|not healthy"
end() { echo "== $(ts) d_b END"; exit 0; }
health_rel() {  # run name: finished, last validation finite, < 100 skipped, last <= 50% of step 0 or < 0.03
  local v0 vf sk
  v0=$(grep -E "VAL step 0 " $A/run_$1.log | sed -E 's/.*flow=([0-9.eE+-]+|nan|inf).*/\1/')
  vf=$(grep -E "VAL step " $A/run_$1.log | tail -n 1 | sed -E 's/.*flow=([0-9.eE+-]+|nan|inf).*/\1/')
  sk=$(grep -E " step [0-9]+ .*skipped=" $A/run_$1.log | tail -n 1 | sed -E 's/.*skipped=([0-9.]+).*/\1/')
  echo "$1: params_6000 $([ -f $A/run_$1/params_6000.npz ] && echo present || echo MISSING); step-0 validation $v0, last $vf; skipped $sk; spikes $(cat $A/run_$1/spikes.jsonl 2>/dev/null | wc -l)"
  [ -f $A/run_$1/params_6000.npz ] && awk -v a="$v0" -v v="$vf" -v sk="$sk" 'BEGIN{exit !(v != "" && v+0 == v && a+0 > 0 && (v+0 <= 0.5 * a || v+0 < 0.03) && sk != "" && sk+0 < 100)}'
}
echo "== $(ts) d_b c2 start (pid $$); aw_train.py $(md5sum $A/aw_train.py | cut -c1-8), aw_server.py $(md5sum $A/aw_server.py | cut -c1-8)"
gpus_free 4 7 || { echo "GPUs 4-7 busy: nothing started"; end; }
ls -d $A/data_pi0 > /dev/null 2>&1 || { echo "no data_pi0: nothing started"; end; }
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$A timeout 900 $PY $A/aw_train.py --bb pi0 --data $A/data_pi0 --kind awx \
  --bs 32 --lam_loc 0.1 --lam_coord 0.2 --seed 0 --val_only --out $A/val0_awx_p0 > $A/val0_awx_p0.log 2>&1 < /dev/null
grep -E "data roots|frames [0-9]+: train|VAL0ONLY|Traceback|Error" $A/val0_awx_p0.log | cut -c25-300 | head -n 6
FA=$(fpv awx_p0 flow); CA=$(fpv awx_p0 coord)
[ -n "$FA" ] && [ -n "$CA" ] || { echo "no fingerprint: awx3p0 not started"; tail -n 5 $A/val0_awx_p0.log | cut -c1-300; end; }
grep -E "action convention" $A/val0_awx_p0.log | cut -c25-200
awk -v f="$FA" 'BEGIN{exit !(f+0 < 0.5)}' || { echo "released pi0 does not fit its own teacher data (step-0 flow $FA >= 0.5): action convention still wrong, not started"; end; }
mem_wait 40
start_train awx3p0 4,5,6,7 --bb pi0 --data $A/data_pi0 --kind awx --geo_aug
if fp_ok $A/run_awx3p0.log $FA $CA; then echo "awx3p0 healthy, kept"; else echo "awx3p0 NOT healthy: stopped"; tail -n 5 $A/run_awx3p0.log | cut -c1-300; stop_train awx3p0; end; fi
wait_train awx3p0 180 || { echo "awx3p0 still running after 180 min: stopping here"; end; }
grep -E "VAL step|Traceback" $A/run_awx3p0.log | cut -c1-300 | tail -n 13
health_rel awx3p0 || { echo "awx3p0 does not pass the training health rule: not evaluated"; end; }
gpus_free 4 7 || { echo "GPUs 4-7 busy: not evaluated"; end; }
[ -f $CH/p0_equiv.ok ] || { echo "no $CH/p0_equiv.ok (pi0 server path not verified): not evaluated"; end; }
lock_eval
mem_wait 13
if ! BB=pi0 servers_up awx $P awx3p0 "$S"; then echo "servers not healthy: no rollouts"; servers_down awx $P "$S"; unlock_eval; end; fi
mem_wait 20
echo "== $(ts) awx3p0 (pi0): development split, dev2 extension, test split (16 workers)"
BB=pi0 PHASE=workers $A/aw_eval.sh awx $P awx3p0 "$S" 2 | grep -E "$G"
BB=pi0 PHASE=workers $A/aw_eval.sh awx $P awx3p0 "$S" 1 dev2ext | grep -E "$G"
BB=pi0 PHASE=workers $A/aw_eval.sh awx $P awx3p0 "$S" 3 test | grep -E "$G"
sleep 120
wait_workers "awx3p0" "$N" 460 80
servers_down awx $P "$S"
unlock_eval
cd $A
$PY aw_summary.py pi0:awdawx3p0 pi0:awpawx3p0 pi0:awtawx3p0 2>/dev/null
pack d_b "$N"
echo "== $(ts) wait for d_a (bc3p0), then the paired comparison"
for i in $(seq 1 360); do [ "$(pgrep -fc 'd_a_bc3p0[.]sh')" -eq 0 ] && break; sleep 30; done
$PY aw_dev2.py pi0/awx3p0 pi0/bc3p0 --pairs pi0/awx3p0:pi0/bc3p0
$PY aw_paired.py pi0:awtawx3p0 pi0:awtbc3p0
end
