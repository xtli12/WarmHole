#!/bin/bash
# k_master.sh c1 -- DETACHED on the server (log $A/chain/k_master.log).  Round k of the handover 7.17 items 1-2 (11-02):
# DIAGNOSTICS ONLY of the final round-j models -- no selection, no test split, no training, nothing decided here.
#   1  wormhole diagnostics of awx3j / awx3s1j on the development split (3.6: aw_off, aw_bias0, aw_hl; 100 switch
#      episodes each; the same specs and client (bh_client_lens12) as for awx2c on 10-31): servers 8761-8764 on GPUs 0-3,
#      4 workers per server; result sets pi05_awxawx3j_p*, pi05_awxawx3s1j_p*
#   2  (after 1, cards free) localisation-head measurement of awx3j, awx3s1j (designated heads 6-13:0-3) and bc3j, bc3s1j
#      (top heads) on the same 100 switch frames as 10-29 / 11-01 (pi05_awd0hr_p0/steps/*_flush_*.npz), two at a time
#      on GPUs 4-5; compact copies small_hr_<tag>.npz for the figure -> pack/heads_k.tgz
#   3  read-only posture replay (one LIBERO process, EGL GPU 7): for EVERY B->A devR episode with t_s >= 32 (80 per
#      model, all classes) of awx3, awx3j, awx3s1, awx3s1j, bc3, bc3j, bc3s1, bc3s1j: elbow margin at the switch, share
#      at the limit, first step at the limit, reach -> does the student with the safe-exit data enter the stretched
#      configuration less often?  Plus the hand-back posture of the new A->B teacher data (jab) against the original.
#   4  summaries and packs: pack/aw_results_k_diag.tgz, pack/heads_k.tgz, pack/k_logs.tgz
source ${CHLIB:-${BH:-/data/blackhole}/lens/aw/chain/lib.sh}
OP=$BH/openpi
EVAL7=$A/aw_eval.sh
ST="$BH/runs/lens/pi05_awd0hr_p0/steps/*_flush_*.npz"
G="episodes ->|DOWN|not healthy"
end() { echo "== $(ts) k_master END"; exit 0; }
P() { echo $A/run_$1/params_6000.npz; }
alive() { pgrep -fc 'aw_server(_c9)?[.]py|lens_server[.]py|bh_client_lens[.]py|aw_train(_c11)?[.]py|[e-j]_master(_c[0-9])?[.]sh|pi_heads[.]py|posture_c2[.]py'; }
S0="8761:0 8762:1"; S1="8763:2 8764:3"
echo "== $(ts) k_master c1 start (pid $$): diagnostics of the round-j models (handover 7.17 items 1-2); nothing is selected"
(cd $A && md5sum chain/lib.sh chain/k_master.sh aw_eval.sh aw_eval.sh aw_server.py pi_heads.py heads_small.py posture.py aw_summary.py ../bh_client_lens.py ../bh_client_lens.py ../make_specs_lens.py)
for k in awx3j awx3s1j bc3j bc3s1j; do [ -f $(P $k) ] || { echo "no $(P $k): nothing started"; end; }; done
for n in awxawx3j awxawx3s1j; do [ -d $BH/runs/lens/pi05_${n}_p0 ] && { echo "results of $n exist already: nothing started"; end; }; done
[ "$(alive)" -eq 0 ] || { echo "processes of this project alive ($(alive)): nothing started"; end; }
gpus_free 0 7 || { echo "GPUs 0-7 not free: nothing started"; end; }
echo "$(ts) /data $(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9) GiB free; anon $(anon) GiB; step files $(ls $ST 2>/dev/null | wc -l)"

# ---------------------------------------------------------------- 1: wormhole diagnostics (GPUs 0-3)
echo "== $(ts) phase 1: wormhole diagnostics aw_off / aw_bias0 / aw_hl of awx3j, awx3s1j (development split, 300 episodes each)"
lock_eval
mem_wait 26
if ! { servers_up awx $(P awx3j) awx3j "$S0" && servers_up awx $(P awx3s1j) awx3s1j "$S1"; }; then
  echo "servers not healthy: no rollouts"; servers_down awx $(P awx3j) "$S0"; servers_down awx $(P awx3s1j) "$S1"; unlock_eval; end
fi
mem_wait 22
export EGLOFF=0
DIAGNAME=awxawx3j   CLIENT=bh_client_lens.py PHASE=workers $EVAL7 awx $(P awx3j)   awx3j   "$S0" 4 diag aw_off,aw_bias0,aw_hl | grep -E "$G"
DIAGNAME=awxawx3s1j CLIENT=bh_client_lens.py PHASE=workers $EVAL7 awx $(P awx3s1j) awx3s1j "$S1" 4 diag aw_off,aw_bias0,aw_hl | grep -E "$G"
ND="awxawx3j awxawx3s1j"
sleep 120
wait_workers "wormhole diagnostics" "$ND" 600 60
[ "$(nwk "$ND")" -eq 0 ] || { echo "$(ts) diagnostics workers still running after 60 min: stopped"; kill_workers "$ND"; }
servers_down awx $(P awx3j) "$S0"; servers_down awx $(P awx3s1j) "$S1"
unlock_eval
cd $A
echo "-- diagnostics summary (committed = t_s >= 12; 4.11 for comparison: awx2c aw_off 0%, aw_hl 0%, aw_bias0 95%)"
$PY aw_summary.py awxawx3j awxawx3s1j awxawx2c 2>/dev/null
pack k_diag "$ND"

# ---------------------------------------------------------------- 2: localisation heads (GPUs 4-5)
echo "== $(ts) phase 2: localisation heads of awx3j, awx3s1j, bc3j, bc3s1j on the 100 switch frames (GPUs 4-5, two at a time)"
gpus_free 0 7 || { echo "GPUs busy: heads not measured"; end; }
heads() {  # tag kind gpu
  mem_wait 12
  CUDA_VISIBLE_DEVICES=$3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 PYTHONPATH=$A setsid nohup $PY $A/pi_heads.py $A/heads_hr_$1 --kind $2 \
    --params $(P $1) --frames "$ST" --max 100 > $A/heads_hr_$1.log 2>&1 < /dev/null &
}
for batch in "awx3j:awx:4 awx3s1j:awx:5" "bc3j:bc:4 bc3s1j:bc:5"; do
  for j in $batch; do IFS=: read t k g <<< "$j"; heads $t $k $g; sleep 20; done
  for i in $(seq 1 80); do [ "$(pgrep -fc 'pi_heads[.]py')" -eq 0 ] && break; sleep 15; done
done
cd $A
for t in awx3j awx3s1j; do $PY heads_small.py $A/heads_hr_$t $A/small_hr_$t.npz --frames "$ST" --k 5 --designated "6-13:0-3" > /dev/null 2>&1; done
for t in bc3j bc3s1j; do $PY heads_small.py $A/heads_hr_$t $A/small_hr_$t.npz --frames "$ST" --k 5 > /dev/null 2>&1; done
for t in awx3j awx3s1j bc3j bc3s1j; do
  echo "-- heads $t: $(grep -E 'Traceback|Error' $A/heads_hr_$t.log | head -n 2 | cut -c1-150)"
  $PY -c "
import json; j = json.load(open('$A/heads_hr_$t.json'))
print({k: v for k, v in j.items() if k not in ('top_heads', 'per_head')})
print('top heads', [(h['layer'], h['head'], round(h.get('score', h.get('acc_new', 0)), 3)) for h in j.get('top_heads', [])[:3]])" 2>&1 | cut -c1-400
  ls -la $A/small_hr_$t.npz 2>/dev/null | awk '{print "   small:", $5, "bytes"}'
done
(cd $A && tar czf $A/pack/heads_k.tgz heads_hr_{awx3j,awx3s1j,bc3j,bc3s1j}.json small_hr_{awx3j,awx3s1j,bc3j,bc3s1j}.npz 2>&1 | tail -n 2); md5sum $A/pack/heads_k.tgz

# ---------------------------------------------------------------- 3: posture replay of every B->A devR episode (t_s >= 32)
echo "== $(ts) phase 3: elbow at the switch, every B->A devR episode with t_s >= 32 (8 models) + hand-back posture of jab"
HB=""; ls $BH/runs/lens/pi05_jab_p0/traj/*.npz > /dev/null 2>&1 && HB="--handback jab"
mem_wait 6
(export PYTHONPATH=$OP/third_party/libero:$OP/packages/openpi-client/src:$L:$BH/code
 export LIBERO_CONFIG_PATH=$BH/libero_cfg MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=7 BH_DEPTH=1
 export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
 cd $A && nice -n 10 taskset -c 112-127 timeout 5400 $BH/venvs/libero/bin/python posture.py $BH/runs/lens \
   awrawx3 awrawx3j awrawx3s1 awrawx3s1j awrbc3 awrbc3j awrbc3s1 awrbc3s1j --max 80 $HB 2>&1 \
   | grep -E "^==|^  [a-z]" | cut -c1-400 | tee $CH/k_posture.txt)
echo "$(ts) posture done; anon $(anon) GiB"

# ---------------------------------------------------------------- 4: packs
(cd $A && tar czf pack/k_logs.tgz chain/k_master.log chain/k_posture.txt heads_hr_{awx3j,awx3s1j,bc3j,bc3s1j}.log 2>&1 | tail -n 2; md5sum pack/k_logs.tgz pack/aw_results_k_diag.tgz pack/heads_k.tgz)
end
