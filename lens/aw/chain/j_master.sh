#!/bin/bash
# j_master.sh c1 -- DETACHED on the server (log $A/chain/j_master.log).  Round j of the handover 7.16 (plan and every
# rule written 11-01 before any round-j result and before the development result of round i): a kinematically safe
# wormhole exit for the teacher data, for AW (kind awx) and its matched BC, two seeds each.
#   0  wait (up to 9 h) until no process of this project runs (round i included) and GPUs 0-7 are free
#   1  smoke: 2 teacher servers; 12 safe-exit episodes (M = 1.2) on layouts r130-131 (both directions) and 8 reproduction
#      episodes (inits 20, 21; t_s 22, 42; A->B; safe exit and the original wo_off_h15); j_check smoke / repro; spec
#      generator r25 against r24; the original teacher directories without their A->B switch episodes, and counts
#   2  pilot: M in {1.0, 1.2}, A->B, inits 20-39, t_s 22 / 42 (2 x 40 episodes), 8 teacher servers; rule -> j_pilot.txt
#   3  the new A->B teacher data (260 episodes: inits 20-39, t_s 2-62 every 5) with the selected M; dataset data_pi05j
#      (the original B->A switch and no-switch episodes + the new A->B switch episodes)
#   4  single-card fingerprints of data_pi05j; AW-j training awx3j (seed 0, GPUs 0-3) and awx3s1j (seed 1, GPUs 4-7)
#   5  AW-j development evaluation and the rule of 7.16 (strictly more pooled dev successes than awx3 + awx3s1)
#      -> $CH/j_decision.txt.  NOGO: stop here
#   7  GO: AW-j test split once each while bc3j trains (GPUs 4-7); then bc3s1j (GPUs 0-3)
#   8  BC-j: dev (+ no-switch), dev2 extension, devR and test split once each, whatever the result
#   9  tables and result packs in $A/pack
source ${JLIB:-${BH:-/data/blackhole}/lens/aw/chain/j_lib.sh}
end() { echo "== $(ts) j_master END"; exit 0; }
TR0="$TR"; TR1="${TR/--seed 0/--seed 1}"
AWS="awx3j awx3s1j"; BCS="bc3j bc3s1j"
declare -A KIND=([awx3j]=awx [awx3s1j]=awx [bc3j]=bc [bc3s1j]=bc)
declare -A SRV=([awx3j]="8761:0 8762:1" [awx3s1j]="8763:2 8764:3" [bc3j]="8765:0 8766:1" [bc3s1j]="8767:2 8768:3")
TS1="8781:0 8782:1"
TS8="8781:0 8782:1 8783:2 8784:3 8785:4 8786:5 8787:6 8788:7"
TSA="8781:0 8782:1 8783:2 8784:3"; TSB="8785:4 8786:5 8787:6 8788:7"
TEACH="$BH/runs/lens/pi05_teach_p0 $BH/runs/lens/pi05_teach_p1"
DJ=$A/data_pi05j
JP_INITS=${JP_INITS:-20-39}; JP_TS=${JP_TS:-22,42}; JD_INITS=${JD_INITS:-20-39}; JDMIN=${JDMIN:-50000}
JD_TS_ARG=""; [ -n "$JD_TS" ] && JD_TS_ARG="--m_ts $JD_TS"     # default: every t_s of the teacher data (2, 7, ..., 62)
fps() { F0=$(fpv j_awx0 flow); C0=$(fpv j_awx0 coord); F1=$(fpv j_awx1 flow); C1=$(fpv j_awx1 coord); FB0=$(fpv j_bc0 flow); FB1=$(fpv j_bc1 flow); }
P() { echo $A/run_$1/params_6000.npz; }
alive() { pgrep -fc 'aw_server(_c9)?[.]py|lens_server[.]py|bh_client_lens[.]py|aw_train(_c11)?[.]py|[e-i]_master(_c[0-9])?[.]sh'; }
up_all() { local k ok=0; for k in $1; do servers_up ${KIND[$k]} $(P $k) $k "${SRV[$k]}" || ok=1; done; return $ok; }
down_all() { local k; for k in $1; do servers_down ${KIND[$k]} $(P $k) "${SRV[$k]}"; done; }
healthy_all() { local k sg; for k in $1; do for sg in ${SRV[$k]}; do curl -s -m 3 localhost:${sg%%:*}/healthz > /dev/null || return 1; done; done; }
finish() {
  local mods=$1 spl=$2 want=$3 lab=$4 mx=$5 k names="" n
  for k in $want; do names="$names ${k%%:*}"; done
  n=$(echo $want | tr ' ' '\n' | awk -F: '{s+=$2} END{print s}')
  sleep 90
  wait_workers "$lab" "$names" $n $mx
  [ "$(nwk "$names")" -eq 0 ] || { echo "$(ts) $lab: workers still running after $mx min: stopped"; kill_workers "$names"; }
  (cd $A && $PY e_check.py runs $BH/runs/lens $want) && return 0
  healthy_all "$mods" || { echo "$(ts) $lab: incomplete and a server is down: no re-run"; return 1; }
  echo "$(ts) $lab: incomplete -> aborted rows moved to errors_*.jsonl, missing episodes re-run once"
  (cd $A && $PY e_check.py clean $BH/runs/lens $names)
  for k in $mods; do ev ${KIND[$k]} $(P $k) $k "${SRV[$k]}" $spl; done
  sleep 60
  wait_workers "$lab (re-run)" "$names" $n $mx
  [ "$(nwk "$names")" -eq 0 ] || { echo "$(ts) $lab: re-run workers still running: stopped"; kill_workers "$names"; }
  (cd $A && $PY e_check.py runs $BH/runs/lens $want)
}
FROM=${FROM:-0}
echo "== $(ts) j_master c1 start (pid $$), from phase $FROM; training flags seed 0: $TR0"
(cd $A && md5sum j_check.py chain/j_lib.sh chain/j_master.sh chain/g_lib.sh ../bh_client_lens.py ../make_specs_lens.py \
   aw_build_data.py aw_train.py aw_dev3.py aw_pool.py e_check.py chain/e_lib.sh aw_eval.sh aw_server.py \
   aw_server.py ../bh_client_lens.py ../bh_client_lens.py ../make_specs_lens.py posture.py si_diag.py)

# ---------------------------------------------------------------- 0: wait for round i and free cards
phase0() {
for i in $(seq 1 1080); do
  [ "$(alive)" -eq 0 ] && gpus_free 0 7 && break
  [ $((i % 40)) -eq 1 ] && echo "$(ts) waiting: processes of this project $(alive) (i_master $(pgrep -fc 'i_master[.]sh')); $(grep -E '^== ' $CH/i_master.log 2>/dev/null | tail -n 1 | cut -c1-120)"
  sleep 30
done
{ [ "$(alive)" -eq 0 ] && gpus_free 0 7; } || { echo "GPUs 0-7 not free or processes of this project alive after 9 h: nothing started"; end; }
AV=$(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9)
echo "$(ts) /data: ${AV} GiB available; memory anon $(anon) GiB; $(free -g | sed -n 2p)"
[ "${AV:-0}" -ge 300 ] || { echo "less than 300 GiB free on /data: nothing started"; end; }
}

# ---------------------------------------------------------------- 1: smoke
phase1() {
echo "== $(ts) phase 1: smoke"
cd $A
SOK=1
for d in jsmk jrep; do rm -rf $BH/runs/lens/pi05_${d}_p*; done
lock_eval
mem_wait 14
if ! t_up "$TS1"; then echo "smoke servers not healthy: nothing else started"; t_down "$TS1"; unlock_eval; end; fi
$PY $MS25 $BH/specs/lens --teach_m jsmk --m_margin 1.2 --m_inits r130-131 --m_ts 22,42,62 --m_dirs AB,BA --bbs pi05 --vlm_offset "$OFF"
$PY $MS25 $BH/specs/lens --teach_m jrep --m_margin 1.2 --m_inits 20-21 --m_ts 22,42 --m_dirs AB --m_orig --bbs pi05 --vlm_offset "$OFF"
dg_run jsmk "$(tg "$TS1")" 3
dg_run jrep "$(tg "$TS1")" 2
dg_wait "smoke" "jsmk jrep" 20 40
[ "$(dgwk "jsmk jrep")" -eq 0 ] || { echo "smoke workers still running: stopped"; dg_kill "jsmk jrep"; }
t_down "$TS1"
unlock_eval
grep -hE "Traceback|Error" $BH/runs/lens/pi05_jsmk_p*/worker_*.log $BH/runs/lens/pi05_jrep_p*/worker_*.log | sort | uniq -c | head -n 5
$PY $JCHK smoke $BH/runs/lens jsmk || SOK=0
echo "-- reproduction of the original teacher (report only)"
$PY $JCHK repro $BH/runs/lens jrep $TEACH
echo "-- spec generator r25 against r24 (every existing call)"
rm -rf $A/spec_chk5; mkdir -p $A/spec_chk5/a $A/spec_chk5/b
for d in a b; do F=$L/make_specs_lens.py; [ $d = b ] && F=$MS25
  $PY $F $A/spec_chk5/$d --dev flush --name awdx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk5/$d --plain awpx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk5/$d --test flush --name awdxx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk5/$d --dev flush --name awex --ts 52,62 --bbs pi05 > /dev/null
  $PY $F $A/spec_chk5/$d --devr awrx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk5/$d --selfplay spx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk5/$d --teach --vlm_offset "$OFF" > /dev/null
  $PY $F $A/spec_chk5/$d --cfdag gx --cf_layouts 20-37,r101-124 --bbs pi05 --vlm_offset "$OFF" > /dev/null
  $PY $F $A/spec_chk5/$d --swdag ix --bbs pi05 --vlm_offset "$OFF" > /dev/null
done
if diff -rq $A/spec_chk5/a $A/spec_chk5/b; then echo "specs identical ($(ls $A/spec_chk5/a | wc -l) files)"; else SOK=0; echo "SPECS DIFFER"; fi
rm -rf $A/spec_chk5
echo "-- the original teacher data without its A->B switch episodes"
for k in 0 1; do rm -rf $BH/runs/lens/pi05_teachjBA_p$k; $PY $JCHK filter $BH/runs/lens/pi05_teach_p$k $BH/runs/lens/pi05_teachjBA_p$k || SOK=0; done
$PY $JCHK count $TEACH $BH/runs/lens/pi05_teachjBA_p0 $BH/runs/lens/pi05_teachjBA_p1
echo "$(ts) smoke gates passed: $SOK"
[ "$SOK" -eq 1 ] || { echo "nothing else started"; end; }
}

# ---------------------------------------------------------------- 2: pilot (selects M)
phase2() {
echo "== $(ts) phase 2: pilot (selects the elbow margin M only; 7.16)"
for d in jp10 jp12 jab; do [ -d $BH/runs/lens/pi05_${d}_p0 ] && { echo "results of $d exist already: nothing started"; end; }; done
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing started"; end; }
lock_eval
mem_wait 52
if ! t_up "$TS8"; then echo "teacher servers not healthy: nothing started"; t_down "$TS8"; unlock_eval; end; fi
$PY $MS25 $BH/specs/lens --teach_m jp10 --m_margin 1.0 --m_inits $JP_INITS --m_ts $JP_TS --m_dirs AB --bbs pi05 --vlm_offset "$OFF"
$PY $MS25 $BH/specs/lens --teach_m jp12 --m_margin 1.2 --m_inits $JP_INITS --m_ts $JP_TS --m_dirs AB --bbs pi05 --vlm_offset "$OFF"
NP=$(wc -l < $BH/specs/lens/pi05_jp10.jsonl)
mem_wait 30
dg_run jp10 "$(tg "$TSA")" 3
dg_run jp12 "$(tg "$TSB")" 3
dg_finish "pilot" "jp10:$NP jp12:$NP" "jp10=$(tg "$TSA");jp12=$(tg "$TSB")" 3 60 || echo "pilot incomplete after one re-run (the rule uses what is there)"
cd $A
$PY $JCHK pilot $BH/runs/lens $CH/j_pilot.txt jp10:1.0 jp12:1.2 -- $TEACH || { echo "pilot rule (7.16): NOGO, nothing else started"; t_down "$TS8"; unlock_eval; end; }
}

# ---------------------------------------------------------------- 3: new A->B teacher data and the dataset
phase3() {
M=$(sed -n 's/^M=//p' $CH/j_pilot.txt 2>/dev/null | head -n 1)
[ -n "$M" ] || { echo "no M selected: stop"; t_down "$TS8"; unlock_eval; end; }
echo "== $(ts) phase 3: the new A->B teacher data with M = $M (layouts $JD_INITS)"
[ -d $BH/runs/lens/pi05_jab_p0 ] && { echo "results of jab exist already: nothing started"; t_down "$TS8"; unlock_eval; end; }
t_healthy "$TS8" || { echo "teacher servers down: stop"; t_down "$TS8"; unlock_eval; end; }
$PY $MS25 $BH/specs/lens --teach_m jab --m_margin $M --m_inits $JD_INITS $JD_TS_ARG --m_dirs AB --bbs pi05 --vlm_offset "$OFF"
NJ=$(wc -l < $BH/specs/lens/pi05_jab.jsonl)
mem_wait 30
dg_run jab "$(tg "$TS8")" 3
dg_finish "new A->B teacher data" "jab:$NJ" "jab=$(tg "$TS8")" 3 90 || echo "new A->B teacher data incomplete after one re-run (the dataset uses what is there)"
t_down "$TS8"
unlock_eval
cd $A
$PY $JCHK count $BH/runs/lens/pi05_teachjBA_p0 $BH/runs/lens/pi05_teachjBA_p1 $(ls -d $BH/runs/lens/pi05_jab_p*) | tee $CH/j_data.txt
$PY $JCHK pilot $BH/runs/lens $CH/j_data_vs_orig.txt jab:$M -- $TEACH | tee -a $CH/j_data.txt
pack j_teach "jab jp10 jp12 jsmk jrep"
echo "-- dataset data_pi05j"
mem_wait 16
rm -rf $DJ
taskset -c 64-127 timeout 5400 $PY $TRAIN7 $DJ $BH/runs/lens/pi05_teachjBA_p0 $BH/runs/lens/pi05_teachjBA_p1 $(ls -d $BH/runs/lens/pi05_jab_p*) \
  --procs 16 > $CH/j_build.log 2>&1 < /dev/null
grep -vE "^\s*$" $CH/j_build.log | tail -n 6 | cut -c1-300
$PY $JCHK data $DJ --min $JDMIN || { echo "data_pi05j not consistent: nothing trained"; end; }
du -sh $DJ | awk '{print "dataset: " $1}'; echo "/data $(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9) GiB free"
}

# ---------------------------------------------------------------- 4: fingerprints, AW-j training
phase4() {
echo "== $(ts) phase 4: single-card fingerprints of data_pi05j; AW-j training (awx3j seed 0 GPUs 0-3, awx3s1j seed 1 GPUs 4-7)"
for k in $AWS $BCS; do for n in $(nm $k dev dev2ext devr test); do [ -d $BH/runs/lens/pi05_${n}_p0 ] && { echo "results of $n exist already: nothing trained"; end; }; done; done
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing trained"; end; }
mem_wait 40
(val0s j_awx0 0 $DJ awx 0 > $CH/j_fp_awx0.txt) & (val0s j_awx1 1 $DJ awx 1 > $CH/j_fp_awx1.txt) &
(val0s j_bc0 2 $DJ bc 0 > $CH/j_fp_bc0.txt) & (val0s j_bc1 3 $DJ bc 1 > $CH/j_fp_bc1.txt) &
wait
cat $CH/j_fp_*.txt
fps
echo "fingerprints of data_pi05j: awx seed 0 flow $F0 readout $C0; awx seed 1 flow $F1 readout $C1; bc seed 0 flow $FB0; bc seed 1 flow $FB1" | tee $CH/j_fp.txt
for v in "$F0" "$C0" "$F1" "$C1" "$FB0" "$FB1"; do [ -n "$v" ] || { echo "a fingerprint is missing: nothing trained"; end; }; done
mem_wait 40
TR="$TR0"; start_train awx3j 0,1,2,3 --data $DJ --kind awx --geo_aug
grep -E "data roots|category shares|trainable|Traceback" $A/run_awx3j.log | cut -c25-330
if fp_ok $A/run_awx3j.log $F0 $C0; then echo "awx3j healthy, kept"; else echo "awx3j NOT healthy: stopped"; tail -n 5 $A/run_awx3j.log | cut -c1-300; stop_train awx3j; end; fi
mem_wait 40
TR="$TR1"; start_train awx3s1j 4,5,6,7 --data $DJ --kind awx --geo_aug; TR="$TR0"
grep -E "data roots|category shares|trainable|Traceback" $A/run_awx3s1j.log | cut -c25-330
if fp_ok $A/run_awx3s1j.log $F1 $C1; then echo "awx3s1j healthy, kept"; else echo "awx3s1j NOT healthy: stopped (awx3j continues; this chain stops)"; tail -n 5 $A/run_awx3s1j.log | cut -c1-300; stop_train awx3s1j; end; fi
for k in $AWS; do $PY -c "import json; d=json.load(open('$A/run_$k/args.json')); print('$k args: seed', d.get('seed'), 'data', d.get('data'))"; done
wait_train awx3j 220 || { echo "awx3j still running after 220 min: stopping here"; end; }
wait_train awx3s1j 60 || { echo "awx3s1j still running: stopping here"; end; }
for k in $AWS; do grep -E "VAL step|Traceback" $A/run_$k.log | cut -c1-300 | tail -n 13; done
{ train_health awx3j && train_health awx3s1j; } || { echo "an AW-j model does not pass the health rule: by 7.7 it is retrained once (not automatic)"; end; }
}

# ---------------------------------------------------------------- 5: AW-j development evaluation and the rule of 7.16
phase5() {
echo "== $(ts) phase 5: AW-j development evaluation"
gpus_free 0 3 || { echo "GPUs 0-3 busy: not evaluated"; end; }
lock_eval
mem_wait 26
if ! up_all "$AWS"; then echo "servers not healthy: no rollouts"; down_all "$AWS"; unlock_eval; end; fi
mem_wait 45
for k in $AWS; do ev awx $(P $k) $k "${SRV[$k]}" dev:2 dev2ext:1 devr:4; done
W=""; for k in $AWS; do W="$W awd$k:100 awp$k:40 awe$k:40 awr$k:240"; done
finish "$AWS" "dev:1 dev2ext:1 devr:1" "$W" "AW-j dev" 90 || echo "AW-j development split incomplete after one re-run (the rule then gives NOGO)"
cd $A
$PY aw_summary.py awdawx3j awpawx3j awdawx3s1j awpawx3s1j 2>/dev/null
$PY aw_dev3.py awx3 awx3s1 bc3 bc3s1 awx3j awx3s1j --pairs awx3j:awx3,awx3s1j:awx3s1,awx3j:bc3,awx3s1j:bc3s1 \
  --pool "awx3j+awx3s1j:awx3+awx3s1,awx3j+awx3s1j:bc3+bc3s1" --select "awx3j+awx3s1j:awx3+awx3s1" --out $CH/j_decision.txt | tee $CH/j_dev_aw.txt
pack j_awj_dev "$(nm awx3j dev dev2ext devr) $(nm awx3s1j dev dev2ext devr)"
DC=$(head -n 1 $CH/j_decision.txt 2>/dev/null)
echo "$(ts) decision (7.16): ${DC:-none}"
case "$DC" in
  GO*) ;;
  *) echo "not GO: by 7.16 no test split and no BC-j; this chain stops here"; down_all "$AWS"; unlock_eval; diag_j awrawx3j awrawx3s1j; end ;;
esac
}

# diagnostics (not part of any rule): failure classes / switch states (si_diag_c2) and the elbow at the switch
# (posture_c2, one LIBERO process; only if anon <= 90 GiB)
diag_j() {
  echo "-- diagnostics (not part of the rule): failure classes, switch states, elbow at the switch"
  (cd $A && taskset -c 96-127 timeout 900 $PY si_diag.py $BH/runs/lens --sets awrawx3 awrawx3s1 "$@" --teach $TEACH 2>&1 \
     | grep -E "^sets|B->A .*NN " | cut -c1-260 | head -n 40)
  if [ "$(anon)" -le 90 ]; then
    (export PYTHONPATH=$OP/third_party/libero:$OP/packages/openpi-client/src:$L:$BH/code
     export LIBERO_CONFIG_PATH=$BH/libero_cfg MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=7 BH_DEPTH=1
     export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
     cd $A && nice -n 10 taskset -c 120-127 timeout 1500 $BH/venvs/libero/bin/python posture.py $BH/runs/lens "$@" --max 20 2>&1 \
       | grep -E "^==|^  [a-z]" | cut -c1-400)
  fi
}

# ---------------------------------------------------------------- 7: AW-j test split; BC-j training
phase7() {
echo "== $(ts) phase 7: AW-j test split (once each, same servers); bc3j training on GPUs 4-7"
fps
[ -n "$FB0" ] && [ -n "$FB1" ] || { echo "no BC fingerprint of data_pi05j: stop"; down_all "$AWS"; unlock_eval; end; }
mem_wait 20
for k in $AWS; do ev awx $(P $k) $k "${SRV[$k]}" test:4; done
BC0=no
if gpus_free 4 7; then
  mem_wait 40
  TR="$TR0"; start_train bc3j 4,5,6,7 --data $DJ --kind bc --geo_aug
  grep -E "data roots|category shares|Traceback" $A/run_bc3j.log | cut -c25-330
  if fp_bc $A/run_bc3j.log $FB0; then BC0=yes; echo "bc3j healthy, kept"; else echo "bc3j NOT healthy: stopped"; stop_train bc3j; BC0=bad; fi
else echo "GPUs 4-7 busy: bc3j starts after the test split"; fi
finish "$AWS" "test:2" "awtawx3j:280 awtawx3s1j:280" "AW-j test" 60 || echo "AW-j test split incomplete after one re-run"
down_all "$AWS"
unlock_eval
cd $A
pack j_awj_test "awtawx3j awtawx3s1j"
$PY aw_pool.py "awx3j+awx3s1j:awx3+awx3s1" "awx3j+awx3s1j:bc3+bc3s1" | tee $CH/j_test_aw.txt
[ "$BC0" = bad ] && { echo "bc3j fingerprint failed: BC-j not trained (decided by hand)"; end; }
gpus_free 0 3 || { echo "GPUs 0-3 busy: bc3s1j not started"; end; }
mem_wait 40
TR="$TR1"; start_train bc3s1j 0,1,2,3 --data $DJ --kind bc --geo_aug; TR="$TR0"
grep -E "data roots|category shares|Traceback" $A/run_bc3s1j.log | cut -c25-330
if fp_bc $A/run_bc3s1j.log $FB1; then echo "bc3s1j healthy, kept"; else echo "bc3s1j NOT healthy: stopped"; stop_train bc3s1j; end; fi
if [ "$BC0" = no ]; then
  gpus_free 4 7 || { echo "GPUs 4-7 busy: bc3j not started"; end; }
  mem_wait 40
  TR="$TR0"; start_train bc3j 4,5,6,7 --data $DJ --kind bc --geo_aug
  fp_bc $A/run_bc3j.log $FB0 || { echo "bc3j NOT healthy: stopped"; stop_train bc3j; end; }
fi
for k in $BCS; do $PY -c "import json; d=json.load(open('$A/run_$k/args.json')); print('$k args: seed', d.get('seed'), 'data', d.get('data'))"; done
wait_train bc3j 220 || { echo "bc3j still running after 220 min: stopping here"; end; }
wait_train bc3s1j 120 || { echo "bc3s1j still running: stopping here"; end; }
for k in $BCS; do grep -E "VAL step|Traceback" $A/run_$k.log | cut -c1-300 | tail -n 13; done
{ train_health bc3j && train_health bc3s1j; } || { echo "a BC-j model does not pass the health rule: not evaluated"; end; }
}

# ---------------------------------------------------------------- 8: BC-j development and test splits
phase8() {
echo "== $(ts) phase 8: BC-j development splits and test split (once each)"
gpus_free 0 3 || { echo "GPUs 0-3 busy: not evaluated"; end; }
lock_eval
mem_wait 26
if ! up_all "$BCS"; then echo "servers not healthy: no rollouts"; down_all "$BCS"; unlock_eval; end; fi
mem_wait 52
for k in $BCS; do ev bc $(P $k) $k "${SRV[$k]}" dev:2 dev2ext:1 devr:3 test:3; done
W=""; for k in $BCS; do W="$W awd$k:100 awp$k:40 awe$k:40 awr$k:240 awt$k:280"; done
finish "$BCS" "dev:1 dev2ext:1 devr:1 test:2" "$W" "BC-j dev + test" 120 || echo "BC-j splits incomplete after one re-run"
down_all "$BCS"
unlock_eval
}

# ---------------------------------------------------------------- 9: tables and packs
phase9() {
echo "== $(ts) phase 9: tables"
cd $A
$PY aw_dev3.py awx3 awx3s1 bc3 bc3s1 awx3j awx3s1j bc3j bc3s1j --pairs awx3j:bc3j,awx3s1j:bc3s1j,bc3j:bc3,bc3s1j:bc3s1 \
  --pool "awx3j+awx3s1j:bc3j+bc3s1j,bc3j+bc3s1j:bc3+bc3s1" | tee $CH/j_dev_all.txt
$PY aw_pool.py "awx3j+awx3s1j:bc3j+bc3s1j" "awx3j+awx3s1j:awx3+awx3s1" "bc3j+bc3s1j:bc3+bc3s1" | tee $CH/j_test.txt
$PY aw_summary.py awtawx3j awtawx3s1j awtbc3j awtbc3s1j awpbc3j awpbc3s1j 2>/dev/null
for pr in "awtawx3j awtbc3j" "awtawx3s1j awtbc3s1j" "awtawx3j awtawx3" "awtawx3s1j awtawx3s1" "awtbc3j awtbc3" "awtbc3s1j awtbc3s1"; do
  set -- $pr; $PY aw_paired.py $1 $2
done
diag_j awrawx3j awrawx3s1j awrbc3j awrbc3s1j
pack j_bcj "$(nm bc3j dev dev2ext devr test) $(nm bc3s1j dev dev2ext devr test)"
(cd $A && tar czf pack/j_logs.tgz run_awx3j.log run_awx3s1j.log run_bc3j.log run_bc3s1j.log chain/j_master.log chain/j_*.txt \
   chain/j_build.log val0_j_*.log 2>&1 | tail -n 2; md5sum pack/j_logs.tgz)
}

for ph in 0 1 2 3 4 5 7 8 9; do [ $ph -ge $FROM ] && phase$ph; done
end
