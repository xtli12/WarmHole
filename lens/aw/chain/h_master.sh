#!/bin/bash
# h_master.sh c1 -- DETACHED on the server (log $A/chain/h_master.log).  Round h of the handover 7.14 (plan and every
# rule written 11-01 19:05 before any round-h result): round g (7.13) with the CORRIDOR TEACHER on the new-instruction
# branch.  The B->A freeze point is a kinematic trap (elbow joint at its limit, wrist on the cabinet; 4.20, 216); the
# round-g teacher crossed at the trap height and failed there.  The corridor teacher first lifts the end effector to H
# above the new bowl, crosses at that height and goes down to the unchanged hand-back point.
#   0  wait until no process of this project runs and GPUs 0-7 are free; >= 300 GiB on /data
#   1  pilot (selects H, nothing else): the 205 freeze-triggered round-g snapshots x {original teacher replayed,
#      corridor 0.30, corridor 0.36}; 8 teacher servers (GPUs 0-7, ports 8781-8788), 3 workers each, no step records.
#      Rule: the H with more successes (tie 0.30).  Plus: which limit of joint 4 the freeze snapshots touch (diagnostic)
#   2  round-h data: the corridor teacher (selected H) from every round-g take-over snapshot of the 4 students (817),
#      step records on; then the teacher servers stop
#   3  datasets data_h6_<student> (category 6); data_g7_<student> of round g (category 7) checked; data rule >= 60 each
#   4  AW-H training: awx3h (seed 0, GPUs 0-3), awx3s1h (seed 1, GPUs 4-7); fingerprints, health rule
#   5  AW-H development evaluation and the rule of 7.14 (strictly more pooled dev successes than awx3 + awx3s1) ->
#      $CH/h_decision.txt.  NOGO: stop here
#   7  GO: AW-H test split once each while bc3h trains (GPUs 4-7); then bc3s1h (GPUs 0-3)
#   8  BC-H: dev (+ no-switch), dev2 extension, devR and test split once each, whatever the result
#   9  tables and result packs in $A/pack
source ${HLIB:-${BH:-/data/blackhole}/lens/aw/chain/h_lib.sh}
end() { echo "== $(ts) h_master END"; exit 0; }
TR0="$TR"; TR1="${TR/--seed 0/--seed 1}"
STU="awx3 awx3s1 bc3 bc3s1"; AWS="awx3h awx3s1h"; BCS="bc3h bc3s1h"
declare -A KIND=([awx3]=awx [awx3s1]=awx [bc3]=bc [bc3s1]=bc [awx3h]=awx [awx3s1h]=awx [bc3h]=bc [bc3s1h]=bc)
declare -A SRV=([awx3h]="8771:0 8772:1" [awx3s1h]="8773:2 8774:3" [bc3h]="8775:0 8776:1" [bc3s1h]="8777:2 8778:3")
declare -A TPAIR=([awx3]="8781:0 8782:1" [awx3s1]="8783:2 8784:3" [bc3]="8785:4 8786:5" [bc3s1]="8787:6 8788:7")
TS="8781:0 8782:1 8783:2 8784:3 8785:4 8786:5 8787:6 8788:7"
SHARE="--share 6:0.15,7:0.15"
P() { echo $A/run_$1/params_6000.npz; }
D() { echo $A/data_pi05,$A/data_h6_$1,$A/data_g7_$1; }      # training roots of the round-h model of student $1
alive() { pgrep -fc 'aw_server(_c9)?[.]py|lens_server[.]py|bh_client_lens[.]py|aw_train(_c11)?[.]py|[efg]_master(_c2)?[.]sh|h0_trap'; }
up_all() { local k ok=0; for k in $1; do servers_up ${KIND[$k]} $(P $k) $k "${SRV[$k]}" || ok=1; done; return $ok; }
down_all() { local k; for k in $1; do servers_down ${KIND[$k]} $(P $k) "${SRV[$k]}"; done; }
healthy_all() { local k sg; for k in $1; do for sg in ${SRV[$k]}; do curl -s -m 3 localhost:${sg%%:*}/healthz > /dev/null || return 1; done; done; }
# finish (as in g_master) for the evaluation servers of the round-h models
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
echo "== $(ts) h_master c1 start (pid $$), from phase $FROM; training flags seed 0: $TR0"
(cd $A && md5sum h_check.py chain/h_lib.sh chain/h_master.sh aw_build_data.py g_check.py chain/g_lib.sh ../bh_client_lens.py \
   aw_train.py aw_dev3.py aw_pool.py e_check.py chain/e_lib.sh aw_eval.sh aw_server.py aw_server.py \
   ../bh_client_lens.py ../bh_client_lens.py h0_trap.py freeze_anatomy.py)

# ---------------------------------------------------------------- 0: wait for free cards
phase0() {
for i in $(seq 1 360); do
  [ "$(alive)" -eq 0 ] && gpus_free 0 7 && break
  [ $((i % 20)) -eq 1 ] && echo "$(ts) waiting: processes of this project $(alive); GPUs 0-7 free: $(gpus_free 0 7 && echo yes || echo no)"
  sleep 30
done
{ [ "$(alive)" -eq 0 ] && gpus_free 0 7; } || { echo "GPUs 0-7 not free or processes of this project alive after 3 h: nothing started"; end; }
AV=$(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9)
echo "$(ts) /data: ${AV} GiB available; memory anon $(anon) GiB; $(free -g | sed -n 2p)"
[ "${AV:-0}" -ge 300 ] || { echo "less than 300 GiB free on /data: nothing started"; end; }
for s in $STU; do [ -f $A/data_g7_$s/meta.npz ] || { echo "no $A/data_g7_$s (round g): nothing started"; end; }; done
}

# ---------------------------------------------------------------- 1: pilot (selects H)
phase1() {
echo "== $(ts) phase 1: pilot (selects the corridor height H only; 7.14)"
[ -d $BH/runs/lens/pi05_h1p_p0 ] && { echo "results of h1p exist already: nothing started"; end; }
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing started"; end; }
cd $A
$PY $HCHK specs $BH/runs/lens $BH/specs/lens h1p pilot "$STU" --only freeze || { echo "pilot specs failed: nothing started"; end; }
NP=$(wc -l < $BH/specs/lens/pi05_h1p.jsonl)
echo "-- which limit of joint 4 the freeze snapshots touch (diagnostic, LIBERO venv, CPU 112-127, EGL GPU 0)"
(export PYTHONPATH=$OP/third_party/libero:$OP/packages/openpi-client/src:$L:$BH/code
 export LIBERO_CONFIG_PATH=$BH/libero_cfg MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
 export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
 cd $A && taskset -c 112-127 timeout 900 $BH/venvs/libero/bin/python $A/h0_side.py $BH/runs/lens gawx3 gbc3 \
   > $CH/h0_side.log 2>&1 < /dev/null) &
SIDE=$!
lock_eval
mem_wait 52
t_up "$TS" || { echo "teacher servers not healthy: nothing started"; t_down "$TS"; unlock_eval; end; }
mem_wait 40
dg_run h1p "$(tg "$TS")" 3
dg_finish "pilot" "h1p:$NP" "h1p=$(tg "$TS")" 3 60 || echo "pilot incomplete after one re-run (the selection uses what is there)"
wait $SIDE
grep -E "^SIDE" $CH/h0_side.log | cut -c1-250 || echo "joint-4 side diagnostic: no output (see $CH/h0_side.log)"
cd $A
$PY $HCHK pilot $BH/runs/lens h1p $CH/h_pilot.txt | tee $CH/h_pilot_sum.txt
echo "$(ts) pilot selection: $(cat $CH/h_pilot.txt 2>/dev/null)"
[ -s $CH/h_pilot.txt ] || { echo "no selection: stop"; t_down "$TS"; unlock_eval; end; }
unlock_eval                       # the teacher servers keep running for phase 2
}

# ---------------------------------------------------------------- 2: round-h data (corridor teacher from every take-over)
phase2() {
H=$(sed -n 's/^H=//p' $CH/h_pilot.txt 2>/dev/null | head -n 1)
[ -n "$H" ] || { echo "no H selected: stop"; t_down "$TS"; end; }
echo "== $(ts) phase 2: round-h data, corridor teacher H=$H from every round-g take-over of $STU"
for s in $STU; do [ -d $BH/runs/lens/pi05_h${s}_p0 ] && { echo "results of h$s exist already: nothing started"; t_down "$TS"; end; }; done
cd $A
for s in $STU; do $PY $HCHK specs $BH/runs/lens $BH/specs/lens h$s corridor "$s" --H $H --rec || { echo "specs h$s failed: stop"; t_down "$TS"; end; }; done
lock_eval
if ! t_healthy "$TS"; then
  t_down "$TS"; mem_wait 52
  t_up "$TS" || { echo "teacher servers not healthy: no rollouts"; t_down "$TS"; unlock_eval; end; }
fi
mem_wait 40
W=""; SP=""
for s in $STU; do
  dg_run h$s "$(tg "${TPAIR[$s]}")" 3
  W="$W h$s:$(wc -l < $BH/specs/lens/pi05_h$s.jsonl)"; SP="$SP;h$s=$(tg "${TPAIR[$s]}")"
done
dg_finish "round-h data" "${W# }" "${SP#;}" 3 120 || echo "round-h data incomplete after one re-run (the datasets use what is there)"
t_down "$TS"
unlock_eval
cd $A
$PY $HCHK hsum $BH/runs/lens $(for s in $STU; do printf 'h%s ' $s; done) | tee $CH/h_data.txt
pack h_data "h1p $(for s in $STU; do printf 'h%s ' $s; done)"
du -sh $(for s in $STU; do echo $BH/runs/lens/pi05_h${s}_p*; done) $BH/runs/lens/pi05_h1p_p* 2>/dev/null | awk '{s=s" "$1} END{print "run directories:" s}'
}

# ---------------------------------------------------------------- 3: datasets and the data-volume rule
phase3() {
echo "== $(ts) phase 3: round-h datasets"
cd $A
mem_wait 16
BP=""
for s in $STU; do
  rm -rf $A/data_h6_$s
  (taskset -c 64-127 timeout 3600 $PY $TRAIN6 $A/data_h6_$s $(ls -d $BH/runs/lens/pi05_h${s}_p*) --cat_as 6 --procs 6 \
     > $CH/h_build6_$s.log 2>&1 < /dev/null) &
  BP="$BP $!"
done
wait $BP
for s in $STU; do echo "-- data_h6_$s"; grep -vE "^\s*$" $CH/h_build6_$s.log | tail -n 6 | cut -c1-300; done
OKD=1
$PY g_check.py data 6 $(for s in $STU; do printf "$A/data_h6_%s " $s; done) || OKD=0
$PY g_check.py data 7 $(for s in $STU; do printf "$A/data_g7_%s " $s; done) || OKD=0
for s in $STU; do $PY $HCHK enough $BH/runs/lens h$s 60 || OKD=0; done
du -sh $(for s in $STU; do echo $A/data_h6_$s; done) | awk '{s=s" "$1} END{print "datasets:" s}'; echo "/data $(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9) GiB free"
[ $OKD -eq 1 ] || { echo "a round-h dataset is not consistent or too small (7.14): nothing trained"; end; }
}

# ---------------------------------------------------------------- 4: AW-H training
phase4() {
echo "== $(ts) phase 4: AW-H training (awx3h seed 0 GPUs 0-3, awx3s1h seed 1 GPUs 4-7)"
for k in $AWS $BCS; do for n in $(nm $k dev dev2ext devr test); do [ -d $BH/runs/lens/pi05_${n}_p0 ] && { echo "results of $n exist already: nothing trained"; end; }; done; done
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing trained"; end; }
mem_wait 40
TR="$TR0"; start_train awx3h 0,1,2,3 --data $(D awx3) --kind awx --geo_aug $SHARE
grep -E "data roots|category shares|trainable|Traceback" $A/run_awx3h.log | cut -c25-330
if fp_ok $A/run_awx3h.log 0.0541 1.1345; then echo "awx3h healthy, kept"; else echo "awx3h NOT healthy: stopped"; tail -n 5 $A/run_awx3h.log | cut -c1-300; stop_train awx3h; end; fi
mem_wait 40
TR="$TR1"; start_train awx3s1h 4,5,6,7 --data $(D awx3s1) --kind awx --geo_aug $SHARE; TR="$TR0"
grep -E "data roots|category shares|trainable|Traceback" $A/run_awx3s1h.log | cut -c25-330
if fp_ok $A/run_awx3s1h.log 0.0540 0.8999; then echo "awx3s1h healthy, kept"; else echo "awx3s1h NOT healthy: stopped (awx3h continues; this chain stops)"; tail -n 5 $A/run_awx3s1h.log | cut -c1-300; stop_train awx3s1h; end; fi
for k in $AWS; do $PY -c "import json; d=json.load(open('$A/run_$k/args.json')); print('$k args: seed', d.get('seed'), 'share', d.get('share'), 'data', d.get('data'))"; done
wait_train awx3h 220 || { echo "awx3h still running after 220 min: stopping here"; end; }
wait_train awx3s1h 60 || { echo "awx3s1h still running: stopping here"; end; }
for k in $AWS; do grep -E "VAL step|Traceback" $A/run_$k.log | cut -c1-300 | tail -n 13; done
{ train_health awx3h && train_health awx3s1h; } || { echo "an AW-H model does not pass the health rule: by 7.7 it is retrained once (not automatic)"; end; }
}

# ---------------------------------------------------------------- 5: AW-H development evaluation and the rule of 7.14
phase5() {
echo "== $(ts) phase 5: AW-H development evaluation"
gpus_free 0 3 || { echo "GPUs 0-3 busy: not evaluated"; end; }
lock_eval
mem_wait 26
if ! up_all "$AWS"; then echo "servers not healthy: no rollouts"; down_all "$AWS"; unlock_eval; end; fi
mem_wait 45
for k in $AWS; do ev awx $(P $k) $k "${SRV[$k]}" dev:2 dev2ext:1 devr:4; done
W=""; for k in $AWS; do W="$W awd$k:100 awp$k:40 awe$k:40 awr$k:240"; done
finish "$AWS" "dev:1 dev2ext:1 devr:1" "$W" "AW-H dev" 90 || echo "AW-H development split incomplete after one re-run (the rule then gives NOGO)"
cd $A
$PY aw_summary.py awdawx3h awpawx3h awdawx3s1h awpawx3s1h 2>/dev/null
$PY aw_dev3.py awx3 awx3s1 bc3 bc3s1 awx3h awx3s1h --pairs awx3h:awx3,awx3s1h:awx3s1,awx3h:bc3,awx3s1h:bc3s1 \
  --pool "awx3h+awx3s1h:awx3+awx3s1,awx3h+awx3s1h:bc3+bc3s1" --select "awx3h+awx3s1h:awx3+awx3s1" --out $CH/h_decision.txt | tee $CH/h_dev_aw.txt
$PY freeze_anatomy.py $BH/runs/lens awrawx3h awrawx3s1h 2>&1 | grep -vE "path every" | cut -c1-330
pack h_aw_dev "$(nm awx3h dev dev2ext devr) $(nm awx3s1h dev dev2ext devr)"
DC=$(head -n 1 $CH/h_decision.txt 2>/dev/null)
echo "$(ts) decision (7.14): ${DC:-none}"
case "$DC" in
  GO*) ;;
  *) echo "not GO: by 7.14 no test split and no BC-H; this chain stops here"; down_all "$AWS"; unlock_eval; end ;;
esac
}

# ---------------------------------------------------------------- 7: AW-H test split; BC-H training
phase7() {
echo "== $(ts) phase 7: AW-H test split (once each, same servers); bc3h training on GPUs 4-7"
mem_wait 20
for k in $AWS; do ev awx $(P $k) $k "${SRV[$k]}" test:4; done
BC0=no
if gpus_free 4 7; then
  mem_wait 40
  TR="$TR0"; start_train bc3h 4,5,6,7 --data $(D bc3) --kind bc --geo_aug $SHARE
  grep -E "data roots|category shares|Traceback" $A/run_bc3h.log | cut -c25-330
  if fp_bc $A/run_bc3h.log 0.0410; then BC0=yes; echo "bc3h healthy, kept"; else echo "bc3h NOT healthy: stopped"; stop_train bc3h; BC0=bad; fi
else echo "GPUs 4-7 busy: bc3h starts after the test split"; fi
finish "$AWS" "test:2" "awtawx3h:280 awtawx3s1h:280" "AW-H test" 60 || echo "AW-H test split incomplete after one re-run"
down_all "$AWS"
unlock_eval
cd $A
pack h_aw_test "awtawx3h awtawx3s1h"
$PY aw_pool.py "awx3h+awx3s1h:awx3+awx3s1" "awx3h+awx3s1h:bc3+bc3s1" | tee $CH/h_test_aw.txt
[ "$BC0" = bad ] && { echo "bc3h fingerprint failed: BC-H not trained (decided by hand)"; end; }
gpus_free 0 3 || { echo "GPUs 0-3 busy: bc3s1h not started"; end; }
mem_wait 40
TR="$TR1"; start_train bc3s1h 0,1,2,3 --data $(D bc3s1) --kind bc --geo_aug $SHARE; TR="$TR0"
grep -E "data roots|category shares|Traceback" $A/run_bc3s1h.log | cut -c25-330
if fp_bc $A/run_bc3s1h.log 0.0409; then echo "bc3s1h healthy, kept"; else echo "bc3s1h NOT healthy: stopped"; stop_train bc3s1h; end; fi
if [ "$BC0" = no ]; then
  gpus_free 4 7 || { echo "GPUs 4-7 busy: bc3h not started"; end; }
  mem_wait 40
  TR="$TR0"; start_train bc3h 4,5,6,7 --data $(D bc3) --kind bc --geo_aug $SHARE
  fp_bc $A/run_bc3h.log 0.0410 || { echo "bc3h NOT healthy: stopped"; stop_train bc3h; end; }
fi
for k in $BCS; do $PY -c "import json; d=json.load(open('$A/run_$k/args.json')); print('$k args: seed', d.get('seed'), 'share', d.get('share'), 'data', d.get('data'))"; done
wait_train bc3h 220 || { echo "bc3h still running after 220 min: stopping here"; end; }
wait_train bc3s1h 120 || { echo "bc3s1h still running: stopping here"; end; }
for k in $BCS; do grep -E "VAL step|Traceback" $A/run_$k.log | cut -c1-300 | tail -n 13; done
{ train_health bc3h && train_health bc3s1h; } || { echo "a BC-H model does not pass the health rule: not evaluated"; end; }
}

# ---------------------------------------------------------------- 8: BC-H development and test splits
phase8() {
echo "== $(ts) phase 8: BC-H development splits and test split (once each)"
gpus_free 0 3 || { echo "GPUs 0-3 busy: not evaluated"; end; }
lock_eval
mem_wait 26
if ! up_all "$BCS"; then echo "servers not healthy: no rollouts"; down_all "$BCS"; unlock_eval; end; fi
mem_wait 52
for k in $BCS; do ev bc $(P $k) $k "${SRV[$k]}" dev:2 dev2ext:1 devr:3 test:3; done
W=""; for k in $BCS; do W="$W awd$k:100 awp$k:40 awe$k:40 awr$k:240 awt$k:280"; done
finish "$BCS" "dev:1 dev2ext:1 devr:1 test:2" "$W" "BC-H dev + test" 120 || echo "BC-H splits incomplete after one re-run"
down_all "$BCS"
unlock_eval
}

# ---------------------------------------------------------------- 9: tables and packs
phase9() {
echo "== $(ts) phase 9: tables"
cd $A
$PY aw_dev3.py awx3 awx3s1 bc3 bc3s1 awx3h awx3s1h bc3h bc3s1h --pairs awx3h:bc3h,awx3s1h:bc3s1h,bc3h:bc3,bc3s1h:bc3s1 \
  --pool "awx3h+awx3s1h:bc3h+bc3s1h,bc3h+bc3s1h:bc3+bc3s1" | tee $CH/h_dev_all.txt
$PY aw_pool.py "awx3h+awx3s1h:bc3h+bc3s1h" "awx3h+awx3s1h:awx3+awx3s1" "bc3h+bc3s1h:bc3+bc3s1" | tee $CH/h_test.txt
$PY aw_summary.py awtawx3h awtawx3s1h awtbc3h awtbc3s1h awpbc3h awpbc3s1h 2>/dev/null
for pr in "awtawx3h awtbc3h" "awtawx3s1h awtbc3s1h" "awtawx3h awtawx3" "awtawx3s1h awtawx3s1" "awtbc3h awtbc3" "awtbc3s1h awtbc3s1"; do
  set -- $pr; $PY aw_paired.py $1 $2
done
$PY freeze_anatomy.py $BH/runs/lens awrbc3h awrbc3s1h 2>&1 | grep -vE "path every" | cut -c1-330
pack h_bc "$(nm bc3h dev dev2ext devr test) $(nm bc3s1h dev dev2ext devr test)"
(cd $A && tar czf pack/h_logs.tgz run_awx3h.log run_awx3s1h.log run_bc3h.log run_bc3s1h.log chain/h_master.log chain/h_*.txt \
   chain/h_build*.log chain/h0_side.log 2>&1 | tail -n 2; md5sum pack/h_logs.tgz)
}

for ph in 0 1 2 3 4 5 7 8 9; do [ $ph -ge $FROM ] && phase$ph; done
end
