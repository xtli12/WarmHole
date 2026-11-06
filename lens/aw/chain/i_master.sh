#!/bin/bash
# i_master.sh c1 -- DETACHED on the server (log $A/chain/i_master.log).  Round i of the handover 7.15 (plan and every
# rule written 11-01 22:xx before any round-i result): switch-state relabeling with a kinematics-aware teacher, for AW
# (kind awx) and its matched BC, two seeds each.
#   0  wait until no process of this project runs and GPUs 0-7 are free (jobs of other users: wait, never touched)
#   1  smoke: 12 switch-state relabeling episodes of awx3 on layout r130 (outside every split; both directions, t_s 22 /
#      42 / 62); checks of the take-over at the switch, the step-record sources and the B->A corridor (i_check smoke);
#      data builder c7 (category 6, transit + first 40 policy frames); CPU training smokes with --share 6:0.3 (awx, bc;
#      validation batch unchanged); spec generator r24 against r23.  Nothing else starts unless every gate passes.
#   2  switch-state relabeling of the 4 students (training layouts 20-37 + r101-124, t_s 12-62 every 5 steps, A->B one
#      policy noise, B->A two: 1386 episodes each), 4 groups of student + teacher server
#   3  datasets data_i6_<student> (category 6) and the data-volume rule (>= 600 successful episodes, >= 400 of them
#      B->A, per student)
#   4  AW-i: awx3i (seed 0, GPUs 0-3) and awx3s1i (seed 1, GPUs 4-7), fingerprint-checked, health rule
#   5  AW-i development evaluation (dev + no-switch, dev2 extension, devR) and the rule of 7.15 (aw_dev3_c2 --select:
#      strictly more pooled dev successes than awx3 + awx3s1) -> $CH/i_decision.txt.  NOGO: stop here
#   7  GO: AW-i test split once each on the same servers while bc3i trains (GPUs 4-7); then bc3s1i (GPUs 0-3)
#   8  BC-i: health rule; dev (+ no-switch), dev2 extension, devR and test split once each, whatever the result
#   9  tables and result packs in $A/pack
source ${ILIB:-${BH:-/data/blackhole}/lens/aw/chain/i_lib.sh}
end() { echo "== $(ts) i_master END"; exit 0; }
TR0="$TR"; TR1="${TR/--seed 0/--seed 1}"
STU="awx3 awx3s1 bc3 bc3s1"; AWS="awx3i awx3s1i"; BCS="bc3i bc3s1i"
declare -A KIND=([awx3]=awx [awx3s1]=awx [bc3]=bc [bc3s1]=bc [awx3i]=awx [awx3s1i]=awx [bc3i]=bc [bc3s1i]=bc)
declare -A GRP=([awx3]="8751:8752:0:1" [awx3s1]="8753:8754:2:3" [bc3]="8755:8756:4:5" [bc3s1]="8757:8758:6:7")
declare -A SRV=([awx3i]="8761:0 8762:1" [awx3s1i]="8763:2 8764:3" [bc3i]="8765:0 8766:1" [bc3s1i]="8767:2 8768:3")
SHARE="--share 6:0.3"
SW_LAYOUTS=${SW_LAYOUTS:-20-37,r101-124}
SW_TS=${SW_TS:-12,17,22,27,32,37,42,47,52,57,62}
SW_NOISE=${SW_NOISE:-AB:1,BA:2}
IMIN=${IMIN:-600}; IMINBA=${IMINBA:-400}
TEACH="$BH/runs/lens/pi05_teach_p0 $BH/runs/lens/pi05_teach_p1"
P() { echo $A/run_$1/params_6000.npz; }
D() { echo $A/data_pi05,$A/data_i6_$1; }      # training roots of the round-i model of student $1
alive() { pgrep -fc 'aw_server(_c9)?[.]py|lens_server[.]py|bh_client_lens[.]py|aw_train(_c11)?[.]py|[e-h]_master(_c[0-9])?[.]sh'; }
up_all() { local k ok=0; for k in $1; do servers_up ${KIND[$k]} $(P $k) $k "${SRV[$k]}" || ok=1; done; return $ok; }
down_all() { local k; for k in $1; do servers_down ${KIND[$k]} $(P $k) "${SRV[$k]}"; done; }
healthy_all() { local k sg; for k in $1; do for sg in ${SRV[$k]}; do curl -s -m 3 localhost:${sg%%:*}/healthz > /dev/null || return 1; done; done; }
# finish (as in e_master / g_master) for the evaluation servers of the round-i models
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
echo "== $(ts) i_master c1 start (pid $$), from phase $FROM; training flags seed 0: $TR0"
echo "   relabeling layouts $SW_LAYOUTS, t_s $SW_TS, noises $SW_NOISE; data rule >= $IMIN successes, >= $IMINBA B->A"
(cd $A && md5sum aw_build_data.py i_check.py si_diag.py chain/i_lib.sh chain/i_master.sh chain/g_lib.sh ../bh_client_lens.py \
   ../make_specs_lens.py aw_train.py aw_dev3.py aw_pool.py e_check.py chain/e_lib.sh aw_eval.sh aw_server.py \
   aw_server.py ../bh_client_lens.py ../bh_client_lens.py ../make_specs_lens.py)

# ---------------------------------------------------------------- 0: free cards
phase0() {
for i in $(seq 1 360); do
  [ "$(alive)" -eq 0 ] && gpus_free 0 7 && break
  [ $((i % 20)) -eq 1 ] && echo "$(ts) waiting: processes of this project $(alive)"
  sleep 30
done
{ [ "$(alive)" -eq 0 ] && gpus_free 0 7; } || { echo "GPUs 0-7 not free or processes of this project alive after 3 h: nothing started"; end; }
AV=$(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9)
echo "$(ts) /data: ${AV} GiB available; memory anon $(anon) GiB; $(free -g | sed -n 2p)"
[ "${AV:-0}" -ge 400 ] || { echo "less than 400 GiB free on /data: nothing started"; end; }
}

# ---------------------------------------------------------------- 1: smoke
phase1() {
echo "== $(ts) phase 1: smoke"
cd $A
SOK=1
G0="8790:8791:0:1"
rm -rf $BH/runs/lens/pi05_ismk_p*
rm -rf $A/data_i6_smk $A/smoke_i0 $A/smoke_i1 $A/smoke_i2
lock_eval
mem_wait 20
if ! dg_up awx $(P awx3) "$G0"; then echo "smoke servers not healthy: nothing else started"; dg_down awx $(P awx3) "$G0"; unlock_eval; end; fi
$PY $MS24 $BH/specs/lens --swdag ismk --sw_layouts r130 --sw_ts 22,42,62 --sw_noise AB:2,BA:2 --bbs pi05 --vlm_offset "$OFF"
dg_run ismk "$G0" 4
dg_wait "smoke relabeling" ismk 12 30
[ "$(dgwk ismk)" -eq 0 ] || { echo "smoke workers still running: stopped"; dg_kill ismk; }
dg_down awx $(P awx3) "$G0"
unlock_eval
grep -hE "Traceback|Error" $BH/runs/lens/pi05_ismk_p0/worker_*.log | sort | uniq -c | head -n 5
$PY i_check.py smoke $BH/runs/lens ismk || SOK=0
$PY i_check.py dsum $BH/runs/lens ismk
taskset -c 64-95 timeout 900 $PY $TRAIN7 $A/data_i6_smk $BH/runs/lens/pi05_ismk_p0 --cat_as 6 --policy_max 40 --procs 4 2>&1 | tail -n 7 | cut -c1-300
$PY i_check.py data $A/data_i6_smk || SOK=0
echo "-- spec generator r24 against r23 (every existing call)"
rm -rf $A/spec_chk4; mkdir -p $A/spec_chk4/a $A/spec_chk4/b
for d in a b; do F=$MS23; [ $d = b ] && F=$MS24
  $PY $F $A/spec_chk4/$d --dev flush --name awdx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk4/$d --plain awpx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk4/$d --test flush --name awdxx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk4/$d --dev flush --name awex --ts 52,62 --bbs pi05 > /dev/null
  $PY $F $A/spec_chk4/$d --devr awrx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk4/$d --selfplay spx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk4/$d --dev cig05,cig10 --name awdg --ts 12,22,32,42,52,62 --bbs pi05 > /dev/null
  $PY $F $A/spec_chk4/$d --teach --vlm_offset "$OFF" > /dev/null
  $PY $F $A/spec_chk4/$d --dagger dg10,dg30,dgb20 --name dagx --bbs pi05 --vlm_offset "$OFF" > /dev/null
  $PY $F $A/spec_chk4/$d --cfdag gx --cf_layouts 20-37,r101-124 --bbs pi05 --vlm_offset "$OFF" > /dev/null
done
if diff -rq $A/spec_chk4/a $A/spec_chk4/b; then echo "specs identical ($(ls $A/spec_chk4/a | wc -l) files)"; else SOK=0; echo "SPECS DIFFER"; fi
rm -rf $A/spec_chk4
echo "-- CPU training smokes"
SMK="--smoke --bs 32 --lam_loc 0.1 --lam_coord 0.2 --skip_gn 1000 --seed 0"
mem_wait 24
(CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$A taskset -c 80-95 timeout 1500 $PY $A/aw_train.py $SMK --data $A/data_pi05 \
   --kind awx --geo_aug --out $A/smoke_i0 > $A/smoke_i0.log 2>&1 < /dev/null) &
PS0=$!
(CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$A taskset -c 96-111 timeout 1500 $PY $A/aw_train.py $SMK \
   --data $A/data_pi05,$A/data_i6_smk --kind awx --geo_aug $SHARE --out $A/smoke_i1 > $A/smoke_i1.log 2>&1 < /dev/null) &
PS1=$!
(CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$A taskset -c 112-127 timeout 1500 $PY $A/aw_train.py $SMK \
   --data $A/data_pi05,$A/data_i6_smk --kind bc --geo_aug $SHARE --out $A/smoke_i2 > $A/smoke_i2.log 2>&1 < /dev/null) &
PS2=$!
wait $PS0 $PS1 $PS2
for k in i0 i1 i2; do
  echo "-- smoke_$k"; grep -E "data roots|frames [0-9]+: train|category shares|step 0 |VAL step 0|smoke done|Error|Traceback" $A/smoke_$k.log | cut -c25-330
  grep -q "smoke done" $A/smoke_$k.log || { SOK=0; tail -n 8 $A/smoke_$k.log | cut -c1-300; }
done
$PY e_check.py val0 $A/smoke_i0.log $A/smoke_i1.log || SOK=0
for k in i1 i2; do grep -E "category shares" $A/smoke_$k.log | grep -qE "6: 0[.]3[},]" \
  || { SOK=0; echo "category 6 does not have the share 0.3 in smoke $k"; }; done
echo "$(ts) smoke gates passed: $SOK"
[ "$SOK" -eq 1 ] || { echo "nothing else started"; end; }
rm -rf $A/smoke_i0 $A/smoke_i1 $A/smoke_i2
}

# ---------------------------------------------------------------- 2: switch-state relabeling of the 4 students
phase2() {
echo "== $(ts) phase 2: switch-state relabeling of $STU"
for s in $STU; do [ -d $BH/runs/lens/pi05_i${s}_p0 ] && { echo "results of i$s exist already: nothing started"; end; }; done
for s in $STU; do [ -f $(P $s) ] || { echo "no $(P $s): nothing started"; end; }; done
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing started"; end; }
lock_eval
mem_wait 52
OKS=1; for s in $STU; do dg_up ${KIND[$s]} $(P $s) "${GRP[$s]}" || OKS=0; done
[ $OKS -eq 1 ] || { echo "servers not healthy: no rollouts"; for s in $STU; do dg_down ${KIND[$s]} $(P $s) "${GRP[$s]}"; done; unlock_eval; end; }
echo "$(ts) servers up; anon $(anon) GiB"
for s in $STU; do $PY $MS24 $BH/specs/lens --swdag i$s --sw_layouts $SW_LAYOUTS --sw_ts $SW_TS --sw_noise $SW_NOISE \
  --bbs pi05 --vlm_offset "$OFF"; done
NI=$(wc -l < $BH/specs/lens/pi05_iawx3.jsonl)
mem_wait 40
for s in $STU; do dg_run i$s "${GRP[$s]}" 6; done
W=""; SP=""; for s in $STU; do W="$W i$s:$NI"; SP="$SP;i$s=${GRP[$s]}"; done
dg_finish "switch-state relabeling" "$W" "${SP#;}" 6 240 || { echo "relabeling rollouts incomplete after one re-run (the data rule decides)"; }
for s in $STU; do dg_down ${KIND[$s]} $(P $s) "${GRP[$s]}"; done
unlock_eval
cd $A
$PY i_check.py dsum $BH/runs/lens $(for s in $STU; do printf 'i%s ' $s; done) | tee $CH/i_data.txt
pack i_data "$(for s in $STU; do printf 'i%s ' $s; done)"
du -sh $(for s in $STU; do echo $BH/runs/lens/pi05_i${s}_p*; done) 2>/dev/null | awk '{s=s" "$1} END{print "run directories:" s}'
echo "-- switch states of the relabelled episodes against the teacher data (diagnostic, 220)"
taskset -c 96-127 timeout 900 $PY si_diag.py $BH/runs/lens --variants dg0c --sets $(for s in $STU; do printf 'i%s ' $s; done) --teach $TEACH 2>&1 \
  | grep -E "^teacher|^sets| B->A " | grep -vE "part B|new bowl to plate" | cut -c1-300 | head -n 40
}

# ---------------------------------------------------------------- 3: datasets and the data-volume rule
phase3() {
echo "== $(ts) phase 3: round-i datasets"
mem_wait 16
BP=""
for s in $STU; do
  rm -rf $A/data_i6_$s
  (taskset -c 64-127 timeout 5400 $PY $TRAIN7 $A/data_i6_$s $(ls -d $BH/runs/lens/pi05_i${s}_p*) --cat_as 6 --policy_max 40 --procs 6 \
     > $CH/i_build6_$s.log 2>&1 < /dev/null) &
  BP="$BP $!"
done
wait $BP
for s in $STU; do echo "-- data_i6_$s"; grep -vE "^\s*$" $CH/i_build6_$s.log | tail -n 6 | cut -c1-300; done
OKD=1
$PY i_check.py data $(for s in $STU; do printf "$A/data_i6_%s " $s; done) || OKD=0
for s in $STU; do $PY i_check.py enough $BH/runs/lens i$s $IMIN $IMINBA || OKD=0; done
du -sh $(for s in $STU; do echo $A/data_i6_$s; done) | awk '{s=s" "$1} END{print "datasets:" s}'; echo "/data $(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9) GiB free"
[ $OKD -eq 1 ] || { echo "a round-i dataset is not consistent or too small (7.15): nothing trained"; end; }
}

# ---------------------------------------------------------------- 4: AW-i training
phase4() {
echo "== $(ts) phase 4: AW-i training (awx3i seed 0 GPUs 0-3, awx3s1i seed 1 GPUs 4-7)"
for k in $AWS $BCS; do for n in $(nm $k dev dev2ext devr test); do [ -d $BH/runs/lens/pi05_${n}_p0 ] && { echo "results of $n exist already: nothing trained"; end; }; done; done
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing trained"; end; }
mem_wait 40
TR="$TR0"; start_train awx3i 0,1,2,3 --data $(D awx3) --kind awx --geo_aug $SHARE
grep -E "data roots|category shares|trainable|Traceback" $A/run_awx3i.log | cut -c25-330
if fp_ok $A/run_awx3i.log 0.0541 1.1345; then echo "awx3i healthy, kept"; else echo "awx3i NOT healthy: stopped"; tail -n 5 $A/run_awx3i.log | cut -c1-300; stop_train awx3i; end; fi
mem_wait 40
TR="$TR1"; start_train awx3s1i 4,5,6,7 --data $(D awx3s1) --kind awx --geo_aug $SHARE; TR="$TR0"
grep -E "data roots|category shares|trainable|Traceback" $A/run_awx3s1i.log | cut -c25-330
if fp_ok $A/run_awx3s1i.log 0.0540 0.8999; then echo "awx3s1i healthy, kept"; else echo "awx3s1i NOT healthy: stopped (awx3i continues; this chain stops)"; tail -n 5 $A/run_awx3s1i.log | cut -c1-300; stop_train awx3s1i; end; fi
for k in $AWS; do $PY -c "import json; d=json.load(open('$A/run_$k/args.json')); print('$k args: seed', d.get('seed'), 'share', d.get('share'), 'data', d.get('data'))"; done
wait_train awx3i 220 || { echo "awx3i still running after 220 min: stopping here"; end; }
wait_train awx3s1i 60 || { echo "awx3s1i still running: stopping here"; end; }
for k in $AWS; do grep -E "VAL step|Traceback" $A/run_$k.log | cut -c1-300 | tail -n 13; done
{ train_health awx3i && train_health awx3s1i; } || { echo "an AW-i model does not pass the health rule: by 7.7 it is retrained once (not automatic)"; end; }
}

# ---------------------------------------------------------------- 5: AW-i development evaluation and the rule of 7.15
phase5() {
echo "== $(ts) phase 5: AW-i development evaluation"
gpus_free 0 3 || { echo "GPUs 0-3 busy: not evaluated"; end; }
lock_eval
mem_wait 26
if ! up_all "$AWS"; then echo "servers not healthy: no rollouts"; down_all "$AWS"; unlock_eval; end; fi
mem_wait 45
for k in $AWS; do ev awx $(P $k) $k "${SRV[$k]}" dev:2 dev2ext:1 devr:4; done
W=""; for k in $AWS; do W="$W awd$k:100 awp$k:40 awe$k:40 awr$k:240"; done
finish "$AWS" "dev:1 dev2ext:1 devr:1" "$W" "AW-i dev" 90 || echo "AW-i development split incomplete after one re-run (the rule then gives NOGO)"
cd $A
$PY aw_summary.py awdawx3i awpawx3i awdawx3s1i awpawx3s1i 2>/dev/null
$PY aw_dev3.py awx3 awx3s1 bc3 bc3s1 awx3i awx3s1i --pairs awx3i:awx3,awx3s1i:awx3s1,awx3i:bc3,awx3s1i:bc3s1 \
  --pool "awx3i+awx3s1i:awx3+awx3s1,awx3i+awx3s1i:bc3+bc3s1" --select "awx3i+awx3s1i:awx3+awx3s1" --out $CH/i_decision.txt | tee $CH/i_dev_aw.txt
echo "-- failure classes and switch states on devR / dev2 (diagnostic, not part of the rule)"
taskset -c 96-127 timeout 900 $PY si_diag.py $BH/runs/lens --sets awrawx3 awrawx3s1 awrawx3i awrawx3s1i awdawx3i+aweawx3i awdawx3s1i+aweawx3s1i \
  --teach $TEACH 2>&1 | grep -E "^sets|NN " | cut -c1-260 | head -n 60
pack i_awi_dev "$(nm awx3i dev dev2ext devr) $(nm awx3s1i dev dev2ext devr)"
DC=$(head -n 1 $CH/i_decision.txt 2>/dev/null)
echo "$(ts) decision (7.15): ${DC:-none}"
case "$DC" in
  GO*) ;;
  *) echo "not GO: by 7.15 no test split and no BC-i; this chain stops here"; down_all "$AWS"; unlock_eval; end ;;
esac
}

# ---------------------------------------------------------------- 7: AW-i test split; BC-i training
phase7() {
echo "== $(ts) phase 7: AW-i test split (once each, same servers); bc3i training on GPUs 4-7"
mem_wait 20
for k in $AWS; do ev awx $(P $k) $k "${SRV[$k]}" test:4; done
BC0=no
if gpus_free 4 7; then
  mem_wait 40
  TR="$TR0"; start_train bc3i 4,5,6,7 --data $(D bc3) --kind bc --geo_aug $SHARE
  grep -E "data roots|category shares|Traceback" $A/run_bc3i.log | cut -c25-330
  if fp_bc $A/run_bc3i.log 0.0410; then BC0=yes; echo "bc3i healthy, kept"; else echo "bc3i NOT healthy: stopped"; stop_train bc3i; BC0=bad; fi
else echo "GPUs 4-7 busy: bc3i starts after the test split"; fi
finish "$AWS" "test:2" "awtawx3i:280 awtawx3s1i:280" "AW-i test" 60 || echo "AW-i test split incomplete after one re-run"
down_all "$AWS"
unlock_eval
cd $A
pack i_awi_test "awtawx3i awtawx3s1i"
$PY aw_pool.py "awx3i+awx3s1i:awx3+awx3s1" "awx3i+awx3s1i:bc3+bc3s1" | tee $CH/i_test_aw.txt
[ "$BC0" = bad ] && { echo "bc3i fingerprint failed: BC-i not trained (decided by hand)"; end; }
gpus_free 0 3 || { echo "GPUs 0-3 busy: bc3s1i not started"; end; }
mem_wait 40
TR="$TR1"; start_train bc3s1i 0,1,2,3 --data $(D bc3s1) --kind bc --geo_aug $SHARE; TR="$TR0"
grep -E "data roots|category shares|Traceback" $A/run_bc3s1i.log | cut -c25-330
if fp_bc $A/run_bc3s1i.log 0.0409; then echo "bc3s1i healthy, kept"; else echo "bc3s1i NOT healthy: stopped"; stop_train bc3s1i; end; fi
if [ "$BC0" = no ]; then
  gpus_free 4 7 || { echo "GPUs 4-7 busy: bc3i not started"; end; }
  mem_wait 40
  TR="$TR0"; start_train bc3i 4,5,6,7 --data $(D bc3) --kind bc --geo_aug $SHARE
  fp_bc $A/run_bc3i.log 0.0410 || { echo "bc3i NOT healthy: stopped"; stop_train bc3i; end; }
fi
for k in $BCS; do $PY -c "import json; d=json.load(open('$A/run_$k/args.json')); print('$k args: seed', d.get('seed'), 'share', d.get('share'), 'data', d.get('data'))"; done
wait_train bc3i 220 || { echo "bc3i still running after 220 min: stopping here"; end; }
wait_train bc3s1i 120 || { echo "bc3s1i still running: stopping here"; end; }
for k in $BCS; do grep -E "VAL step|Traceback" $A/run_$k.log | cut -c1-300 | tail -n 13; done
{ train_health bc3i && train_health bc3s1i; } || { echo "a BC-i model does not pass the health rule: not evaluated"; end; }
}

# ---------------------------------------------------------------- 8: BC-i development and test splits
phase8() {
echo "== $(ts) phase 8: BC-i development splits and test split (once each)"
gpus_free 0 3 || { echo "GPUs 0-3 busy: not evaluated"; end; }
lock_eval
mem_wait 26
if ! up_all "$BCS"; then echo "servers not healthy: no rollouts"; down_all "$BCS"; unlock_eval; end; fi
mem_wait 52
for k in $BCS; do ev bc $(P $k) $k "${SRV[$k]}" dev:2 dev2ext:1 devr:3 test:3; done
W=""; for k in $BCS; do W="$W awd$k:100 awp$k:40 awe$k:40 awr$k:240 awt$k:280"; done
finish "$BCS" "dev:1 dev2ext:1 devr:1 test:2" "$W" "BC-i dev + test" 120 || echo "BC-i splits incomplete after one re-run"
down_all "$BCS"
unlock_eval
}

# ---------------------------------------------------------------- 9: tables and packs
phase9() {
echo "== $(ts) phase 9: tables"
cd $A
$PY aw_dev3.py awx3 awx3s1 bc3 bc3s1 awx3i awx3s1i bc3i bc3s1i --pairs awx3i:bc3i,awx3s1i:bc3s1i,bc3i:bc3,bc3s1i:bc3s1 \
  --pool "awx3i+awx3s1i:bc3i+bc3s1i,bc3i+bc3s1i:bc3+bc3s1" | tee $CH/i_dev_all.txt
$PY aw_pool.py "awx3i+awx3s1i:bc3i+bc3s1i" "awx3i+awx3s1i:awx3+awx3s1" "bc3i+bc3s1i:bc3+bc3s1" | tee $CH/i_test.txt
$PY aw_summary.py awtawx3i awtawx3s1i awtbc3i awtbc3s1i awpbc3i awpbc3s1i 2>/dev/null
for pr in "awtawx3i awtbc3i" "awtawx3s1i awtbc3s1i" "awtawx3i awtawx3" "awtawx3s1i awtawx3s1" "awtbc3i awtbc3" "awtbc3s1i awtbc3s1"; do
  set -- $pr; $PY aw_paired.py $1 $2
done
taskset -c 96-127 timeout 900 $PY si_diag.py $BH/runs/lens --sets awrbc3 awrbc3s1 awrbc3i awrbc3s1i --teach $TEACH 2>&1 \
  | grep -E "^sets|B->A .*NN " | cut -c1-260 | head -n 30
pack i_bci "$(nm bc3i dev dev2ext devr test) $(nm bc3s1i dev dev2ext devr test)"
(cd $A && tar czf pack/i_logs.tgz run_awx3i.log run_awx3s1i.log run_bc3i.log run_bc3s1i.log chain/i_master.log chain/i_*.txt \
   chain/i_build*.log 2>&1 | tail -n 2; md5sum pack/i_logs.tgz)
}

for ph in 0 1 2 3 4 5 7 8 9; do [ $ph -ge $FROM ] && phase$ph; done
end
