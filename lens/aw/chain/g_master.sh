#!/bin/bash
# g_master.sh c1 -- DETACHED on the server (log $A/chain/g_master.log).  Round g of the handover 7.13 (plan and every
# rule written 11-01 16:20 before any round-g result and before the development results of round f): counterfactual
# relabeling at the students' own attractor states, for AW (kind awx) and its matched BC, two seeds each.
#   0  wait for round f (f_master c2) to end and for GPUs 0-7 to be free (jobs of other users: wait, never touched)
#   1  smoke: freeze probe of awx3 / bc3 (diagnostic only); 12 conditional take-over episodes of awx3 on layout r130
#      (outside every split: k = 1 forces a take-over, k = 100 mostly skips) and their counterfactual branches; checks of
#      the take-overs, skips, step-record sources and restores (g_check smoke); data builder c6 (categories 6 / 7);
#      CPU training smokes with --share 6:0.15,7:0.15 (awx, bc; validation batch unchanged); spec generator r23 against
#      r22.  Nothing else starts unless every gate passes.
#   2  conditional take-overs of the 4 students (1512 B->A episodes each, training layouts 20-37 + r101-124, t_s 12-62,
#      k 40 / 70 / 100, 2 noises), 4 groups of student + teacher server; then the counterfactual branch of every take-over
#   3  datasets data_g6_<student> (new-instruction branch, category 6) and data_g7_<student> (counterfactual, category 7);
#      the data-volume rule (>= 60 successful take-overs per student)
#   4  AW-CF: awx3cf (seed 0, GPUs 0-3) and awx3s1cf (seed 1, GPUs 4-7), fingerprint-checked, health rule
#   5  AW-CF development evaluation (dev + no-switch, dev2 extension, devR) and the rule of 7.13 (aw_dev3_c2 --select:
#      strictly more pooled dev successes than awx3 + awx3s1) -> $CH/g_decision.txt.  NOGO: stop here
#   7  GO: AW-CF test split once each on the same servers while bc3cf trains (GPUs 4-7); then bc3s1cf (GPUs 0-3)
#   8  BC-CF: health rule; dev (+ no-switch), dev2 extension, devR and test split once each, whatever the result
#   9  tables and result packs in $A/pack
source ${GLIB:-${BH:-/data/blackhole}/lens/aw/chain/g_lib.sh}
end() { echo "== $(ts) g_master END"; exit 0; }
TR0="$TR"; TR1="${TR/--seed 0/--seed 1}"
STU="awx3 awx3s1 bc3 bc3s1"; AWS="awx3cf awx3s1cf"; BCS="bc3cf bc3s1cf"
declare -A KIND=([awx3]=awx [awx3s1]=awx [bc3]=bc [bc3s1]=bc [awx3cf]=awx [awx3s1cf]=awx [bc3cf]=bc [bc3s1cf]=bc)
declare -A GRP=([awx3]="8751:8752:0:1" [awx3s1]="8753:8754:2:3" [bc3]="8755:8756:4:5" [bc3s1]="8757:8758:6:7")
declare -A SRV=([awx3cf]="8761:0 8762:1" [awx3s1cf]="8763:2 8764:3" [bc3cf]="8765:0 8766:1" [bc3s1cf]="8767:2 8768:3")
SHARE="--share 6:0.15,7:0.15"
P() { echo $A/run_$1/params_6000.npz; }
D() { echo $A/data_pi05,$A/data_g6_$1,$A/data_g7_$1; }      # training roots of the round-g model of student $1
alive() { pgrep -fc 'aw_server(_c9)?[.]py|lens_server[.]py|bh_client_lens[.]py|aw_train(_c11)?[.]py|f_master(_c2)?[.]sh|e_master[.]sh'; }
up_all() { local k ok=0; for k in $1; do servers_up ${KIND[$k]} $(P $k) $k "${SRV[$k]}" || ok=1; done; return $ok; }
down_all() { local k; for k in $1; do servers_down ${KIND[$k]} $(P $k) "${SRV[$k]}"; done; }
healthy_all() { local k sg; for k in $1; do for sg in ${SRV[$k]}; do curl -s -m 3 localhost:${sg%%:*}/healthz > /dev/null || return 1; done; done; }
# finish (as in e_master) for the evaluation servers of the round-g models
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
echo "== $(ts) g_master c1 start (pid $$), from phase $FROM; training flags seed 0: $TR0"
(cd $A && md5sum aw_build_data.py g_check.py freeze_probe.py chain/g_lib.sh chain/g_master.sh ../bh_client_lens.py \
   ../make_specs_lens.py aw_train.py aw_dev3.py aw_pool.py e_check.py chain/e_lib.sh aw_eval.sh aw_server.py \
   aw_server.py ../bh_client_lens.py ../bh_client_lens.py ../make_specs_lens.py)

# ---------------------------------------------------------------- 0: wait for round f and free cards
phase0() {
for i in $(seq 1 360); do
  [ "$(alive)" -eq 0 ] && gpus_free 0 7 && break
  [ $((i % 20)) -eq 1 ] && echo "$(ts) waiting: processes of this project $(alive) (f_master $(pgrep -fc 'f_master(_c2)?[.]sh')); $(grep -E '^== ' $CH/f_master2.log 2>/dev/null | tail -n 1 | cut -c1-120)"
  sleep 30
done
{ [ "$(alive)" -eq 0 ] && gpus_free 0 7; } || { echo "GPUs 0-7 not free or processes of this project alive after 3 h: nothing started"; end; }
AV=$(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9)
echo "$(ts) /data: ${AV} GiB available; memory anon $(anon) GiB; $(free -g | sed -n 2p)"
[ "${AV:-0}" -ge 300 ] || { echo "less than 300 GiB free on /data: nothing started"; end; }
}

# ---------------------------------------------------------------- 1: smoke
phase1() {
echo "== $(ts) phase 1: smoke"
lock_eval
cd $OP
echo "-- freeze probe (diagnostic, 7.13; no gate)"
for kg in "awx:awx3:0:spawx3" "bc:bc3:1:spbc3"; do
  IFS=: read k m g sp <<< "$kg"
  CUDA_VISIBLE_DEVICES=$g OPENPI_DATA_HOME=$BH/openpi_data XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 PYTHONPATH=$A timeout 1200 \
    $PY $A/freeze_probe.py $k $(P $m) "$BH/runs/lens/pi05_${sp}_p*" 24 > $CH/g_probe_$k.log 2>&1 < /dev/null &
done
wait
for k in awx bc; do echo "-- probe $k"; grep -vE "^INFO|^WARNING|^I[0-9]{4}|^W[0-9]{4}|^\s*$" $CH/g_probe_$k.log | cut -c1-300 | tail -n 14; done
cd $A
SOK=1
G0="8790:8791:0:1"
for d in gsmk gsmkcf; do rm -rf $BH/runs/lens/pi05_${d}_p*; done
rm -rf $A/data_g6_smk $A/data_g7_smk $A/smoke_g0 $A/smoke_g1 $A/smoke_g2
mem_wait 20
if ! dg_up awx $(P awx3) "$G0"; then echo "smoke servers not healthy: nothing else started"; dg_down awx $(P awx3) "$G0"; unlock_eval; end; fi
$PY $MS23 $BH/specs/lens --cfdag gsmk --cf_layouts r130 --cf_ts 12,22,32,42,52,62 --cf_ks 1,100 --cf_noise 1 --bbs pi05 --vlm_offset "$OFF"
dg_run gsmk "$G0" 3
dg_wait "smoke take-overs" gsmk 12 30
[ "$(dgwk gsmk)" -eq 0 ] || { echo "smoke workers still running: stopped"; dg_kill gsmk; }
$PY $MS23 $BH/specs/lens --cfbranch gsmkcf --cf_from gsmk --cf_runs $BH/runs/lens --bbs pi05 --vlm_offset "$OFF"
NB=$(wc -l < $BH/specs/lens/pi05_gsmkcf.jsonl)
dg_run gsmkcf "$G0" 2
dg_wait "smoke branches" gsmkcf $NB 30
[ "$(dgwk gsmkcf)" -eq 0 ] || { echo "smoke branch workers still running: stopped"; dg_kill gsmkcf; }
dg_down awx $(P awx3) "$G0"
unlock_eval
grep -hE "Traceback|Error" $BH/runs/lens/pi05_gsmk*_p0/worker_*.log | sort | uniq -c | head -n 5
$PY g_check.py smoke $BH/runs/lens gsmk gsmkcf || SOK=0
$PY g_check.py dsum $BH/runs/lens gsmk
$PY g_check.py csum $BH/runs/lens gsmkcf
taskset -c 64-95 timeout 900 $PY $TRAIN6 $A/data_g6_smk $BH/runs/lens/pi05_gsmk_p0 --cat_as 6 --need_takeover --procs 4 2>&1 | tail -n 7 | cut -c1-300
taskset -c 64-95 timeout 900 $PY $TRAIN6 $A/data_g7_smk $BH/runs/lens/pi05_gsmkcf_p0 --cat_as 7 --procs 4 2>&1 | tail -n 7 | cut -c1-300
$PY g_check.py data 6 $A/data_g6_smk || SOK=0
$PY g_check.py data 7 $A/data_g7_smk || SOK=0
echo "-- spec generator r23 against r22 (every existing call)"
rm -rf $A/spec_chk3; mkdir -p $A/spec_chk3/a $A/spec_chk3/b
for d in a b; do F=$L/make_specs_lens.py; [ $d = b ] && F=$MS23
  $PY $F $A/spec_chk3/$d --dev flush --name awdx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk3/$d --plain awpx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk3/$d --test flush --name awdxx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk3/$d --dev flush --name awex --ts 52,62 --bbs pi05 > /dev/null
  $PY $F $A/spec_chk3/$d --devr awrx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk3/$d --selfplay spx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk3/$d --dev cig05,cig10 --name awdg --ts 12,22,32,42,52,62 --bbs pi05 > /dev/null
  $PY $F $A/spec_chk3/$d --teach --vlm_offset "$OFF" > /dev/null
  $PY $F $A/spec_chk3/$d --dagger dg10,dg30,dgb20 --name dagx --bbs pi05 --vlm_offset "$OFF" > /dev/null
done
if diff -rq $A/spec_chk3/a $A/spec_chk3/b; then echo "specs identical ($(ls $A/spec_chk3/a | wc -l) files)"; else SOK=0; echo "SPECS DIFFER"; fi
rm -rf $A/spec_chk3
echo "-- CPU training smokes"
SMK="--smoke --bs 32 --lam_loc 0.1 --lam_coord 0.2 --skip_gn 1000 --seed 0"
mem_wait 24
(CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$A taskset -c 80-95 timeout 1500 $PY $A/aw_train.py $SMK --data $A/data_pi05 \
   --kind awx --geo_aug --out $A/smoke_g0 > $A/smoke_g0.log 2>&1 < /dev/null) &
PS0=$!
(CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$A taskset -c 96-111 timeout 1500 $PY $A/aw_train.py $SMK \
   --data $A/data_pi05,$A/data_g6_smk,$A/data_g7_smk --kind awx --geo_aug $SHARE --out $A/smoke_g1 > $A/smoke_g1.log 2>&1 < /dev/null) &
PS1=$!
(CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$A taskset -c 112-127 timeout 1500 $PY $A/aw_train.py $SMK \
   --data $A/data_pi05,$A/data_g6_smk,$A/data_g7_smk --kind bc --geo_aug $SHARE --out $A/smoke_g2 > $A/smoke_g2.log 2>&1 < /dev/null) &
PS2=$!
wait $PS0 $PS1 $PS2
for k in g0 g1 g2; do
  echo "-- smoke_$k"; grep -E "data roots|frames [0-9]+: train|category shares|step 0 |VAL step 0|smoke done|Error|Traceback" $A/smoke_$k.log | cut -c25-330
  grep -q "smoke done" $A/smoke_$k.log || { SOK=0; tail -n 8 $A/smoke_$k.log | cut -c1-300; }
done
$PY e_check.py val0 $A/smoke_g0.log $A/smoke_g1.log || SOK=0
for k in g1 g2; do grep -E "category shares" $A/smoke_$k.log | grep -qE "6: 0[.]15[},]" && grep -E "category shares" $A/smoke_$k.log | grep -qE "7: 0[.]15[},]" \
  || { SOK=0; echo "categories 6 / 7 do not have the share 0.15 in smoke $k"; }; done
echo "$(ts) smoke gates passed: $SOK"
[ "$SOK" -eq 1 ] || { echo "nothing else started"; end; }
rm -rf $A/smoke_g0 $A/smoke_g1 $A/smoke_g2
}

# ---------------------------------------------------------------- 2: conditional take-overs and counterfactual branches
phase2() {
echo "== $(ts) phase 2: conditional take-overs of $STU (1512 B->A episodes each), then the counterfactual branches"
for s in $STU; do for n in g$s g${s}cf; do [ -d $BH/runs/lens/pi05_${n}_p0 ] && { echo "results of $n exist already: nothing started"; end; }; done; done
for s in $STU; do [ -f $(P $s) ] || { echo "no $(P $s): nothing started"; end; }; done
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing started"; end; }
lock_eval
mem_wait 52
OKS=1; for s in $STU; do dg_up ${KIND[$s]} $(P $s) "${GRP[$s]}" || OKS=0; done
[ $OKS -eq 1 ] || { echo "servers not healthy: no rollouts"; for s in $STU; do dg_down ${KIND[$s]} $(P $s) "${GRP[$s]}"; done; unlock_eval; end; }
echo "$(ts) servers up; anon $(anon) GiB"
for s in $STU; do $PY $MS23 $BH/specs/lens --cfdag g$s --cf_layouts 20-37,r101-124 --cf_ts 12,22,32,42,52,62 --cf_ks 40,70,100 \
  --cf_noise 2 --bbs pi05 --vlm_offset "$OFF"; done
mem_wait 40
for s in $STU; do dg_run g$s "${GRP[$s]}" 6; done
W=""; SP=""; for s in $STU; do W="$W g$s:1512"; SP="$SP;g$s=${GRP[$s]}"; done
dg_finish "take-overs" "$W" "${SP#;}" 6 180 || { echo "take-over rollouts incomplete after one re-run: nothing trained"; }
cd $A
$PY g_check.py dsum $BH/runs/lens $(for s in $STU; do printf 'g%s ' $s; done) | tee $CH/g_takeovers.txt
for s in $STU; do $PY $MS23 $BH/specs/lens --cfbranch g${s}cf --cf_from g$s --cf_runs $BH/runs/lens --bbs pi05 --vlm_offset "$OFF"; done
W=""; SP=""; for s in $STU; do W="$W g${s}cf:$(wc -l < $BH/specs/lens/pi05_g${s}cf.jsonl)"; SP="$SP;g${s}cf=${GRP[$s]}"; done
echo "$(ts) counterfactual branches:$W"
for s in $STU; do dg_run g${s}cf "${GRP[$s]}" 4; done
dg_finish "counterfactual branches" "$W" "${SP#;}" 4 120 || echo "counterfactual branches incomplete after one re-run (the datasets use what is there)"
for s in $STU; do dg_down ${KIND[$s]} $(P $s) "${GRP[$s]}"; done
unlock_eval
cd $A
$PY g_check.py csum $BH/runs/lens $(for s in $STU; do printf 'g%scf ' $s; done) | tee -a $CH/g_takeovers.txt
pack g_dag "$(for s in $STU; do printf 'g%s g%scf ' $s $s; done)"
du -sh $(for s in $STU; do echo $BH/runs/lens/pi05_g${s}_p* $BH/runs/lens/pi05_g${s}cf_p*; done) 2>/dev/null | awk '{s=s" "$1} END{print "run directories:" s}'
}

# ---------------------------------------------------------------- 3: datasets and the data-volume rule
phase3() {
echo "== $(ts) phase 3: round-g datasets"
mem_wait 16
BP=""
for s in $STU; do
  rm -rf $A/data_g6_$s $A/data_g7_$s
  (taskset -c 64-127 timeout 3600 $PY $TRAIN6 $A/data_g6_$s $(ls -d $BH/runs/lens/pi05_g${s}_p*) --cat_as 6 --need_takeover --procs 6 \
     > $CH/g_build6_$s.log 2>&1 < /dev/null; \
   taskset -c 64-127 timeout 3600 $PY $TRAIN6 $A/data_g7_$s $(ls -d $BH/runs/lens/pi05_g${s}cf_p*) --cat_as 7 --procs 6 \
     > $CH/g_build7_$s.log 2>&1 < /dev/null) &
  BP="$BP $!"
done
wait $BP
for s in $STU; do for c in 6 7; do echo "-- data_g${c}_$s"; grep -vE "^\s*$" $CH/g_build${c}_$s.log | tail -n 6 | cut -c1-300; done; done
OKD=1
$PY g_check.py data 6 $(for s in $STU; do printf "$A/data_g6_%s " $s; done) || OKD=0
$PY g_check.py data 7 $(for s in $STU; do printf "$A/data_g7_%s " $s; done) || OKD=0
for s in $STU; do $PY g_check.py enough $BH/runs/lens g$s 60 || OKD=0; done
du -sh $(for s in $STU; do echo $A/data_g6_$s $A/data_g7_$s; done) | awk '{s=s" "$1} END{print "datasets:" s}'; echo "/data $(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9) GiB free"
[ $OKD -eq 1 ] || { echo "a round-g dataset is not consistent or too small (7.13): nothing trained"; end; }
}

# ---------------------------------------------------------------- 4: AW-CF training
phase4() {
echo "== $(ts) phase 4: AW-CF training (awx3cf seed 0 GPUs 0-3, awx3s1cf seed 1 GPUs 4-7)"
for k in $AWS $BCS; do for n in $(nm $k dev dev2ext devr test); do [ -d $BH/runs/lens/pi05_${n}_p0 ] && { echo "results of $n exist already: nothing trained"; end; }; done; done
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing trained"; end; }
mem_wait 40
TR="$TR0"; start_train awx3cf 0,1,2,3 --data $(D awx3) --kind awx --geo_aug $SHARE
grep -E "data roots|category shares|trainable|Traceback" $A/run_awx3cf.log | cut -c25-330
if fp_ok $A/run_awx3cf.log 0.0541 1.1345; then echo "awx3cf healthy, kept"; else echo "awx3cf NOT healthy: stopped"; tail -n 5 $A/run_awx3cf.log | cut -c1-300; stop_train awx3cf; end; fi
mem_wait 40
TR="$TR1"; start_train awx3s1cf 4,5,6,7 --data $(D awx3s1) --kind awx --geo_aug $SHARE; TR="$TR0"
grep -E "data roots|category shares|trainable|Traceback" $A/run_awx3s1cf.log | cut -c25-330
if fp_ok $A/run_awx3s1cf.log 0.0540 0.8999; then echo "awx3s1cf healthy, kept"; else echo "awx3s1cf NOT healthy: stopped (awx3cf continues; this chain stops)"; tail -n 5 $A/run_awx3s1cf.log | cut -c1-300; stop_train awx3s1cf; end; fi
for k in $AWS; do $PY -c "import json; d=json.load(open('$A/run_$k/args.json')); print('$k args: seed', d.get('seed'), 'share', d.get('share'), 'data', d.get('data'))"; done
wait_train awx3cf 220 || { echo "awx3cf still running after 220 min: stopping here"; end; }
wait_train awx3s1cf 60 || { echo "awx3s1cf still running: stopping here"; end; }
for k in $AWS; do grep -E "VAL step|Traceback" $A/run_$k.log | cut -c1-300 | tail -n 13; done
{ train_health awx3cf && train_health awx3s1cf; } || { echo "an AW-CF model does not pass the health rule: by 7.7 it is retrained once (not automatic)"; end; }
}

# ---------------------------------------------------------------- 5: AW-CF development evaluation and the rule of 7.13
phase5() {
echo "== $(ts) phase 5: AW-CF development evaluation"
gpus_free 0 3 || { echo "GPUs 0-3 busy: not evaluated"; end; }
lock_eval
mem_wait 26
if ! up_all "$AWS"; then echo "servers not healthy: no rollouts"; down_all "$AWS"; unlock_eval; end; fi
mem_wait 45
for k in $AWS; do ev awx $(P $k) $k "${SRV[$k]}" dev:2 dev2ext:1 devr:4; done
W=""; for k in $AWS; do W="$W awd$k:100 awp$k:40 awe$k:40 awr$k:240"; done
finish "$AWS" "dev:1 dev2ext:1 devr:1" "$W" "AW-CF dev" 90 || echo "AW-CF development split incomplete after one re-run (the rule then gives NOGO)"
cd $A
$PY aw_summary.py awdawx3cf awpawx3cf awdawx3s1cf awpawx3s1cf 2>/dev/null
$PY aw_dev3.py awx3 awx3s1 bc3 bc3s1 awx3cf awx3s1cf --pairs awx3cf:awx3,awx3s1cf:awx3s1,awx3cf:bc3,awx3s1cf:bc3s1 \
  --pool "awx3cf+awx3s1cf:awx3+awx3s1,awx3cf+awx3s1cf:bc3+bc3s1" --select "awx3cf+awx3s1cf:awx3+awx3s1" --out $CH/g_decision.txt | tee $CH/g_dev_aw.txt
$PY freeze_anatomy.py $BH/runs/lens awrawx3cf awrawx3s1cf 2>&1 | grep -vE "path every" | cut -c1-330
pack g_awcf_dev "$(nm awx3cf dev dev2ext devr) $(nm awx3s1cf dev dev2ext devr)"
DC=$(head -n 1 $CH/g_decision.txt 2>/dev/null)
echo "$(ts) decision (7.13): ${DC:-none}"
case "$DC" in
  GO*) ;;
  *) echo "not GO: by 7.13 no test split and no BC-CF; this chain stops here"; down_all "$AWS"; unlock_eval; end ;;
esac
}

# ---------------------------------------------------------------- 7: AW-CF test split; BC-CF training
phase7() {
echo "== $(ts) phase 7: AW-CF test split (once each, same servers); bc3cf training on GPUs 4-7"
mem_wait 20
for k in $AWS; do ev awx $(P $k) $k "${SRV[$k]}" test:4; done
BC0=no
if gpus_free 4 7; then
  mem_wait 40
  TR="$TR0"; start_train bc3cf 4,5,6,7 --data $(D bc3) --kind bc --geo_aug $SHARE
  grep -E "data roots|category shares|Traceback" $A/run_bc3cf.log | cut -c25-330
  if fp_bc $A/run_bc3cf.log 0.0410; then BC0=yes; echo "bc3cf healthy, kept"; else echo "bc3cf NOT healthy: stopped"; stop_train bc3cf; BC0=bad; fi
else echo "GPUs 4-7 busy: bc3cf starts after the test split"; fi
finish "$AWS" "test:2" "awtawx3cf:280 awtawx3s1cf:280" "AW-CF test" 60 || echo "AW-CF test split incomplete after one re-run"
down_all "$AWS"
unlock_eval
cd $A
pack g_awcf_test "awtawx3cf awtawx3s1cf"
$PY aw_pool.py "awx3cf+awx3s1cf:awx3+awx3s1" "awx3cf+awx3s1cf:bc3+bc3s1" | tee $CH/g_test_aw.txt
[ "$BC0" = bad ] && { echo "bc3cf fingerprint failed: BC-CF not trained (decided by hand)"; end; }
gpus_free 0 3 || { echo "GPUs 0-3 busy: bc3s1cf not started"; end; }
mem_wait 40
TR="$TR1"; start_train bc3s1cf 0,1,2,3 --data $(D bc3s1) --kind bc --geo_aug $SHARE; TR="$TR0"
grep -E "data roots|category shares|Traceback" $A/run_bc3s1cf.log | cut -c25-330
if fp_bc $A/run_bc3s1cf.log 0.0409; then echo "bc3s1cf healthy, kept"; else echo "bc3s1cf NOT healthy: stopped"; stop_train bc3s1cf; end; fi
if [ "$BC0" = no ]; then
  gpus_free 4 7 || { echo "GPUs 4-7 busy: bc3cf not started"; end; }
  mem_wait 40
  TR="$TR0"; start_train bc3cf 4,5,6,7 --data $(D bc3) --kind bc --geo_aug $SHARE
  fp_bc $A/run_bc3cf.log 0.0410 || { echo "bc3cf NOT healthy: stopped"; stop_train bc3cf; end; }
fi
for k in $BCS; do $PY -c "import json; d=json.load(open('$A/run_$k/args.json')); print('$k args: seed', d.get('seed'), 'share', d.get('share'), 'data', d.get('data'))"; done
wait_train bc3cf 220 || { echo "bc3cf still running after 220 min: stopping here"; end; }
wait_train bc3s1cf 120 || { echo "bc3s1cf still running: stopping here"; end; }
for k in $BCS; do grep -E "VAL step|Traceback" $A/run_$k.log | cut -c1-300 | tail -n 13; done
{ train_health bc3cf && train_health bc3s1cf; } || { echo "a BC-CF model does not pass the health rule: not evaluated"; end; }
}

# ---------------------------------------------------------------- 8: BC-CF development and test splits
phase8() {
echo "== $(ts) phase 8: BC-CF development splits and test split (once each)"
gpus_free 0 3 || { echo "GPUs 0-3 busy: not evaluated"; end; }
lock_eval
mem_wait 26
if ! up_all "$BCS"; then echo "servers not healthy: no rollouts"; down_all "$BCS"; unlock_eval; end; fi
mem_wait 52
for k in $BCS; do ev bc $(P $k) $k "${SRV[$k]}" dev:2 dev2ext:1 devr:3 test:3; done
W=""; for k in $BCS; do W="$W awd$k:100 awp$k:40 awe$k:40 awr$k:240 awt$k:280"; done
finish "$BCS" "dev:1 dev2ext:1 devr:1 test:2" "$W" "BC-CF dev + test" 120 || echo "BC-CF splits incomplete after one re-run"
down_all "$BCS"
unlock_eval
}

# ---------------------------------------------------------------- 9: tables and packs
phase9() {
echo "== $(ts) phase 9: tables"
cd $A
$PY aw_dev3.py awx3 awx3s1 bc3 bc3s1 awx3cf awx3s1cf bc3cf bc3s1cf --pairs awx3cf:bc3cf,awx3s1cf:bc3s1cf,bc3cf:bc3,bc3s1cf:bc3s1 \
  --pool "awx3cf+awx3s1cf:bc3cf+bc3s1cf,bc3cf+bc3s1cf:bc3+bc3s1" | tee $CH/g_dev_all.txt
$PY aw_pool.py "awx3cf+awx3s1cf:bc3cf+bc3s1cf" "awx3cf+awx3s1cf:awx3+awx3s1" "bc3cf+bc3s1cf:bc3+bc3s1" | tee $CH/g_test.txt
$PY aw_summary.py awtawx3cf awtawx3s1cf awtbc3cf awtbc3s1cf awpbc3cf awpbc3s1cf 2>/dev/null
for pr in "awtawx3cf awtbc3cf" "awtawx3s1cf awtbc3s1cf" "awtawx3cf awtawx3" "awtawx3s1cf awtawx3s1" "awtbc3cf awtbc3" "awtbc3s1cf awtbc3s1"; do
  set -- $pr; $PY aw_paired.py $1 $2
done
$PY freeze_anatomy.py $BH/runs/lens awrbc3cf awrbc3s1cf 2>&1 | grep -vE "path every" | cut -c1-330
pack g_bccf "$(nm bc3cf dev dev2ext devr test) $(nm bc3s1cf dev dev2ext devr test)"
(cd $A && tar czf pack/g_logs.tgz run_awx3cf.log run_awx3s1cf.log run_bc3cf.log run_bc3s1cf.log chain/g_master.log chain/g_*.txt \
   chain/g_build*.log chain/g_probe_*.log 2>&1 | tail -n 2; md5sum pack/g_logs.tgz)
}

for ph in 0 1 2 3 4 5 7 8 9; do [ $ph -ge $FROM ] && phase$ph; done
end
