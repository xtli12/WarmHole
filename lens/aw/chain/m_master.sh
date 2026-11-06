#!/bin/bash
# m_master.sh c1 -- DETACHED on the server (log $A/chain/m_master.log).  Round m of the handover 7.19 (11-02, rules written
# before any result): round j (kinematically safe wormhole exit of the teacher data) on the SECOND BACKBONE pi0, decided
# on 11-02.  Confirmatory design as in round d (7.10): no selection on pi0; every model is
# trained once per seed and evaluated once on dev (+ no-switch), dev2 extension and the test split whatever its
# development result.  Everything with the backbone prefix pi0 (BBP=pi0; teacher = the released pi0, lens_server
# --config pi0_libero; training/serving with --bb pi0 as in round d; aw_train.py c10 = the trainer validated for pi0).
#   0  wait until no process of this project runs and GPUs 0-7 are free
#   1  smoke: 2 pi0 teacher servers; 12 safe-exit episodes (M = 1.2) on layouts r130-131; m_check smoke; the original pi0
#      teacher directories without their A->B switch episodes (pi0_teachjBA_p0/p1) and their counts
#   2  pilot (selects M only): M in {1.0, 1.2}, A->B, inits 20-39, t_s 22 / 42 (2 x 40 episodes), 8 teacher servers;
#      the rule of 7.16 (more successes, tie -> larger M; NOGO if more than 1 success below the original pi0 teacher on
#      the same conditions) -> m_pilot.txt; NOGO stops the chain (reported as such)
#   3  the new A->B pi0 teacher data (260 episodes: inits 20-39, t_s 2-62 every 5) with the selected M -> pi0_jab_p*;
#      dataset data_pi0j (the original pi0 B->A switch and no-switch episodes + the new A->B switch episodes)
#   4  single-card fingerprints of data_pi0j (awx / bc, seeds 0 and 1, --bb pi0); training awx3jp0 (GPUs 0-3) and
#      bc3jp0 (GPUs 4-7), seed 0, step 0 checked, health rule of 7.10 (health_rel)
#   5  evaluation of the seed-0 pair: dev (+ no-switch), dev2 extension, test split, once each (servers 8761-8764)
#   6  training awx3s1jp0 / bc3s1jp0 (seed 1); 7  their evaluation
#   8  tables: per seed (aw_summary, aw_dev2 pi0/..., aw_paired pi0:...), pooled over two seeds (aw_pool_c3, BBP=pi0),
#      against the single-seed round-d models awx3p0 / bc3p0; packs in $A/pack
source ${JLIB:-${BH:-/data/blackhole}/lens/aw/chain/j_lib.sh}
export BBP=pi0
export BB=pi0
end() { echo "== $(ts) m_master END"; exit 0; }
TR0="$TR"; TR1="${TR/--seed 0/--seed 1}"
MCHK=$A/m_check.py
POOL3=$A/aw_pool.py
AW0=awx3jp0; BC0=bc3jp0; AW1=awx3s1jp0; BC1=bc3s1jp0
declare -A KIND=([awx3jp0]=awx [bc3jp0]=bc [awx3s1jp0]=awx [bc3s1jp0]=bc)
declare -A SRV=([awx3jp0]="8761:0 8762:1" [bc3jp0]="8763:2 8764:3" [awx3s1jp0]="8761:0 8762:1" [bc3s1jp0]="8763:2 8764:3")
TS1="8781:0 8782:1"
TS8="8781:0 8782:1 8783:2 8784:3 8785:4 8786:5 8787:6 8788:7"
TSA="8781:0 8782:1 8783:2 8784:3"; TSB="8785:4 8786:5 8787:6 8788:7"
TEACH="$BH/runs/lens/pi0_teach_p0 $BH/runs/lens/pi0_teach_p1"
DJ=$A/data_pi0j
JP_INITS=${JP_INITS:-20-39}; JP_TS=${JP_TS:-22,42}; JD_INITS=${JD_INITS:-20-39}; JDMIN=${JDMIN:-40000}
P() { echo $A/run_$1/params_6000.npz; }
alive() { pgrep -fc 'aw_server(_c9)?[.]py|lens_server[.]py|bh_client_lens[.]py|aw_train(_c11)?[.]py|[e-l]_master(_c[0-9])?[.]sh|pi_heads[.]py|posture_c2[.]py'; }
# pi0 teacher servers (the released pi0; as t_up of j_lib with the pi0 config)
t_up_p0() {
  local x p g i ok n
  for x in $1; do p=${x%%:*}; curl -s -m 3 localhost:$p/healthz > /dev/null && { echo "port $p already serving: refusing"; return 1; }; done
  for x in $1; do
    p=${x%%:*}; g=${x##*:}
    (cd $OP && CUDA_VISIBLE_DEVICES=$g OPENPI_DATA_HOME=$BH/openpi_data XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      PYTHONPATH=$BH/code/overlay:$BH/code:$L setsid nohup $PY $L/lens_server.py --config pi0_libero --ckpt $CK/pi0_libero \
      --port $p > $BH/logs/lens_server_$p.log 2>&1 < /dev/null &)
    sleep 10
  done
  for i in $(seq 1 60); do
    ok=0; n=0
    for x in $1; do n=$((n+1)); curl -s -m 3 localhost:${x%%:*}/healthz > /dev/null && ok=$((ok+1)); done
    [ $ok -eq $n ] && break
    sleep 10
  done
  for x in $1; do curl -s -m 3 localhost:${x%%:*}/healthz > /dev/null || { echo "teacher ${x%%:*} DOWN"; tail -n 5 $BH/logs/lens_server_${x%%:*}.log | cut -c1-200; }; done
  echo "$(ts) pi0 teacher servers ($1): $ok/$n healthy; anon $(anon) GiB"
  [ $ok -eq $n ]
}
t_down_p0() {
  local x
  for x in $1; do pkill -f "lens_server[.]py --config pi0_libero --ckpt $CK/pi0_libero --port ${x%%:*}( |$)"; done
  sleep 20
  for x in $1; do pkill -9 -f "lens_server[.]py --config pi0_libero --ckpt $CK/pi0_libero --port ${x%%:*}( |$)"; done
  sleep 5
}
t_healthy_p0() { local x; for x in $1; do curl -s -m 3 localhost:${x%%:*}/healthz > /dev/null || return 1; done; }
# single-card fingerprint on pi0: NAME GPU DATA KIND SEED (aw_train.py c10, --bb pi0, as round d)
val0p() {
  CUDA_VISIBLE_DEVICES=$2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$A timeout 1200 $PY $A/aw_train.py --bb pi0 --data $3 --kind $4 \
    --bs 32 --lam_loc 0.1 --lam_coord 0.2 --seed $5 --val_only --out $A/val0_$1 > $A/val0_$1.log 2>&1 < /dev/null
  grep -E "VAL0ONLY|action convention|Traceback" $A/val0_$1.log | cut -c25-300
}
# training on pi0 with aw_train.py c10 (lib2's start_train), health rule of 7.10
start_train_p0() {  # run name, gpus, args...
  local k=$1 g=$2 i; shift 2
  rm -rf $A/run_$k
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$A setsid nohup $PY $A/aw_train.py "$@" $TR --out $A/run_$k > $A/run_$k.log 2>&1 < /dev/null &
  for i in $(seq 1 60); do grep -qE " step 50 |Traceback" $A/run_$k.log && break; sleep 10; done
}
wait_train_p0() { local i; for i in $(seq 1 $(($2 * 2))); do [ "$(pgrep -fc "aw_train[.]py .*--out $A/run_$1( |$)")" -eq 0 ] && return 0; [ $((i % 20)) -eq 1 ] && echo "$(ts) $1 $(grep -E ' step [0-9]+ ' $A/run_$1.log | tail -n 1 | cut -c25-110)"; sleep 30; done; return 1; }
stop_train_p0() { pkill -f "aw_train[.]py .*--out $A/run_$1( |$)"; }
health_rel() {  # run name: finished, last validation finite, < 100 skipped, last <= 50% of step 0 or < 0.03 (7.10)
  local v0 vf sk
  v0=$(grep -E "VAL step 0 " $A/run_$1.log | sed -E 's/.*flow=([0-9.eE+-]+|nan|inf).*/\1/')
  vf=$(grep -E "VAL step " $A/run_$1.log | tail -n 1 | sed -E 's/.*flow=([0-9.eE+-]+|nan|inf).*/\1/')
  sk=$(grep -E " step [0-9]+ .*skipped=" $A/run_$1.log | tail -n 1 | sed -E 's/.*skipped=([0-9.]+).*/\1/')
  echo "$1: params_6000 $([ -f $A/run_$1/params_6000.npz ] && echo present || echo MISSING); step-0 validation $v0, last $vf; skipped $sk; spikes $(cat $A/run_$1/spikes.jsonl 2>/dev/null | wc -l)"
  [ -f $A/run_$1/params_6000.npz ] && awk -v a="$v0" -v v="$vf" -v sk="$sk" 'BEGIN{exit !(v != "" && v+0 == v && a+0 > 0 && (v+0 <= 0.5 * a || v+0 < 0.03) && sk != "" && sk+0 < 100)}'
}
# evaluation servers / workers on pi0 (aw_eval.sh with BB=pi0 as round d; lens12 client)
up_all() { local k ok=0; for k in $1; do BB=pi0 servers_up ${KIND[$k]} $(P $k) $k "${SRV[$k]}" || ok=1; done; return $ok; }
down_all() { local k; for k in $1; do servers_down ${KIND[$k]} $(P $k) "${SRV[$k]}"; done; }
healthy_all() { local k sg; for k in $1; do for sg in ${SRV[$k]}; do curl -s -m 3 localhost:${sg%%:*}/healthz > /dev/null || return 1; done; done; }
G="episodes ->|DOWN|not healthy"
evp() {  # KIND PARAMS TAG "PORT:GPU ..." : dev (+ no-switch) 2 workers/server, dev2ext 1, test 3 -- as round d
  BB=pi0 PHASE=workers $A/aw_eval.sh $1 $2 $3 "$4" 2 | grep -E "$G"
  BB=pi0 PHASE=workers $A/aw_eval.sh $1 $2 $3 "$4" 1 dev2ext | grep -E "$G"
  BB=pi0 PHASE=workers $A/aw_eval.sh $1 $2 $3 "$4" 3 test | grep -E "$G"
}
names_of() { echo "awd$1 awp$1 awe$1 awt$1"; }
finish_p0() {  # "models" "names" expected label maxmin: wait; re-run missing episodes once
  local mods=$1 names=$2 n=$3 lab=$4 mx=$5 k
  sleep 90
  wait_workers "$lab" "$names" $n $mx
  [ "$(nwk "$names")" -eq 0 ] || { echo "$(ts) $lab: workers still running after $mx min: stopped"; kill_workers "$names"; }
  (cd $A && BBP=pi0 $PY $MCHK runs $BH/runs/lens $(for k in $mods; do echo "awd$k:100 awp$k:40 awe$k:40 awt$k:280"; done)) && return 0
  healthy_all "$mods" || { echo "$(ts) $lab: incomplete and a server is down: no re-run"; return 1; }
  echo "$(ts) $lab: incomplete -> aborted rows moved to errors_*.jsonl, missing episodes re-run once"
  (cd $A && BBP=pi0 $PY $MCHK clean $BH/runs/lens $names)
  for k in $mods; do evp ${KIND[$k]} $(P $k) $k "${SRV[$k]}"; done
  sleep 60
  wait_workers "$lab (re-run)" "$names" $n $mx
  [ "$(nwk "$names")" -eq 0 ] || { echo "$(ts) $lab: re-run workers still running: stopped"; kill_workers "$names"; }
  (cd $A && BBP=pi0 $PY $MCHK runs $BH/runs/lens $(for k in $mods; do echo "awd$k:100 awp$k:40 awe$k:40 awt$k:280"; done))
}
FROM=${FROM:-0}
echo "== $(ts) m_master c1 start (pid $$), from phase $FROM; BBP=$BBP; training flags seed 0: $TR0"
(cd $A && md5sum m_check.py aw_pool.py chain/m_master.sh chain/j_lib.sh chain/g_lib.sh chain/e_lib.sh chain/lib2.sh aw_train.py aw_server.py aw_eval.sh \
   aw_build_data.py aw_summary.py aw_paired.py aw_dev2.py ../bh_client_lens.py ../bh_client_lens.py ../make_specs_lens.py)
[ -f $CH/p0_equiv.ok ] || { echo "no $CH/p0_equiv.ok (pi0 server path not verified, 2.2 #20): nothing started"; end; }
for d in pi0_teach_p0 pi0_teach_p1; do ls -d $BH/runs/lens/$d/steps > /dev/null || { echo "no $d: nothing started"; end; }; done
for x in jsmk jp10 jp12 jab teachjBA; do [ -d $BH/runs/lens/pi0_${x}_p0 ] && [ $FROM -le 1 ] && { echo "results of pi0_$x exist already: nothing started"; end; }; done
for k in $AW0 $BC0 $AW1 $BC1; do [ -d $A/run_$k ] && [ $FROM -le 4 ] && { echo "run_$k exists already: nothing started"; end; }; done

# ---------------------------------------------------------------- 0: free cards
phase0() {
for i in $(seq 1 360); do
  [ "$(alive)" -eq 0 ] && gpus_free 0 7 && break
  [ $((i % 20)) -eq 1 ] && echo "$(ts) waiting: processes of this project $(alive)"
  sleep 30
done
{ [ "$(alive)" -eq 0 ] && gpus_free 0 7; } || { echo "GPUs 0-7 not free or processes of this project alive after 3 h: nothing started"; end; }
AV=$(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9)
echo "$(ts) /data: ${AV} GiB available; memory anon $(anon) GiB"
[ "${AV:-0}" -ge 200 ] || { echo "less than 200 GiB free on /data: nothing started"; end; }
}

# ---------------------------------------------------------------- 1: smoke
phase1() {
echo "== $(ts) phase 1: smoke (pi0 teacher, M = 1.2, layouts r130-131)"
cd $A
SOK=1
rm -rf $BH/runs/lens/pi0_jsmk_p*
lock_eval
mem_wait 14
if ! t_up_p0 "$TS1"; then echo "smoke servers not healthy: nothing else started"; t_down_p0 "$TS1"; unlock_eval; end; fi
$PY $MS25 $BH/specs/lens --teach_m jsmk --m_margin 1.2 --m_inits r130-131 --m_ts 22,42,62 --m_dirs AB,BA --bbs pi0 --vlm_offset "$OFF"
dg_run jsmk "$(tg "$TS1")" 3
dg_wait "smoke" "jsmk" 12 40
[ "$(dgwk "jsmk")" -eq 0 ] || { echo "smoke workers still running: stopped"; dg_kill "jsmk"; }
t_down_p0 "$TS1"
unlock_eval
grep -hE "Traceback|Error" $BH/runs/lens/pi0_jsmk_p*/worker_*.log | sort | uniq -c | head -n 5
BBP=pi0 $PY $MCHK smoke $BH/runs/lens jsmk || SOK=0
echo "-- the original pi0 teacher data without its A->B switch episodes"
for k in 0 1; do rm -rf $BH/runs/lens/pi0_teachjBA_p$k; BBP=pi0 $PY $MCHK filter $BH/runs/lens/pi0_teach_p$k $BH/runs/lens/pi0_teachjBA_p$k || SOK=0; done
BBP=pi0 $PY $MCHK count $TEACH $BH/runs/lens/pi0_teachjBA_p0 $BH/runs/lens/pi0_teachjBA_p1
echo "$(ts) smoke gates passed: $SOK"
[ "$SOK" -eq 1 ] || { echo "nothing else started"; end; }
}

# ---------------------------------------------------------------- 2: pilot (selects M)
phase2() {
echo "== $(ts) phase 2: pilot on pi0 (selects the elbow margin M only; rule of 7.16)"
for d in jp10 jp12 jab; do [ -d $BH/runs/lens/pi0_${d}_p0 ] && { echo "results of pi0_$d exist already: nothing started"; end; }; done
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing started"; end; }
lock_eval
mem_wait 52
if ! t_up_p0 "$TS8"; then echo "teacher servers not healthy: nothing started"; t_down_p0 "$TS8"; unlock_eval; end; fi
$PY $MS25 $BH/specs/lens --teach_m jp10 --m_margin 1.0 --m_inits $JP_INITS --m_ts $JP_TS --m_dirs AB --bbs pi0 --vlm_offset "$OFF"
$PY $MS25 $BH/specs/lens --teach_m jp12 --m_margin 1.2 --m_inits $JP_INITS --m_ts $JP_TS --m_dirs AB --bbs pi0 --vlm_offset "$OFF"
NP=$(wc -l < $BH/specs/lens/pi0_jp10.jsonl)
mem_wait 30
dg_run jp10 "$(tg "$TSA")" 3
dg_run jp12 "$(tg "$TSB")" 3
sleep 60
dg_wait "pilot" "jp10 jp12" $((2 * NP)) 60
[ "$(dgwk "jp10 jp12")" -eq 0 ] || { echo "pilot workers still running: stopped"; dg_kill "jp10 jp12"; }
(cd $A && BBP=pi0 $PY $MCHK runs $BH/runs/lens jp10:$NP jp12:$NP) || echo "pilot incomplete (the rule uses what is there)"
cd $A
BBP=pi0 $PY $MCHK pilot $BH/runs/lens $CH/m_pilot.txt jp10:1.0 jp12:1.2 -- $TEACH || { echo "pilot rule (7.16 on pi0): NOGO, nothing else started"; t_down_p0 "$TS8"; unlock_eval; end; }
}

# ---------------------------------------------------------------- 3: new A->B teacher data and the dataset
phase3() {
M=$(sed -n 's/^M=//p' $CH/m_pilot.txt 2>/dev/null | head -n 1)
[ -n "$M" ] || { echo "no M selected: stop"; t_down_p0 "$TS8"; unlock_eval; end; }
echo "== $(ts) phase 3: the new A->B pi0 teacher data with M = $M (layouts $JD_INITS)"
[ -d $BH/runs/lens/pi0_jab_p0 ] && { echo "results of pi0_jab exist already: nothing started"; t_down_p0 "$TS8"; unlock_eval; end; }
t_healthy_p0 "$TS8" || { echo "teacher servers down: stop"; t_down_p0 "$TS8"; unlock_eval; end; }
$PY $MS25 $BH/specs/lens --teach_m jab --m_margin $M --m_inits $JD_INITS --m_dirs AB --bbs pi0 --vlm_offset "$OFF"
NJ=$(wc -l < $BH/specs/lens/pi0_jab.jsonl)
mem_wait 30
dg_run jab "$(tg "$TS8")" 3
sleep 60
dg_wait "new A->B teacher data" "jab" $NJ 90
[ "$(dgwk "jab")" -eq 0 ] || { echo "jab workers still running: stopped"; dg_kill "jab"; }
(cd $A && BBP=pi0 $PY $MCHK runs $BH/runs/lens jab:$NJ) || echo "new A->B teacher data incomplete (the dataset uses what is there)"
t_down_p0 "$TS8"
unlock_eval
cd $A
BBP=pi0 $PY $MCHK count $BH/runs/lens/pi0_teachjBA_p0 $BH/runs/lens/pi0_teachjBA_p1 $(ls -d $BH/runs/lens/pi0_jab_p*) | tee $CH/m_data.txt
BBP=pi0 $PY $MCHK pilot $BH/runs/lens $CH/m_data_vs_orig.txt jab:$M -- $TEACH | tee -a $CH/m_data.txt
pack m_teach "jab jp10 jp12 jsmk"
echo "-- dataset data_pi0j"
mem_wait 16
rm -rf $DJ
taskset -c 64-127 timeout 5400 $PY $TRAIN7 $DJ $BH/runs/lens/pi0_teachjBA_p0 $BH/runs/lens/pi0_teachjBA_p1 $(ls -d $BH/runs/lens/pi0_jab_p*) \
  --procs 16 > $CH/m_build.log 2>&1 < /dev/null
grep -vE "^\s*$" $CH/m_build.log | tail -n 6 | cut -c1-300
BBP=pi0 $PY $MCHK data $DJ --min $JDMIN || { echo "data_pi0j not consistent: nothing trained"; end; }
du -sh $DJ | awk '{print "dataset: " $1}'; echo "/data $(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9) GiB free"
}

# ---------------------------------------------------------------- 4: fingerprints, seed-0 training
phase4() {
echo "== $(ts) phase 4: single-card fingerprints of data_pi0j (pi0); training $AW0 (GPUs 0-3) and $BC0 (GPUs 4-7), seed 0"
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing trained"; end; }
mem_wait 40
(val0p m_awx0 0 $DJ awx 0 > $CH/m_fp_awx0.txt) & (val0p m_awx1 1 $DJ awx 1 > $CH/m_fp_awx1.txt) &
(val0p m_bc0 4 $DJ bc 0 > $CH/m_fp_bc0.txt) & (val0p m_bc1 5 $DJ bc 1 > $CH/m_fp_bc1.txt) &
wait
cat $CH/m_fp_*.txt
F0=$(fpv m_awx0 flow); C0=$(fpv m_awx0 coord); F1=$(fpv m_awx1 flow); C1=$(fpv m_awx1 coord); FB0=$(fpv m_bc0 flow); FB1=$(fpv m_bc1 flow)
echo "fingerprints of data_pi0j: awx seed 0 flow $F0 readout $C0; awx seed 1 flow $F1 readout $C1; bc seed 0 flow $FB0; bc seed 1 flow $FB1" | tee $CH/m_fp.txt
for v in "$F0" "$C0" "$F1" "$C1" "$FB0" "$FB1"; do [ -n "$v" ] || { echo "a fingerprint is missing: nothing trained"; end; }; done
awk -v f="$FB0" 'BEGIN{exit !(f+0 < 0.5)}' || { echo "released pi0 does not fit data_pi0j (bc step-0 flow $FB0 >= 0.5): action convention wrong, nothing trained"; end; }
mem_wait 40
TR="$TR0"; start_train_p0 $AW0 0,1,2,3 --bb pi0 --data $DJ --kind awx --geo_aug
grep -E "data roots|category shares|trainable|action convention|Traceback" $A/run_$AW0.log | cut -c25-330
if fp_ok $A/run_$AW0.log $F0 $C0; then echo "$AW0 healthy, kept"; else echo "$AW0 NOT healthy: stopped"; tail -n 5 $A/run_$AW0.log | cut -c1-300; stop_train_p0 $AW0; end; fi
mem_wait 40
TR="$TR0"; start_train_p0 $BC0 4,5,6,7 --bb pi0 --data $DJ --kind bc --geo_aug
grep -E "data roots|category shares|action convention|Traceback" $A/run_$BC0.log | cut -c25-330
if fp_bc $A/run_$BC0.log $FB0; then echo "$BC0 healthy, kept"; else echo "$BC0 NOT healthy: stopped ($AW0 continues; this chain stops)"; stop_train_p0 $BC0; end; fi
for k in $AW0 $BC0; do $PY -c "import json; d=json.load(open('$A/run_$k/args.json')); print('$k args: seed', d.get('seed'), 'bb', d.get('bb'), 'data', d.get('data'), 'kind', d.get('kind'))"; done
wait_train_p0 $AW0 220 || { echo "$AW0 still running after 220 min: stopping here"; end; }
wait_train_p0 $BC0 60 || { echo "$BC0 still running: stopping here"; end; }
for k in $AW0 $BC0; do grep -E "VAL step|Traceback" $A/run_$k.log | cut -c1-300 | tail -n 13; done
{ health_rel $AW0 && health_rel $BC0; } || { echo "a seed-0 pi0 model does not pass the health rule (7.10): not evaluated, chain stops"; end; }
}

# ---------------------------------------------------------------- 5: evaluation of the seed-0 pair
phase5() {
echo "== $(ts) phase 5: dev (+ no-switch), dev2 extension and test split of $AW0, $BC0 (once each)"
gpus_free 0 3 || { echo "GPUs 0-3 busy: not evaluated"; end; }
lock_eval
mem_wait 26
if ! up_all "$AW0 $BC0"; then echo "servers not healthy: no rollouts"; down_all "$AW0 $BC0"; unlock_eval; end; fi
mem_wait 40
for k in $AW0 $BC0; do evp ${KIND[$k]} $(P $k) $k "${SRV[$k]}"; done
finish_p0 "$AW0 $BC0" "$(names_of $AW0) $(names_of $BC0)" 920 "seed-0 pi0 splits" 100 || echo "seed-0 pi0 splits incomplete after one re-run"
down_all "$AW0 $BC0"
unlock_eval
cd $A
$PY aw_summary.py pi0:awd$AW0 pi0:awp$AW0 pi0:awt$AW0 pi0:awd$BC0 pi0:awp$BC0 pi0:awt$BC0 2>/dev/null
pack m_seed0 "$(names_of $AW0) $(names_of $BC0)"
}

# ---------------------------------------------------------------- 6: seed-1 training
phase6() {
echo "== $(ts) phase 6: training $AW1 (GPUs 0-3) and $BC1 (GPUs 4-7), seed 1"
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing trained"; end; }
F1=$(fpv m_awx1 flow); C1=$(fpv m_awx1 coord); FB1=$(fpv m_bc1 flow)
mem_wait 40
TR="$TR1"; start_train_p0 $AW1 0,1,2,3 --bb pi0 --data $DJ --kind awx --geo_aug
if fp_ok $A/run_$AW1.log $F1 $C1; then echo "$AW1 healthy, kept"; else echo "$AW1 NOT healthy: stopped"; tail -n 5 $A/run_$AW1.log | cut -c1-300; stop_train_p0 $AW1; TR="$TR0"; end; fi
mem_wait 40
TR="$TR1"; start_train_p0 $BC1 4,5,6,7 --bb pi0 --data $DJ --kind bc --geo_aug; TR="$TR0"
if fp_bc $A/run_$BC1.log $FB1; then echo "$BC1 healthy, kept"; else echo "$BC1 NOT healthy: stopped ($AW1 continues; this chain stops)"; stop_train_p0 $BC1; end; fi
for k in $AW1 $BC1; do $PY -c "import json; d=json.load(open('$A/run_$k/args.json')); print('$k args: seed', d.get('seed'), 'bb', d.get('bb'), 'data', d.get('data'), 'kind', d.get('kind'))"; done
wait_train_p0 $AW1 220 || { echo "$AW1 still running after 220 min: stopping here"; end; }
wait_train_p0 $BC1 60 || { echo "$BC1 still running: stopping here"; end; }
for k in $AW1 $BC1; do grep -E "VAL step|Traceback" $A/run_$k.log | cut -c1-300 | tail -n 13; done
{ health_rel $AW1 && health_rel $BC1; } || { echo "a seed-1 pi0 model does not pass the health rule (7.10): not evaluated, chain stops"; end; }
}

# ---------------------------------------------------------------- 7: evaluation of the seed-1 pair
phase7() {
echo "== $(ts) phase 7: dev (+ no-switch), dev2 extension and test split of $AW1, $BC1 (once each)"
gpus_free 0 3 || { echo "GPUs 0-3 busy: not evaluated"; end; }
lock_eval
mem_wait 26
if ! up_all "$AW1 $BC1"; then echo "servers not healthy: no rollouts"; down_all "$AW1 $BC1"; unlock_eval; end; fi
mem_wait 40
for k in $AW1 $BC1; do evp ${KIND[$k]} $(P $k) $k "${SRV[$k]}"; done
finish_p0 "$AW1 $BC1" "$(names_of $AW1) $(names_of $BC1)" 920 "seed-1 pi0 splits" 100 || echo "seed-1 pi0 splits incomplete after one re-run"
down_all "$AW1 $BC1"
unlock_eval
cd $A
$PY aw_summary.py pi0:awd$AW1 pi0:awp$AW1 pi0:awt$AW1 pi0:awd$BC1 pi0:awp$BC1 pi0:awt$BC1 2>/dev/null
pack m_seed1 "$(names_of $AW1) $(names_of $BC1)"
}

# ---------------------------------------------------------------- 8: tables
phase8() {
echo "== $(ts) phase 8: tables (pi0: per seed, pooled over two seeds, against the round-d models awx3p0 / bc3p0)"
cd $A
$PY aw_dev2.py pi0/awx3p0 pi0/bc3p0 pi0/$AW0 pi0/$BC0 pi0/$AW1 pi0/$BC1 --pairs pi0/$AW0:pi0/$BC0,pi0/$AW1:pi0/$BC1,pi0/$AW0:pi0/awx3p0,pi0/$BC0:pi0/bc3p0 | tee $CH/m_dev.txt
for pr in "$AW0 $BC0" "$AW1 $BC1" "$AW0 awx3p0" "$BC0 bc3p0" "$AW1 awx3p0" "$BC1 bc3p0"; do set -- $pr; $PY aw_paired.py pi0:awt$1 pi0:awt$2; done | tee $CH/m_paired.txt
BBP=pi0 $PY $POOL3 "$AW0+$AW1:$BC0+$BC1" "$AW0+$AW1:awx3p0+awx3p0" "$BC0+$BC1:bc3p0+bc3p0" | tee $CH/m_test.txt
(cd $A && tar czf pack/m_logs.tgz run_$AW0.log run_$BC0.log run_$AW1.log run_$BC1.log chain/m_master.log chain/m_*.txt chain/m_build.log val0_m_*.log 2>&1 | tail -n 2; md5sum pack/m_logs.tgz)
}

for ph in 0 1 2 3 4 5 6 7 8; do [ $ph -ge $FROM ] && phase$ph; done
end
