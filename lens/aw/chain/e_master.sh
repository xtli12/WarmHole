#!/bin/bash
# e_master.sh c1 -- DETACHED on the server (log $A/chain/e_master.log).  Round e of the handover 7.11 (plan and every rule
# written 11-01 09:30 before any self-play result; order of execution and the NOGO stop clarified in 7.11 at 09:55,
# still before any result): switch self-play + group-relative self-imitation for AW (kind awx) and its matched BC, two
# seeds each.
#   0  wait for round d (d_a / d_b) to end and for GPUs 0-7 to be free (jobs of other users: wait, never touched)
#   1  smoke (GPU 0): one awx3 server; 24 self-play + 12 devR-type episodes (layout r521, lens14) and 8 official development episodes with
#      lens14 (compared with the lens12 results of awx3, information only); data builder c5; spec generator r21 against
#      the installed one; CPU training smokes: c11 = c10 without self-play data, validation batch unchanged with it,
#      --share 5:0.3 for awx and bc.  Nothing else starts unless every gate passes.
#   2  self-play of the 4 students awx3, awx3s1, bc3, bc3s1 (1512 episodes each) and devR of the same models (240 each):
#      8 servers (2 per student), 4 workers per server (3 if memory would exceed 100 GiB); a student's devR starts when its
#      self-play is done
#   3  4 self-play datasets (aw_build_data_c5 --self)
#   4  AW-SP: awx3sp (seed 0, GPUs 0-3) and awx3s1sp (seed 1, GPUs 4-7), fingerprint-checked, health rule
#   5  AW-SP development evaluation (2 servers each, GPUs 0-3): dev (+ no-switch), dev2 extension, devR
#   6  rule of 7.11 (aw_dev3.py --decide) -> $CH/e_decision.txt.  NOGO: stop here (no BC-SP, no test split)
#   7  GO: AW-SP test split once each on the same servers while bc3sp trains (GPUs 4-7); then bc3s1sp (GPUs 0-3)
#   8  BC-SP: health rule; dev (+ no-switch), dev2 extension, devR and test split once each, whatever the result
#   9  tables (aw_dev3, aw_pool, aw_paired, aw_summary) and result packs in $A/pack
source ${ELIB:-${BH:-/data/blackhole}/lens/aw/chain/e_lib.sh}
end() { echo "== $(ts) e_master END"; exit 0; }
TR0="$TR"; TR1="${TR/--seed 0/--seed 1}"
STU="awx3 awx3s1 bc3 bc3s1"; AWS="awx3sp awx3s1sp"; BCS="bc3sp bc3s1sp"
declare -A KIND=([awx3]=awx [awx3s1]=awx [bc3]=bc [bc3s1]=bc [awx3sp]=awx [awx3s1sp]=awx [bc3sp]=bc [bc3s1sp]=bc)
declare -A SRV=([awx3]="8711:0 8712:1" [awx3s1]="8713:2 8714:3" [bc3]="8715:4 8716:5" [bc3s1]="8717:6 8718:7"
                [awx3sp]="8721:0 8722:1" [awx3s1sp]="8723:2 8724:3" [bc3sp]="8725:0 8726:1" [bc3s1sp]="8727:2 8728:3")
P() { echo $A/run_$1/params_6000.npz; }
alive() { pgrep -fc 'aw_server[.]py|bh_client_lens[.]py|aw_train(_c11)?[.]py'; }
up_all() {  # "models": start their servers; 0 only if all are healthy
  local k ok=0
  for k in $1; do servers_up ${KIND[$k]} $(P $k) $k "${SRV[$k]}" || ok=1; done
  return $ok
}
down_all() { local k; for k in $1; do servers_down ${KIND[$k]} $(P $k) "${SRV[$k]}"; done; }
healthy_all() { local k sg; for k in $1; do for sg in ${SRV[$k]}; do curl -s -m 3 localhost:${sg%%:*}/healthz > /dev/null || return 1; done; done; }
# finish MODELS "split:nw ..." "NAME:N ..." LABEL MAXMIN: wait for the workers; if rows are missing or aborted, move the
# aborted rows aside and re-run the missing episodes once (servers must still be healthy); 0 if complete
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
FROM=${FROM:-0}      # first phase to run (resume after a stop; the phases before it must have completed)
echo "== $(ts) e_master c1 start (pid $$), from phase $FROM; training flags seed 0: $TR0"
(cd $A && md5sum aw_train.py aw_build_data.py aw_eval.sh aw_dev3.py aw_pool.py e_check.py chain/e_lib.sh chain/e_master.sh \
   ../bh_client_lens.py ../make_specs_lens.py ../make_specs_lens.py ../bh_client_lens.py aw_train.py aw_server.py aw_eval.sh)

# ---------------------------------------------------------------- 0: wait for round d and free cards
phase0() {
for i in $(seq 1 480); do
  [ "$(pgrep -fc 'd_a_bc3p0[.]sh|d_b_awx3p0[.]sh')" -eq 0 ] && break
  [ $((i % 20)) -eq 1 ] && echo "$(ts) round d running (d_a $(pgrep -fc 'd_a_bc3p0[.]sh'), d_b $(pgrep -fc 'd_b_awx3p0[.]sh')): $(tail -n 1 $CH/d_b.log | cut -c1-160)"
  sleep 30
done
[ "$(pgrep -fc 'd_a_bc3p0[.]sh|d_b_awx3p0[.]sh')" -eq 0 ] || { echo "round d still running after 4 h: nothing started"; end; }
echo "$(ts) round d ended:"; tail -n 2 $CH/d_a.log | cut -c1-200; tail -n 2 $CH/d_b.log | cut -c1-200
for i in $(seq 1 120); do
  [ "$(alive)" -eq 0 ] && gpus_free 0 7 && break
  [ $((i % 10)) -eq 1 ] && echo "$(ts) waiting: processes of this project $(alive); GPU memory $(timeout 60 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
  sleep 30
done
{ [ "$(alive)" -eq 0 ] && gpus_free 0 7; } || { echo "GPUs 0-7 not free or processes of this project alive after 60 min: nothing started"; end; }
AV=$(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9)
echo "$(ts) /data: ${AV} GiB available; memory anon $(anon) GiB; $(free -g | sed -n 2p)"
[ "${AV:-0}" -ge 500 ] || { echo "less than 500 GiB free on /data (round e writes about 300 GiB): nothing started"; end; }

}
# ---------------------------------------------------------------- 1: smoke
phase1() {
echo "== $(ts) phase 1: smoke"
S0="8790:0"; P3=$(P awx3)
for d in spsmk awrsmk awqsmk; do rm -rf $BH/runs/lens/pi05_${d}_p*; done
rm -rf $A/data_sp_smk $A/smoke_c10 $A/smoke_c11a $A/smoke_c11b $A/smoke_c11c
SMK="--smoke --bs 32 --lam_loc 0.1 --lam_coord 0.2 --skip_gn 1000 --seed 0"
mem_wait 16
(CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$A taskset -c 96-111 timeout 1500 $PY $A/aw_train.py $SMK --data $A/data_pi05 \
   --kind awx --geo_aug --out $A/smoke_c10 > $A/smoke_c10.log 2>&1 < /dev/null) &
PS1=$!
(CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$A taskset -c 112-127 timeout 1500 $PY $A/aw_train.py $SMK --data $A/data_pi05 \
   --kind awx --geo_aug --out $A/smoke_c11a > $A/smoke_c11a.log 2>&1 < /dev/null) &
PS2=$!
lock_eval
mem_wait 18
if ! servers_up awx $P3 smk "$S0"; then echo "smoke server not healthy: nothing else started"; servers_down awx $P3 "$S0"; unlock_eval; wait $PS1 $PS2; end; fi
SPL="20,r101" SPK=1 ev awx $P3 smk "$S0" selfplay:4
DRL="r521" ev awx $P3 smk "$S0" devr:2      # r521: outside devR (r501-520) and self-play (< 500)
# 8 official development episodes (init 40 / 41, t_s 12 / 42, both directions) with lens14, to compare with awdawx3 (lens12)
$PY $L/make_specs_lens.py $BH/specs/lens --dev flush --name awqsmk --ts 12,42 --bbs pi05 > /dev/null
grep -E '_i4[01]"' $BH/specs/lens/pi05_awqsmk.jsonl > $BH/specs/lens/pi05_awqsmk_part0.jsonl
mkdir -p $BH/runs/lens/pi05_awqsmk_p0
(export PYTHONPATH=$BH/openpi/third_party/libero:$BH/openpi/packages/openpi-client/src:$L:$BH/code
 export LIBERO_CONFIG_PATH=$BH/libero_cfg MUJOCO_GL=egl PYOPENGL_PLATFORM=egl BH_DEPTH=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
 for w in 0 1; do
   MUJOCO_EGL_DEVICE_ID=0 setsid nohup $BH/venvs/libero/bin/python $L/bh_client_lens.py --spec $BH/specs/lens/pi05_awqsmk_part0.jsonl \
     --out $BH/runs/lens/pi05_awqsmk_p0 --port ${S0%%:*} --worker $w --nworkers 2 > $BH/runs/lens/pi05_awqsmk_p0/worker_$w.log 2>&1 < /dev/null &
 done)
echo "$(ts) smoke: $(wc -l < $BH/specs/lens/pi05_spsmk.jsonl) self-play, $(wc -l < $BH/specs/lens/pi05_awrsmk.jsonl) devR, $(wc -l < $BH/specs/lens/pi05_awqsmk_part0.jsonl) official episodes"
sleep 30
WMAX=0; SRVM=0      # largest RssAnon of a self-play worker / of the server during the smoke (GiB): sizes phase 2
for i in $(seq 1 100); do
  [ "$(nwk "spsmk awrsmk awqsmk")" -eq 0 ] && break
  for p in $(pgrep -f "bh_client_lens[.]py --spec $BH/specs/lens/pi05_spsmk_part"); do WMAX=$(awk -v a=$WMAX -v b="$(rssanon $p)" 'BEGIN{print (b+0>a+0)?b+0:a+0}'); done
  SRVM=$(awk -v a=$SRVM -v b="$(rssanon $(pidtag awx $P3 ${S0%%:*}))" 'BEGIN{print (b+0>a+0)?b+0:a+0}')
  [ $((i % 8)) -eq 1 ] && echo "$(ts) smoke: workers $(nwk "spsmk awrsmk awqsmk"), results $(nres "spsmk awrsmk awqsmk")/44, aborted $(nerr "spsmk awrsmk awqsmk"); anon $(anon) GiB"
  sleep 15
done
[ "$(nwk "spsmk awrsmk awqsmk")" -eq 0 ] || { echo "smoke workers still running after 25 min: stopped"; kill_workers "spsmk awrsmk awqsmk"; }
echo "$(ts) smoke rollouts: results $(nres "spsmk awrsmk awqsmk")/44, aborted $(nerr "spsmk awrsmk awqsmk"); largest RssAnon: self-play worker $WMAX GiB, server $SRVM GiB"
servers_down awx $P3 "$S0"
unlock_eval
cd $A
SOK=1
grep -hE "Traceback|Error" $BH/runs/lens/pi05_spsmk_p0/worker_*.log $BH/runs/lens/pi05_awrsmk_p0/worker_*.log | sort | uniq -c | head -n 5
$PY e_check.py smoke $BH/runs/lens || SOK=0
$PY e_check.py equiv $BH/runs/lens awqsmk awdawx3
taskset -c 64-95 timeout 900 $PY $A/aw_build_data.py $A/data_sp_smk $BH/runs/lens/pi05_spsmk_p0 --self --procs 4 2>&1 | tail -n 14 | cut -c1-300
$PY e_check.py data $A/data_sp_smk || SOK=0
echo "-- spec generator r21 against the installed one (every existing call)"
rm -rf $A/spec_chk; mkdir -p $A/spec_chk/a $A/spec_chk/b
for d in a b; do F=$L/make_specs_lens.py; [ $d = b ] && F=$L/make_specs_lens.py
  $PY $F $A/spec_chk/$d --dev flush --name awdx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk/$d --plain awpx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk/$d --test flush --name awdxx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk/$d --dev flush --name awex --ts 52,62 --bbs pi05 > /dev/null
  $PY $F $A/spec_chk/$d --dev flush --name awdx0 --bbs pi0 > /dev/null
  $PY $F $A/spec_chk/$d --dev aw_off,aw_bias0,aw_hl --name awxx --bbs pi05 > /dev/null
  $PY $F $A/spec_chk/$d --teach --vlm_offset '{"pi05": [-0.017, 0.044], "pi0": [-0.012, 0.044]}' > /dev/null
  $PY $F $A/spec_chk/$d --dagger dg10,dg30,dgb20 --name dagx --bbs pi05 --vlm_offset '{"pi05": [-0.017, 0.044], "pi0": [-0.012, 0.044]}' > /dev/null
done
echo "spec files compared: $(ls $A/spec_chk/a | wc -l)"
if diff -rq $A/spec_chk/a $A/spec_chk/b; then echo "specs identical"; else SOK=0; echo "SPECS DIFFER"; fi
rm -rf $A/spec_chk
echo "-- CPU training smokes"
wait $PS1 $PS2
mem_wait 16
(CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$A taskset -c 96-111 timeout 1500 $PY $A/aw_train.py $SMK --data $A/data_pi05,$A/data_sp_smk \
   --kind awx --geo_aug --share 5:0.3 --out $A/smoke_c11b > $A/smoke_c11b.log 2>&1 < /dev/null) &
PS3=$!
(CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$A taskset -c 112-127 timeout 1500 $PY $A/aw_train.py $SMK --data $A/data_pi05,$A/data_sp_smk \
   --kind bc --geo_aug --share 5:0.3 --out $A/smoke_c11c > $A/smoke_c11c.log 2>&1 < /dev/null) &
PS4=$!
wait $PS3 $PS4
for k in c10 c11a c11b c11c; do
  echo "-- smoke_$k"; grep -E "data roots|frames [0-9]+: train|step 0 |VAL step 0|smoke done|Error|Traceback" $A/smoke_$k.log | cut -c25-330
  grep -q "smoke done" $A/smoke_$k.log || { SOK=0; tail -n 8 $A/smoke_$k.log | cut -c1-300; }
done
$PY e_check.py val0 $A/smoke_c10.log $A/smoke_c11a.log $A/smoke_c11b.log || SOK=0
[ "$(grep -E 'category shares' $A/smoke_c10.log | sed 's/.*shares//')" = "$(grep -E 'category shares' $A/smoke_c11a.log | sed 's/.*shares//')" ] \
  || { SOK=0; echo "c11 sampling differs from c10 without self-play data"; }
for k in c11b c11c; do grep -E "category shares" $A/smoke_$k.log | grep -qE "5: 0[.]3[},]" || { SOK=0; echo "category 5 share is not 0.3 in smoke $k"; }; done
echo "$(ts) smoke gates passed: $SOK"
[ "$SOK" -eq 1 ] || { echo "nothing else started"; end; }
rm -rf $A/smoke_c10 $A/smoke_c11a $A/smoke_c11b $A/smoke_c11c

}
# ---------------------------------------------------------------- 2: self-play rollouts and devR of the students
phase2() {
echo "== $(ts) phase 2: switch self-play of $STU (1512 episodes each) and their devR (240 each)"
SPN=""; DRN=""; for s in $STU; do SPN="$SPN sp$s"; DRN="$DRN awr$s"; done; SPN=$(echo $SPN); DRN=$(echo $DRN)
for n in $SPN $DRN; do [ -d $BH/runs/lens/pi05_${n}_p0 ] && { echo "results of $n exist already: nothing started"; end; }; done
for s in $STU; do [ -f $(P $s) ] || { echo "no $(P $s): nothing started"; end; }; done
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing started"; end; }
lock_eval
mem_wait 52
if ! up_all "$STU"; then echo "servers not healthy: no rollouts"; down_all "$STU"; unlock_eval; end; fi
echo "$(ts) servers up; RssAnon (GiB): $(for s in $STU; do for sg in ${SRV[$s]}; do printf '%s ' $(rssanon $(pidtag ${KIND[$s]} $(P $s) ${sg%%:*})); done; done); anon $(anon) GiB"
# workers per server: 4, unless 32 workers of the size measured in the smoke (+ 0.15 GiB) would take this user's anonymous
# memory above 100 GiB (2.2 #18); then 3 (or 2)
AN=$(anon); NWS=$(awk -v a="${AN:-60}" -v w="${WMAX:-1.3}" 'BEGIN{w=(w+0>0.5)?w+0:1.3; n=4; while (n>2 && a+8*n*(w+0.15)>100) n--; print n}')
echo "$(ts) self-play workers per server: $NWS (anon now ${AN} GiB, worker size ${WMAX:-?} GiB)"
mem_wait $(awk -v n=$NWS -v w="${WMAX:-1.3}" 'BEGIN{w=(w+0>0.5)?w+0:1.3; printf "%d", 8*n*(w+0.15)+0.5}')
for s in $STU; do ev ${KIND[$s]} $(P $s) $s "${SRV[$s]}" selfplay:$NWS; done
sleep 60
declare -A DR
for i in $(seq 1 $((200 * 4))); do
  for s in $STU; do
    if [ -z "${DR[$s]}" ] && [ "$(nwk sp$s)" -eq 0 ]; then
      echo "$(ts) $s: self-play done ($(nres sp$s)/1512 results, $(nerr sp$s) aborted) -> its devR"
      ev ${KIND[$s]} $(P $s) $s "${SRV[$s]}" devr:$NWS
      DR[$s]=1
    fi
  done
  [ ${#DR[@]} -eq 4 ] && [ "$(nwk "$DRN")" -eq 0 ] && break
  [ $((i % 40)) -eq 1 ] && echo "$(ts) self-play workers $(nwk "$SPN"), results $(nres "$SPN")/6048, aborted $(nerr "$SPN"); devR workers $(nwk "$DRN"), results $(nres "$DRN")/960; anon $(anon) GiB; worker RssAnon $(for p in $(pgrep -f 'bh_client_lens[.]py' | head -n 3); do printf '%s ' $(rssanon $p); done); /data $(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9) GiB free"
  sleep 15
done
W=""; for s in $STU; do W="$W sp$s:1512:1497 awr$s:240"; done
if ! finish "$STU" "selfplay:2 devr:2" "$W" "self-play + devR" 120; then
  echo "self-play or devR incomplete after one re-run: nothing trained"; down_all "$STU"; unlock_eval; end
fi
down_all "$STU"
unlock_eval
cd $A
$PY e_check.py spsum $BH/runs/lens $SPN | tee $CH/e_selfplay.txt
$PY aw_dev3.py $STU | tee $CH/e_devr_students.txt
pack e_sp "$SPN $DRN"
du -sh $(for n in $SPN; do echo $BH/runs/lens/pi05_${n}_p*/steps; done) 2>/dev/null | awk '{print $1}' | tr '\n' ' '; echo "(step records)"

}
# ---------------------------------------------------------------- 3: self-play datasets
phase3() {
echo "== $(ts) phase 3: self-play datasets"
mem_wait 16
BP=""
for s in $STU; do
  rm -rf $A/data_sp_$s
  taskset -c 64-127 timeout 5400 $PY $A/aw_build_data.py $A/data_sp_$s $(ls -d $BH/runs/lens/pi05_sp${s}_p*) --self --procs 8 \
    > $CH/e_build_$s.log 2>&1 < /dev/null &
  BP="$BP $!"
done
wait $BP
for s in $STU; do echo "-- data_sp_$s"; grep -vE "^\s*$" $CH/e_build_$s.log | tail -n 14 | cut -c1-300; done
$PY e_check.py data $A/data_sp_awx3 $A/data_sp_awx3s1 $A/data_sp_bc3 $A/data_sp_bc3s1 || { echo "a self-play dataset is not consistent: nothing trained"; end; }
du -sh $A/data_sp_awx3 $A/data_sp_awx3s1 $A/data_sp_bc3 $A/data_sp_bc3s1 | awk '{print $1}' | tr '\n' ' '; echo "(datasets); /data $(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9) GiB free"

}
# ---------------------------------------------------------------- 4: AW-SP training
phase4() {
echo "== $(ts) phase 4: AW-SP training (awx3sp seed 0 GPUs 0-3, awx3s1sp seed 1 GPUs 4-7)"
for k in $AWS $BCS; do for n in $(nm $k dev dev2ext devr test); do [ -d $BH/runs/lens/pi05_${n}_p0 ] && { echo "results of $n exist already: nothing trained"; end; }; done; done
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing trained"; end; }
mem_wait 40
TR="$TR0"; start_train awx3sp 0,1,2,3 --data $A/data_pi05,$A/data_sp_awx3 --kind awx --geo_aug --share 5:0.3
grep -E "data roots|category shares|trainable|Traceback" $A/run_awx3sp.log | cut -c25-330
if fp_ok $A/run_awx3sp.log 0.0541 1.1345; then echo "awx3sp healthy, kept"; else echo "awx3sp NOT healthy: stopped"; tail -n 5 $A/run_awx3sp.log | cut -c1-300; stop_train awx3sp; end; fi
mem_wait 40
TR="$TR1"; start_train awx3s1sp 4,5,6,7 --data $A/data_pi05,$A/data_sp_awx3s1 --kind awx --geo_aug --share 5:0.3; TR="$TR0"
grep -E "data roots|category shares|trainable|Traceback" $A/run_awx3s1sp.log | cut -c25-330
if fp_ok $A/run_awx3s1sp.log 0.0540 0.8999; then echo "awx3s1sp healthy, kept"; else echo "awx3s1sp NOT healthy: stopped (awx3sp continues; this chain stops)"; tail -n 5 $A/run_awx3s1sp.log | cut -c1-300; stop_train awx3s1sp; end; fi
for k in $AWS; do $PY -c "import json; d=json.load(open('$A/run_$k/args.json')); print('$k args: seed', d.get('seed'), 'share', d.get('share'), 'data', d.get('data'))"; done
wait_train awx3sp 200 || { echo "awx3sp still running after 200 min: stopping here"; end; }
wait_train awx3s1sp 60 || { echo "awx3s1sp still running: stopping here"; end; }
for k in $AWS; do grep -E "VAL step|Traceback" $A/run_$k.log | cut -c1-300 | tail -n 13; done
{ train_health awx3sp && train_health awx3s1sp; } || { echo "an AW-SP model does not pass the health rule: by 7.7 it is retrained once (not automatic)"; end; }

}
# ---------------------------------------------------------------- 5 / 6: AW-SP development evaluation and the rule of 7.11
phase5() {
echo "== $(ts) phase 5: AW-SP development evaluation"
gpus_free 0 3 || { echo "GPUs 0-3 busy: not evaluated"; end; }
lock_eval
mem_wait 26
if ! up_all "$AWS"; then echo "servers not healthy: no rollouts"; down_all "$AWS"; unlock_eval; end; fi
mem_wait 45
for k in $AWS; do ev awx $(P $k) $k "${SRV[$k]}" dev:2 dev2ext:1 devr:4; done
W=""; for k in $AWS; do W="$W awd$k:100 awp$k:40 awe$k:40 awr$k:240"; done
finish "$AWS" "dev:1 dev2ext:1 devr:1" "$W" "AW-SP dev" 90 || echo "AW-SP development split incomplete after one re-run (the rule then gives NOGO)"
cd $A
$PY aw_summary.py awdawx3sp awpawx3sp awdawx3s1sp awpawx3s1sp 2>/dev/null
$PY aw_dev3.py awx3 awx3s1 bc3 bc3s1 awx3sp awx3s1sp --pairs awx3sp:awx3,awx3s1sp:awx3s1,awx3sp:bc3,awx3s1sp:bc3s1 \
  --pool "awx3sp+awx3s1sp:bc3+bc3s1" --decide "awx3sp+awx3s1sp:awx3+awx3s1" --out $CH/e_decision.txt | tee $CH/e_dev_aw.txt
pack e_awsp_dev "$(nm awx3sp dev dev2ext devr) $(nm awx3s1sp dev dev2ext devr)"
D=$(head -n 1 $CH/e_decision.txt 2>/dev/null)
echo "$(ts) decision (7.11): ${D:-none}"
case "$D" in
  GO*) ;;
  *) echo "not GO: by 7.11 no test split and no BC-SP; this chain stops here"; down_all "$AWS"; unlock_eval; end ;;
esac

}
# ---------------------------------------------------------------- 7: AW-SP test split; BC-SP training
phase7() {
echo "== $(ts) phase 7: AW-SP test split (once each, same servers); bc3sp training on GPUs 4-7"
mem_wait 20
for k in $AWS; do ev awx $(P $k) $k "${SRV[$k]}" test:4; done
BC0=no
if gpus_free 4 7; then
  mem_wait 40
  TR="$TR0"; start_train bc3sp 4,5,6,7 --data $A/data_pi05,$A/data_sp_bc3 --kind bc --geo_aug --share 5:0.3
  grep -E "data roots|category shares|Traceback" $A/run_bc3sp.log | cut -c25-330
  if fp_bc $A/run_bc3sp.log 0.0410; then BC0=yes; echo "bc3sp healthy, kept"; else echo "bc3sp NOT healthy: stopped"; stop_train bc3sp; BC0=bad; fi
else echo "GPUs 4-7 busy: bc3sp starts after the test split"; fi
finish "$AWS" "test:2" "awtawx3sp:280 awtawx3s1sp:280" "AW-SP test" 60 || echo "AW-SP test split incomplete after one re-run"
down_all "$AWS"
unlock_eval
cd $A
pack e_awsp_test "awtawx3sp awtawx3s1sp"
$PY aw_pool.py "awx3sp+awx3s1sp:awx3+awx3s1" "awx3sp+awx3s1sp:bc3+bc3s1" | tee $CH/e_test_aw.txt
[ "$BC0" = bad ] && { echo "bc3sp fingerprint failed: BC-SP not trained (decided by hand)"; end; }
gpus_free 0 3 || { echo "GPUs 0-3 busy: bc3s1sp not started"; end; }
mem_wait 40
TR="$TR1"; start_train bc3s1sp 0,1,2,3 --data $A/data_pi05,$A/data_sp_bc3s1 --kind bc --geo_aug --share 5:0.3; TR="$TR0"
grep -E "data roots|category shares|Traceback" $A/run_bc3s1sp.log | cut -c25-330
if fp_bc $A/run_bc3s1sp.log 0.0409; then echo "bc3s1sp healthy, kept"; else echo "bc3s1sp NOT healthy: stopped"; stop_train bc3s1sp; end; fi
if [ "$BC0" = no ]; then
  gpus_free 4 7 || { echo "GPUs 4-7 busy: bc3sp not started"; end; }
  mem_wait 40
  TR="$TR0"; start_train bc3sp 4,5,6,7 --data $A/data_pi05,$A/data_sp_bc3 --kind bc --geo_aug --share 5:0.3
  fp_bc $A/run_bc3sp.log 0.0410 || { echo "bc3sp NOT healthy: stopped"; stop_train bc3sp; end; }
fi
for k in $BCS; do $PY -c "import json; d=json.load(open('$A/run_$k/args.json')); print('$k args: seed', d.get('seed'), 'share', d.get('share'), 'data', d.get('data'))"; done
wait_train bc3sp 200 || { echo "bc3sp still running after 200 min: stopping here"; end; }
wait_train bc3s1sp 120 || { echo "bc3s1sp still running: stopping here"; end; }
for k in $BCS; do grep -E "VAL step|Traceback" $A/run_$k.log | cut -c1-300 | tail -n 13; done
{ train_health bc3sp && train_health bc3s1sp; } || { echo "a BC-SP model does not pass the health rule: not evaluated"; end; }

}
# ---------------------------------------------------------------- 8: BC-SP development and test splits
phase8() {
echo "== $(ts) phase 8: BC-SP development splits and test split (once each)"
gpus_free 0 3 || { echo "GPUs 0-3 busy: not evaluated"; end; }
lock_eval
mem_wait 26
if ! up_all "$BCS"; then echo "servers not healthy: no rollouts"; down_all "$BCS"; unlock_eval; end; fi
mem_wait 52
for k in $BCS; do ev bc $(P $k) $k "${SRV[$k]}" dev:2 dev2ext:1 devr:3 test:3; done
W=""; for k in $BCS; do W="$W awd$k:100 awp$k:40 awe$k:40 awr$k:240 awt$k:280"; done
finish "$BCS" "dev:1 dev2ext:1 devr:1 test:2" "$W" "BC-SP dev + test" 120 || echo "BC-SP splits incomplete after one re-run"
down_all "$BCS"
unlock_eval

}
# ---------------------------------------------------------------- 9: tables and packs
phase9() {
echo "== $(ts) phase 9: tables"
cd $A
$PY aw_dev3.py awx3 awx3s1 bc3 bc3s1 awx3sp awx3s1sp bc3sp bc3s1sp --pairs awx3sp:bc3sp,awx3s1sp:bc3s1sp,bc3sp:bc3,bc3s1sp:bc3s1 \
  --pool "awx3sp+awx3s1sp:bc3sp+bc3s1sp,bc3sp+bc3s1sp:bc3+bc3s1" | tee $CH/e_dev_all.txt
$PY aw_pool.py "awx3sp+awx3s1sp:bc3sp+bc3s1sp" "awx3sp+awx3s1sp:awx3+awx3s1" "bc3sp+bc3s1sp:bc3+bc3s1" | tee $CH/e_test.txt
$PY aw_summary.py awtawx3sp awtawx3s1sp awtbc3sp awtbc3s1sp awpbc3sp awpbc3s1sp 2>/dev/null
for pr in "awtawx3sp awtbc3sp" "awtawx3s1sp awtbc3s1sp" "awtawx3sp awtawx3" "awtawx3s1sp awtawx3s1" "awtbc3sp awtbc3" "awtbc3s1sp awtbc3s1"; do
  set -- $pr; $PY aw_paired.py $1 $2
done
pack e_bcsp "$(nm bc3sp dev dev2ext devr test) $(nm bc3s1sp dev dev2ext devr test)"
(cd $A && tar czf pack/e_logs.tgz run_awx3sp.log run_awx3s1sp.log run_bc3sp.log run_bc3s1sp.log data_sp_*/conditions.json \
   data_sp_*/episodes.json chain/e_master.log chain/e_*.txt chain/e_build_*.log 2>&1 | tail -n 2; md5sum pack/e_logs.tgz)
}

for ph in 0 1 2 3 4 5 7 8 9; do [ $ph -ge $FROM ] && phase$ph; done
end
