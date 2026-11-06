#!/bin/bash
# l_master.sh c1 -- DETACHED on the server (log $A/chain/l_master.log).  Round l of the handover 7.18 (11-02, rules written
# before any result): a THIRD training seed (seed 2) of the round-j pair, awx3j / bc3j, on the same data (data_pi05j),
# the same augmentation and schedule -- confirmatory only.  NOTHING is selected: the development splits are run for the
# record, the test split is run once per model whatever the development result, and every number is reported.
#   0  wait (up to 3 h) until no process of this project runs (round k included) and GPUs 0-7 are free
#   1  single-card fingerprints of data_pi05j for seed 2 (awx on GPU 0, bc on GPU 4)
#   2  training awx3js2 (kind awx, GPUs 0-3) and bc3js2 (kind bc, GPUs 4-7), step 0 checked against the fingerprints,
#      health rule
#   3  development splits (dev + no-switch, dev2 extension, devR) and the test split once each, both models together
#      on servers 8761-8764 (GPUs 0-3)
#   4  tables: per-seed numbers, and the pooled comparison AW-j vs BC-j over THREE seeds (pairs within seeds), the
#      two-seed comparisons of 4.25 unchanged; packs in $A/pack
source ${JLIB:-${BH:-/data/blackhole}/lens/aw/chain/j_lib.sh}
end() { echo "== $(ts) l_master END"; exit 0; }
TR2="${TR/--seed 0/--seed 2}"
AW2=awx3js2; BC2=bc3js2
declare -A KIND=([awx3js2]=awx [bc3js2]=bc)
declare -A SRV=([awx3js2]="8761:0 8762:1" [bc3js2]="8763:2 8764:3")
DJ=$A/data_pi05j
P() { echo $A/run_$1/params_6000.npz; }
alive() { pgrep -fc 'aw_server(_c9)?[.]py|lens_server[.]py|bh_client_lens[.]py|aw_train(_c11)?[.]py|[e-k]_master(_c[0-9])?[.]sh|pi_heads[.]py|posture_c2[.]py'; }
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
echo "== $(ts) l_master c1 start (pid $$): third seed of the round-j pair; training flags seed 2: $TR2"
(cd $A && md5sum chain/j_lib.sh chain/g_lib.sh chain/e_lib.sh chain/lib2.sh chain/l_master.sh aw_train.py aw_eval.sh aw_server.py e_check.py aw_pool.py aw_dev3.py aw_summary.py aw_paired.py ../bh_client_lens.py ../bh_client_lens.py)
for k in $AW2 $BC2; do [ -d $A/run_$k ] && { echo "run_$k exists already: nothing started"; end; }; done
for k in $AW2 $BC2; do for n in $(nm $k dev dev2ext devr test); do [ -d $BH/runs/lens/pi05_${n}_p0 ] && { echo "results of $n exist already: nothing started"; end; }; done; done
[ -d $DJ ] || { echo "no $DJ: nothing started"; end; }

# ---------------------------------------------------------------- 0: wait for round k and free cards
for i in $(seq 1 360); do
  [ "$(alive)" -eq 0 ] && gpus_free 0 7 && break
  [ $((i % 20)) -eq 1 ] && echo "$(ts) waiting: processes of this project $(alive) (k_master $(pgrep -fc 'k_master[.]sh'))"
  sleep 30
done
{ [ "$(alive)" -eq 0 ] && gpus_free 0 7; } || { echo "GPUs 0-7 not free or processes of this project alive after 3 h: nothing started"; end; }
AV=$(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9)
echo "$(ts) /data: ${AV} GiB available; memory anon $(anon) GiB"
[ "${AV:-0}" -ge 200 ] || { echo "less than 200 GiB free on /data: nothing started"; end; }

# ---------------------------------------------------------------- 1: fingerprints of seed 2
echo "== $(ts) phase 1: single-card fingerprints of data_pi05j, seed 2"
mem_wait 30
(val0s j_awx2 0 $DJ awx 2 > $CH/j_fp_awx2.txt) & (val0s j_bc2 4 $DJ bc 2 > $CH/j_fp_bc2.txt) &
wait
cat $CH/j_fp_awx2.txt $CH/j_fp_bc2.txt
F2=$(fpv j_awx2 flow); C2=$(fpv j_awx2 coord); FB2=$(fpv j_bc2 flow)
echo "fingerprints of data_pi05j, seed 2: awx flow $F2 readout $C2; bc flow $FB2" | tee $CH/l_fp.txt
for v in "$F2" "$C2" "$FB2"; do [ -n "$v" ] || { echo "a fingerprint is missing: nothing trained"; end; }; done

# ---------------------------------------------------------------- 2: training
echo "== $(ts) phase 2: training $AW2 (GPUs 0-3) and $BC2 (GPUs 4-7), seed 2"
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing trained"; end; }
mem_wait 40
TR="$TR2"; start_train $AW2 0,1,2,3 --data $DJ --kind awx --geo_aug
grep -E "data roots|category shares|trainable|Traceback" $A/run_$AW2.log | cut -c25-330
if fp_ok $A/run_$AW2.log $F2 $C2; then echo "$AW2 healthy, kept"; else echo "$AW2 NOT healthy: stopped"; tail -n 5 $A/run_$AW2.log | cut -c1-300; stop_train $AW2; end; fi
mem_wait 40
TR="$TR2"; start_train $BC2 4,5,6,7 --data $DJ --kind bc --geo_aug
grep -E "data roots|category shares|Traceback" $A/run_$BC2.log | cut -c25-330
if fp_bc $A/run_$BC2.log $FB2; then echo "$BC2 healthy, kept"; else echo "$BC2 NOT healthy: stopped ($AW2 continues; this chain stops)"; stop_train $BC2; end; fi
for k in $AW2 $BC2; do $PY -c "import json; d=json.load(open('$A/run_$k/args.json')); print('$k args: seed', d.get('seed'), 'data', d.get('data'), 'kind', d.get('kind'))"; done
wait_train $AW2 220 || { echo "$AW2 still running after 220 min: stopping here"; end; }
wait_train $BC2 60 || { echo "$BC2 still running: stopping here"; end; }
for k in $AW2 $BC2; do grep -E "VAL step|Traceback" $A/run_$k.log | cut -c1-300 | tail -n 13; done
{ train_health $AW2 && train_health $BC2; } || { echo "a seed-2 model does not pass the health rule: not evaluated (by 7.7 it is retrained once, not automatic)"; end; }

# ---------------------------------------------------------------- 3: development splits and the test split, once each
echo "== $(ts) phase 3: development splits and test split of $AW2, $BC2 (once each, whatever the development result)"
gpus_free 0 3 || { echo "GPUs 0-3 busy: not evaluated"; end; }
lock_eval
mem_wait 26
if ! up_all "$AW2 $BC2"; then echo "servers not healthy: no rollouts"; down_all "$AW2 $BC2"; unlock_eval; end; fi
mem_wait 52
for k in $AW2 $BC2; do ev ${KIND[$k]} $(P $k) $k "${SRV[$k]}" dev:2 dev2ext:1 devr:3 test:3; done
W=""; for k in $AW2 $BC2; do W="$W awd$k:100 awp$k:40 awe$k:40 awr$k:240 awt$k:280"; done
finish "$AW2 $BC2" "dev:1 dev2ext:1 devr:1 test:2" "$W" "seed-2 dev + test" 120 || echo "seed-2 splits incomplete after one re-run"
down_all "$AW2 $BC2"
unlock_eval

# ---------------------------------------------------------------- 4: tables and packs
echo "== $(ts) phase 4: tables (per seed; pooled over three seeds AW-j vs BC-j)"
cd $A
$PY aw_summary.py awd$AW2 awp$AW2 awd$BC2 awp$BC2 awt$AW2 awt$BC2 2>/dev/null
$PY aw_dev3.py awx3j awx3s1j $AW2 bc3j bc3s1j $BC2 --pairs $AW2:$BC2 \
  --pool "awx3j+awx3s1j+$AW2:bc3j+bc3s1j+$BC2" | tee $CH/l_dev.txt
$PY aw_pool.py "awx3j+awx3s1j+$AW2:bc3j+bc3s1j+$BC2" "$AW2:$BC2" | tee $CH/l_test.txt
$PY aw_paired.py awt$AW2 awt$BC2
pack l_seed2 "$(nm $AW2 dev dev2ext devr test) $(nm $BC2 dev dev2ext devr test)"
(cd $A && tar czf pack/l_logs.tgz run_$AW2.log run_$BC2.log chain/l_master.log chain/l_*.txt val0_j_awx2.log val0_j_bc2.log 2>&1 | tail -n 2; md5sum pack/l_logs.tgz)
end
