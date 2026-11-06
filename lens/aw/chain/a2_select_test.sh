#!/bin/bash
# a2.sh -- DETACHED on the server (log $A/chain/a2.log).  Lane A (GPUs 0-3): the rest of the selection rule of 7.7 (fixed
# 10-30 08:14 before any new result; baseline awx2c since 09:12), after a1 ($A/chain/a1_decision.txt):
#   NEXT=awx5: wait for awx5 -> health rule -> awx5 on dev2 -> the candidate is the best dev2 of awx3 / awx4 / awx5 (ties ->
#              fewer changes: awx3 < awx4 < awx5) -> train its matched BC (bc3 / bc4 / bc5: kind bc, same data and
#              augmentation, seed 0, --host_place, fingerprint-checked);
#   NEXT=bc3 / bc4: the candidate is awx3 / awx4, its matched BC is being trained by a1;
#   then: wait for the matched BC -> health rule -> matched BC on dev2 (record) -> candidate and matched BC on the test split,
#   once each, whatever the dev2 result -> tables and paired tests (candidate vs matched BC, candidate vs awx2c,
#   matched BC vs bcc).  NEXT=none / anything else: nothing to do (negative result is reported by a1).
# Evaluation phases hold $A/chain/eval.lock and wait for memory (2.2 #18).
source ${CHLIB:-${BH:-/data/blackhole}/lens/aw/chain/lib.sh}
G="episodes ->|DOWN|not healthy"
end() { echo "== $(ts) a2 END"; exit 0; }
echo "== $(ts) a2 start (pid $$)"
for i in $(seq 1 120); do [ "$(pgrep -fc 'a1_dev2_select[.]sh')" -eq 0 ] && break; sleep 30; done
D=$(head -n 1 $CH/a1_decision.txt 2>/dev/null)
NEXT=$(echo "$D" | sed -E 's/^NEXT=([a-z0-9_]+).*/\1/')
echo "a1 decision: $D"
case "$D" in *stopped*|*not_started*|*no_fingerprint*) echo "the training a1 called for is not running: nothing to do"; end;; esac
case $NEXT in
  awx5) ;;
  bc3) CAND=awx3; MBC=bc3 ;;
  bc4) CAND=awx4; MBC=bc4 ;;
  *) echo "nothing to do for NEXT=$NEXT"; end ;;
esac
if [ "$NEXT" = "awx5" ]; then
  wait_train awx5 150 || { echo "awx5 still running after 150 min: stopping here"; end; }
  grep -E "VAL step|Traceback" $A/run_awx5.log | cut -c1-300 | tail -n 13
  train_health awx5 || { echo "awx5 does not pass the health rule: by 7.7 it is retrained once (not automatic)"; end; }
  gpus_free 0 3 || { echo "GPUs 0-3 busy: nothing started"; end; }
  S5="8665:0 8666:1"; P5=$A/run_awx5/params_6000.npz; N5="awdawx5 awpawx5 aweawx5"
  lock_eval
  mem_wait 13
  if ! servers_up awx $P5 awx5 "$S5"; then echo "servers not healthy: no rollouts"; servers_down awx $P5 "$S5"; unlock_eval; end; fi
  mem_wait 12
  echo "== $(ts) awx5 dev2 (10 workers)"
  PHASE=workers $A/aw_eval.sh awx $P5 awx5 "$S5" 2 | grep -E "$G"
  PHASE=workers $A/aw_eval.sh awx $P5 awx5 "$S5" 1 dev2ext | grep -E "$G"
  sleep 120
  wait_workers "awx5 dev2" "$N5" 180 60
  servers_down awx $P5 "$S5"
  unlock_eval
  cd $A
  $PY aw_summary.py awdawx5 awpawx5 2>/dev/null
  $PY aw_dev2.py awx2c awx3 awx4 awx5 bcc --pairs awx5:awx2c,awx5:awx3,awx5:awx4 | tee $CH/dev2_a2.txt
  pack a2_awx5 "$N5"
  n2c=$(cnt awx2c $CH/dev2_a2.txt); n3=$(cnt awx3 $CH/dev2_a2.txt); n4=$(cnt awx4 $CH/dev2_a2.txt); n5=$(cnt awx5 $CH/dev2_a2.txt)
  echo "committed dev2 successes of 120: awx2c ${n2c:-?}, awx3 ${n3:-?}, awx4 ${n4:-?}, awx5 ${n5:-?}"
  [ -n "$n2c" ] && [ -n "$n3" ] && [ -n "$n4" ] && [ -n "$n5" ] || { echo "a dev2 result is incomplete: no decision"; end; }
  CAND=awx3; B=$n3
  [ "$n4" -gt "$B" ] && { CAND=awx4; B=$n4; }
  [ "$n5" -gt "$B" ] && { CAND=awx5; B=$n5; }
  [ "$B" -ge "$n2c" ] || { echo "best candidate below awx2c: no new test split"; end; }
  MBC=bc${CAND#awx}
  echo "final candidate $CAND (dev2 $B/120 >= awx2c $n2c/120); matched BC $MBC" | tee $CH/a2_decision.txt
  gpus_free 0 3 || { echo "GPUs 0-3 busy: $MBC not started"; end; }
  case $MBC in
    bc3)
      start_train bc3 0,1,2,3 --data $A/data_pi05 --kind bc --geo_aug
      if fp_bc $A/run_bc3.log 0.0410; then echo "bc3 healthy, kept"; else echo "bc3 NOT healthy: stopped"; stop_train bc3; end; fi ;;
    bc4|bc5)
      val0 bc_dag 0 $DAG bc
      FB=$(fpv bc_dag flow); [ -n "$FB" ] || { echo "no fingerprint: $MBC not started"; end; }
      if [ "$MBC" = "bc4" ]; then start_train bc4 0,1,2,3 --data $DAG --kind bc; else start_train bc5 0,1,2,3 --data $DAG --kind bc --geo_aug; fi
      if fp_bc $A/run_$MBC.log $FB; then echo "$MBC healthy, kept"; else echo "$MBC NOT healthy: stopped"; stop_train $MBC; end; fi ;;
  esac
fi
echo "== $(ts) candidate $CAND, matched BC $MBC: wait for $MBC"
wait_train $MBC 150 || { echo "$MBC still running after 150 min: stopping here"; end; }
grep -E "VAL step|Traceback" $A/run_$MBC.log | cut -c1-300 | tail -n 13
train_health $MBC || { echo "$MBC does not pass the health rule: by 7.7 it is retrained once (not automatic)"; end; }
gpus_free 0 3 || { echo "GPUs 0-3 busy: nothing started"; end; }
SCD="8667:0 8668:1"; SMB="8653:2 8654:3"
PCD=$A/run_$CAND/params_6000.npz; PMB=$A/run_$MBC/params_6000.npz
NT="awd$MBC awp$MBC awe$MBC awt$MBC awt$CAND"
lock_eval
mem_wait 26
if ! { servers_up awx $PCD $CAND "$SCD" && servers_up bc $PMB $MBC "$SMB"; }; then
  echo "servers not healthy: no rollouts"; servers_down awx $PCD "$SCD"; servers_down bc $PMB "$SMB"; unlock_eval; end
fi
mem_wait 27
echo "== $(ts) $MBC: development split, dev2 extension, test split; $CAND: test split (22 workers)"
PHASE=workers $A/aw_eval.sh bc $PMB $MBC "$SMB" 2 | grep -E "$G"
PHASE=workers $A/aw_eval.sh bc $PMB $MBC "$SMB" 1 dev2ext | grep -E "$G"
PHASE=workers $A/aw_eval.sh bc $PMB $MBC "$SMB" 3 test | grep -E "$G"
PHASE=workers $A/aw_eval.sh awx $PCD $CAND "$SCD" 3 test | grep -E "$G"
sleep 120
wait_workers "test" "$NT" 740 80
servers_down awx $PCD "$SCD"; servers_down bc $PMB "$SMB"
unlock_eval
cd $A
$PY aw_summary.py awd$MBC awp$MBC awt$CAND awt$MBC awtawx2c awtbcc 2>/dev/null
$PY aw_dev2.py awx2c bcc $CAND $MBC --pairs $CAND:$MBC,$MBC:bcc,$CAND:awx2c
for pr in "awt$CAND awt$MBC" "awt$CAND awtawx2c" "awt$MBC awtbcc" "awt$CAND awtbcc"; do set -- $pr; $PY aw_paired.py $1 $2; done
pack a2 "$NT"
end
