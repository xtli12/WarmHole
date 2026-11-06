#!/bin/bash
# b2.sh -- DETACHED on the server (log $A/chain/b2.log).  Lane B (GPUs 4-7), rules of the handover 7.8 (written 10-31 08:57,
# before any awc / awac result):
#  (1) wait for awc (started by b1) -> training health rule -> awc gets, once each and whatever the result: development
#      split (+ no-switch), dev2 extension, test split, and the diagnostics run on the 10-28 AW (aw_off, aw_tn, aw_to;
#      aw_ht, aw_ht_off, aw_int, aw_int_ht); bcc gets the language-path probes run on the old BC (aw_ht, aw_int, aw_int_ht).
#      Servers awc 8675/8676 (GPUs 4/5), bcc 8657/8658 (GPUs 6/7).  Phase A 20 workers, phase B 12 workers.  Workers render
#      on the server's card; if no result comes within 4 min they are restarted rendering on GPUs 0-3 (EGLOFF=-4).
#  (2) awac = the 10-29 awa + --host_place: single-card fingerprint on GPU 4, training on GPUs 4-7 (checked at step 0/50).
#  (3) wait for awac -> health rule -> development split, dev2 extension, test split, aw_off, once each.
# Evaluation phases hold $A/chain/eval.lock (one evaluation at a time on the server) and wait for memory (2.2 #18).
source ${CHLIB:-${BH:-/data/blackhole}/lens/aw/chain/lib.sh}
SC="8675:4 8676:5"; SB="8657:6 8658:7"; SAC="8677:4 8678:5"
PC=$A/run_awc/params_6000.npz; PB=$A/run_bcc/params_6000.npz; PAC=$A/run_awac/params_6000.npz
NA="awdawc awpawc aweawc awtawc awybcc"; NB="awxawc awyawc"; NC="awdawac awpawac aweawac awtawac awxawac"
G="episodes ->|DOWN|not healthy"
end() { echo "== $(ts) b2 END"; exit 0; }
echo "== $(ts) b2 start (pid $$)"
[ -f $PC ] || [ "$(pgrep -fc "aw_train[.]py .*--out $A/run_awc( |$)")" -gt 0 ] || { echo "awc is neither running nor finished"; end; }
wait_train awc 150 || { echo "awc still running after 150 min: stopping here"; end; }
grep -E "VAL step|Traceback" $A/run_awc.log | cut -c1-300 | tail -n 13
train_health awc || { echo "awc does not pass the health rule: not evaluated (by 7.7 it is retrained once, not automatic)"; end; }
gpus_free 4 7 || { echo "GPUs 4-7 busy: nothing started"; end; }
lock_eval
mem_wait 26
if ! { servers_up aw $PC awc "$SC" && servers_up bc $PB bcc "$SB"; }; then
  echo "servers not healthy: no rollouts"; servers_down aw $PC "$SC"; servers_down bc $PB "$SB"; unlock_eval; end
fi
mem_wait 25
launchA() {
  PHASE=workers $A/aw_eval.sh aw $PC awc "$SC" 2 | grep -E "$G"
  PHASE=workers $A/aw_eval.sh aw $PC awc "$SC" 1 dev2ext | grep -E "$G"
  PHASE=workers $A/aw_eval.sh aw $PC awc "$SC" 3 test | grep -E "$G"
  DIAGNAME=awybcc PHASE=workers $A/aw_eval.sh bc $PB bcc "$SB" 2 diag aw_ht,aw_int,aw_int_ht | grep -E "$G"
}
export EGLOFF=0
echo "== $(ts) phase A: awc development split, dev2 extension, test split; bcc language-path probes (20 workers)"
launchA
sleep 240
echo "$(ts) after 4 min: workers $(nwk "$NA"), results $(nres "$NA")"
if [ "$(nres "$NA")" -eq 0 ]; then
  echo "no result in 4 min with rendering on GPUs 4-7; worker state:"
  ps -eo pid=,stat=,wchan:30=,args= | grep "[b]h_client_lens12" | head -n 3 | cut -c1-200
  kill_workers "$NA"; export EGLOFF=-4; echo "$(ts) restart rendering on GPUs 0-3 (EGLOFF=-4)"; launchA; sleep 120
fi
wait_workers "phase A" "$NA" 760 70
echo "== $(ts) phase B: awc diagnostics (12 workers)"
mem_wait 15
DIAGNAME=awxawc PHASE=workers $A/aw_eval.sh aw $PC awc "$SC" 3 diag aw_off,aw_tn,aw_to | grep -E "$G"
DIAGNAME=awyawc PHASE=workers $A/aw_eval.sh aw $PC awc "$SC" 3 diag aw_ht,aw_ht_off,aw_int,aw_int_ht | grep -E "$G"
sleep 60
wait_workers "phase B" "$NB" 700 70
servers_down aw $PC "$SC"; servers_down bc $PB "$SB"
unlock_eval
cd $A
$PY aw_summary.py awdaw6k awdawc awpawc awtaw6k awtawc awxaw6k awxawc awyaw6k awyawc awybc6k awybcc 2>/dev/null
$PY aw_dev2.py awc awx2c bcc --pairs awc:bcc,awc:awx2c
for pr in "awtawc awtaw6k" "awtawc awtbcc" "awtawc awtawx2c" "awdawc awdaw6k"; do set -- $pr; $PY aw_paired.py $1 $2; done
pack b2 "$NA $NB"
echo "== $(ts) awac: single-card fingerprint on GPU 4, then training on GPUs 4-7"
gpus_free 4 7 || { echo "GPUs 4-7 busy: awac not started"; end; }
val0 awa_s0 4 $A/data_pi05 awa
FA=$(fpv awa_s0 flow); CA=$(fpv awa_s0 coord)
[ -n "$FA" ] && [ -n "$CA" ] || { echo "no fingerprint: awac not started"; end; }
start_train awac 4,5,6,7 --data $A/data_pi05 --kind awa
if fp_ok $A/run_awac.log $FA $CA; then echo "awac healthy, kept"; else echo "awac NOT healthy: stopped"; stop_train awac; end; fi
wait_train awac 150 || { echo "awac still running after 150 min: stopping here"; end; }
grep -E "VAL step|Traceback" $A/run_awac.log | cut -c1-300 | tail -n 13
train_health awac || { echo "awac does not pass the health rule: not evaluated"; end; }
gpus_free 4 7 || { echo "GPUs 4-7 busy: awac not evaluated"; end; }
lock_eval
mem_wait 13
if ! servers_up awa $PAC awac "$SAC"; then echo "servers not healthy: no rollouts"; servers_down awa $PAC "$SAC"; unlock_eval; end; fi
mem_wait 25
echo "== $(ts) awac: development split, dev2 extension, test split, aw_off (20 workers, EGLOFF=$EGLOFF)"
PHASE=workers $A/aw_eval.sh awa $PAC awac "$SAC" 2 | grep -E "$G"
PHASE=workers $A/aw_eval.sh awa $PAC awac "$SAC" 1 dev2ext | grep -E "$G"
PHASE=workers $A/aw_eval.sh awa $PAC awac "$SAC" 3 test | grep -E "$G"
DIAGNAME=awxawac PHASE=workers $A/aw_eval.sh awa $PAC awac "$SAC" 2 diag aw_off | grep -E "$G"
sleep 120
wait_workers "awac" "$NC" 560 70
servers_down awa $PAC "$SAC"
unlock_eval
cd $A
$PY aw_summary.py awdawa awdawac awpawac awtawa awtawac awxawa awxawac 2>/dev/null
$PY aw_dev2.py awac awc awx2c bcc --pairs awac:awx2c,awac:bcc
for pr in "awtawac awtawa" "awtawac awtbcc" "awtawac awtawx2c"; do set -- $pr; $PY aw_paired.py $1 $2; done
pack b3 "$NC"
end
