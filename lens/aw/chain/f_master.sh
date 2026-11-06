#!/bin/bash
# f_master.sh c2 -- DETACHED on the server (log $A/chain/f_master2.log).  c2 (11-01 16:20, after the first run stopped at
# the smoke gate): self-check gate (1) is a tolerance -- guided w = 0 must agree with plain sampling to bf16 numerics
# (<= 0.01; 198: the guided program is compiled differently, plain vs plain is exactly 0, same-prompt branches differ
# by up to 2e-3) and w = 1 must differ by more than 10x that.  Nothing else changed; launched with FROM=2 (phase 1
# passed under this gate in the first run, f_selftest_*.log).
# c1 header:  Round f of the handover 7.12 (plan and every rule
# written before any guided result): contrastive instruction guidance (CIG) at inference on the trained self-trained
# wormhole (awx3 seed 0, awx3s1 seed 1) and, with exactly the same procedure, on the matched behaviour cloning (bc3,
# bc3s1).  CIG: after the switch every flow step uses v + w (v - v_old), v_old = the same observation with the old
# instruction (for awx the two branches differ only in the [LOC] token); the client re-queries at the switch as flush.
#   0  nothing of this project running, GPUs 0-7 free
#   1  smoke: aw_server_c9 self-checks for awx and bc (w = 0 gives exactly the plain actions, w = 1 does not); one guided
#      server, 12 devR-type episodes (layout r521, outside devR) with w = 1
#   2  development splits of the 4 models x w in {0.5, 1.0}: dev2 (devg: states 40-49, t_s 12-62, n = 120) and devR
#      (procedural layouts r501-r520, n = 240) = 360 each, 2880 episodes; 8 servers (2 per model), 4 workers per server
#   3  selection per method on dev (aw_dev3_c2 --select): w* = most pooled dev (dev2 + devR, both seeds) successes, ties to
#      0.5; GO iff strictly more than the same models unguided (AW 629 / 720, BC 593 / 720)
#   4  GO methods: the test split once per model at w* (the same servers)
#   5  tables (primary: the better AW vs the better BC on the test split, pooled over seeds) and packs
source ${FLIB:-${BH:-/data/blackhole}/lens/aw/chain/f_lib.sh}
end() { echo "== $(ts) f_master END"; exit 0; }
MOD="awx3 awx3s1 bc3 bc3s1"
declare -A KIND=([awx3]=awx [awx3s1]=awx [bc3]=bc [bc3s1]=bc)
declare -A SRV=([awx3]="8741:0 8742:1" [awx3s1]="8743:2 8744:3" [bc3]="8745:4 8746:5" [bc3s1]="8747:6 8748:7")
declare -A WV=([05]=cig05 [10]=cig10)
P() { echo $A/run_$1/params_6000.npz; }
alive() { pgrep -fc 'aw_server(_c9)?[.]py|bh_client_lens[.]py|aw_train(_c11)?[.]py'; }
up_all() { local k ok=0; for k in $1; do servers_up ${KIND[$k]} $(P $k) $k "${SRV[$k]}" || ok=1; done; return $ok; }
down_all() { local k; for k in $1; do servers_down ${KIND[$k]} $(P $k) "${SRV[$k]}"; done; }
healthy_all() { local k sg; for k in $1; do for sg in ${SRV[$k]}; do curl -s -m 3 localhost:${sg%%:*}/healthz > /dev/null || return 1; done; done; }
# fin LABEL MAXMIN "NAME:N ..." "MODEL:WW:SPLIT+SPLIT ..." -- wait for the workers; if rows are missing or aborted, move
# the aborted rows aside and re-run the missing episodes once (servers must be healthy); 0 if complete
fin() {
  local lab=$1 mx=$2 want="$3" recipe="$4" names="" n x m ww sp a
  for x in $want; do names="$names ${x%%:*}"; done
  n=$(echo $want | tr ' ' '\n' | awk -F: '{s+=$2} END{print s}')
  sleep 90
  wait_workers "$lab" "$names" $n $mx
  [ "$(nwk "$names")" -eq 0 ] || { echo "$(ts) $lab: workers still running after $mx min: stopped"; kill_workers "$names"; }
  (cd $A && $PY e_check.py runs $BH/runs/lens $want) && return 0
  healthy_all "$MOD" || { echo "$(ts) $lab: incomplete and a server is down: no re-run"; return 1; }
  echo "$(ts) $lab: incomplete -> aborted rows moved to errors_*.jsonl, missing episodes re-run once"
  (cd $A && $PY e_check.py clean $BH/runs/lens $names)
  for x in $recipe; do
    m=${x%%:*}; ww=$(echo $x | cut -d: -f2); sp=$(echo $x | cut -d: -f3 | tr '+' ' ')
    a=""; for s in $sp; do a="$a $s:1"; done
    evg ${KIND[$m]} $(P $m) ${m}g$ww "${SRV[$m]}" ${WV[$ww]} $a
  done
  sleep 60
  wait_workers "$lab (re-run)" "$names" $n $mx
  [ "$(nwk "$names")" -eq 0 ] || { echo "$(ts) $lab: re-run workers still running: stopped"; kill_workers "$names"; }
  (cd $A && $PY e_check.py runs $BH/runs/lens $want)
}
FROM=${FROM:-0}
echo "== $(ts) f_master c1 start (pid $$), from phase $FROM"
(cd $A && md5sum aw_model.py aw_server.py aw_eval.sh aw_dev3.py aw_pool.py e_check.py chain/f_lib.sh chain/f_master.sh \
   ../make_specs_lens.py ../bh_client_lens.py ../bh_client_lens.py aw_model.py aw_gemma.py)

# ---------------------------------------------------------------- 0
phase0() {
for i in $(seq 1 120); do
  [ "$(alive)" -eq 0 ] && gpus_free 0 7 && break
  [ $((i % 10)) -eq 1 ] && echo "$(ts) waiting: processes of this project $(alive); GPU memory $(timeout 60 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
  sleep 30
done
{ [ "$(alive)" -eq 0 ] && gpus_free 0 7; } || { echo "GPUs 0-7 not free or processes of this project alive after 60 min: nothing started"; end; }
echo "$(ts) memory anon $(anon) GiB; $(free -g | sed -n 2p); /data $(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9) GiB free"
}

# ---------------------------------------------------------------- 1: smoke
phase1() {
echo "== $(ts) phase 1: smoke"
lock_eval
SOK=1
cd $BH/openpi
for kg in "awx:awx3:0:8790" "bc:bc3:1:8791"; do
  IFS=: read k m g p <<< "$kg"
  CUDA_VISIBLE_DEVICES=$g OPENPI_DATA_HOME=$BH/openpi_data XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 PYTHONPATH=$A timeout 900 \
    $PY $A/aw_server.py --kind $k --bb pi05 --params $(P $m) --port $p --selftest > $CH/f_selftest_$k.log 2>&1 < /dev/null &
done
wait
for k in awx bc; do
  grep -E "warm-up|CIG check|Traceback|Error" $CH/f_selftest_$k.log | cut -c1-200 | tail -n 6
  d0=$(grep -E "CIG check w=0.0" $CH/f_selftest_$k.log | sed -E 's/.*= //'); d1=$(grep -E "CIG check w=1.0" $CH/f_selftest_$k.log | sed -E 's/.*= //')
  awk -v a="$d0" -v b="$d1" 'BEGIN{exit !(a != "" && b != "" && a+0 <= 0.01 && b+0 > 1e-4 && b+0 > 10 * (a+0))}' && echo "$k: w = 0 equal to plain within bf16 numerics ($d0), w = 1 differs ($d1)" \
    || { SOK=0; echo "$k: CIG self-check FAILED (w=0: '$d0', w=1: '$d1')"; }
done
cd $A
S0="8790:0"
rm -rf $BH/runs/lens/pi05_awrsmkg_p*
mem_wait 16
if servers_up awx $(P awx3) smkg "$S0"; then
  DRL="r521" evg awx $(P awx3) smkg "$S0" cig10 devr:3
  sleep 60
  wait_workers "guided smoke" "awrsmkg" 12 25
  [ "$(nwk awrsmkg)" -eq 0 ] || { echo "smoke workers still running: stopped"; kill_workers awrsmkg; }
  $PY e_check.py runs $BH/runs/lens awrsmkg:12 || SOK=0
  $PY - <<'PY' || SOK=0
import glob, json, os
def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")
rows = [json.loads(l) for f in glob.glob(os.environ.get("BH", "/data/blackhole") + "/runs/lens/pi05_awrsmkg_p*/results_*.jsonl") for l in open(f)]
old = [json.loads(l) for f in glob.glob(os.environ.get("BH", "/data/blackhole") + "/runs/lens/pi05_awrsmk_p*/results_*.jsonl") for l in open(f)]
ok = len(rows) == 12 and all(r.get("variant") == "cig10" and r.get("w") == 1.0 and not r.get("error") for r in rows)
print("guided smoke: variant / w right:", ok, "| outcomes", {o: sum(r["outcome"] == o for r in rows) for o in {r["outcome"] for r in rows}},
      "| wall s/episode guided %.1f vs flush smoke %.1f" % (mean(r.get("wall_s") for r in rows), mean(r.get("wall_s") for r in old)),
      "| ms/query guided %.0f vs flush %.0f" % (mean(r.get("infer_ms") for r in rows), mean(r.get("infer_ms") for r in old)))
raise SystemExit(0 if ok else 1)
PY
else
  SOK=0; echo "guided smoke server not healthy"
fi
servers_down awx $(P awx3) "$S0"
unlock_eval
echo "$(ts) smoke gates passed: $SOK"
[ "$SOK" -eq 1 ] || { echo "nothing else started"; end; }
}

# ---------------------------------------------------------------- 2: development splits, 4 models x 2 guidance scales
phase2() {
echo "== $(ts) phase 2: development splits of $MOD with CIG w = 0.5 and 1.0 (dev2 + devR, 360 each)"
W=""; R=""
for m in $MOD; do for ww in 05 10; do
  for n in awd${m}g$ww awr${m}g$ww; do [ -d $BH/runs/lens/pi05_${n}_p0 ] && { echo "results of $n exist already: nothing started"; end; }; done
  W="$W awd${m}g$ww:120 awr${m}g$ww:240"; R="$R $m:$ww:devg+devr"
done; done
gpus_free 0 7 || { echo "GPUs 0-7 busy: nothing started"; end; }
lock_eval
mem_wait 52
if ! up_all "$MOD"; then echo "servers not healthy: no rollouts"; down_all "$MOD"; unlock_eval; end; fi
echo "$(ts) servers up; RssAnon (GiB): $(for m in $MOD; do for sg in ${SRV[$m]}; do printf '%s ' $(rssanon $(pidtag ${KIND[$m]} $(P $m) ${sg%%:*})); done; done); anon $(anon) GiB"
mem_wait 42
for m in $MOD; do for ww in 05 10; do evg ${KIND[$m]} $(P $m) ${m}g$ww "${SRV[$m]}" ${WV[$ww]} devg:1 devr:1; done; done
fin "dev (guided)" 150 "$W" "$R" || echo "a development split is incomplete after one re-run (the selection then excludes it)"
cd $A
pack f_dev "$(for x in $W; do printf '%s ' ${x%%:*}; done)"
}

# ---------------------------------------------------------------- 3 / 4: selection per method; test split once at w*
phase3() {
echo "== $(ts) phase 3: selection on dev (7.12)"
cd $A
$PY aw_dev3.py awx3 awx3s1 awx3g05 awx3s1g05 awx3g10 awx3s1g10 --pairs awx3g05:awx3,awx3g10:awx3,awx3s1g05:awx3s1,awx3s1g10:awx3s1 \
  --pool "awx3g05+awx3s1g05:awx3+awx3s1,awx3g10+awx3s1g10:awx3+awx3s1" \
  --select "awx3g05+awx3s1g05,awx3g10+awx3s1g10:awx3+awx3s1" --out $CH/f_sel_aw.txt | tee $CH/f_dev_aw.txt
$PY aw_dev3.py bc3 bc3s1 bc3g05 bc3s1g05 bc3g10 bc3s1g10 --pairs bc3g05:bc3,bc3g10:bc3,bc3s1g05:bc3s1,bc3s1g10:bc3s1 \
  --pool "bc3g05+bc3s1g05:bc3+bc3s1,bc3g10+bc3s1g10:bc3+bc3s1" \
  --select "bc3g05+bc3s1g05,bc3g10+bc3s1g10:bc3+bc3s1" --out $CH/f_sel_bc.txt | tee $CH/f_dev_bc.txt
SA=$(head -n 1 $CH/f_sel_aw.txt); SB=$(head -n 1 $CH/f_sel_bc.txt)
echo "$(ts) selection AW: ${SA:-none}"; echo "$(ts) selection BC: ${SB:-none}"
WA=""; WB=""
case "$SA" in GO*) WA=$(echo "$SA" | sed -E 's/^GO awx3g([0-9]+)\+.*/\1/') ;; esac
case "$SB" in GO*) WB=$(echo "$SB" | sed -E 's/^GO bc3g([0-9]+)\+.*/\1/') ;; esac
echo "guided test split: AW at w = ${WA:-none}, BC at w = ${WB:-none}"
TW=""; TR=""
for m in awx3 awx3s1; do [ -n "$WA" ] && { TW="$TW awt${m}g$WA:280"; TR="$TR $m:$WA:test"; }; done
for m in bc3 bc3s1; do [ -n "$WB" ] && { TW="$TW awt${m}g$WB:280"; TR="$TR $m:$WB:test"; }; done
if [ -n "$TW" ]; then
  echo "== $(ts) phase 4: test split once per model at w*"
  for x in $TW; do [ -d $BH/runs/lens/pi05_${x%%:*}_p0 ] && { echo "results of ${x%%:*} exist already: no test"; down_all "$MOD"; unlock_eval; end; }; done
  healthy_all "$MOD" || { echo "a server is down: no test"; down_all "$MOD"; unlock_eval; end; }
  mem_wait 30
  for x in $TR; do m=${x%%:*}; ww=$(echo $x | cut -d: -f2); evg ${KIND[$m]} $(P $m) ${m}g$ww "${SRV[$m]}" ${WV[$ww]} test:4; done
  fin "test (guided)" 90 "$TW" "$TR" || echo "a guided test split is incomplete after one re-run"
  pack f_test "$(for x in $TW; do printf '%s ' ${x%%:*}; done)"
else
  echo "no method improved on dev with guidance: no test split (7.12)"
fi
down_all "$MOD"
unlock_eval
}

# ---------------------------------------------------------------- 5: tables
phase5() {
echo "== $(ts) phase 5: tables"
cd $A
SA=$(head -n 1 $CH/f_sel_aw.txt 2>/dev/null); SB=$(head -n 1 $CH/f_sel_bc.txt 2>/dev/null)
WA=""; WB=""
case "$SA" in GO*) WA=$(echo "$SA" | sed -E 's/^GO awx3g([0-9]+)\+.*/\1/') ;; esac
case "$SB" in GO*) WB=$(echo "$SB" | sed -E 's/^GO bc3g([0-9]+)\+.*/\1/') ;; esac
AB="awx3+awx3s1"; [ -n "$WA" ] && AB="awx3g$WA+awx3s1g$WA"
BB_="bc3+bc3s1"; [ -n "$WB" ] && BB_="bc3g$WB+bc3s1g$WB"
SPECS="$AB:$BB_"
[ -n "$WA" ] && SPECS="$SPECS $AB:awx3+awx3s1"
[ -n "$WA" ] && [ -n "$WB" ] && SPECS="$SPECS $AB:bc3+bc3s1"
[ -n "$WB" ] && SPECS="$SPECS $BB_:bc3+bc3s1 awx3+awx3s1:$BB_"
if [ -n "$WA$WB" ]; then
  echo "primary (7.12): $AB vs $BB_ on the test split"
  $PY aw_pool.py $SPECS | tee $CH/f_test.txt
  for m in awx3 awx3s1; do [ -n "$WA" ] && $PY aw_summary.py awt${m}g$WA 2>/dev/null; done
  for m in bc3 bc3s1; do [ -n "$WB" ] && $PY aw_summary.py awt${m}g$WB 2>/dev/null; done
fi
(cd $A && tar czf pack/f_logs.tgz chain/f_master.log chain/f_master2.log chain/f_*.txt chain/f_selftest_*.log 2>&1 | tail -n 2; md5sum pack/f_logs.tgz)
}

for ph in 0 1 2 3 5; do [ $ph -ge $FROM ] && phase$ph; done
end
