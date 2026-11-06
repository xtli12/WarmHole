#!/bin/bash
# Development evaluation of a trained wormhole / control policy with plain inference (no external model, no scripted
# motion): the client only re-queries the policy when the instruction changes (method flush) or runs it unchanged
# (no-switch rollouts).  Refuses to run if a policy server is not healthy (never falls back to another model).
# usage: aw_eval.sh KIND PARAMS TAG "PORT:GPU PORT:GPU ..." NWORKERS_PER_SERVER [SPLIT] [VARIANTS]
#   SPLIT = dev (default: initial states 40-49, t_s 2-42, plus no-switch rollouts) | test (initial states 0-19, t_s 2-62)
#   | diag (development states, the wormhole diagnostics VARIANTS, e.g. aw_off,aw_tn,aw_to; no no-switch rollouts)
#   | dev2ext (development states, the late switches t_s 52 and 62 only: pi05_awe${TAG}; with the dev runs (t_s 12-42)
#     this gives the harder development split "dev2" of 10-30, t_s 12-62 as in the test split, n = 120)
#   writes $BH/specs/lens/pi05_awd${TAG}.jsonl (switch) and pi05_awp${TAG}.jsonl (no switch), results in
#   $BH/runs/lens/pi05_awd${TAG}_p$k and pi05_awp${TAG}_p$k
# c7 (11-01, handover 7.11): workers run $CLIENT (default bh_client_lens.py = lens13 + procedural layouts, identical
#   to lens12 on official initial states) and specs come from $MS (default make_specs_lens.py, identical to the
#   old generator for every old call); two new splits:
#   SPLIT = selfplay: switch self-play of the policy itself on training layouts (SPL, default 20-37,r101-124), SPK
#     rollouts per condition (default 3), every control step recorded -> pi05_sp${TAG}
#   SPLIT = devr: development split on procedural layouts (DRL, default r501-520), t_s 12-62 -> pi05_awr${TAG}
# c8 (11-01, handover 7.12): servers run $A/$AWSERVER (default aw_server.py; aw_server.py serves contrastive
#   instruction guidance), specs from make_specs_lens.py (= r21 for every old call); VARIANTS (default flush) now
#   also applies to dev2ext and devr, and SPLIT = devg is dev2 under one name (development states 40-49, t_s 12-62,
#   no no-switch rollouts, n = 120) -> pi05_awd${TAG}
BH=${BH:-/data/blackhole}
L=$BH/lens
A=$L/aw
OP=$BH/openpi
PY=$OP/.venv/bin/python
KIND=$1; PARAMS=$2; TAG=$3; SERVERS=$4; NW=$5; SPLIT=${6:-dev}; VARIANTS=${7:-flush}
# PHASE=servers: only start the servers; PHASE=workers: servers must already be up, only launch the rollouts
PHASE=${PHASE:-all}
# EGLOFF=k: workers render on EGL device (server GPU + k), e.g. -4 to render on GPUs 0-3 for servers on GPUs 4-7 (c6)
EGLOFF=${EGLOFF:-0}
BB=${BB:-pi05}                     # backbone of the trained policy: pi05 | pi0
RECF=""; [ "${REC:-0}" = "1" ] && RECF="--rec_steps"   # REC=1: record every control step of the switch episodes
CLIENT=${CLIENT:-bh_client_lens.py}   # c7
MS=${MS:-make_specs_lens.py}        # c8
AWSERVER=${AWSERVER:-aw_server.py}      # c8
[ -f "$PARAMS" ] || { echo "no params $PARAMS"; exit 1; }
cd $OP
if [ "$PHASE" != "workers" ]; then
for sg in $SERVERS; do
  port=${sg%%:*}; gpu=${sg##*:}
  curl -s -m 3 localhost:$port/healthz > /dev/null && { echo "port $port already serving: refusing"; exit 1; }
  CUDA_VISIBLE_DEVICES=$gpu OPENPI_DATA_HOME=$BH/openpi_data XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 \
    PYTHONPATH=$A setsid nohup $PY $A/$AWSERVER --kind $KIND --bb $BB --params $PARAMS --port $port \
    > $BH/logs/aw_server_$port.log 2>&1 < /dev/null &
  sleep 15
done
fi
for i in $(seq 1 60); do
  ok=1; for sg in $SERVERS; do curl -s -m 3 localhost:${sg%%:*}/healthz > /dev/null || ok=0; done
  [ $ok -eq 1 ] && break; sleep 10
done
for sg in $SERVERS; do curl -s -m 3 localhost:${sg%%:*}/healthz > /dev/null && echo "${sg%%:*} ok" || { echo "${sg%%:*} DOWN"; tail -n 20 $BH/logs/aw_server_${sg%%:*}.log; }; done
[ $ok -eq 1 ] || { echo "servers not healthy: no episodes launched"; exit 1; }
for sg in $SERVERS; do grep -h "AW policy\|warm-up 2" $BH/logs/aw_server_${sg%%:*}.log | tail -n 2 | cut -c1-200; done
[ "$PHASE" = "servers" ] && exit 0
if [ "$SPLIT" = "selfplay" ]; then
  $PY $L/$MS $BH/specs/lens --selfplay sp${TAG} --bbs $BB --sp_layouts "${SPL:-20-37,r101-124}" --sp_k ${SPK:-3} > /dev/null
  NAMES="sp${TAG}"
elif [ "$SPLIT" = "devr" ]; then
  $PY $L/$MS $BH/specs/lens --devr awr${TAG} --bbs $BB --devr_layouts "${DRL:-r501-520}" --devr_variant $VARIANTS > /dev/null
  NAMES="awr${TAG}"
elif [ "$SPLIT" = "devg" ]; then
  $PY $L/$MS $BH/specs/lens --dev $VARIANTS --name awd${TAG} --ts 12,22,32,42,52,62 --bbs $BB > /dev/null
  NAMES="awd${TAG}"
elif [ "$SPLIT" = "test" ]; then
  $PY $L/$MS $BH/specs/lens --test $VARIANTS --name awd${TAG}x --bbs $BB > /dev/null
  mv $BH/specs/lens/${BB}_lt.jsonl $BH/specs/lens/${BB}_awt${TAG}.jsonl
  NAMES="awt${TAG}"
elif [ "$SPLIT" = "dev2ext" ]; then
  $PY $L/$MS $BH/specs/lens --dev $VARIANTS --name awe${TAG} --ts 52,62 --bbs $BB $RECF > /dev/null
  NAMES="awe${TAG}"
elif [ "$SPLIT" = "diag" ]; then
  DN=${DIAGNAME:-awx${TAG}}          # spec name of the diagnostics (DIAGNAME to run another set)
  $PY $L/$MS $BH/specs/lens --dev $VARIANTS --name $DN --bbs $BB $RECF
  NAMES="$DN"
else
  $PY $L/$MS $BH/specs/lens --dev flush --name awd${TAG} --bbs $BB $RECF
  $PY $L/$MS $BH/specs/lens --plain awp${TAG} --bbs $BB
  NAMES="awd${TAG} awp${TAG}"
fi
export PYTHONPATH=$OP/third_party/libero:$OP/packages/openpi-client/src:$L:$BH/code
export LIBERO_CONFIG_PATH=$BH/libero_cfg MUJOCO_GL=egl PYOPENGL_PLATFORM=egl BH_DEPTH=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
n=$(echo $SERVERS | wc -w)
for NAME in $NAMES; do
  SPEC=$BH/specs/lens/${BB}_${NAME}.jsonl
  k=0
  for sg in $SERVERS; do
    port=${sg%%:*}; gpu=${sg##*:}
    part=$BH/specs/lens/${BB}_${NAME}_part$k.jsonl
    awk -v n=$n -v k=$k '(NR-1)%n==k' $SPEC > $part
    out=$BH/runs/lens/${BB}_${NAME}_p$k
    mkdir -p $out
    for w in $(seq 0 $((NW-1))); do
      MUJOCO_EGL_DEVICE_ID=$((gpu + EGLOFF)) setsid nohup $BH/venvs/libero/bin/python $L/$CLIENT --spec $part --out $out \
        --port $port --worker $w --nworkers $NW > $out/worker_$w.log 2>&1 < /dev/null &
    done
    echo "[$(date '+%F %T')] $NAME part $k: $(wc -l < $part) episodes -> port $port ($KIND, $PARAMS), EGL device $((gpu + EGLOFF))"
    k=$((k+1))
  done
done
