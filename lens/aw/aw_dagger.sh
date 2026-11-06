#!/bin/bash
# DAgger data for the self-trained wormhole policy (10-30), on the TRAINING initial states 20-39 only.
# The student (a trained policy on aw_server.py) acts before the switch and dg_k control steps after it; then the
# teacher of the original data takes over from the state the student reached: the oracle wormhole wo_off_h15 (scripted
# traversal to 15 cm above the new object, own grasp offset) and, after the hand-back, the released pi0.5 (lens_server.py).
# Every control step is recorded (rec_steps); student actions are marked (src -3 / -4) and never become labels.
# usage: aw_dagger.sh KIND PARAMS NAME "SPORT:TPORT:GPU ..." NWORKERS_PER_GROUP [VARIANTS]
#   PHASE=servers: start the student and teacher servers of every group (one GPU per group); PHASE=workers: only the
#   rollouts (servers must be up).  Specs: $BH/specs/lens/pi05_${NAME}.jsonl, results: $BH/runs/lens/pi05_${NAME}_p$k
BH=${BH:-/data/blackhole}
L=$BH/lens
A=$L/aw
OP=$BH/openpi
PY=$OP/.venv/bin/python
CK=$BH/openpi_data/openpi-assets/checkpoints
KIND=$1; PARAMS=$2; NAME=$3; SGROUPS=$4; NW=$5; VARIANTS=${6:-dg10,dg30,dgb20}
PHASE=${PHASE:-all}
[ -f "$PARAMS" ] || { echo "no params $PARAMS"; exit 1; }
cd $OP
if [ "$PHASE" != "workers" ]; then
for g in $SGROUPS; do
  IFS=: read sp tp gpu <<< "$g"
  for port in $sp $tp; do curl -s -m 3 localhost:$port/healthz > /dev/null && { echo "port $port already serving: refusing"; exit 1; }; done
  CUDA_VISIBLE_DEVICES=$gpu OPENPI_DATA_HOME=$BH/openpi_data XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 \
    PYTHONPATH=$A setsid nohup $PY $A/aw_server.py --kind $KIND --bb pi05 --params $PARAMS --port $sp \
    > $BH/logs/aw_server_$sp.log 2>&1 < /dev/null &
  sleep 15
  CUDA_VISIBLE_DEVICES=$gpu OPENPI_DATA_HOME=$BH/openpi_data XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONPATH=$BH/code/overlay:$BH/code:$L setsid nohup $PY $L/lens_server.py --config pi05_libero --ckpt $CK/pi05_libero \
    --port $tp > $BH/logs/lens_server_$tp.log 2>&1 < /dev/null &
  sleep 15
done
fi
for i in $(seq 1 60); do
  ok=1; for g in $SGROUPS; do IFS=: read sp tp gpu <<< "$g"; for port in $sp $tp; do curl -s -m 3 localhost:$port/healthz > /dev/null || ok=0; done; done
  [ $ok -eq 1 ] && break; sleep 10
done
for g in $SGROUPS; do IFS=: read sp tp gpu <<< "$g"
  curl -s -m 3 localhost:$sp/healthz > /dev/null && echo "student $sp ok" || { echo "student $sp DOWN"; tail -n 20 $BH/logs/aw_server_$sp.log; }
  curl -s -m 3 localhost:$tp/healthz > /dev/null && echo "teacher $tp ok" || { echo "teacher $tp DOWN"; tail -n 20 $BH/logs/lens_server_$tp.log; }
done
[ $ok -eq 1 ] || { echo "servers not healthy: no episodes launched"; exit 1; }
[ "$PHASE" = "servers" ] && exit 0
$PY $L/make_specs_lens.py $BH/specs/lens --dagger $VARIANTS --name $NAME --bbs pi05 \
  --vlm_offset '{"pi05": [-0.017, 0.044], "pi0": [-0.012, 0.044]}'
export PYTHONPATH=$OP/third_party/libero:$OP/packages/openpi-client/src:$L:$BH/code
export LIBERO_CONFIG_PATH=$BH/libero_cfg MUJOCO_GL=egl PYOPENGL_PLATFORM=egl BH_DEPTH=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SPEC=$BH/specs/lens/pi05_${NAME}.jsonl
n=$(echo $SGROUPS | wc -w)
k=0
for g in $SGROUPS; do
  IFS=: read sp tp gpu <<< "$g"
  part=$BH/specs/lens/pi05_${NAME}_part$k.jsonl
  awk -v n=$n -v k=$k '(NR-1)%n==k' $SPEC > $part
  out=$BH/runs/lens/pi05_${NAME}_p$k
  mkdir -p $out
  for w in $(seq 0 $((NW-1))); do
    MUJOCO_EGL_DEVICE_ID=$gpu setsid nohup $BH/venvs/libero/bin/python $L/bh_client_lens.py --spec $part --out $out \
      --port $sp --teacher_port $tp --worker $w --nworkers $NW > $out/worker_$w.log 2>&1 < /dev/null &
  done
  echo "[$(date '+%F %T')] $NAME part $k: $(wc -l < $part) episodes -> student $sp, teacher $tp (GPU $gpu; $KIND, $PARAMS)"
  k=$((k+1))
done
