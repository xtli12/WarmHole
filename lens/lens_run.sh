#!/bin/bash
# usage: lens_run.sh CONFIG CKPT SPEC OUTDIR NWORKERS PORT [SERVER_GPU] [RENDER_GPU] [FOCUS]
# Like run_exp.sh, but the policy server is lens_server.py (all bh_server modes plus the attention-lens modes)
# and the clients run bh_client_lens.py with depth rendering on (BH_DEPTH=1).
BH=${BH:-/data/blackhole}
OP=$BH/openpi
L=$BH/lens
CONFIG=$1; CKPT=$2; SPEC=$3; OUT=$4; NW=$5; PORT=$6; SGPU=${7:-0}; RGPU=${8:-$SGPU}; FOCUS=${9:-}
mkdir -p $OUT $BH/logs
if ! curl -s -m 3 localhost:$PORT/healthz >/dev/null; then
  echo "starting lens server $CONFIG on $PORT (GPU $SGPU)"
  cd $OP
  CUDA_VISIBLE_DEVICES=$SGPU OPENPI_DATA_HOME=$BH/openpi_data XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$BH/code/overlay:$BH/code:$L setsid nohup $OP/.venv/bin/python \
    $L/lens_server.py --config $CONFIG --ckpt $CKPT --port $PORT ${FOCUS:+--focus $FOCUS} ${LENS_W:+--lens_w $LENS_W} \
    > $BH/logs/lens_server_$PORT.log 2>&1 < /dev/null &
  for k in $(seq 1 160); do curl -s -m 3 localhost:$PORT/healthz >/dev/null && break; sleep 5; done
fi
curl -s -m 3 localhost:$PORT/healthz || { echo "server not healthy"; tail -30 $BH/logs/lens_server_$PORT.log; exit 1; }
export PYTHONPATH=$OP/third_party/libero:$OP/packages/openpi-client/src:$L:$BH/code
export LIBERO_CONFIG_PATH=$BH/libero_cfg MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$RGPU BH_DEPTH=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for w in $(seq 0 $((NW-1))); do
  setsid nohup $BH/venvs/libero/bin/python $L/${LENS_CLIENT:-bh_client_lens.py} --spec $SPEC --out $OUT --port $PORT \
     --worker $w --nworkers $NW > $OUT/worker_$w.log 2>&1 < /dev/null &
done
echo "launched $NW workers -> $OUT"
