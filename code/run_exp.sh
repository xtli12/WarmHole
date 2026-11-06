#!/bin/bash
# usage: run_exp.sh CONFIG CKPT SPEC OUTDIR NWORKERS PORT [SERVER_GPU] [RENDER_GPU]
BH=${BH:-/data/blackhole}
OP=$BH/openpi
CONFIG=$1; CKPT=$2; SPEC=$3; OUT=$4; NW=$5; PORT=$6; SGPU=${7:-0}; RGPU=${8:-3}
# optional override (lets the rendering GPU be moved between stages without restarting a driver)
[ -f $BH/logs/render_gpu_$PORT ] && RGPU=$(cat $BH/logs/render_gpu_$PORT)
[ -f $BH/logs/nworkers_$PORT ] && NW=$(cat $BH/logs/nworkers_$PORT)
echo "rendering on GPU $RGPU, $NW workers"
mkdir -p $OUT $BH/logs
if ! curl -s -m 3 localhost:$PORT/healthz >/dev/null; then
  echo "starting server $CONFIG on $PORT (GPU $SGPU)"
  cd $OP
  CUDA_VISIBLE_DEVICES=$SGPU OPENPI_DATA_HOME=$BH/openpi_data XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45} \
    PYTHONPATH=$BH/code/overlay setsid nohup $OP/.venv/bin/python $BH/code/bh_server.py --config $CONFIG --ckpt $CKPT \
    --port $PORT > $BH/logs/server_$PORT.log 2>&1 < /dev/null &
  for k in $(seq 1 120); do curl -s -m 3 localhost:$PORT/healthz >/dev/null && break; sleep 5; done
fi
curl -s -m 3 localhost:$PORT/healthz || { echo "server not healthy"; tail -20 $BH/logs/server_$PORT.log; exit 1; }
export PYTHONPATH=$OP/third_party/libero:$OP/packages/openpi-client/src
export LIBERO_CONFIG_PATH=$BH/libero_cfg MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$RGPU
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for w in $(seq 0 $((NW-1))); do
  setsid nohup $BH/venvs/libero/bin/python $BH/code/bh_client.py --spec $SPEC --out $OUT --port $PORT \
     --worker $w --nworkers $NW > $OUT/worker_$w.log 2>&1 < /dev/null &
done
echo "launched $NW workers -> $OUT"
