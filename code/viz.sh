#!/bin/bash
# Runtime internals for the mechanism figures (attention over prefix keys; Wormhole internals at a switch).
BH=${BH:-/data/blackhole}; OP=$BH/openpi; LOG=$BH/logs/viz.log
G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
echo "[$(date '+%F %T')] using GPU $G" >> $LOG
cd $OP
B="pick up the black bowl on the wooden cabinet and place it on the plate"
for bb in pi05 pi0; do
  CFG=${bb}_libero; CK=$BH/openpi_data/openpi-assets/checkpoints/$CFG
  CUDA_VISIBLE_DEVICES=$G OPENPI_DATA_HOME=$BH/openpi_data XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
    PYTHONPATH=$BH/code/overlay:$BH/code .venv/bin/python $BH/code/viz_runtime.py --config $CFG --ckpt $CK \
    --obs $BH/runs/${bb}_e0/obs/${bb}_e0_spAB_old_i0.npz --traj $BH/runs/${bb}_e0/traj/${bb}_e0_spAB_old_i0.npz \
    --new_prompt "$B" --library $BH/lib/${bb}_spAB.npz --attn_steps 0,5,10,20,30,40 --switch_step 30 \
    --out $BH/viz/${bb}_viz_spAB_i0.npz >> $LOG 2>&1
done
echo "[$(date '+%F %T')] viz done" >> $LOG
touch $BH/logs/viz.done
