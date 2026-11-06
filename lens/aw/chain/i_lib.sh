# i_lib.sh -- round i (handover 7.15, switch-state relabeling with a kinematics-aware teacher): g_lib.sh with the
#   rollouts on bh_client_lens.py (the B->A corridor of the teacher, lens_corridor) and specs make_specs_lens.py.
#   Sourced, not run.  Overrides dgwk / dg_kill / dg_run of g_lib.sh (dg_up, dg_down, dg_wait, dg_finish unchanged:
#   they call these by name).
source ${GLIBI:-${BH:-/data/blackhole}/lens/aw/chain/g_lib.sh}
MS24=$L/make_specs_lens.py
TRAIN7=$A/aw_build_data.py
CL16=$L/bh_client_lens.py
dgwk() { pgrep -fc "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; }
dg_kill() {
  pkill -f "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; sleep 15
  pkill -9 -f "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; sleep 5
}
# dg_run NAME "SPORT:TPORT:SGPU:TGPU ..." NW: as in g_lib.sh, with bh_client_lens.py
dg_run() {
  local name=$1 groups="$2" nw=$3 spec=$BH/specs/lens/${BBP}_$1.jsonl n k g sp tp sg tg part out w
  n=$(echo $groups | wc -w); k=0
  for g in $groups; do
    IFS=: read sp tp sg tg <<< "$g"
    part=$BH/specs/lens/${BBP}_${name}_part$k.jsonl
    awk -v n=$n -v k=$k '(NR-1)%n==k' $spec > $part
    out=$BH/runs/lens/${BBP}_${name}_p$k
    mkdir -p $out
    for w in $(seq 0 $((nw-1))); do
      (export PYTHONPATH=$OP/third_party/libero:$OP/packages/openpi-client/src:$L:$BH/code
       export LIBERO_CONFIG_PATH=$BH/libero_cfg MUJOCO_GL=egl PYOPENGL_PLATFORM=egl BH_DEPTH=1
       export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
       MUJOCO_EGL_DEVICE_ID=$sg setsid nohup $BH/venvs/libero/bin/python $CL16 --spec $part --out $out \
         --port $sp --teacher_port $tp --worker $w --nworkers $nw > $out/worker_$w.log 2>&1 < /dev/null &)
    done
    echo "$(ts) $name part $k: $(wc -l < $part) episodes -> student $sp, teacher $tp ($nw workers)"
    k=$((k+1))
  done
}
