# j_lib.sh -- round j (handover 7.16, kinematically safe wormhole exit): g_lib.sh plus teacher-only servers (as h_lib)
#   and rollouts on bh_client_lens.py (lens_margin).  Sourced, not run.  The teacher episodes need only the released
#   pi0.5 (lens_server.py); the client's --port and --teacher_port are the same server.
source ${GLIBJ:-${BH:-/data/blackhole}/lens/aw/chain/g_lib.sh}
MS25=$L/make_specs_lens.py
TRAIN7=$A/aw_build_data.py
CL17=$L/bh_client_lens.py
JCHK=$A/j_check.py
t_up() {
  local x p g i ok n
  for x in $1; do p=${x%%:*}; curl -s -m 3 localhost:$p/healthz > /dev/null && { echo "port $p already serving: refusing"; return 1; }; done
  for x in $1; do
    p=${x%%:*}; g=${x##*:}
    (cd $OP && CUDA_VISIBLE_DEVICES=$g OPENPI_DATA_HOME=$BH/openpi_data XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      PYTHONPATH=$BH/code/overlay:$BH/code:$L setsid nohup $PY $L/lens_server.py --config pi05_libero --ckpt $CK/pi05_libero \
      --port $p > $BH/logs/lens_server_$p.log 2>&1 < /dev/null &)
    sleep 10
  done
  for i in $(seq 1 60); do
    ok=0; n=0
    for x in $1; do n=$((n+1)); curl -s -m 3 localhost:${x%%:*}/healthz > /dev/null && ok=$((ok+1)); done
    [ $ok -eq $n ] && break
    sleep 10
  done
  for x in $1; do curl -s -m 3 localhost:${x%%:*}/healthz > /dev/null || { echo "teacher ${x%%:*} DOWN"; tail -n 5 $BH/logs/lens_server_${x%%:*}.log | cut -c1-200; }; done
  echo "$(ts) teacher servers ($1): $ok/$n healthy; anon $(anon) GiB"
  [ $ok -eq $n ]
}
t_down() {
  local x
  for x in $1; do pkill -f "lens_server[.]py --config pi05_libero --ckpt $CK/pi05_libero --port ${x%%:*}( |$)"; done
  sleep 20
  for x in $1; do pkill -9 -f "lens_server[.]py --config pi05_libero --ckpt $CK/pi05_libero --port ${x%%:*}( |$)"; done
  sleep 5
}
t_healthy() { local x; for x in $1; do curl -s -m 3 localhost:${x%%:*}/healthz > /dev/null || return 1; done; }
tg() { local x o=""; for x in $1; do o="$o ${x%%:*}:${x%%:*}:${x##*:}:${x##*:}"; done; echo ${o# }; }
dgwk() { pgrep -fc "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; }
dg_kill() {
  pkill -f "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; sleep 15
  pkill -9 -f "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; sleep 5
}
dg_healthy() { local g sp tp sg tg; for g in $1; do IFS=: read sp tp sg tg <<< "$g"; for port in $sp $tp; do curl -s -m 3 localhost:$port/healthz > /dev/null || return 1; done; done; }
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
       MUJOCO_EGL_DEVICE_ID=$sg setsid nohup $BH/venvs/libero/bin/python $CL17 --spec $part --out $out \
         --port $sp --teacher_port $tp --worker $w --nworkers $nw > $out/worker_$w.log 2>&1 < /dev/null &)
    done
    echo "$(ts) $name part $k: $(wc -l < $part) episodes -> teacher $sp ($nw workers)"
    k=$((k+1))
  done
}
# val0s NAME GPU DATA KIND SEED: single-card initial-model validation (the fingerprint of a new dataset) -> $A/val0_NAME.log
val0s() {
  CUDA_VISIBLE_DEVICES=$2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$A timeout 1200 $PY $A/aw_train.py --data $3 --kind $4 \
    --bs 32 --lam_loc 0.1 --lam_coord 0.2 --seed $5 --val_only --out $A/val0_$1 > $A/val0_$1.log 2>&1 < /dev/null
  grep -E "VAL0ONLY|Traceback" $A/val0_$1.log | cut -c25-300
}
