# g_lib.sh -- round g (handover 7.13, counterfactual attractor relabeling): e_lib.sh plus the helpers of round g.
#   Sourced, not run.  Rollouts of round g: bh_client_lens.py (conditional DAgger take-over, counterfactual branch from
#   a stored simulator state) with a student server (aw_server.py, the trained policy) and a teacher server
#   (lens_server.py, the released pi0.5 -- the teacher of the original data) per group; specs make_specs_lens.py.
source ${ELIBG:-${BH:-/data/blackhole}/lens/aw/chain/e_lib.sh}
OP=$BH/openpi
CK=$BH/openpi_data/openpi-assets/checkpoints
MS23=$L/make_specs_lens.py
OFF='{"pi05": [-0.017, 0.044], "pi0": [-0.012, 0.044]}'     # the teacher's own grasp offset (as for the teacher data)
TRAIN6=$A/aw_build_data.py
# dg_up KIND PARAMS "SPORT:TPORT:SGPU:TGPU ...": start the student and teacher servers of these groups; 0 if all healthy
dg_up() {
  local k=$1 p=$2 g sp tp sg tg i ok n
  for g in $3; do
    IFS=: read sp tp sg tg <<< "$g"
    for port in $sp $tp; do curl -s -m 3 localhost:$port/healthz > /dev/null && { echo "port $port already serving: refusing"; return 1; }; done
    (cd $OP && CUDA_VISIBLE_DEVICES=$sg OPENPI_DATA_HOME=$BH/openpi_data XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 \
      PYTHONPATH=$A setsid nohup $PY $A/aw_server.py --kind $k --bb $BBP --params $p --port $sp \
      > $BH/logs/aw_server_$sp.log 2>&1 < /dev/null &)
    sleep 10
    (cd $OP && CUDA_VISIBLE_DEVICES=$tg OPENPI_DATA_HOME=$BH/openpi_data XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      PYTHONPATH=$BH/code/overlay:$BH/code:$L setsid nohup $PY $L/lens_server.py --config pi05_libero --ckpt $CK/pi05_libero \
      --port $tp > $BH/logs/lens_server_$tp.log 2>&1 < /dev/null &)
    sleep 10
  done
  for i in $(seq 1 60); do
    ok=0; n=0
    for g in $3; do IFS=: read sp tp sg tg <<< "$g"; for port in $sp $tp; do n=$((n+1)); curl -s -m 3 localhost:$port/healthz > /dev/null && ok=$((ok+1)); done; done
    [ $ok -eq $n ] && break
    sleep 10
  done
  for g in $3; do IFS=: read sp tp sg tg <<< "$g"
    curl -s -m 3 localhost:$sp/healthz > /dev/null && echo "student $sp ok" || { echo "student $sp DOWN"; tail -n 5 $BH/logs/aw_server_$sp.log | cut -c1-200; }
    curl -s -m 3 localhost:$tp/healthz > /dev/null && echo "teacher $tp ok" || { echo "teacher $tp DOWN"; tail -n 5 $BH/logs/lens_server_$tp.log | cut -c1-200; }
  done
  echo "$(ts) groups $k ($3): $ok/$n servers healthy"
  [ $ok -eq $n ]
}
# dg_down KIND PARAMS "SPORT:TPORT:SGPU:TGPU ...": stop exactly these servers
dg_down() {
  local g sp tp sg tg
  for g in $3; do IFS=: read sp tp sg tg <<< "$g"
    pkill -f "aw_server[.]py --kind $1 --bb $BBP --params $2 --port $sp( |$)"
    pkill -f "lens_server[.]py --config pi05_libero --ckpt $CK/pi05_libero --port $tp( |$)"
  done
  sleep 20
  for g in $3; do IFS=: read sp tp sg tg <<< "$g"
    pkill -9 -f "aw_server[.]py --kind $1 --bb $BBP --params $2 --port $sp( |$)"
    pkill -9 -f "lens_server[.]py --config pi05_libero --ckpt $CK/pi05_libero --port $tp( |$)"
  done
  sleep 5
}
dg_healthy() { local g sp tp sg tg; for g in $1; do IFS=: read sp tp sg tg <<< "$g"; for port in $sp $tp; do curl -s -m 3 localhost:$port/healthz > /dev/null || return 1; done; done; }
dgwk() { pgrep -fc "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; }
dg_kill() {
  pkill -f "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; sleep 15
  pkill -9 -f "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; sleep 5
}
# dg_run NAME "SPORT:TPORT:SGPU:TGPU ..." NW: split $BH/specs/lens/pi05_NAME.jsonl over the groups (part k -> group k),
#   NW workers per group; results in $BH/runs/lens/pi05_NAME_p<k>.  Re-running skips the episodes already in results.
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
       MUJOCO_EGL_DEVICE_ID=$sg setsid nohup $BH/venvs/libero/bin/python $L/bh_client_lens.py --spec $part --out $out \
         --port $sp --teacher_port $tp --worker $w --nworkers $nw > $out/worker_$w.log 2>&1 < /dev/null &)
    done
    echo "$(ts) $name part $k: $(wc -l < $part) episodes -> student $sp, teacher $tp ($nw workers)"
    k=$((k+1))
  done
}
# dg_wait LABEL "NAMES" EXPECTED MAXMIN: wait for the round-g workers of these runs
dg_wait() {
  local i
  for i in $(seq 1 $(($4 * 4))); do
    [ "$(dgwk "$2")" -eq 0 ] && break
    [ $((i % 40)) -eq 1 ] && echo "$(ts) $1: workers $(dgwk "$2"), results $(nres "$2")/$3, errors $(nerr "$2"); memory anon $(anon) GiB; /data $(df -BG --output=avail /data | tail -n 1 | tr -dc 0-9) GiB free"
    sleep 15
  done
  echo "$(ts) $1: workers left $(dgwk "$2"); results $(nres "$2")/$3; errors $(nerr "$2"); $(ps -eo stat=,wchan:40= | awk '$1 ~ /^D/ && $2 ~ /mem_cgroup/' | wc -l) processes throttled by the memory cgroup"
}
# dg_finish LABEL "NAME:N ..." "GROUPS_OF_NAME ..." NW MAXMIN: wait; if rows are missing or aborted, move the aborted rows
#   aside and re-run the missing episodes once (servers must be healthy); 0 if complete.  $3 = "NAME=GROUPS;..." pairs
dg_finish() {
  local lab=$1 want="$2" spec="$3" nw=$4 mx=$5 names="" n x nm gr
  for x in $want; do names="$names ${x%%:*}"; done
  n=$(echo $want | tr ' ' '\n' | awk -F: '{s+=$2} END{print s}')
  sleep 60
  dg_wait "$lab" "$names" $n $mx
  [ "$(dgwk "$names")" -eq 0 ] || { echo "$(ts) $lab: workers still running after $mx min: stopped"; dg_kill "$names"; }
  (cd $A && $PY e_check.py runs $BH/runs/lens $want) && return 0
  echo "$(ts) $lab: incomplete -> aborted rows moved to errors_*.jsonl, missing episodes re-run once"
  (cd $A && $PY e_check.py clean $BH/runs/lens $names)
  IFS=';' read -ra PAIRS_ <<< "$spec"
  for x in "${PAIRS_[@]}"; do nm=${x%%=*}; gr=${x#*=}; dg_healthy "$gr" && dg_run $nm "$gr" $nw || echo "$(ts) $nm: a server is down, not re-run"; done
  sleep 60
  dg_wait "$lab (re-run)" "$names" $n $mx
  [ "$(dgwk "$names")" -eq 0 ] || { echo "$(ts) $lab: re-run workers still running: stopped"; dg_kill "$names"; }
  (cd $A && $PY e_check.py runs $BH/runs/lens $want)
}
