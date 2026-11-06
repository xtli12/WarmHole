# h_lib.sh -- round h (handover 7.14, kinematics-aware attractor relabeling): g_lib.sh plus teacher-only servers.
#   Sourced, not run.  The rollouts of round h resume round-g take-over snapshots with dg_k 0 (the teacher drives from
#   the first step), so only teacher servers run (lens_server.py, the released pi0.5 -- the teacher of the original data);
#   the client's --port and --teacher_port are the same server (the student port is never queried in these episodes).
source ${GLIBH:-${BH:-/data/blackhole}/lens/aw/chain/g_lib.sh}
HCHK=$A/h_check.py
# t_up "PORT:GPU ...": one teacher server per entry; 0 if all healthy
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
# t_down "PORT:GPU ...": stop exactly these teacher servers
t_down() {
  local x
  for x in $1; do pkill -f "lens_server[.]py --config pi05_libero --ckpt $CK/pi05_libero --port ${x%%:*}( |$)"; done
  sleep 20
  for x in $1; do pkill -9 -f "lens_server[.]py --config pi05_libero --ckpt $CK/pi05_libero --port ${x%%:*}( |$)"; done
  sleep 5
}
t_healthy() { local x; for x in $1; do curl -s -m 3 localhost:${x%%:*}/healthz > /dev/null || return 1; done; }
# tg "PORT:GPU ...": the same servers as dg_run / dg_finish groups SPORT:TPORT:SGPU:TGPU (student port = teacher port)
tg() { local x o=""; for x in $1; do o="$o ${x%%:*}:${x%%:*}:${x##*:}:${x##*:}"; done; echo ${o# }; }
