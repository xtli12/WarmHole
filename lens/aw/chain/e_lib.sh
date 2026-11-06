# e_lib.sh -- round e (handover 7.11, switch self-play): lib2.sh plus the helpers of round e.  Sourced, not run.
#   training with aw_train.py (category 5 + per-frame weights + --share); workers of bh_client_lens12 OR lens14.
source ${CHLIB2:-${BH:-/data/blackhole}/lens/aw/chain/lib2.sh}
TRAINPY=${TRAINPY:-$A/aw_train.py}
EVAL7=$A/aw_eval.sh
G="episodes ->|DOWN|not healthy"
start_train() {  # run name, gpus, training args...: detached; returns after step 50 or a traceback (aw_train.py)
  local k=$1 g=$2 i; shift 2
  rm -rf $A/run_$k
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$A setsid nohup $PY $TRAINPY "$@" $TR --out $A/run_$k > $A/run_$k.log 2>&1 < /dev/null &
  for i in $(seq 1 60); do grep -qE " step 50 |Traceback" $A/run_$k.log && break; sleep 10; done
}
# wait_train of lib2 matches "aw_train[.]py"; the c11 trainer is aw_train.py
wait_train() {  # run name, max minutes: returns 0 once no training process writes to run_<name>
  local i
  for i in $(seq 1 $(($2 * 2))); do
    [ "$(pgrep -fc "aw_train(_c11)?[.]py .*--out $A/run_$1( |$)")" -eq 0 ] && return 0
    [ $((i % 20)) -eq 1 ] && echo "$(ts) $1 $(grep -E ' step [0-9]+ ' $A/run_$1.log | tail -n 1 | cut -c25-110)"
    sleep 30
  done
  return 1
}
stop_train() { pkill -f "aw_train(_c11)?[.]py .*--out $A/run_$1( |$)"; }
nwk() { pgrep -fc "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; }
kill_workers() {
  pkill -f "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; sleep 15
  pkill -9 -f "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; sleep 5
}
# ev KIND PARAMS TAG "PORT:GPU ..." SPLIT:NW [SPLIT:NW ...] -- launch the workers of these splits on running servers.
#   dev (awd + awp), dev2ext (awe), test (awt): bh_client_lens.py, exactly as for every earlier model;
#   devr (awr), selfplay (sp): bh_client_lens.py (procedural layouts).  NW = workers per server and spec.
ev() {
  local k=$1 p=$2 t=$3 s="$4" x sp nw cl; shift 4
  for x in "$@"; do
    sp=${x%%:*}; nw=${x##*:}
    case $sp in dev|dev2ext|test) cl=bh_client_lens.py ;; devr|selfplay) cl=bh_client_lens.py ;; *) echo "unknown split $sp"; continue ;; esac
    CLIENT=$cl PHASE=workers $EVAL7 $k $p $t "$s" $nw $sp | grep -E "$G"
  done
}
# names of the result directories of the splits of a tag (for nwk / nres / nerr / pack)
nm() {  # tag split [split ...]
  local t=$1 x o=""; shift
  for x in "$@"; do
    case $x in dev) o="$o awd$t awp$t" ;; dev2ext) o="$o awe$t" ;; test) o="$o awt$t" ;; devr) o="$o awr$t" ;; selfplay) o="$o sp$t" ;; esac
  done
  echo $o
}
# expected episodes per split: dev 100 + no-switch 40, dev2ext 40, test 280, devr 240, selfplay 1512
nexp() { local x n=0; for x in "$@"; do case $x in dev) n=$((n+140)) ;; dev2ext) n=$((n+40)) ;; test) n=$((n+280)) ;; devr) n=$((n+240)) ;; selfplay) n=$((n+1512)) ;; esac; done; echo $n; }
pidtag() { pgrep -f "aw_server[.]py --kind $1 --bb $BBP --params $2 --port $3( |$)" | head -n 1; }
rssanon() { awk '/RssAnon/{printf "%.2f", $2/1048576}' /proc/$1/status 2>/dev/null; }   # GiB
