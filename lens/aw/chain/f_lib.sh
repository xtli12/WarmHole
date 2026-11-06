# f_lib.sh -- round f (handover 7.12, contrastive instruction guidance on the trained policies): lib2.sh plus the helpers
#   of round f.  Servers are aw_server.py (aw_model: bh.mode = "cig"), workers bh_client_lens12 (official
#   states) or lens14 (procedural layouts), specs make_specs_lens.py.  Sourced, not run.
source ${CHLIB2:-${BH:-/data/blackhole}/lens/aw/chain/lib2.sh}
EVAL8=$A/aw_eval.sh
SRVPY=aw_server.py
G="episodes ->|DOWN|not healthy"
servers_up() {  # kind, params, tag, "port:gpu ...": start aw_server_c9 servers; 0 only if all are healthy
  local sg ok=0 n=0
  AWSERVER=$SRVPY PHASE=servers $EVAL8 $1 $2 $3 "$4" 1 | grep -E " ok$|DOWN|refus|not healthy"
  for sg in $4; do n=$((n+1)); curl -s -m 3 localhost:${sg%%:*}/healthz > /dev/null && ok=$((ok+1)); done
  echo "$(ts) servers $3 ($4): $ok/$n healthy"
  [ $ok -eq $n ]
}
servers_down() {  # kind, params, "port:gpu ...": stop exactly these servers
  local sg
  for sg in $3; do pkill -f "aw_server(_c9)?[.]py --kind $1 --bb $BBP --params $2 --port ${sg%%:*}( |$)"; done
  sleep 20
  for sg in $3; do pkill -9 -f "aw_server(_c9)?[.]py --kind $1 --bb $BBP --params $2 --port ${sg%%:*}( |$)"; done
  sleep 5
}
nwk() { pgrep -fc "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; }
kill_workers() {
  pkill -f "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; sleep 15
  pkill -9 -f "bh_client_lens[.]py --spec $BH/specs/lens/${BBP}_($(echo $1 | tr ' ' '|'))_part"; sleep 5
}
# evg KIND PARAMS TAG "PORT:GPU ..." VARIANT SPLIT:NW [SPLIT:NW ...] -- workers of these splits on running servers.
#   devg = the development split on the official states 40-49 with t_s 12-62 (dev2, n = 120) -> awd<TAG>;
#   devr (n = 240, procedural layouts) -> awr<TAG>; test (n = 280) -> awt<TAG>.  lens12 for official states (as for
#   every earlier model), lens14 for procedural layouts.
evg() {
  local k=$1 p=$2 t=$3 s="$4" v=$5 x sp nw; shift 5
  for x in "$@"; do
    sp=${x%%:*}; nw=${x##*:}
    case $sp in
      devg) CLIENT=bh_client_lens.py PHASE=workers $EVAL8 $k $p $t "$s" $nw devg $v | grep -E "$G" ;;
      devr) CLIENT=bh_client_lens.py PHASE=workers $EVAL8 $k $p $t "$s" $nw devr $v | grep -E "$G" ;;
      test) CLIENT=bh_client_lens.py PHASE=workers $EVAL8 $k $p $t "$s" $nw test $v | grep -E "$G" ;;
      *) echo "unknown split $sp" ;;
    esac
  done
}
pidtag() { pgrep -f "aw_server(_c9)?[.]py --kind $1 --bb $BBP --params $2 --port $3( |$)" | head -n 1; }
rssanon() { awk '/RssAnon/{printf "%.2f", $2/1048576}' /proc/$1/status 2>/dev/null; }   # GiB
