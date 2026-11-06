#!/bin/bash
# Launch one spec file of the attention-lens experiments across several lens servers.
# usage: lens_launch.sh NAME BB "PORT:GPU PORT:GPU ..." NWORKERS_PER_SERVER [FOCUS]
# $BH/specs/lens/${BB}_${NAME}.jsonl is split round-robin; part k runs on server k -> $BH/runs/lens/${BB}_${NAME}_p$k
BH=${BH:-/data/blackhole}
NAME=$1; BB=$2; SERVERS=$3; NW=$4; FOCUS=${5:-}
CFG=${BB}_libero
CK=$BH/openpi_data/openpi-assets/checkpoints/$CFG
SPEC=$BH/specs/lens/${BB}_${NAME}.jsonl
[ -f $SPEC ] || { echo "no spec $SPEC"; exit 1; }
n=$(echo $SERVERS | wc -w); k=0
for sg in $SERVERS; do
  port=${sg%%:*}; gpu=${sg##*:}
  part=$BH/specs/lens/${BB}_${NAME}_part$k.jsonl
  awk -v n=$n -v k=$k '(NR-1)%n==k' $SPEC > $part
  # skip episodes already written by an earlier attempt
  out=$BH/runs/lens/${BB}_${NAME}_p$k
  if ls $out/results_*.jsonl >/dev/null 2>&1; then
    $BH/openpi/.venv/bin/python - $part $out <<'PY'
import glob, json, sys
part, out = sys.argv[1], sys.argv[2]
done = set()
for f in glob.glob(out + "/results_*.jsonl"):
    for l in open(f):
        try:
            done.add(json.loads(l)["id"])
        except Exception:
            pass
rows = [l for l in open(part) if json.loads(l)["id"] not in done]
open(part, "w").writelines(rows)
print(f"{len(done)} done, {len(rows)} to run")
PY
  fi
  echo "[$(date '+%F %T')] $NAME/$BB part $k: $(wc -l < $part) episodes -> port $port GPU $gpu"
  [ -s $part ] && $BH/lens/lens_run.sh $CFG $CK $part $out $NW $port $gpu $gpu $FOCUS
  k=$((k+1))
done
