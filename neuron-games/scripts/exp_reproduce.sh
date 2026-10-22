#!/usr/bin/env bash
# Restore and rerun any recorded experiment exactly.
#
# Usage (from neuron-games/):
#   scripts/exp_reproduce.sh experiments/batch_NN/<ID>_<slug> [--checkout-only] [--out <run_dir>]
#
# Reads <experiment>/experiment.json (commit + exact command), creates a git
# worktree of that commit under experiments/_restore/<batch>-<ID>-<commit7>/,
# and reruns the same command there (same settings, same seed) into a new run
# directory. --checkout-only stops after creating the worktree, to inspect or
# branch from that exact version of the code. The original run is untouched.
set -euo pipefail

[ $# -ge 1 ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
EXP_DIR="$(cd "$1" && pwd)"; shift
CHECKOUT_ONLY=0; OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --checkout-only) CHECKOUT_ONLY=1; shift ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done

NG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"; PY="${PY:-python3}"
read -r BATCH ID COMMIT < <("$PY" -c "import json,sys; e=json.load(open(sys.argv[1])); print(e['batch'], e['id'], e['commit'])" "$EXP_DIR/experiment.json")

WT="$NG_ROOT/experiments/_restore/${BATCH}-${ID}-${COMMIT:0:7}"
if [ ! -d "$WT" ]; then
    git -C "$NG_ROOT" worktree add --detach "$WT" "$COMMIT"
fi
echo "[reproduce] code of batch $BATCH / $ID (commit $COMMIT) is at: $WT"
[ "$CHECKOUT_ONLY" -eq 1 ] && exit 0

OUT="${OUT:-$NG_ROOT/experiments/_restore/runs/${BATCH}-${ID}-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
# Same command, new run directory; MNIST is reused from the main checkout.
mapfile -t CMD < <("$PY" - "$EXP_DIR/experiment.json" "$OUT" "$NG_ROOT/data/raw" <<'PY'
import json, sys
e = json.load(open(sys.argv[1])); out, data = sys.argv[2], sys.argv[3]
cmd, res, i = e["command"][1:], [], 0
while i < len(cmd):
    if cmd[i] == "--run-dir":
        res += ["--run-dir", out]; i += 2
    else:
        res.append(cmd[i]); i += 1
if "--data-dir" not in res:
    res += ["--data-dir", data]
print("\n".join(res))
PY
)
echo "[reproduce] rerunning: $PY ${CMD[*]}"
cd "$WT/neuron-games"
"$PY" "${CMD[@]}"
echo "[reproduce] results: $OUT/results.json"
