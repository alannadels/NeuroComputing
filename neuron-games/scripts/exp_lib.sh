#!/usr/bin/env bash
# Shared runner for experiment batches (Iterative Experiment Refinement).
# Sourced by experiments/batch_NN/run_batch_NN.sh; see docs/EXPERIMENTS.md.
#
#   source "$(dirname "$0")/../../scripts/exp_lib.sh"
#   batch_init 01 "short title"
#   run_exp E01 three_factor "what it tests" --ref C01 -- --model three_factor
#   ...
#   batch_finish
#
# Guarantees, so any result can be traced back and rerun exactly:
#   - refuses to run unless the working tree is clean (code == a commit);
#   - tags that commit exp/batch-NN and stores a source snapshot of it;
#   - records environment (python, packages, GPU) and the test-suite result;
#   - writes experiment.json per experiment: id, description, reference,
#     commit, exact command, overrides, status, timings;
#   - generates BATCH_REPORT.md + batch_results.json and appends every
#     experiment to experiments/VERSIONS.md (scripts/exp_report.py).
set -uo pipefail

EXP_NG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # neuron-games/
EXP_ROOT="$EXP_NG_ROOT/experiments"
EXP_EPOCHS="${EXP_EPOCHS:-50}"
EXP_SEED="${EXP_SEED:-1}"
EXP_DEVICE="${EXP_DEVICE:-cuda}"
EXP_DATASET="${EXP_DATASET:-mnist}"

_exp_die() { echo "[batch] ERROR: $*" >&2; exit 2; }

_exp_python() {
    if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then echo "$VIRTUAL_ENV/bin/python"
    else echo python3; fi
}

batch_init() {   # batch_init <NN> <title>
    BATCH_ID="$1"; BATCH_TITLE="$2"
    BATCH_DIR="$EXP_ROOT/batch_$BATCH_ID"
    PY="$(_exp_python)"
    mkdir -p "$BATCH_DIR" || _exp_die "cannot create $BATCH_DIR"
    cd "$EXP_NG_ROOT" || _exp_die "cannot cd to $EXP_NG_ROOT"

    # 1. code must be exactly a commit
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        git status --short --untracked-files=no
        _exp_die "working tree has uncommitted changes; commit them first so the batch is reproducible"
    fi
    BATCH_COMMIT="$(git rev-parse HEAD)"
    BATCH_TAG="exp/batch-$BATCH_ID"
    if git rev-parse -q --verify "refs/tags/$BATCH_TAG" >/dev/null; then
        [ "$(git rev-parse "$BATCH_TAG^{commit}")" = "$BATCH_COMMIT" ] \
            || _exp_die "tag $BATCH_TAG already exists on a different commit; use a new batch number"
    else
        # lightweight tag: needs no git identity; the title is in batch_meta.json
        git tag "$BATCH_TAG" "$BATCH_COMMIT" || _exp_die "cannot create tag $BATCH_TAG"
    fi

    # 2. log everything from here on
    exec 3>&1
    exec > >(tee -a "$BATCH_DIR/batch.log") 2>&1
    BATCH_STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[batch $BATCH_ID] $BATCH_TITLE"
    echo "[batch $BATCH_ID] commit $BATCH_COMMIT (tag $BATCH_TAG) python $PY"
    echo "[batch $BATCH_ID] defaults: dataset=$EXP_DATASET epochs=$EXP_EPOCHS seed=$EXP_SEED device=$EXP_DEVICE"

    # 3. source snapshot + environment
    mkdir -p "$BATCH_DIR/env"
    git -C "$EXP_NG_ROOT/.." archive --format=tar.gz -o "$BATCH_DIR/env/source_${BATCH_COMMIT:0:7}.tar.gz" "$BATCH_COMMIT" \
        || _exp_die "cannot snapshot source"
    "$PY" -m pip freeze > "$BATCH_DIR/env/packages.txt" 2>/dev/null \
        || uv pip freeze --python "$PY" > "$BATCH_DIR/env/packages.txt" 2>/dev/null \
        || echo "(package list unavailable)" > "$BATCH_DIR/env/packages.txt"
    "$PY" - > "$BATCH_DIR/env/gpu.txt" 2>&1 <<'PY' || true
import sys, torch
print("python:", sys.version.split()[0], "| torch:", torch.__version__, "| cuda:", torch.version.cuda)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu: {p.name} | {p.total_memory / 2**30:.1f} GiB")
PY

    # 4. tests must pass before anything runs
    echo "[batch $BATCH_ID] running test suite"
    "$PY" -m pytest tests -q > "$BATCH_DIR/env/pytest.txt" 2>&1 \
        || { tail -20 "$BATCH_DIR/env/pytest.txt"; _exp_die "tests failed"; }
    tail -1 "$BATCH_DIR/env/pytest.txt"

    "$PY" - "$BATCH_DIR/batch_meta.json" "$BATCH_ID" "$BATCH_TITLE" "$BATCH_COMMIT" "$BATCH_TAG" \
        "$BATCH_STARTED" "$EXP_DATASET" "$EXP_EPOCHS" "$EXP_SEED" "$EXP_DEVICE" <<'PY'
import json, sys
out, bid, title, commit, tag, started, ds, ep, seed, dev = sys.argv[1:]
json.dump({"batch": bid, "title": title, "commit": commit, "tag": tag, "started_utc": started,
           "defaults": {"dataset": ds, "epochs": int(ep), "seed": int(seed), "device": dev}},
          open(out, "w"), indent=2)
PY
    BATCH_N=0; BATCH_FAILED=()
}

run_exp() {   # run_exp <ID> <slug> <description> [--ref <ID>] -- <run_snn.py args>
    local id="$1" slug="$2" desc="$3"; shift 3
    local ref=""
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do
        case "$1" in --ref) ref="$2"; shift 2 ;; *) _exp_die "run_exp: unknown option $1" ;; esac
    done
    [ "${1:-}" = "--" ] && shift
    local run_dir="$BATCH_DIR/${id}_${slug}"
    [ -e "$run_dir/results.json" ] && { echo "[batch] $id already done, skipping"; return 0; }
    mkdir -p "$run_dir"
    local cmd=("$PY" -u scripts/run_snn.py --mode train --dataset "$EXP_DATASET" --device "$EXP_DEVICE"
               --epochs "$EXP_EPOCHS" --seed "$EXP_SEED" --run-dir "$run_dir" --progress "$@")
    BATCH_N=$((BATCH_N + 1))
    echo
    echo "=== [$BATCH_ID/$id] $desc"
    echo "    ${cmd[*]}"
    local t0; t0=$(date +%s)
    local started; started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    NG_PROGRESS_FD=3 "${cmd[@]}" 2>&1 | tee "$run_dir/train.log" \
        | grep --line-buffered -E "^\[(epoch [0-9]+/[0-9]+|lr|audit|done)\]" | grep --line-buffered -v artifacts
    local rc=${PIPESTATUS[0]}
    local explain_rc=-1
    if [ "$rc" -eq 0 ]; then
        PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m neuron_games explain --run "$run_dir" \
            --device "$EXP_DEVICE" > "$run_dir/explain.log" 2>&1
        explain_rc=$?
        [ "$explain_rc" -eq 0 ] || echo "[batch] explain failed for $id (see $run_dir/explain.log)"
    else
        BATCH_FAILED+=("$id (exit $rc)")
        echo "[batch] FAILED: $id (exit $rc)"; tail -15 "$run_dir/train.log"
    fi
    "$PY" - "$run_dir/experiment.json" "$BATCH_ID" "$id" "$slug" "$desc" "$ref" "$BATCH_COMMIT" \
        "$BATCH_TAG" "$started" "$(( $(date +%s) - t0 ))" "$rc" "$explain_rc" -- "${cmd[@]}" <<'PY'
import json, sys
a = sys.argv[1:]
sep = a.index("--")
out, bid, eid, slug, desc, ref, commit, tag, started, wall, rc, erc = a[:sep]
cmd = a[sep + 1:]
# user-facing overrides = everything after the fixed run_snn.py arguments
fixed = {"--mode", "--dataset", "--device", "--epochs", "--seed", "--run-dir"}
overrides, i, args = [], 0, cmd[3:]
while i < len(args):
    if args[i] in fixed:
        i += 2
    elif args[i] == "--progress":
        i += 1
    else:
        overrides.append(args[i]); i += 1
json.dump({"batch": bid, "id": eid, "slug": slug, "description": desc, "reference": ref or None,
           "commit": commit, "tag": tag, "started_utc": started, "wall_s": int(wall),
           "exit_code": int(rc), "explain_exit_code": int(erc),
           "command": cmd, "overrides": overrides}, open(out, "w"), indent=2)
PY
}

batch_finish() {
    echo
    echo "[batch $BATCH_ID] generating report"
    "$PY" scripts/exp_report.py "$BATCH_DIR" || echo "[batch] report generation failed"
    echo "[batch $BATCH_ID] done: $((BATCH_N - ${#BATCH_FAILED[@]}))/$BATCH_N experiments succeeded"
    echo "[batch $BATCH_ID] report: $BATCH_DIR/BATCH_REPORT.md"
    if [ ${#BATCH_FAILED[@]} -gt 0 ]; then
        echo "[batch $BATCH_ID] failed:"; for f in "${BATCH_FAILED[@]}"; do echo "  - $f"; done
        return 1
    fi
}
