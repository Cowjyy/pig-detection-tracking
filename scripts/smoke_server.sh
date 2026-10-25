#!/usr/bin/env bash
# One-iteration integration checks only; all outputs are explicitly smoke runs.
set -uo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
STAGE="${1:?Use detection or tracking}"
SMOKE_ROOT="${2:?Supply a new experiments/smoke/<run> path}"
[[ "$SMOKE_ROOT" == experiments/smoke/* && "$SMOKE_ROOT" != *..* ]] || exit 2
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$SMOKE_ROOT/logs"
test ! -e "$SMOKE_ROOT/$STAGE.tsv" || exit 2
printf 'component\texit_code\n' > "$SMOKE_ROOT/$STAGE.tsv"
FAILED=0
if [[ "$STAGE" == detection ]]; then
    PYTHON="$PROJECT_ROOT/.venvs/detection/bin/python"
    read -r -a MODELS <<< "${3:-faster_rcnn retinanet fcos yolox rtmdet deformable_detr}"
    for model in "${MODELS[@]}"; do
        "$PYTHON" -B -u scripts/train_detection.py --model "$model" \
            --output "$SMOKE_ROOT/detection/$model" --smoke-iters 1 --workers 0 \
            > "$SMOKE_ROOT/logs/$model.log" 2>&1
        CODE=$?
        printf '%s\t%s\n' "$model" "$CODE" >> "$SMOKE_ROOT/$STAGE.tsv"
        [[ "$CODE" == 0 ]] || FAILED=1
    done
    if [[ "$FAILED" == 0 ]]; then
        "$PYTHON" -B -u scripts/cache_detections.py --subset tuning --smoke \
            --training-run "$SMOKE_ROOT/detection/faster_rcnn" --max-frames 5 \
            --output "$SMOKE_ROOT/cache" > "$SMOKE_ROOT/logs/cache.log" 2>&1
        CODE=$?
        printf 'detection_cache\t%s\n' "$CODE" >> "$SMOKE_ROOT/$STAGE.tsv"
        [[ "$CODE" == 0 ]] || FAILED=1
    fi
elif [[ "$STAGE" == tracking ]]; then
    PYTHON="$PROJECT_ROOT/.venvs/tracking/bin/python"
    read -r -a METHODS <<< "${3:-bytetrack botsort ocsort strongsort deepocsort boosttrack}"
    for method in "${METHODS[@]}"; do
        "$PYTHON" -B -u scripts/run_tracking.py --method "$method" --smoke \
            --cache "$SMOKE_ROOT/cache" --output "$SMOKE_ROOT/tracking/$method" \
            > "$SMOKE_ROOT/logs/$method.log" 2>&1
        CODE=$?
        printf '%s\t%s\n' "$method" "$CODE" >> "$SMOKE_ROOT/$STAGE.tsv"
        [[ "$CODE" == 0 ]] || FAILED=1
    done
else
    exit 2
fi
cat "$SMOKE_ROOT/$STAGE.tsv"
exit "$FAILED"
