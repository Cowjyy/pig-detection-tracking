#!/usr/bin/env bash
# Complete the dedicated environments; never install into the shared base env.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
DETECTION_PYTHON="$PROJECT_ROOT/.venvs/detection/bin/python"
TRACKING_PYTHON="$PROJECT_ROOT/.venvs/tracking/bin/python"
test -x "$DETECTION_PYTHON"
test -x "$TRACKING_PYTHON"
mkdir -p experiments/setup .cache/matplotlib
"$DETECTION_PYTHON" -m pip install --no-deps --no-build-isolation -e tools/vendor/mmdetection
"$DETECTION_PYTHON" -m pip check
"$DETECTION_PYTHON" -m pip freeze > experiments/setup/detection.lock.txt
"$TRACKING_PYTHON" -m pip install -c configs/environments/tracking-constraints.txt \
    torchvision==0.17.2 opencv-python==4.10.0.84 matplotlib==3.8.4 pycocotools==2.0.7
"$TRACKING_PYTHON" -m pip install -c configs/environments/tracking-constraints.txt -e tools/vendor/boxmot
"$TRACKING_PYTHON" -m pip check
"$TRACKING_PYTHON" -m pip freeze > experiments/setup/tracking.lock.txt
printf '\nPROJECT_DEPENDENCIES_READY\n'
