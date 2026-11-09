#!/usr/bin/env bash
# Run from any directory. Usage: ./run_review.sh CPU_JOB_ID [all|voices|frames|preview|check]
set -euo pipefail
EXPLAINER_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPLAINER_JOB="${1:?Pass the ID of your running CPU allocation}"
EXPLAINER_STAGE="${2:-all}"
export EXPLAINER_ENV="${EXPLAINER_ENV:-/shared/results/common/kargin/projects/krasnal-id/cache/explainer-env}"
export EXPLAINER_CACHE="${EXPLAINER_CACHE:-/shared/results/common/kargin/projects/krasnal-id/cache/explainer-review}"
export PATH="$EXPLAINER_ENV/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=4
export XDG_CACHE_HOME="$EXPLAINER_CACHE/xdg" MPLCONFIGDIR="$EXPLAINER_CACHE/matplotlib"
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 PYTHONDONTWRITEBYTECODE=1
case "$EXPLAINER_STAGE" in all|voices|frames|preview|check) ;; *) echo 'Unknown review stage' >&2; exit 2;; esac
srun --overlap --jobid "$EXPLAINER_JOB" --ntasks 1 --cpus-per-task 4 --gres=none --time 00:30:00 \
  bash -s -- "$EXPLAINER_SOURCE" "$EXPLAINER_STAGE" <<'INNER'
set -euo pipefail
cd "$1"
stage="$2"
python build_review.py docs
if [[ "$stage" == all || "$stage" == voices ]]; then python narrate.py --cache "$EXPLAINER_CACHE"; fi
if [[ "$stage" == all || "$stage" == frames ]]; then python build_review.py frames --cache "$EXPLAINER_CACHE"; fi
if [[ "$stage" == all || "$stage" == preview ]]; then python build_review.py preview --cache "$EXPLAINER_CACHE"; fi
if [[ "$stage" == all || "$stage" == check ]]; then python build_review.py check; fi
INNER
