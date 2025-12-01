#!/bin/sh
### LSF batch job: Evaluate Prophet ARRIVALS models for selected clusters (TEST set)

#BSUB -q hpc
#BSUB -J prophet_eval_arrivals
#BSUB -n 4
#BSUB -W 04:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o logs/prophet_eval_%J.out
#BSUB -e logs/prophet_eval_%J.err

set -eu

echo "=== JOB STARTED on $(date) ==="
echo "Host: $(hostname)"
echo "PWD : $(pwd)"

# Usage:
#   ./batch_eval_prophet_arrivals.sh "<models_dir>" "<cluster_ids>" "<run_id>" "<save_hourly:0/1>" "<recent_minutes>"
MODELS_DIR="${1:?models_dir required}"
CLUSTERS="${2:?cluster_ids required}"        # e.g. 3,7,8 or all
RUN_ID="${3:?run_id required}"
SAVE_HOURLY="${4:-0}"
RECENT_MINUTES="${5:-10}"   # default: last 10 minutes


echo "Loading modules..."
module purge
module load python3/3.10.13

echo "Activating venv..."
source "$BLACKHOLE/deep_learning/deep_learning/venv_deep_learning/bin/activate"

BASE="$BLACKHOLE/deep_learning/deep_learning"
EVAL_SCRIPT="$BASE/test_prophet_arrivals_json_models.py"
DATA_PATH="$BASE/cluster_hourly.parquet"

OUT_DIR="$BASE/prophet_eval_outputs"
mkdir -p "$OUT_DIR" "$BASE/logs"

HOURLY_FLAG=""
if [ "$SAVE_HOURLY" -eq 1 ]; then
  HOURLY_FLAG="--save_hourly"
fi

echo "----------------------------------------"
echo "RUN_ID      : $RUN_ID"
echo "MODELS_DIR  : $MODELS_DIR"
echo "CLUSTERS    : $CLUSTERS"
echo "SAVE_HOURLY : $SAVE_HOURLY"
echo "DATA_PATH   : $DATA_PATH"
echo "OUT_DIR     : $OUT_DIR"
echo "EVAL_SCRIPT : $EVAL_SCRIPT"
echo "----------------------------------------"

python "$EVAL_SCRIPT" \
  --data_path "$DATA_PATH" \
  --models_dir "$MODELS_DIR" \
  --cluster_ids "$CLUSTERS" \
  --job_id "$RUN_ID" \
  --output_dir "$OUT_DIR" \
  --recent_minutes "$RECENT_MINUTES" \
  $HOURLY_FLAG


echo "=== JOB FINISHED on $(date) ==="
