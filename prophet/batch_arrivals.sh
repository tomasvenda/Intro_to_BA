#!/bin/bash
### LSF batch job script for Prophet Single Run (Simplified)

# Submit like:
# bsub < ./batch_arrivals.sh multiplicative 0.05 10.0

# or override environment knobs:
#   DAILY_FOURIER=20 bsub < ./batch_arrivals.sh multiplicative 0.05

# ======= QUEUE AND RESOURCES =======
#BSUB -q hpc
#BSUB -J prophet_single_run
#BSUB -n 4
#BSUB -W 1:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"

# ======= NOTIFICATIONS =======
#BSUB -B
#BSUB -N

# ======= OUTPUT FILES =======
#BSUB -o logs/prophet_single_%J.out
#BSUB -e logs/prophet_single_%J.err

set -eu

echo "=== JOB STARTED on $(date) ==="
echo "Host: $(hostname)"
echo "PWD : $(pwd)"

mode="${1:-}"
cps="${2:-}"
sps="${3:-}"

if [ -z "$mode" ] || [ -z "$cps" ] || [ -z "$sps" ]; then
  echo "ERROR: need <seasonality_mode> <changepoint_prior_scale> <seasonality_prior_scale>"
  echo "Usage: $0 multiplicative 0.05 10.0"
  exit 1
fi


mkdir -p logs

# Defaults for the simplified knobs
DAILY_FOURIER="${DAILY_FOURIER:-15}"
WEEKLY_FOURIER="${WEEKLY_FOURIER:-10}"
USE_US_HOLIDAYS="${USE_US_HOLIDAYS:-0}"

echo "Loading modules..."
module purge
module load python3/3.10.13

echo "Activating venv..."
source "$BLACKHOLE/deep_learning/deep_learning/venv_deep_learning/bin/activate"

SCRIPT_PATH="$BLACKHOLE/deep_learning/deep_learning/train_prophet.py"

# Simplified RUN_ID (removed ms, log, cap, etc.)
RUN_ID="prophet_${mode}_cps${cps}_sps${sps}_dfo${DAILY_FOURIER}_wfo${WEEKLY_FOURIER}_hol${USE_US_HOLIDAYS}"

echo "------------------------------------------------"
echo "RUN_ID          : $RUN_ID"
echo "MODE            : $mode"
echo "CPS             : $cps"
echo "DAILY_FOURIER   : $DAILY_FOURIER"
echo "WEEKLY_FOURIER  : $WEEKLY_FOURIER"
echo "USE_US_HOLIDAYS : $USE_US_HOLIDAYS"
echo "SCRIPT_PATH     : $SCRIPT_PATH"
echo "OUTPUT_DIR      : ./prophet_results"
echo "------------------------------------------------"

HOL_FLAG=""
[ "$USE_US_HOLIDAYS" = "1" ] && HOL_FLAG="--use_us_holidays"

# Only passing arguments that exist in your new Python script
python "$SCRIPT_PATH" \
  --seasonality_mode "$mode" \
  --changepoint_prior_scale "$cps" \
  --seasonality_prior_scale "$sps" \
  --job_id "$RUN_ID" \
  --output_dir "./prophet_results" \
  --daily_fourier "$DAILY_FOURIER" \
  --weekly_fourier "$WEEKLY_FOURIER" \
  $HOL_FLAG


echo "=== JOB FINISHED on $(date) ==="