#!/bin/bash
set -eu

mkdir -p logs

JOB_SCRIPT="./batch_arrivals.sh"

# ----------------------------
# GRID SEARCH (Simplified)
# ----------------------------

# Prophet knobs to sweep
modes=("additive" "multiplicative") # Keeping only additive as discussed
cps_values=(0.05 0.1 0.2)
sps_values=(5 10 20)


# Fourier orders (daily/weekly seasonality flexibility)
daily_fourier_values=(15 20 25)
weekly_fourier_values=(8 9 10)

# Optional: holiday switch (0/1)
use_holidays_values=(0)

echo "Submitting Prophet jobs (Simplified Grid)..."
echo "GRID: modes=${modes[*]} cps_values=${cps_values[*]}"
echo "GRID: daily_fourier_values=${daily_fourier_values[*]} weekly_fourier_values=${weekly_fourier_values[*]}"
echo "GRID: use_holidays_values=${use_holidays_values[*]}"

for dh in "${use_holidays_values[@]}"; do
  export USE_US_HOLIDAYS="$dh"

  for dfo in "${daily_fourier_values[@]}"; do
    export DAILY_FOURIER="$dfo"

    for wfo in "${weekly_fourier_values[@]}"; do
      export WEEKLY_FOURIER="$wfo"

      for mode in "${modes[@]}"; do
        for cps in "${cps_values[@]}"; do
          for sps in "${sps_values[@]}"; do

            # Simplified ID matching the batch script
            RUN_ID="prophet_${mode}_cps${cps}_sps${sps}_dfo${dfo}_wfo${wfo}_hol${dh}"

            OUT="./logs/${RUN_ID}.out"
            ERR="./logs/${RUN_ID}.err"

            echo "Submitting: $RUN_ID"
            
            # Only passing relevant env vars
            bsub -q hpc \
                -J "$RUN_ID" \
                -W 1:00 \
                -R "rusage[mem=8GB]" \
                -R "span[hosts=1]" \
                -n 4 \
                -o "$OUT" \
                -e "$ERR" \
                "USE_US_HOLIDAYS=$dh DAILY_FOURIER=$dfo WEEKLY_FOURIER=$wfo $JOB_SCRIPT $mode $cps $sps"

          done
        done
      done
    done
  done
done

echo "All jobs submitted. Use: bjobs"