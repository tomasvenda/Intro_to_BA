#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json

TARGET_CLUSTERS = {3, 7, 8}
HARD_CUTOFF = pd.Timestamp("2018-12-31 23:00:00")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train Prophet on hourly ARRIVALS per cluster with robust hourly grid handling."
    )

    # Hyperparameters to tune
    parser.add_argument(
        "--seasonality_mode",
        type=str,
        default="multiplicative",
        choices=["additive", "multiplicative"],
        help="Mode for seasonality (additive vs multiplicative).",
    )
    parser.add_argument(
        "--changepoint_prior_scale",
        type=float,
        default=0.05,
        help="Flexibility of the trend (higher = more flexible).",
    )
    parser.add_argument(
        "--seasonality_prior_scale",
        type=float,
        default=10.0,
        help="Strength of the seasonality (higher = more flexible seasonality).",
    )
    parser.add_argument(
        "--holidays_prior_scale",
        type=float,
        default=10.0,
        help="Strength of the holiday components.",
    )

    # Extra seasonality control (often helps peaks)
    parser.add_argument("--daily_fourier", type=int, default=15, help="Fourier order for 24h seasonality.")
    parser.add_argument("--weekly_fourier", type=int, default=10, help="Fourier order for 168h seasonality.")

    # Job identification
    parser.add_argument("--job_id", type=str, default="default_run", help="Unique identifier for this batch run.")
    parser.add_argument("--output_dir", type=str, default="./prophet_results", help="Directory to save results.")

    # Optional holidays toggle (keep simple)
    parser.add_argument("--use_us_holidays", action="store_true", help="Add US holidays to the model.")

    return parser.parse_args()



def load_and_prep_data(filepath: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Loads parquet, filters to TARGET_CLUSTERS, hard-cuts to HARD_CUTOFF,
    then splits Train (Jan-Oct) / Test (Nov-Dec).

    Expected parquet columns: cluster, datetime_hour, arrivals
    Output: {cluster_id: {"train": train_df, "test": test_df}}
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_parquet(filepath)

    required_cols = {"cluster", "datetime_hour", "arrivals"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in parquet: {sorted(missing)}")

    df = df[df["cluster"].isin(TARGET_CLUSTERS)].copy()
    df = df.rename(columns={"datetime_hour": "ds", "arrivals": "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"]).copy()

    # HARD CUTOFF to avoid stray 2019+
    df = df[df["ds"] <= HARD_CUTOFF].copy()
    if df.empty:
        raise ValueError(f"No data left after applying HARD_CUTOFF={HARD_CUTOFF}.")

    df = df.sort_values(["cluster", "ds"]).reset_index(drop=True)

    cluster_splits: Dict[str, Dict[str, pd.DataFrame]] = {}
    for cid in sorted(TARGET_CLUSTERS):
        df_c = df.loc[df["cluster"] == cid, ["ds", "y"]].copy()
        if df_c.empty:
            continue

        month = df_c["ds"].dt.month
        train_df = df_c.loc[month <= 10, ["ds", "y"]].copy()
        test_df  = df_c.loc[month >= 11, ["ds", "y"]].copy()

        cluster_splits[str(cid)] = {"train": train_df, "test": test_df}

    return cluster_splits

def add_time_regressors(df_: pd.DataFrame) -> pd.DataFrame:
    df_ = df_.copy()

    # Basic calendar fields
    df_["dow"] = df_["ds"].dt.dayofweek  # 0=Mon ... 6=Sun
    df_["is_weekend"] = (df_["dow"] >= 5).astype(int)
    df_["hour"] = df_["ds"].dt.hour

    # ---------- DOW dummies (drop one to avoid collinearity) ----------
    dow_d = pd.get_dummies(df_["dow"], prefix="dow", drop_first=True, dtype=int)
    # Ensures consistent columns dow_1..dow_6
    for i in range(1, 7):
        col = f"dow_{i}"
        if col not in dow_d.columns:
            dow_d[col] = 0
    dow_d = dow_d[[f"dow_{i}" for i in range(1, 7)]]

    # ---------- Hour-of-day dummies (drop one) ----------
    hr_d = pd.get_dummies(df_["hour"], prefix="hr", drop_first=True, dtype=int)
    # Ensures consistent columns hr_1..hr_23
    for h in range(1, 24):
        col = f"hr_{h}"
        if col not in hr_d.columns:
            hr_d[col] = 0
    hr_d = hr_d[[f"hr_{h}" for h in range(1, 24)]]

    # ---------- Interaction: hour × weekend ----------
    # Lets weekends have a DIFFERENT hourly shape than weekdays.
    wk_hr_d = hr_d.mul(df_["is_weekend"], axis=0)
    wk_hr_d.columns = [c + "_wknd" for c in wk_hr_d.columns]

    # Assemble
    df_ = df_.drop(columns=["dow", "hour"])
    df_ = pd.concat([df_, dow_d, hr_d, wk_hr_d], axis=1)

    return df_



def train_models(args):
    os.makedirs(args.output_dir, exist_ok=True)
    cluster_data = load_and_prep_data("cluster_hourly.parquet")

    for cluster_id, splits in cluster_data.items():
        train_df = add_time_regressors(splits["train"])
        test_df  = add_time_regressors(splits["test"])  # not used in training, but good practice


        if train_df.empty:
            print(f"[Cluster {cluster_id}] Empty train set, skipping.")
            continue

        # NOTE: yearly seasonality OFF (only 2018)
        model = Prophet(
            seasonality_mode=args.seasonality_mode,
            changepoint_prior_scale=args.changepoint_prior_scale,
            seasonality_prior_scale=args.seasonality_prior_scale,
            holidays_prior_scale=args.holidays_prior_scale,
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
        )

        # Explicit seasonalities for hourly data
        model.add_seasonality(name="daily", period=1, fourier_order=args.daily_fourier)
        model.add_seasonality(name="weekly", period=7, fourier_order=args.weekly_fourier)
        # Keep this
        model.add_regressor("is_weekend", standardize=False)

        # Add all generated dummy regressors
        for col in train_df.columns:
            if col.startswith(("dow_", "hr_")):
                model.add_regressor(col, standardize=False)



        if args.use_us_holidays:
            model.add_country_holidays(country_name="US")

        print(f"[Cluster {cluster_id}] Training Prophet with: {vars(args)}")
        model.fit(train_df)

        # Save model and also save a tiny metadata file to remember transforms
        model_filename = os.path.join(args.output_dir, f"prophet_model_{cluster_id}_{args.job_id}.json")
        with open(model_filename, "w") as fout:
            fout.write(model_to_json(model))
        print(f"[Cluster {cluster_id}] Saved -> {model_filename}")

        meta = {
            "cluster": int(cluster_id),
            "job_id": args.job_id,
            "seasonality_mode": args.seasonality_mode,
            "changepoint_prior_scale": float(args.changepoint_prior_scale),
            "seasonality_prior_scale": float(args.seasonality_prior_scale),
            "holidays_prior_scale": float(args.holidays_prior_scale),
            "daily_fourier": int(args.daily_fourier),
            "weekly_fourier": int(args.weekly_fourier),
            "use_us_holidays": bool(args.use_us_holidays),
        }
        meta_path = os.path.join(args.output_dir, f"prophet_model_{cluster_id}_{args.job_id}_meta.json")
        pd.Series(meta).to_json(meta_path)
        print(f"[Cluster {cluster_id}] Meta  -> {meta_path}")


if __name__ == "__main__":
    args = parse_arguments()
    train_models(args)
