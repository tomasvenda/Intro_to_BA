#!/usr/bin/env python3
import argparse
import os
import re
import sys
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from prophet.serialize import model_from_json

warnings.filterwarnings("ignore")

MODEL_RE = re.compile(r"^prophet_model_(?P<cluster>\d+?)_.+\.json$")
HARD_CUTOFF_DEFAULT = pd.Timestamp("2018-12-31 23:00:00")


def parse_args():
    p = argparse.ArgumentParser(
        description="Simple Prophet evaluation on TEST split (months >= 11)."
    )
    p.add_argument("--data_path", type=str, default="cluster_hourly.parquet")
    p.add_argument("--models_dir", type=str, default="prophet_results")
    p.add_argument("--output_dir", type=str, default="prophet_eval_outputs")
    p.add_argument("--job_id", type=str, default="eval_run")

    p.add_argument("--cluster_ids", type=str, default="all",
                   help='Comma-separated cluster ids (e.g. "3,7,8") or "all".')

    p.add_argument("--cluster_col", type=str, default="cluster")
    p.add_argument("--time_col", type=str, default="datetime_hour")
    p.add_argument("--target_col", type=str, default="arrivals")

    p.add_argument("--train_month_end", type=int, default=10)
    p.add_argument("--test_month_start", type=int, default=11)

    p.add_argument("--hard_cutoff", type=str, default=str(HARD_CUTOFF_DEFAULT))
    p.add_argument("--save_hourly", action="store_true")

    return p.parse_args()


def parse_cluster_ids(raw: str, all_ids: List[int]) -> List[int]:
    raw = raw.strip().lower()
    if raw == "all":
        return sorted(all_ids)
    out = []
    for x in raw.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return sorted(set(out))


def extract_cluster_from_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    if base.endswith("_meta.json"):
        return None
    m = MODEL_RE.match(base)
    if not m:
        return None
    return int(m.group("cluster"))


def list_model_files(models_dir: str) -> List[str]:
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"models_dir not found: {models_dir}")
    files = []
    for fn in os.listdir(models_dir):
        if not fn.endswith(".json"):
            continue
        if fn.endswith("_meta.json"):
            continue
        fp = os.path.join(models_dir, fn)
        if extract_cluster_from_filename(fp) is not None:
            files.append(fp)
    return sorted(files)


def load_prophet_model_json(path: str):
    with open(path, "r") as f:
        return model_from_json(f.read())


def metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> Dict[str, float]:
    err = y_pred - y_true
    ae = np.abs(err)
    se = err ** 2

    mae = float(np.mean(ae))
    rmse = float(np.sqrt(np.mean(se)))
    bias = float(np.mean(err))

    denom = np.abs(y_true) + np.abs(y_pred) + eps
    smape = float(np.mean(2.0 * ae / denom))

    sum_true = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(ae) / (sum_true + eps)) if sum_true > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "Bias": bias, "sMAPE": smape, "WAPE": wape}

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

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.data_path):
        sys.exit(f"ERROR: data_path not found: {args.data_path}")

    cutoff = pd.Timestamp(args.hard_cutoff)

    df = pd.read_parquet(args.data_path)
    needed = {args.cluster_col, args.time_col, args.target_col}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"ERROR: Missing columns in parquet: {sorted(missing)}")

    df = df.copy()
    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    df = df.dropna(subset=[args.time_col]).copy()
    df = df[df[args.time_col] <= cutoff].copy()
    if df.empty:
        sys.exit(f"ERROR: No rows left after cutoff <= {cutoff}")

    all_cluster_ids = sorted(df[args.cluster_col].dropna().astype(int).unique().tolist())
    wanted = parse_cluster_ids(args.cluster_ids, all_cluster_ids)
    if not wanted:
        sys.exit("ERROR: no clusters selected")

    df["__month__"] = df[args.time_col].dt.month

    model_files = list_model_files(args.models_dir)
    if not model_files:
        sys.exit(f"ERROR: No model json files found in {args.models_dir}")

    cluster_to_models: Dict[int, List[str]] = {cid: [] for cid in wanted}
    for mp in model_files:
        cid = extract_cluster_from_filename(mp)
        if cid in cluster_to_models:
            cluster_to_models[cid].append(mp)

    summary_rows = []
    hourly_rows = []

    for cid in wanted:
        dfc = df[df[args.cluster_col].astype(int) == cid].copy()
        if dfc.empty:
            summary_rows.append({"cluster": cid, "status": "no_data_after_cutoff"})
            continue

        test_df = dfc[dfc["__month__"] >= args.test_month_start][[args.time_col, args.target_col]].copy()
        if test_df.empty:
            summary_rows.append({"cluster": cid, "status": "no_test_data"})
            continue

        test_df = (
            test_df.groupby(args.time_col, as_index=False)[args.target_col].sum()
                   .sort_values(args.time_col)
                   .reset_index(drop=True)
        )

        ts = pd.DatetimeIndex(test_df[args.time_col].to_numpy())
        y_true = test_df[args.target_col].to_numpy(dtype=float)

        model_paths = cluster_to_models.get(cid, [])
        if not model_paths:
            summary_rows.append({"cluster": cid, "status": "no_model_for_cluster"})
            continue

        for mp in model_paths:
            model_name = os.path.splitext(os.path.basename(mp))[0]
            try:
                model = load_prophet_model_json(mp)
            except Exception as e:
                summary_rows.append({
                    "cluster": cid, "model": model_name, "model_path": mp,
                    "status": f"load_failed:{type(e).__name__}"
                })
                continue

            future = pd.DataFrame({"ds": ts})
            future = add_time_regressors(future)  # must produce same regressor cols

            try:
                fc = model.predict(future)
                y_pred = fc["yhat"].to_numpy(dtype=float)
            except Exception as e:
                summary_rows.append({
                    "cluster": cid, "model": model_name, "model_path": mp,
                    "status": f"predict_failed:{type(e).__name__}"
                })
                continue

            m = metrics(y_true, y_pred)
            summary_rows.append({
                "cluster": int(cid),
                "model": model_name,
                "model_path": mp,
                "n_hours": int(len(y_true)),
                "hard_cutoff": str(cutoff),
                "test_month_start": int(args.test_month_start),
                **m,
                "status": "ok",
            })

            if args.save_hourly:
                for t, yt, yp in zip(ts, y_true, y_pred):
                    hourly_rows.append({
                        "cluster": int(cid),
                        "model": model_name,
                        "ts": t,
                        "y_true": float(yt),
                        "y_pred": float(yp),
                        "error": float(yp - yt),
                        "abs_error": float(abs(yp - yt)),
                    })

            print(f"[Cluster {cid}] {model_name} -> MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} WAPE={m['WAPE']:.3f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, f"prophet_eval_summary_{args.job_id}.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\nWrote:", summary_path)

    if args.save_hourly:
        hourly_df = pd.DataFrame(hourly_rows)
        hourly_path = os.path.join(args.output_dir, f"prophet_eval_hourly_{args.job_id}.csv")
        hourly_df.to_csv(hourly_path, index=False)
        print("Wrote:", hourly_path)


if __name__ == "__main__":
    main()
