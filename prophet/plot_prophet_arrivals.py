#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Pick best Prophet model per cluster from eval CSVs and plot actual vs forecast."
    )
    p.add_argument("--models_dir", type=str, default="prophet_results",
                   help="Folder that contains prophet_model_{cluster}_{job}.json (for naming only).")
    p.add_argument("--eval_dir", type=str, default="prophet_eval_outputs",
                   help="Folder containing prophet_eval_summary_*.csv and prophet_eval_hourly_*.csv")
    p.add_argument("--summary_csv", type=str, default="",
                   help="Optional: explicit path to prophet_eval_summary_<job>.csv")
    p.add_argument("--hourly_csv", type=str, default="",
                   help="Optional: explicit path to prophet_eval_hourly_<job>.csv (requires eval run saved hourly)")
    p.add_argument("--out_dir", type=str, default="prophet_plots_best",
                   help="Where to write plots")

    p.add_argument("--clusters", type=str, default="all",
                   help='Comma list like "3,7,8" or "all"')
    p.add_argument("--rank_metric", type=str, default="MAE",
                   help="Metric to minimize: one of MAE, RMSE, Bias, sMAPE, WAPE (from the simple eval script)")
    p.add_argument("--start", type=str, default="",
                   help='Optional plot start (e.g. "2018-11-01")')
    p.add_argument("--end", type=str, default="",
                   help='Optional plot end (e.g. "2018-11-07 23:00:00")')
    p.add_argument("--dpi", type=int, default=250)
    return p.parse_args()


def newest_matching(eval_dir: Path, prefix: str) -> Path:
    files = sorted(eval_dir.glob(f"{prefix}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise SystemExit(f"ERROR: Could not find {prefix}*.csv in {eval_dir}")
    return files[0]


def parse_cluster_list(raw: str, present_clusters: list[int]) -> list[int]:
    raw = raw.strip().lower()
    if raw == "all":
        return sorted(present_clusters)
    out = []
    for x in raw.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return sorted(set(out))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise SystemExit(f"ERROR: eval_dir not found: {eval_dir}")

    summary_csv = Path(args.summary_csv) if args.summary_csv else newest_matching(eval_dir, "prophet_eval_summary_")
    hourly_csv = Path(args.hourly_csv) if args.hourly_csv else newest_matching(eval_dir, "prophet_eval_hourly_")

    if not summary_csv.exists():
        raise SystemExit(f"ERROR: summary_csv not found: {summary_csv}")
    if not hourly_csv.exists():
        raise SystemExit(f"ERROR: hourly_csv not found: {hourly_csv}")

    summary = pd.read_csv(summary_csv)
    required = {"cluster", "model", "status", args.rank_metric}
    missing = required - set(summary.columns)
    if missing:
        raise SystemExit(f"ERROR: summary CSV missing columns: {sorted(missing)}")

    summary = summary.copy()
    summary["cluster"] = pd.to_numeric(summary["cluster"], errors="coerce").astype("Int64")
    summary = summary[summary["status"].astype(str) == "ok"].dropna(subset=["cluster"]).copy()
    if summary.empty:
        raise SystemExit("ERROR: No rows with status=='ok' in summary CSV.")

    # metric numeric + choose best per cluster
    summary[args.rank_metric] = pd.to_numeric(summary[args.rank_metric], errors="coerce")
    summary = summary.dropna(subset=[args.rank_metric])

    present_clusters = sorted(summary["cluster"].astype(int).unique().tolist())
    wanted_clusters = parse_cluster_list(args.clusters, present_clusters)

    winners = []
    for cid in wanted_clusters:
        sub = summary[summary["cluster"].astype(int) == cid].copy()
        if sub.empty:
            continue
        winners.append(sub.sort_values(args.rank_metric, ascending=True).iloc[0])

    if not winners:
        raise SystemExit("ERROR: No winners selected. Check --clusters and CSV contents.")

    winners_df = pd.DataFrame(winners)[["cluster", "model", args.rank_metric]].copy()
    winners_df.to_csv(out_dir / "best_models_per_cluster.csv", index=False)
    print("Using summary:", summary_csv)
    print("Using hourly :", hourly_csv)
    print("\nBest per cluster (minimize", args.rank_metric, "):")
    print(winners_df.to_string(index=False))

    # load hourly and filter to winners
    df = pd.read_csv(hourly_csv)
    req_h = {"cluster", "model", "ts", "y_true", "y_pred"}
    miss_h = req_h - set(df.columns)
    if miss_h:
        raise SystemExit(f"ERROR: hourly CSV missing columns: {sorted(miss_h)}")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["ts", "cluster", "model", "y_true", "y_pred"]).copy()
    df["cluster"] = df["cluster"].astype(int)

    if args.start:
        df = df[df["ts"] >= pd.Timestamp(args.start)].copy()
    if args.end:
        df = df[df["ts"] <= pd.Timestamp(args.end)].copy()

    # Keep only winners
    winner_by_cluster = {int(r.cluster): str(r.model) for r in winners_df.itertuples(index=False)}
    df = df[df.apply(lambda r: winner_by_cluster.get(int(r["cluster"])) == str(r["model"]), axis=1)].copy()
    if df.empty:
        raise SystemExit("ERROR: No hourly rows left after filtering to winners and time window.")

    # plot per cluster
    for cid in sorted(df["cluster"].unique()):
        d = df[df["cluster"] == cid].sort_values("ts")
        model_name = winner_by_cluster.get(cid, d["model"].iloc[0])

        plt.figure(figsize=(14, 5))
        plt.plot(d["ts"], d["y_true"], label="Actual")
        plt.plot(d["ts"], d["y_pred"], label="Forecast")
        plt.title(f"Prophet – Cluster {cid} (best: {model_name})")
        plt.xlabel("Time")
        plt.ylabel("Arrivals")
        plt.legend()
        plt.tight_layout()
        plt.gcf().autofmt_xdate()

        out_png = out_dir / f"cluster_{cid}_pred_vs_true.png"
        plt.savefig(out_png, dpi=args.dpi)
        plt.close()

    print(f"\nSaved plots to: {out_dir}")


if __name__ == "__main__":
    main()
