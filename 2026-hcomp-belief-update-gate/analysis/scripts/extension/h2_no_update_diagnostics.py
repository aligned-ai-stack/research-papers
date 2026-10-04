#!/usr/bin/env python
# coding: utf-8

"""Diagnostics for the apparent H2 vs. no-update-model tension.

H2 reports a positive pooled within participant-task slope: observed belief
changes move in the Bayesian direction at roughly half the normative rate.
The behavioral model comparison, however, finds that a no-update model wins
many participant-task trajectories. This script quantifies why both can be
true at the same time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
H2_PANEL = ROOT / "data" / "hypothesis_outputs" / "H2" / "h2_panel_hybrid_S10.csv"
CLASSIFICATION = ROOT / "data" / "followup_outputs" / "robust_belief_model_comparison" / "canonical_classification.csv"
H2_QC = ROOT / "data" / "hypothesis_outputs" / "H2" / "h2_cf_qc_hybrid_S10.csv"
OUTDIR = ROOT / "data" / "followup_outputs" / "h2_no_update_diagnostics"


def within_slope(df: pd.DataFrame) -> float:
    den = float(np.dot(df["norm_dm"], df["norm_dm"]))
    if den == 0:
        return np.nan
    return float(np.dot(df["norm_dm"], df["delta_dm"]) / den)


def add_h2_demeaning(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pid_task"] = out["participant_id"].astype(str) + "·" + out["taskType"].astype(str)
    out["delta_dm"] = out["delta_b"] - out.groupby("pid_task")["delta_b"].transform("mean")
    out["norm_dm"] = out["norm_step"] - out.groupby("pid_task")["norm_step"].transform("mean")
    return out


def build_group_diagnostics(reg: pd.DataFrame) -> pd.DataFrame:
    prior = reg.sort_values("t").groupby(["participant_id", "taskType"])["b_prev"].first().rename("b0")
    posts = reg.set_index(["participant_id", "taskType"]).join(prior)
    gdev = (
        posts.groupby(level=[0, 1])
        .apply(lambda g: float((g["b_post"] - g["b0"]).abs().max()), include_groups=False)
        .rename("max_abs_deviation_from_prior")
        .reset_index()
    )

    grouped = (
        reg.groupby(["participant_id", "taskType", "winner"])
        .agg(
            n=("delta_b", "size"),
            nonzero_updates=("delta_b", lambda s: int((s.abs() >= 1e-12).sum())),
            zero_update_share=("delta_b", lambda s: float((s.abs() < 1e-12).mean())),
            mean_abs_delta=("delta_b", lambda s: float(s.abs().mean())),
            sum_abs_delta=("delta_b", lambda s: float(s.abs().sum())),
        )
        .reset_index()
        .merge(gdev, on=["participant_id", "taskType"], how="left")
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=H2_PANEL)
    parser.add_argument("--classification", type=Path, default=CLASSIFICATION)
    parser.add_argument("--winner-col", default="aicc_winner")
    parser.add_argument("--qc", type=Path, default=H2_QC)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.panel)
    classification = pd.read_csv(args.classification)
    qc = pd.read_csv(args.qc)
    if args.winner_col not in classification.columns:
        args.winner_col = "winner" if "winner" in classification.columns else "bic_winner"
    classification = classification.copy()
    classification["winner"] = classification[args.winner_col]

    df = panel.merge(
        classification[["participant_id", "taskType", "winner"]],
        on=["participant_id", "taskType"],
        how="left",
    )
    reg = add_h2_demeaning(df.dropna(subset=["delta_b", "norm_step"]).copy())

    subset_rows = []
    subsets = {
        "all": reg,
        "exclude_no_update_winner": reg[reg["winner"] != "no_update"],
        "at_least_one_nonzero_update_trajectory": reg.groupby("pid_task").filter(
            lambda g: (g["delta_b"].abs() > 1e-12).any()
        ),
        "at_least_two_nonzero_updates_trajectory": reg.groupby("pid_task").filter(
            lambda g: (g["delta_b"].abs() > 1e-12).sum() >= 2
        ),
        "nonzero_rows_only": reg[reg["delta_b"].abs() > 1e-12],
    }
    for label, sub in subsets.items():
        subset_rows.append(
            {
                "subset": label,
                "rows": int(len(sub)),
                "participant_tasks": int(sub["pid_task"].nunique()),
                "within_slope": within_slope(sub),
                "zero_delta_share": float((sub["delta_b"].abs() < 1e-12).mean()),
            }
        )
    pd.DataFrame(subset_rows).to_csv(args.outdir / "h2_slope_by_subset.csv", index=False)

    eps_rows = [{"threshold": eps, "share": float((reg["delta_b"].abs() < eps).mean())} for eps in [1e-12, 0.001, 0.005, 0.01, 0.02, 0.05]]
    pd.DataFrame(eps_rows).to_csv(args.outdir / "zero_or_near_zero_update_shares.csv", index=False)

    group_diag = build_group_diagnostics(reg)
    winner_diag = (
        group_diag.groupby("winner")
        .agg(
            participant_tasks=("n", "size"),
            mean_nonzero_updates=("nonzero_updates", "mean"),
            median_sum_abs_delta=("sum_abs_delta", "median"),
            median_max_abs_deviation_from_prior=("max_abs_deviation_from_prior", "median"),
        )
        .sort_values("participant_tasks", ascending=False)
        .reset_index()
    )
    winner_diag.to_csv(args.outdir / "winner_update_diagnostics.csv", index=False)

    no_update = group_diag[group_diag["winner"] == "no_update"]
    no_update_summary = pd.DataFrame(
        [
            {"metric": "no_update_winner_trajectories", "value": len(no_update)},
            {"metric": "all_10_deltas_zero", "value": int((no_update["nonzero_updates"] == 0).sum())},
            {"metric": "all_10_deltas_zero_share", "value": float((no_update["nonzero_updates"] == 0).mean())},
            {"metric": "median_sum_abs_delta", "value": float(no_update["sum_abs_delta"].median())},
            {
                "metric": "median_max_abs_deviation_from_prior",
                "value": float(no_update["max_abs_deviation_from_prior"].median()),
            },
        ]
    )
    no_update_summary.to_csv(args.outdir / "no_update_summary.csv", index=False)

    reg["num"] = reg["norm_dm"] * reg["delta_dm"]
    reg["den"] = reg["norm_dm"] * reg["norm_dm"]
    contribution = reg.groupby("winner").agg(num=("num", "sum"), den=("den", "sum"), rows=("den", "size")).reset_index()
    contribution["den_share"] = contribution["den"] / contribution["den"].sum()
    contribution["num_share"] = contribution["num"] / contribution["num"].sum()
    contribution.sort_values("den_share", ascending=False).to_csv(args.outdir / "h2_contribution_by_winner.csv", index=False)

    qc_kept = qc.dropna(subset=["n0"])
    n0_source = classification.merge(
        qc_kept[["participant_id", "taskType", "n0_source", "n0"]],
        on=["participant_id", "taskType"],
        how="left",
    )
    pd.crosstab(n0_source["n0_source"], n0_source["winner"] == "no_update", margins=True).to_csv(
        args.outdir / "no_update_by_n0_source.csv"
    )

    print("[H2 diagnostics] Outputs written to", args.outdir)
    print(pd.DataFrame(subset_rows).to_string(index=False))
    print(no_update_summary.to_string(index=False))


if __name__ == "__main__":
    main()
