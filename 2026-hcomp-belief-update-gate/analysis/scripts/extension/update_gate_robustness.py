#!/usr/bin/env python
# coding: utf-8

"""Robustness checks for the update-gate claim.

This standalone script produces two paper-facing diagnostics:

1. Threshold sensitivity for the update gate, using movement thresholds of
   0, 1, 2, and 5 percentage points.
2. Participant-bootstrap confidence intervals for the H2 slope decomposition.

The script is deliberately separate from the main behavioral pipeline so the
paper can report these checks without changing the canonical analysis objects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The local Anaconda environment has optional pandas accelerators compiled
# against an older NumPy. Mark them unavailable so pandas skips them cleanly.
sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRIAL_PANEL = ROOT / "data" / "followup_outputs" / "behavioral_extension_pipeline" / "trial_panel_with_updater_types.csv"
DEFAULT_H2_PANEL = ROOT / "data" / "hypothesis_outputs" / "H2" / "h2_panel_hybrid_S10.csv"
DEFAULT_CLASSIFICATION = ROOT / "data" / "followup_outputs" / "robust_belief_model_comparison" / "canonical_classification.csv"
DEFAULT_OUTDIR = ROOT / "data" / "followup_outputs" / "update_gate_robustness"

THRESHOLDS = [0.0, 0.01, 0.02, 0.05]
EPS = 1e-12


def prepare_trial_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path)
    for col in ["belief_lag", "belief_post", "delta_belief", "ai_correct", "conf_lag", "delegation"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel["signed_feedback_gap"] = panel["ai_correct"] - panel["belief_lag"]
    panel["abs_feedback_gap"] = panel["signed_feedback_gap"].abs()
    return panel


def design_matrix(df: pd.DataFrame, continuous: list[str], categorical: list[str]) -> tuple[np.ndarray, list[str]]:
    parts = [np.ones((len(df), 1))]
    names = ["Intercept"]
    for col in continuous:
        parts.append(df[[col]].astype(float).to_numpy())
        names.append(col)
    for col in categorical:
        levels = sorted([x for x in df[col].dropna().unique()])
        for level in levels[1:]:
            parts.append((df[col] == level).astype(float).to_numpy().reshape(-1, 1))
            names.append(f"{col}={level}")
    return np.column_stack(parts), names


def cluster_sandwich_cov(
    bread_inv: np.ndarray,
    scores: np.ndarray,
    groups: pd.Series,
    n: int,
    k: int,
) -> tuple[np.ndarray, int]:
    group_values = pd.Series(groups).astype(str).to_numpy()
    unique_groups = pd.unique(group_values)
    meat = np.zeros((k, k))
    for group in unique_groups:
        s_g = scores[group_values == group].sum(axis=0)
        meat += np.outer(s_g, s_g)
    cov = bread_inv @ meat @ bread_inv
    n_clusters = len(unique_groups)
    if n_clusters > 1 and n > k:
        cov *= (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    return cov, n_clusters


def logistic_fit(df: pd.DataFrame, y_col: str, continuous: list[str], categorical: list[str]) -> pd.DataFrame:
    needed = [y_col] + continuous + categorical + ["participant_id"]
    m = df.dropna(subset=needed).copy()
    X, names = design_matrix(m, continuous, categorical)
    y = m[y_col].astype(float).to_numpy()
    beta = np.zeros(X.shape[1])
    ridge = 1e-6
    for _ in range(100):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-6, np.inf)
        z = eta + (y - p) / w
        xtwx = X.T @ (w[:, None] * X) + ridge * np.eye(X.shape[1])
        xtwz = X.T @ (w * z)
        new_beta = np.linalg.solve(xtwx, xtwz)
        if np.max(np.abs(new_beta - beta)) < 1e-7:
            beta = new_beta
            break
        beta = new_beta
    eta = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(p * (1.0 - p), 1e-6, np.inf)
    bread_inv = np.linalg.pinv(X.T @ (w[:, None] * X))
    cluster_cov, n_clusters = cluster_sandwich_cov(bread_inv, X * (y - p)[:, None], m["participant_id"], len(y), X.shape[1])
    se = np.sqrt(np.clip(np.diag(cluster_cov), 0, np.inf))
    ll = float(np.sum(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
    return pd.DataFrame(
        {
            "term": names,
            "coef": beta,
            "se_cluster": se,
            "z_cluster": beta / np.where(se == 0, np.nan, se),
            "odds_ratio": np.exp(np.clip(beta, -20, 20)),
            "n": len(y),
            "n_clusters": n_clusters,
            "log_likelihood": ll,
        }
    )


def ols_fit(df: pd.DataFrame, y_col: str, continuous: list[str], categorical: list[str]) -> pd.DataFrame:
    needed = [y_col] + continuous + categorical + ["participant_id"]
    m = df.dropna(subset=needed).copy()
    X, names = design_matrix(m, continuous, categorical)
    y = m[y_col].astype(float).to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    cluster_cov, n_clusters = cluster_sandwich_cov(xtx_inv, X * resid[:, None], m["participant_id"], n, k)
    se = np.sqrt(np.clip(np.diag(cluster_cov), 0, np.inf))
    r2 = 1.0 - float(np.sum(resid**2) / np.sum((y - y.mean()) ** 2)) if np.sum((y - y.mean()) ** 2) > 0 else np.nan
    return pd.DataFrame(
        {
            "term": names,
            "coef": beta,
            "se_cluster": se,
            "t_cluster": beta / np.where(se == 0, np.nan, se),
            "n": n,
            "n_clusters": n_clusters,
            "r2": r2,
        }
    )


def threshold_sensitivity(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    gate_rows = []
    movement_rows = []
    controls_cont = ["abs_feedback_gap", "belief_lag", "conf_lag"]
    controls_cat = ["taskType", "trialOrdinal", "updater_family"]
    movement_cont = ["signed_feedback_gap", "belief_lag", "conf_lag"]
    for threshold in THRESHOLDS:
        moved_col = f"moved_ge_{int(threshold * 100):02d}pp"
        if threshold == 0.0:
            panel[moved_col] = (panel["delta_belief"].abs() > EPS).astype(int)
        else:
            panel[moved_col] = (panel["delta_belief"].abs() >= threshold).astype(int)
        moved = panel[panel[moved_col] == 1].copy()
        gate = logistic_fit(panel, moved_col, controls_cont, controls_cat)
        movement = ols_fit(moved, "delta_belief", movement_cont, controls_cat)
        gate_key = gate[gate["term"] == "abs_feedback_gap"].iloc[0]
        movement_key = movement[movement["term"] == "signed_feedback_gap"].iloc[0]
        summary_rows.append(
            {
                "threshold": threshold,
                "threshold_label": "exact" if threshold == 0.0 else f">={int(threshold * 100)}pp",
                "moved_trials": int(moved_col in panel and panel[moved_col].sum()),
                "moved_share": float(panel[moved_col].mean()),
                "gate_abs_feedback_gap_coef": float(gate_key["coef"]),
                "gate_abs_feedback_gap_z": float(gate_key["z_cluster"]),
                "gate_abs_feedback_gap_or": float(gate_key["odds_ratio"]),
                "conditional_signed_gap_coef": float(movement_key["coef"]),
                "conditional_signed_gap_t": float(movement_key["t_cluster"]),
                "conditional_rows": int(movement_key["n"]),
                "conditional_r2": float(movement_key["r2"]),
            }
        )
        gate.insert(0, "threshold", threshold)
        movement.insert(0, "threshold", threshold)
        gate_rows.append(gate)
        movement_rows.append(movement)
    return pd.DataFrame(summary_rows), pd.concat(gate_rows, ignore_index=True), pd.concat(movement_rows, ignore_index=True)


def prepare_h2_with_classification(h2_path: Path, class_path: Path) -> pd.DataFrame:
    h2 = pd.read_csv(h2_path)
    classification = pd.read_csv(class_path)
    for col in ["b_prev", "b_post", "delta_b", "norm_step"]:
        h2[col] = pd.to_numeric(h2[col], errors="coerce")
    classification = classification[["participant_id", "taskType", "bic_winner", "nonzero_updates"]].copy()
    h2["participant_id"] = h2["participant_id"].astype(str)
    h2["taskType"] = h2["taskType"].astype(str)
    classification["participant_id"] = classification["participant_id"].astype(str)
    classification["taskType"] = classification["taskType"].astype(str)
    return h2.merge(classification, on=["participant_id", "taskType"], how="inner")


def slope(df: pd.DataFrame) -> float:
    m = df.dropna(subset=["delta_dm", "norm_dm"]).copy()
    if len(m) < 2:
        return np.nan
    x = m["norm_dm"].astype(float).to_numpy()
    y = m["delta_dm"].astype(float).to_numpy()
    den = float(np.dot(x, x))
    if den == 0:
        return np.nan
    return float(np.dot(x, y) / den)


def add_h2_demeaning(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "bootstrap_draw_id" in out.columns:
        out["pid_task"] = (
            out["bootstrap_draw_id"].astype(str)
            + "::"
            + out["participant_id"].astype(str)
            + "::"
            + out["taskType"].astype(str)
        )
    else:
        out["pid_task"] = out["participant_id"].astype(str) + "::" + out["taskType"].astype(str)
    out["delta_dm"] = out["delta_b"] - out.groupby("pid_task")["delta_b"].transform("mean")
    out["norm_dm"] = out["norm_step"] - out.groupby("pid_task")["norm_step"].transform("mean")
    return out


def slope_decomposition(df: pd.DataFrame) -> dict[str, float]:
    reg = add_h2_demeaning(df.dropna(subset=["delta_b", "norm_step"]).copy())
    all_df = reg
    excl = reg[reg["bic_winner"] != "no_update"]
    ge1 = reg.groupby("pid_task").filter(lambda g: (g["delta_b"].abs() > EPS).any())
    ge2 = reg.groupby("pid_task").filter(lambda g: int((g["delta_b"].abs() > EPS).sum()) >= 2)
    nonzero_rows = reg[reg["delta_b"].abs() > EPS]
    return {
        "all": slope(all_df),
        "exclude_no_update_winner": slope(excl),
        "at_least_one_nonzero_update_trajectory": slope(ge1),
        "at_least_two_nonzero_updates_trajectory": slope(ge2),
        "nonzero_rows_only": slope(nonzero_rows),
        "nonzero_minus_all": slope(nonzero_rows) - slope(all_df),
    }


def bootstrap_slope_decomposition(df: pd.DataFrame, reps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    participants = np.asarray(sorted(df["participant_id"].astype(str).unique()))
    point = slope_decomposition(df)
    boot_rows = []
    grouped = {pid: g for pid, g in df.groupby(df["participant_id"].astype(str), sort=False)}
    for rep in range(reps):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        pieces = []
        for draw_id, pid in enumerate(sampled):
            g = grouped[str(pid)].copy()
            g["bootstrap_draw_id"] = draw_id
            pieces.append(g)
        boot_df = pd.concat(pieces, ignore_index=True)
        vals = slope_decomposition(boot_df)
        vals["rep"] = rep
        boot_rows.append(vals)
    boot = pd.DataFrame(boot_rows)
    summary = []
    for key, value in point.items():
        vals = boot[key].dropna().to_numpy()
        summary.append(
            {
                "subset": key,
                "point_estimate": float(value),
                "ci_low": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
                "ci_high": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
                "bootstrap_reps": int(len(vals)),
            }
        )
    return pd.DataFrame(summary), boot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-panel", type=Path, default=DEFAULT_TRIAL_PANEL)
    parser.add_argument("--h2-panel", type=Path, default=DEFAULT_H2_PANEL)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    trial_panel = prepare_trial_panel(args.trial_panel)
    threshold_summary, threshold_gate_models, threshold_movement_models = threshold_sensitivity(trial_panel)
    h2 = prepare_h2_with_classification(args.h2_panel, args.classification)
    slope_summary, slope_bootstrap = bootstrap_slope_decomposition(h2, reps=args.bootstrap_reps, seed=args.random_seed)

    threshold_summary.to_csv(args.outdir / "update_gate_threshold_sensitivity.csv", index=False)
    threshold_gate_models.to_csv(args.outdir / "update_gate_threshold_logit_models.csv", index=False)
    threshold_movement_models.to_csv(args.outdir / "conditional_movement_threshold_ols_models.csv", index=False)
    slope_summary.to_csv(args.outdir / "slope_decomposition_bootstrap_summary.csv", index=False)
    slope_bootstrap.to_csv(args.outdir / "slope_decomposition_bootstrap_draws.csv", index=False)

    manifest = {
        "trial_panel": str(args.trial_panel),
        "h2_panel": str(args.h2_panel),
        "classification": str(args.classification),
        "outdir": str(args.outdir),
        "thresholds": THRESHOLDS,
        "bootstrap_reps": int(args.bootstrap_reps),
        "random_seed": int(args.random_seed),
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[Update-gate robustness] outputs written to", args.outdir)
    print("\nThreshold sensitivity:")
    print(threshold_summary.to_string(index=False))
    print("\nSlope decomposition with participant-bootstrap CIs:")
    print(slope_summary.to_string(index=False))


if __name__ == "__main__":
    main()
