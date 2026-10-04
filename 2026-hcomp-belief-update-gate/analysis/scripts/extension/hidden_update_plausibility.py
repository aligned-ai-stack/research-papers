#!/usr/bin/env python
# coding: utf-8

"""Hidden-update plausibility and participant-pattern clustering.

This diagnostic stays separate from the paper-facing extension pipeline. It asks
whether zero observed report updates look more like plausible hidden latent
updates under coarse reporting, or more like visible inertia despite feedback
that should have moved the report.

The analysis is intentionally assumption-light:

1. Estimate each participant's reporting granularity from all reported belief
   values across the three tasks.
2. Fit a conditional movement slope among rows where reported belief moved.
3. For zero-update rows, compare the expected movement implied by that slope to
   the participant-specific reporting threshold.
4. Summarize participant-level updating patterns and cluster participants using
   transparent k-means over diagnostic features.

These outputs are descriptive. They do not point-identify latent updating.
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
DEFAULT_OUTDIR = ROOT / "data" / "followup_outputs" / "hidden_update_plausibility"

GRID_CANDIDATES = np.asarray([0.01, 0.02, 0.05, 0.10, 0.20], dtype=float)
EPS = 1e-12


def ensure_gate_columns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for col in ["belief_lag", "belief_post", "delta_belief", "ai_correct", "delegation", "conf_lag"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "visible_update" not in out.columns:
        out["visible_update"] = np.where(out["delta_belief"].notna(), (out["delta_belief"].abs() > EPS).astype(int), np.nan)
    if "signed_feedback_gap" not in out.columns:
        out["signed_feedback_gap"] = out["ai_correct"] - out["belief_lag"]
    if "abs_feedback_gap" not in out.columns:
        out["abs_feedback_gap"] = out["signed_feedback_gap"].abs()
    return out


def add_next_trial_behavior(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values(["participant_id", "taskType", "trialOrdinal"]).copy()
    grouped = out.groupby(["participant_id", "taskType"], sort=False)
    out["next_delegation"] = grouped["delegation"].shift(-1)
    out["next_conf_lag"] = grouped["conf_lag"].shift(-1)
    out["delegation_change_next"] = out["next_delegation"] - out["delegation"]
    out["abs_delegation_change_next"] = out["delegation_change_next"].abs()
    out["feedback_consistent_delegation_change"] = np.where(
        out["ai_correct"] >= 0.5,
        out["delegation_change_next"],
        -out["delegation_change_next"],
    )
    out["confidence_change_next"] = out["next_conf_lag"] - out["conf_lag"]
    out["abs_confidence_change_next"] = out["confidence_change_next"].abs()
    return out


def distance_to_grid(values: np.ndarray, grid: float) -> np.ndarray:
    return np.abs(values - np.round(values / grid) * grid)


def estimate_reporting_granularity(panel: pd.DataFrame, grid_share_threshold: float) -> pd.DataFrame:
    rows = []
    for pid, g in panel.groupby("participant_id", sort=True):
        reports = pd.concat([g["belief_lag"], g["belief_post"]], ignore_index=True)
        reports = reports.dropna().astype(float).clip(0.0, 1.0).to_numpy()
        reports = np.unique(np.round(reports, 4))
        deltas = g["delta_belief"].astype(float).abs()
        nonzero = deltas[deltas > EPS]

        shares = {}
        for grid in GRID_CANDIDATES:
            if len(reports) == 0:
                shares[float(grid)] = np.nan
            else:
                shares[float(grid)] = float(np.mean(distance_to_grid(reports, float(grid)) <= 0.005 + EPS))
        eligible = [grid for grid, share in shares.items() if np.isfinite(share) and share >= grid_share_threshold]
        dominant_grid = max(eligible) if eligible else 0.01

        rows.append(
            {
                "participant_id": pid,
                "n_report_values": int(len(reports)),
                "n_distinct_report_values": int(len(np.unique(reports))),
                "min_nonzero_delta": float(nonzero.min()) if len(nonzero) else np.nan,
                "median_nonzero_delta": float(nonzero.median()) if len(nonzero) else np.nan,
                "dominant_grid": float(dominant_grid),
                "reporting_threshold": float(dominant_grid / 2.0),
                "share_on_grid_001": shares[0.01],
                "share_on_grid_002": shares[0.02],
                "share_on_grid_005": shares[0.05],
                "share_on_grid_010": shares[0.10],
                "share_on_grid_020": shares[0.20],
            }
        )
    return pd.DataFrame(rows)


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


def fit_conditional_slope(panel: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    needed = ["delta_belief", "signed_feedback_gap", "belief_lag", "conf_lag", "taskType", "trialOrdinal", "updater_family"]
    moved = panel[(panel["visible_update"] == 1)].dropna(subset=needed).copy()
    X, names = design_matrix(
        moved,
        continuous=["signed_feedback_gap", "belief_lag", "conf_lag"],
        categorical=["taskType", "trialOrdinal", "updater_family"],
    )
    y = moved["delta_belief"].astype(float).to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = float(np.dot(resid, resid) / max(n - k, 1))
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    coef = pd.DataFrame(
        {
            "term": names,
            "coef": beta,
            "se_ols": se,
            "t_ols": beta / np.where(se == 0, np.nan, se),
            "n": n,
            "r2": 1.0 - float(np.sum(resid**2) / np.sum((y - y.mean()) ** 2)) if np.sum((y - y.mean()) ** 2) > 0 else np.nan,
        }
    )
    slope = float(coef.loc[coef["term"] == "signed_feedback_gap", "coef"].iloc[0])
    return slope, coef


def classify_zero_rows(panel: pd.DataFrame, granularity: pd.DataFrame, conditional_slope: float) -> pd.DataFrame:
    merged = panel.merge(granularity[["participant_id", "dominant_grid", "reporting_threshold"]], on="participant_id", how="left")
    zero = merged[(merged["visible_update"] == 0)].copy()
    zero["expected_latent_delta"] = conditional_slope * zero["signed_feedback_gap"].astype(float)
    zero["expected_abs_latent_delta"] = zero["expected_latent_delta"].abs()
    zero["hidden_update_plausible"] = zero["expected_abs_latent_delta"] < zero["reporting_threshold"]
    zero["inertia_plausible"] = ~zero["hidden_update_plausible"]
    zero["zero_type"] = np.where(zero["hidden_update_plausible"], "hidden_update_plausible", "inertia_plausible")
    keep = [
        "participant_id",
        "taskType",
        "taskPosition",
        "trialOrdinal",
        "belief_lag",
        "belief_post",
        "ai_correct",
        "delegation",
        "conf_lag",
        "signed_feedback_gap",
        "abs_feedback_gap",
        "dominant_grid",
        "reporting_threshold",
        "expected_latent_delta",
        "expected_abs_latent_delta",
        "zero_type",
        "next_delegation",
        "delegation_change_next",
        "abs_delegation_change_next",
        "feedback_consistent_delegation_change",
        "next_conf_lag",
        "confidence_change_next",
        "abs_confidence_change_next",
        "updater_model",
        "updater_family",
    ]
    return zero[keep].sort_values(["participant_id", "taskType", "trialOrdinal"])


def summarize_zero_groups(zero_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        zero_rows.groupby("zero_type", as_index=False)
        .agg(
            rows=("zero_type", "size"),
            participants=("participant_id", "nunique"),
            mean_abs_feedback_gap=("abs_feedback_gap", "mean"),
            mean_expected_abs_latent_delta=("expected_abs_latent_delta", "mean"),
            mean_reporting_threshold=("reporting_threshold", "mean"),
            mean_abs_delegation_change_next=("abs_delegation_change_next", "mean"),
            mean_feedback_consistent_delegation_change=("feedback_consistent_delegation_change", "mean"),
            mean_abs_confidence_change_next=("abs_confidence_change_next", "mean"),
        )
        .sort_values("zero_type")
    )


def participant_simple_slopes(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in panel.groupby("participant_id", sort=True):
        moved = g[(g["visible_update"] == 1)].dropna(subset=["delta_belief", "signed_feedback_gap"])
        if len(moved) >= 3 and float(moved["signed_feedback_gap"].var()) > 0:
            x = moved["signed_feedback_gap"].astype(float).to_numpy()
            y = moved["delta_belief"].astype(float).to_numpy()
            x = x - x.mean()
            slope = float(np.dot(x, y - y.mean()) / np.dot(x, x))
        else:
            slope = np.nan
        rows.append({"participant_id": pid, "participant_conditional_slope": slope, "moved_rows_for_slope": int(len(moved))})
    return pd.DataFrame(rows)


def participant_features(panel: pd.DataFrame, granularity: pd.DataFrame, zero_rows: pd.DataFrame) -> pd.DataFrame:
    base = (
        panel.groupby("participant_id", as_index=False)
        .agg(
            trials=("trialOrdinal", "size"),
            tasks=("taskType", "nunique"),
            zero_update_share=("visible_update", lambda s: float(np.mean(s == 0))),
            substantive_update_share=("substantive_update_05", "mean"),
            mean_abs_delta=("delta_belief", lambda s: float(np.nanmean(np.abs(s)))),
            mean_abs_feedback_gap=("abs_feedback_gap", "mean"),
            mean_delegation=("delegation", "mean"),
            mean_confidence=("conf_lag", "mean"),
        )
    )
    task_family = (
        panel.drop_duplicates(["participant_id", "taskType"])
        .groupby("participant_id")
        .agg(
            inertial_task_share=("updater_family", lambda s: float(np.mean(s == "inertial"))),
            evidence_accumulation_task_share=("updater_family", lambda s: float(np.mean(s == "evidence_accumulation"))),
            asymmetric_task_share=("updater_family", lambda s: float(np.mean(s == "asymmetric"))),
            not_classified_task_share=("updater_family", lambda s: float(np.mean(s == "not_classified"))),
        )
        .reset_index()
    )
    zero_summary = (
        zero_rows.groupby("participant_id", as_index=False)
        .agg(
            zero_rows=("zero_type", "size"),
            hidden_update_plausible_zero_share=("zero_type", lambda s: float(np.mean(s == "hidden_update_plausible"))),
            inertia_plausible_zero_share=("zero_type", lambda s: float(np.mean(s == "inertia_plausible"))),
            mean_expected_abs_latent_delta_zero=("expected_abs_latent_delta", "mean"),
            mean_abs_delegation_change_after_zero=("abs_delegation_change_next", "mean"),
            mean_feedback_consistent_delegation_change_after_zero=("feedback_consistent_delegation_change", "mean"),
        )
    )
    slopes = participant_simple_slopes(panel)
    out = base.merge(task_family, on="participant_id", how="left")
    out = out.merge(granularity, on="participant_id", how="left")
    out = out.merge(zero_summary, on="participant_id", how="left")
    out = out.merge(slopes, on="participant_id", how="left")
    fill_zero_cols = ["zero_rows", "hidden_update_plausible_zero_share", "inertia_plausible_zero_share"]
    for col in fill_zero_cols:
        out[col] = out[col].fillna(0.0)
    return out


def standardize_features(features: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    X = features[cols].astype(float).copy()
    impute = {}
    for col in cols:
        median = float(X[col].median()) if X[col].notna().any() else 0.0
        impute[col] = median
        X[col] = X[col].fillna(median)
    means = X.mean(axis=0)
    sds = X.std(axis=0, ddof=0).replace(0, 1.0)
    Z = ((X - means) / sds).to_numpy()
    meta = pd.DataFrame({"feature": cols, "impute_value": [impute[c] for c in cols], "mean": means.values, "sd": sds.values})
    return Z, meta


def kmeans_once(Z: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray, float]:
    n = Z.shape[0]
    centroids = Z[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dist = ((Z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for j in range(k):
            if np.any(labels == j):
                centroids[j] = Z[labels == j].mean(axis=0)
            else:
                centroids[j] = Z[rng.integers(0, n)]
    inertia = float(((Z - centroids[labels]) ** 2).sum())
    return labels, centroids, inertia


def silhouette_score(Z: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return np.nan
    dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for i in range(len(Z)):
        own = labels == labels[i]
        other_labels = [lab for lab in unique if lab != labels[i]]
        a = float(dist[i, own].sum() / max(int(own.sum()) - 1, 1))
        b = min(float(dist[i, labels == lab].mean()) for lab in other_labels)
        denom = max(a, b)
        scores.append(0.0 if denom == 0 else (b - a) / denom)
    return float(np.mean(scores))


def run_kmeans_selection(Z: np.ndarray, seed: int, min_k: int = 2, max_k: int = 6, n_init: int = 100) -> tuple[pd.DataFrame, np.ndarray, int]:
    rows = []
    best_labels = None
    best_k = None
    best_silhouette = -np.inf
    rng = np.random.default_rng(seed)
    for k in range(min_k, max_k + 1):
        best_for_k = None
        for _ in range(n_init):
            labels, centroids, inertia = kmeans_once(Z, k, rng)
            if best_for_k is None or inertia < best_for_k[2]:
                best_for_k = (labels, centroids, inertia)
        labels, _centroids, inertia = best_for_k
        sil = silhouette_score(Z, labels)
        rows.append({"k": k, "inertia": inertia, "silhouette": sil})
        if np.isfinite(sil) and sil > best_silhouette:
            best_silhouette = sil
            best_k = k
            best_labels = labels.copy()
    return pd.DataFrame(rows), best_labels, int(best_k)


def name_clusters(summary: pd.DataFrame) -> dict[int, str]:
    names = {}
    for _, row in summary.iterrows():
        cluster = int(row["cluster"])
        if row["zero_update_share"] >= 0.75 and row["reporting_threshold"] >= 0.025:
            label = "coarse_flat_reporters"
        elif row["zero_update_share"] >= 0.75 and row["inertia_plausible_zero_share"] >= 0.55:
            label = "visible_inertia"
        elif row["hidden_update_plausible_zero_share"] >= 0.65 and row["reporting_threshold"] >= 0.025:
            label = "coarse_hidden_update_plausible"
        elif row["zero_update_share"] <= 0.50 and row["mean_abs_delta"] >= 0.05:
            label = "active_feedback_movers"
        else:
            label = "mixed_or_weak_update"
        suffix = 2
        base = label
        while label in names.values():
            label = f"{base}_{suffix}"
            suffix += 1
        names[cluster] = label
    return names


def cluster_participants(features: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_cols = [
        "zero_update_share",
        "substantive_update_share",
        "mean_abs_delta",
        "dominant_grid",
        "reporting_threshold",
        "hidden_update_plausible_zero_share",
        "inertia_plausible_zero_share",
        "participant_conditional_slope",
        "inertial_task_share",
        "evidence_accumulation_task_share",
        "asymmetric_task_share",
    ]
    Z, feature_meta = standardize_features(features, cluster_cols)
    selection, labels, best_k = run_kmeans_selection(Z, seed=seed)
    assignments = features.copy()
    assignments["cluster"] = labels
    assignments["selected_k"] = best_k
    summary = (
        assignments.groupby("cluster", as_index=False)
        .agg(
            participants=("participant_id", "size"),
            zero_update_share=("zero_update_share", "mean"),
            substantive_update_share=("substantive_update_share", "mean"),
            mean_abs_delta=("mean_abs_delta", "mean"),
            dominant_grid=("dominant_grid", "mean"),
            reporting_threshold=("reporting_threshold", "mean"),
            hidden_update_plausible_zero_share=("hidden_update_plausible_zero_share", "mean"),
            inertia_plausible_zero_share=("inertia_plausible_zero_share", "mean"),
            participant_conditional_slope=("participant_conditional_slope", "mean"),
            inertial_task_share=("inertial_task_share", "mean"),
            evidence_accumulation_task_share=("evidence_accumulation_task_share", "mean"),
            asymmetric_task_share=("asymmetric_task_share", "mean"),
            mean_delegation=("mean_delegation", "mean"),
            mean_confidence=("mean_confidence", "mean"),
        )
        .sort_values(["zero_update_share", "mean_abs_delta"], ascending=[False, True])
    )
    cluster_names = name_clusters(summary)
    assignments["cluster_label"] = assignments["cluster"].map(cluster_names)
    summary["cluster_label"] = summary["cluster"].map(cluster_names)
    ordered_cols = ["cluster", "cluster_label"] + [c for c in summary.columns if c not in {"cluster", "cluster_label"}]
    summary = summary[ordered_cols]
    return assignments, summary, selection, feature_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-panel", type=Path, default=DEFAULT_TRIAL_PANEL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--grid-share-threshold", type=float, default=0.90)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    panel = add_next_trial_behavior(ensure_gate_columns(pd.read_csv(args.trial_panel)))

    granularity = estimate_reporting_granularity(panel, grid_share_threshold=args.grid_share_threshold)
    slope, slope_model = fit_conditional_slope(panel)
    zero_rows = classify_zero_rows(panel, granularity, conditional_slope=slope)
    zero_group_summary = summarize_zero_groups(zero_rows)
    features = participant_features(panel, granularity, zero_rows)
    assignments, cluster_summary, cluster_selection, feature_meta = cluster_participants(features, seed=args.random_seed)

    granularity.to_csv(args.outdir / "participant_reporting_granularity.csv", index=False)
    slope_model.to_csv(args.outdir / "conditional_movement_model.csv", index=False)
    zero_rows.to_csv(args.outdir / "zero_update_plausibility_by_row.csv", index=False)
    zero_group_summary.to_csv(args.outdir / "zero_update_plausibility_summary.csv", index=False)
    features.to_csv(args.outdir / "participant_hidden_update_features.csv", index=False)
    assignments.to_csv(args.outdir / "participant_cluster_assignments.csv", index=False)
    cluster_summary.to_csv(args.outdir / "participant_cluster_summary.csv", index=False)
    cluster_selection.to_csv(args.outdir / "participant_cluster_selection.csv", index=False)
    feature_meta.to_csv(args.outdir / "participant_cluster_feature_scaling.csv", index=False)

    manifest = {
        "trial_panel": str(args.trial_panel),
        "outdir": str(args.outdir),
        "participants": int(panel["participant_id"].nunique()),
        "trials": int(len(panel)),
        "zero_update_rows": int(len(zero_rows)),
        "conditional_movement_slope": float(slope),
        "grid_share_threshold": float(args.grid_share_threshold),
        "selected_k": int(cluster_summary["cluster"].nunique()),
        "random_seed": int(args.random_seed),
        "interpretation": "Descriptive diagnostic; zero reports do not point-identify latent non-updating.",
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[Hidden update plausibility] outputs written to", args.outdir)
    print("Conditional movement slope:", f"{slope:.3f}")
    print("\nZero-update plausibility summary:")
    print(zero_group_summary.to_string(index=False))
    print("\nCluster selection:")
    print(cluster_selection.to_string(index=False))
    print("\nParticipant cluster summary:")
    print(cluster_summary.to_string(index=False))


if __name__ == "__main__":
    main()
