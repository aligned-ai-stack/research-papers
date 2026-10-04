#!/usr/bin/env python
# coding: utf-8

"""Diagnostics for whether self-confidence explains visible belief inertia.

This standalone script asks whether higher self-confidence accounts for
non-updating after controlling for belief level, feedback discrepancy, task
domain/accuracy, trial position, and participant clustering.
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
DEFAULT_TASK_SUMMARY = ROOT / "data" / "followup_outputs" / "behavioral_extension_pipeline" / "task_summary_with_updater_types.csv"
DEFAULT_OUTDIR = ROOT / "data" / "followup_outputs" / "self_confidence_inertia_diagnostics"
EPS = 1e-12
MOVEMENT_THRESHOLDS = [
    ("exact", EPS, "gt"),
    ("ge_1pp", 0.01, "ge"),
    ("ge_2pp", 0.02, "ge"),
    ("ge_5pp", 0.05, "ge"),
]


def clean_trial_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = [
        "belief_lag",
        "conf_lag",
        "belief_post",
        "delta_belief",
        "ai_correct",
        "objective_ai_accuracy",
        "delegation",
        "trialOrdinal",
        "taskPosition",
        "self_was_correct",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["visible_update"] = (df["delta_belief"].abs() > EPS).astype(int)
    df["inertia_trial"] = 1 - df["visible_update"]
    df["signed_feedback_gap"] = df["ai_correct"] - df["belief_lag"]
    df["abs_feedback_gap"] = df["signed_feedback_gap"].abs()
    df["conf_x_abs_gap"] = df["conf_lag"] * df["abs_feedback_gap"]
    df["conf_lag_c"] = df["conf_lag"] - df["conf_lag"].mean()
    df["abs_feedback_gap_c"] = df["abs_feedback_gap"] - df["abs_feedback_gap"].mean()
    df["conf_x_abs_gap_c"] = df["conf_lag_c"] * df["abs_feedback_gap_c"]
    for label, threshold, op in MOVEMENT_THRESHOLDS:
        col = f"visible_update_{label}"
        if op == "gt":
            df[col] = (df["delta_belief"].abs() > threshold).astype(int)
        else:
            df[col] = (df["delta_belief"].abs() >= threshold).astype(int)
    df = df.sort_values(["participant_id", "taskType", "trialOrdinal"]).copy()
    grouped = df.groupby(["participant_id", "taskType"], sort=False)
    df["next_conf_lag"] = grouped["conf_lag"].shift(-1)
    df["confidence_change_next"] = df["next_conf_lag"] - df["conf_lag"]
    df["abs_confidence_change_next"] = df["confidence_change_next"].abs()
    return df


def clean_task_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = [
        "initial_belief",
        "final_belief",
        "objective_ai_accuracy",
        "abs_calibration_error",
        "delegation_rate",
        "mean_belief_lag",
        "mean_conf_lag",
        "mean_abs_delta",
        "nonzero_updates_observed",
        "taskPosition",
        "final_accuracy",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_inertial"] = (df["updater_family"] == "inertial").astype(int)
    df["any_visible_update"] = (df["nonzero_updates_observed"] > 0).astype(int)
    df["mean_conf_lag_c"] = df["mean_conf_lag"] - df["mean_conf_lag"].mean()
    return df


def design_matrix(
    df: pd.DataFrame,
    continuous: list[str],
    categorical: list[str],
    interactions: list[tuple[str, str]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    parts = [np.ones((len(df), 1))]
    names = ["Intercept"]
    cat_levels: dict[str, list] = {}
    for col in continuous:
        parts.append(df[[col]].astype(float).to_numpy())
        names.append(col)
    for col in categorical:
        levels = sorted([x for x in df[col].dropna().unique()])
        cat_levels[col] = levels
        for level in levels[1:]:
            parts.append((df[col] == level).astype(float).to_numpy().reshape(-1, 1))
            names.append(f"{col}={level}")
    if interactions:
        for left, right in interactions:
            if right in cat_levels:
                for level in cat_levels[right][1:]:
                    parts.append((df[left].astype(float) * (df[right] == level).astype(float)).to_numpy().reshape(-1, 1))
                    names.append(f"{left}:{right}={level}")
            else:
                parts.append((df[left].astype(float) * df[right].astype(float)).to_numpy().reshape(-1, 1))
                names.append(f"{left}:{right}")
    return np.column_stack(parts), names


def cluster_sandwich_cov(bread_inv: np.ndarray, scores: np.ndarray, groups: pd.Series, n: int, k: int) -> tuple[np.ndarray, int]:
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


def logistic_fit(
    df: pd.DataFrame,
    y_col: str,
    continuous: list[str],
    categorical: list[str],
    interactions: list[tuple[str, str]] | None = None,
    cluster_col: str = "participant_id",
    model_name: str = "",
) -> pd.DataFrame:
    needed = [y_col] + continuous + categorical + ([cluster_col] if cluster_col else [])
    if interactions:
        needed += [x for pair in interactions for x in pair]
    needed = list(dict.fromkeys(needed))
    m = df.dropna(subset=needed).copy()
    X, names = design_matrix(m, continuous, categorical, interactions)
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
    if cluster_col:
        cov, n_clusters = cluster_sandwich_cov(bread_inv, X * (y - p)[:, None], m[cluster_col], len(y), X.shape[1])
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    else:
        n_clusters = np.nan
        se = np.sqrt(np.clip(np.diag(bread_inv), 0, np.inf))
    ll = float(np.sum(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
    out = pd.DataFrame(
        {
            "model": model_name,
            "outcome": y_col,
            "term": names,
            "coef": beta,
            "se_cluster": se,
            "z_cluster": beta / np.where(se == 0, np.nan, se),
            "odds_ratio": np.exp(np.clip(beta, -20, 20)),
            "n": len(y),
            "n_clusters": n_clusters,
            "mean_outcome": float(np.mean(y)),
            "log_likelihood": ll,
        }
    )
    return out


def ols_fit(
    df: pd.DataFrame,
    y_col: str,
    continuous: list[str],
    categorical: list[str],
    interactions: list[tuple[str, str]] | None = None,
    cluster_col: str = "participant_id",
    model_name: str = "",
) -> pd.DataFrame:
    needed = [y_col] + continuous + categorical + ([cluster_col] if cluster_col else [])
    if interactions:
        needed += [x for pair in interactions for x in pair]
    needed = list(dict.fromkeys(needed))
    m = df.dropna(subset=needed).copy()
    X, names = design_matrix(m, continuous, categorical, interactions)
    y = m[y_col].astype(float).to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    if cluster_col:
        cov, n_clusters = cluster_sandwich_cov(xtx_inv, X * resid[:, None], m[cluster_col], n, k)
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    else:
        n_clusters = np.nan
        sigma2 = float(np.dot(resid, resid) / max(n - k, 1))
        se = np.sqrt(np.clip(np.diag(sigma2 * xtx_inv), 0, np.inf))
    r2 = 1.0 - float(np.sum(resid**2) / np.sum((y - y.mean()) ** 2)) if np.sum((y - y.mean()) ** 2) > 0 else np.nan
    return pd.DataFrame(
        {
            "model": model_name,
            "outcome": y_col,
            "term": names,
            "coef": beta,
            "se_cluster": se,
            "t_cluster": beta / np.where(se == 0, np.nan, se),
            "n": n,
            "n_clusters": n_clusters,
            "r2": r2,
        }
    )


def statsmodels_logit_checks(task: pd.DataFrame, trial: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "model": "statsmodels_unavailable",
                    "term": "",
                    "coef": np.nan,
                    "se_cluster": np.nan,
                    "z_cluster": np.nan,
                    "odds_ratio": np.nan,
                    "n": 0,
                    "status": f"skipped: {type(exc).__name__}: {exc}",
                }
            ]
        )

    specs = [
        (
            "task_inertia_conf_belief_accuracy_position_statsmodels",
            "is_inertial ~ mean_conf_lag_c + initial_belief + objective_ai_accuracy + C(taskPosition)",
            task.dropna(subset=["is_inertial", "mean_conf_lag_c", "initial_belief", "objective_ai_accuracy", "taskPosition", "participant_id"]),
        ),
        (
            "trial_update_conf_gap_interaction_centered_statsmodels",
            "visible_update ~ conf_lag_c + abs_feedback_gap_c + belief_lag + conf_x_abs_gap_c + C(taskType) + C(trialOrdinal)",
            trial.dropna(
                subset=[
                    "visible_update",
                    "conf_lag_c",
                    "abs_feedback_gap_c",
                    "belief_lag",
                    "conf_x_abs_gap_c",
                    "taskType",
                    "trialOrdinal",
                    "participant_id",
                ]
            ),
        ),
    ]
    rows = []
    for model_name, formula, df in specs:
        try:
            robust = smf.logit(formula, data=df).fit(
                disp=False,
                cov_type="cluster",
                cov_kwds={"groups": df["participant_id"].astype(str)},
            )
            for term, coef, se, z in zip(robust.params.index, robust.params, robust.bse, robust.tvalues):
                rows.append(
                    {
                        "model": model_name,
                        "term": term,
                        "coef": float(coef),
                        "se_cluster": float(se),
                        "z_cluster": float(z),
                        "odds_ratio": float(np.exp(np.clip(coef, -20, 20))),
                        "n": int(robust.nobs),
                        "status": "ok",
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "term": "",
                    "coef": np.nan,
                    "se_cluster": np.nan,
                    "z_cluster": np.nan,
                    "odds_ratio": np.nan,
                    "n": int(len(df)),
                    "status": f"skipped: {type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows)


def threshold_confidence_models(trial: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summaries = []
    for label, threshold, op in MOVEMENT_THRESHOLDS:
        outcome = f"visible_update_{label}"
        if outcome not in trial.columns:
            continue
        display_threshold = "exact" if label == "exact" else f">={threshold:.2f}"
        confidence_only = logistic_fit(
            trial,
            outcome,
            continuous=["conf_lag"],
            categorical=[],
            model_name=f"{label}_confidence_only",
        )
        controlled = logistic_fit(
            trial,
            outcome,
            continuous=["conf_lag", "abs_feedback_gap", "belief_lag"],
            categorical=["taskType", "trialOrdinal"],
            model_name=f"{label}_conf_gap_belief_task_trial",
        )
        interaction = logistic_fit(
            trial,
            outcome,
            continuous=["conf_lag_c", "abs_feedback_gap_c", "belief_lag", "conf_x_abs_gap_c"],
            categorical=["taskType", "trialOrdinal"],
            model_name=f"{label}_conf_gap_interaction_centered",
        )
        model_rows = pd.concat([confidence_only, controlled, interaction], ignore_index=True)
        model_rows.insert(0, "threshold_label", label)
        model_rows.insert(1, "movement_threshold", display_threshold)
        rows.append(model_rows)

        controlled_conf = controlled[controlled["term"] == "conf_lag"].iloc[0]
        controlled_gap = controlled[controlled["term"] == "abs_feedback_gap"].iloc[0]
        interaction_conf = interaction[interaction["term"] == "conf_lag_c"].iloc[0]
        interaction_gap = interaction[interaction["term"] == "abs_feedback_gap_c"].iloc[0]
        interaction_term = interaction[interaction["term"] == "conf_x_abs_gap_c"].iloc[0]
        summaries.append(
            {
                "threshold_label": label,
                "movement_threshold": display_threshold,
                "visible_update_share": float(trial[outcome].mean()),
                "n_trials": int(trial[outcome].notna().sum()),
                "controlled_conf_coef": float(controlled_conf["coef"]),
                "controlled_conf_z": float(controlled_conf["z_cluster"]),
                "controlled_conf_or": float(controlled_conf["odds_ratio"]),
                "controlled_abs_gap_coef": float(controlled_gap["coef"]),
                "controlled_abs_gap_z": float(controlled_gap["z_cluster"]),
                "controlled_abs_gap_or": float(controlled_gap["odds_ratio"]),
                "centered_conf_coef": float(interaction_conf["coef"]),
                "centered_conf_z": float(interaction_conf["z_cluster"]),
                "centered_conf_or": float(interaction_conf["odds_ratio"]),
                "centered_abs_gap_coef": float(interaction_gap["coef"]),
                "centered_abs_gap_z": float(interaction_gap["z_cluster"]),
                "centered_abs_gap_or": float(interaction_gap["odds_ratio"]),
                "centered_conf_x_abs_gap_coef": float(interaction_term["coef"]),
                "centered_conf_x_abs_gap_z": float(interaction_term["z_cluster"]),
                "centered_conf_x_abs_gap_or": float(interaction_term["odds_ratio"]),
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def confidence_descriptives(task: pd.DataFrame, trial: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_family = (
        task.groupby("updater_family", as_index=False)
        .agg(
            participant_tasks=("taskType", "size"),
            participants=("participant_id", "nunique"),
            mean_confidence=("mean_conf_lag", "mean"),
            mean_initial_belief=("initial_belief", "mean"),
            mean_objective_ai_accuracy=("objective_ai_accuracy", "mean"),
            mean_abs_delta=("mean_abs_delta", "mean"),
            mean_nonzero_updates=("nonzero_updates_observed", "mean"),
            inertial_share=("is_inertial", "mean"),
        )
        .sort_values("participant_tasks", ascending=False)
    )
    by_task = (
        task.groupby("taskType", as_index=False)
        .agg(
            participant_tasks=("taskType", "size"),
            mean_confidence=("mean_conf_lag", "mean"),
            mean_initial_belief=("initial_belief", "mean"),
            mean_objective_ai_accuracy=("objective_ai_accuracy", "mean"),
            mean_abs_delta=("mean_abs_delta", "mean"),
            inertial_share=("is_inertial", "mean"),
        )
        .sort_values("taskType")
    )
    trial_by_update = (
        trial.groupby("visible_update", as_index=False)
        .agg(
            trials=("trialOrdinal", "size"),
            mean_confidence=("conf_lag", "mean"),
            mean_belief=("belief_lag", "mean"),
            mean_abs_feedback_gap=("abs_feedback_gap", "mean"),
            mean_objective_ai_accuracy=("objective_ai_accuracy", "mean"),
            mean_abs_confidence_change_next=("abs_confidence_change_next", "mean"),
        )
    )
    task_q = task.copy()
    task_q["confidence_quartile"] = pd.qcut(
        task_q["mean_conf_lag"],
        4,
        labels=["Q1 low", "Q2", "Q3", "Q4 high"],
        duplicates="drop",
    )
    by_task_conf_quartile = (
        task_q.groupby("confidence_quartile", observed=True, as_index=False)
        .agg(
            participant_tasks=("taskType", "size"),
            mean_confidence=("mean_conf_lag", "mean"),
            inertial_share=("is_inertial", "mean"),
            mean_initial_belief=("initial_belief", "mean"),
            mean_abs_delta=("mean_abs_delta", "mean"),
        )
    )
    trial_q = trial.copy()
    trial_q["confidence_quartile"] = pd.qcut(
        trial_q["conf_lag"],
        4,
        labels=["Q1 low", "Q2", "Q3", "Q4 high"],
        duplicates="drop",
    )
    by_trial_conf_quartile = (
        trial_q.groupby("confidence_quartile", observed=True, as_index=False)
        .agg(
            trials=("trialOrdinal", "size"),
            mean_confidence=("conf_lag", "mean"),
            visible_update_share=("visible_update", "mean"),
            mean_belief=("belief_lag", "mean"),
            mean_abs_feedback_gap=("abs_feedback_gap", "mean"),
        )
    )
    return {
        "confidence_by_updater_family": by_family,
        "confidence_by_task": by_task,
        "confidence_by_visible_update": trial_by_update,
        "confidence_quartiles_task_level": by_task_conf_quartile,
        "confidence_quartiles_trial_level": by_trial_conf_quartile,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-panel", type=Path, default=DEFAULT_TRIAL_PANEL)
    parser.add_argument("--task-summary", type=Path, default=DEFAULT_TASK_SUMMARY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    trial = clean_trial_panel(args.trial_panel)
    task = clean_task_summary(args.task_summary)

    desc = confidence_descriptives(task, trial)
    for name, df in desc.items():
        df.to_csv(args.outdir / f"{name}.csv", index=False)

    task_models = []
    task_models.append(
        logistic_fit(
            task,
            "is_inertial",
            continuous=["mean_conf_lag"],
            categorical=[],
            model_name="task_inertia_confidence_only",
        )
    )
    task_models.append(
        logistic_fit(
            task,
            "is_inertial",
            continuous=["mean_conf_lag", "initial_belief", "objective_ai_accuracy"],
            categorical=["taskPosition"],
            model_name="task_inertia_conf_belief_accuracy_position",
        )
    )
    task_models.append(
        logistic_fit(
            task,
            "is_inertial",
            continuous=["mean_conf_lag", "initial_belief"],
            categorical=["taskType", "taskPosition"],
            model_name="task_inertia_conf_belief_task_controls",
        )
    )
    task_models.append(
        logistic_fit(
            task,
            "any_visible_update",
            continuous=["mean_conf_lag", "initial_belief", "objective_ai_accuracy"],
            categorical=["taskPosition"],
            model_name="task_any_visible_update_conf_belief_accuracy_position",
        )
    )
    task_models = pd.concat(task_models, ignore_index=True)
    task_models.to_csv(args.outdir / "task_inertia_logit_models.csv", index=False)

    trial_models = []
    trial_models.append(
        logistic_fit(
            trial,
            "visible_update",
            continuous=["conf_lag"],
            categorical=[],
            model_name="trial_update_confidence_only",
        )
    )
    trial_models.append(
        logistic_fit(
            trial,
            "visible_update",
            continuous=["conf_lag", "abs_feedback_gap", "belief_lag"],
            categorical=["taskType", "trialOrdinal"],
            model_name="trial_update_conf_gap_belief_task_trial",
        )
    )
    trial_models.append(
        logistic_fit(
            trial,
            "visible_update",
            continuous=["conf_lag_c", "abs_feedback_gap_c", "belief_lag", "conf_x_abs_gap_c"],
            categorical=["taskType", "trialOrdinal"],
            model_name="trial_update_conf_gap_interaction_centered",
        )
    )
    trial_models = pd.concat(trial_models, ignore_index=True)
    trial_models.to_csv(args.outdir / "trial_visible_update_logit_models.csv", index=False)

    threshold_models, threshold_summary = threshold_confidence_models(trial)
    threshold_models.to_csv(args.outdir / "confidence_threshold_visible_update_logit_models.csv", index=False)
    threshold_summary.to_csv(args.outdir / "confidence_threshold_summary.csv", index=False)

    statsmodels_checks = statsmodels_logit_checks(task, trial)
    statsmodels_checks.to_csv(args.outdir / "statsmodels_logit_crosschecks.csv", index=False)

    moved = trial[trial["visible_update"] == 1].copy()
    movement_models = []
    movement_models.append(
        ols_fit(
            moved,
            "delta_belief",
            continuous=["conf_lag", "signed_feedback_gap", "belief_lag"],
            categorical=["taskType", "trialOrdinal"],
            model_name="conditional_movement_conf_signed_gap",
        )
    )
    movement_models = pd.concat(movement_models, ignore_index=True)
    movement_models.to_csv(args.outdir / "conditional_movement_confidence_ols.csv", index=False)

    zero = trial[trial["visible_update"] == 0].copy()
    zero_conf_change = (
        zero.groupby("updater_family", as_index=False)
        .agg(
            zero_rows=("visible_update", "size"),
            participants=("participant_id", "nunique"),
            mean_confidence=("conf_lag", "mean"),
            mean_abs_feedback_gap=("abs_feedback_gap", "mean"),
            mean_abs_confidence_change_next=("abs_confidence_change_next", "mean"),
            mean_confidence_change_next=("confidence_change_next", "mean"),
        )
        .sort_values("zero_rows", ascending=False)
    )
    zero_conf_change.to_csv(args.outdir / "zero_update_confidence_change_by_family.csv", index=False)

    key_terms = pd.concat(
        [
            task_models[task_models["term"].isin(["mean_conf_lag", "initial_belief", "objective_ai_accuracy"])],
            trial_models[
                trial_models["term"].isin(
                    [
                        "conf_lag",
                        "abs_feedback_gap",
                        "conf_lag_c",
                        "abs_feedback_gap_c",
                        "belief_lag",
                        "conf_x_abs_gap_c",
                    ]
                )
            ],
            movement_models[movement_models["term"].isin(["conf_lag", "signed_feedback_gap", "belief_lag"])],
            statsmodels_checks[
                statsmodels_checks["term"].isin(
                    ["mean_conf_lag_c", "conf_lag_c", "abs_feedback_gap_c", "belief_lag", "conf_x_abs_gap_c"]
                )
            ],
        ],
        ignore_index=True,
    )
    key_terms.to_csv(args.outdir / "key_self_confidence_terms.csv", index=False)

    manifest = {
        "trial_panel": str(args.trial_panel),
        "task_summary": str(args.task_summary),
        "outdir": str(args.outdir),
        "participants": int(trial["participant_id"].nunique()),
        "trials": int(len(trial)),
        "participant_tasks": int(len(task)),
        "interpretation": "Descriptive diagnostics; confidence association is not causal.",
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[Self-confidence inertia diagnostics] outputs written to", args.outdir)
    print("\nConfidence by updater family:")
    print(desc["confidence_by_updater_family"].to_string(index=False))
    print("\nConfidence by visible update:")
    print(desc["confidence_by_visible_update"].to_string(index=False))
    print("\nKey model terms:")
    print(key_terms.to_string(index=False))


if __name__ == "__main__":
    main()
