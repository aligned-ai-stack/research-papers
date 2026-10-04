#!/usr/bin/env python
# coding: utf-8

"""Extension analyses linking updater type to carryover, calibration, and delegation.

The belief-model comparison is useful only if the latent regimes matter for
downstream behavior. This script builds a unified trial/task/transition panel
and runs paper-facing descriptive and regression checks:

1. Updater type distribution by task.
2. Delegation and calibration by updater type.
3. Cross-task prior carryover moderated by previous-task updater type.
4. Trial-level delegation models with belief, confidence, and updater type.
5. Objective AI accuracy reconstruction checks from `aiWrongMap`.
6. Hurdle-style update-gate diagnostics that separate visible report movement
   from the signed movement size conditional on movement.

The models are intentionally transparent and dependency-light. Reported
standard errors are clustered by participant because the panels contain
repeated task and trial observations per person.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "intermediate_outputs" / "valid_sessions.json"
CLASSIFICATION = ROOT / "data" / "followup_outputs" / "robust_belief_model_comparison" / "canonical_classification.csv"
OUTDIR = ROOT / "data" / "followup_outputs" / "behavioral_extension_pipeline"


def load_sessions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def mongo_date_value(value) -> str:
    if isinstance(value, dict):
        return str(value.get("$date", ""))
    return str(value or "")


def load_classification(path: Path = CLASSIFICATION) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing model classification file: {path}")
    out = pd.read_csv(path)
    for col in ["bic_winner", "aicc_winner", "future_block_winner", "loocv_winner"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    missing_aicc = out["aicc_winner"].str.len() == 0
    out.loc[missing_aicc, "aicc_winner"] = out.loc[missing_aicc, "bic_winner"]
    out["classification_source"] = str(path)
    out["participant_id"] = out["participant_id"].astype(str)
    out["taskType"] = out["taskType"].astype(str)
    return out


def is_valid_session(s: dict) -> bool:
    if not s.get("completed", False):
        return False
    if s.get("failedComprehension", False):
        return False
    sac = s.get("solutionAccessCount", {}) or {}
    if isinstance(sac, dict) and any((v or 0) > 1 for v in sac.values()):
        return False
    return True


def select_analysis_sessions(sessions: list[dict], order_condition_quota: int | None = 40) -> list[dict]:
    valid = [s for s in sessions if is_valid_session(s)]
    if order_condition_quota is None or order_condition_quota <= 0:
        return valid

    by_condition: dict[object, list[dict]] = defaultdict(list)
    for session in valid:
        key = session.get("orderCondition")
        if key is None:
            key = tuple(session.get("taskOrder", []) or [])
        by_condition[key].append(session)

    kept: list[dict] = []
    for _, group in sorted(by_condition.items(), key=lambda item: str(item[0])):
        ordered = sorted(
            group,
            key=lambda s: (
                mongo_date_value(s.get("completedAt")),
                mongo_date_value(s.get("createdAt")),
                str(s.get("participant_id", "")),
            ),
        )
        kept.extend(ordered[:order_condition_quota])
    return kept


def group_trials(session: dict) -> dict[str, list[dict]]:
    by_task = defaultdict(list)
    for r in session.get("responses", []):
        by_task[r.get("taskType")].append(r)
    for task in list(by_task):
        by_task[task].sort(key=lambda x: int(x.get("trialOrdinal", 0)))
    return by_task


def ai_correct_from_map(session: dict, task: str, trial_ordinal: int) -> float:
    wrong_map = (session.get("aiWrongMap", {}) or {}).get(task, {}) or {}
    wrong = wrong_map.get(str(int(trial_ordinal)), wrong_map.get(int(trial_ordinal)))
    if wrong is None:
        return 1.0
    return 0.0 if bool(wrong) else 1.0


def objective_ai_accuracy(session: dict, task: str, trials: list[dict]) -> float:
    vals = [ai_correct_from_map(session, task, int(t.get("trialOrdinal", 0))) for t in trials]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def safe_num(x, scale: float = 1.0):
    if x is None:
        return np.nan
    try:
        return float(x) / scale
    except Exception:
        return np.nan


def collapse_updater(model: str) -> str:
    if pd.isna(model) or model == "":
        return "not_classified"
    if model == "not_classified":
        return "not_classified"
    if model == "no_update":
        return "inertial"
    if model == "changepoint":
        return "phase_change"
    if model in {"good_bad_news", "asymmetric"}:
        return "asymmetric"
    if model in {"discounted_bayes_gamma", "discounted_weighted_bayes", "recency", "leaky"}:
        return "recency_discounted"
    if model == "sticky_weighted_bayes":
        return "response_expression"
    if model in {
        "standard_bayes",
        "divisible_weighted_bayes",
        "coarse_weighted_bayes",
        "coarse_bayes",
        "partial_step",
        "threshold_bayes",
        "anchoring_to_prior",
        "sublinear_sample_size",
        "rescorla_wagner",
        "confirmatory_misperception",
    }:
        return "evidence_accumulation"
    return "other"


def nonempty_value(row: dict, col: str, default: str = "not_classified") -> str:
    value = row.get(col, default)
    if pd.isna(value) or str(value) == "":
        return default
    return str(value)


def build_trial_panel(sessions: list[dict], classification: pd.DataFrame) -> pd.DataFrame:
    class_map = classification.set_index(["participant_id", "taskType"]).to_dict("index")
    rows = []
    for s in sessions:
        if not is_valid_session(s):
            continue
        pid = str(s.get("participant_id", ""))
        order = s.get("taskOrder", []) or []
        positions = {task: i + 1 for i, task in enumerate(order)}
        by_task = group_trials(s)
        for task in order:
            trials = by_task.get(task, [])
            if not trials:
                continue
            obj_acc = objective_ai_accuracy(s, task, trials)
            first_ai_correct = ai_correct_from_map(s, task, int(trials[0].get("trialOrdinal", 1))) if trials else np.nan
            first_delta = safe_num(trials[0].get("postBeliefAI"), 100.0) - safe_num(trials[0].get("preBeliefAI"), 100.0) if trials else np.nan
            cinfo = class_map.get((pid, task), {})
            bic_winner = nonempty_value(cinfo, "bic_winner")
            aicc_winner = nonempty_value(cinfo, "aicc_winner", bic_winner)
            future_winner = nonempty_value(cinfo, "future_block_winner", aicc_winner)
            loocv_winner = nonempty_value(cinfo, "loocv_winner", aicc_winner)
            winner = aicc_winner
            family = collapse_updater(winner)
            for idx, tr in enumerate(trials):
                t_ord = int(tr.get("trialOrdinal", 0))
                prev = trials[idx - 1] if idx > 0 else None
                b_prev = safe_num(tr.get("preBeliefAI"), 100.0) if idx == 0 else safe_num(prev.get("postBeliefAI"), 100.0)
                c_lag = safe_num(tr.get("preConfidenceSelf"), 100.0) if idx == 0 else safe_num(prev.get("postConfidenceSelf"), 100.0)
                b_post = safe_num(tr.get("postBeliefAI"), 100.0)
                used_ai = tr.get("usedAI")
                rows.append(
                    {
                        "participant_id": pid,
                        "taskType": task,
                        "taskPosition": positions.get(task),
                        "trialOrdinal": t_ord,
                        "delegation": np.nan if used_ai is None else int(bool(used_ai)),
                        "belief_lag": b_prev,
                        "conf_lag": c_lag,
                        "belief_post": b_post,
                        "delta_belief": b_post - b_prev if np.isfinite(b_prev) and np.isfinite(b_post) else np.nan,
                        "ai_correct": ai_correct_from_map(s, task, t_ord),
                        "objective_ai_accuracy": obj_acc,
                        "first_ai_correct": first_ai_correct,
                        "first_belief_delta": first_delta,
                        "final_was_correct": np.nan if tr.get("finalWasCorrect") is None else int(bool(tr.get("finalWasCorrect"))),
                        "self_was_correct": np.nan if tr.get("selfWasCorrect") is None else int(bool(tr.get("selfWasCorrect"))),
                        "updater_model": winner,
                        "updater_family": family,
                        "bic_updater_model": bic_winner,
                        "bic_updater_family": collapse_updater(bic_winner),
                        "aicc_updater_model": aicc_winner,
                        "aicc_updater_family": collapse_updater(aicc_winner),
                        "future_block_updater_model": future_winner,
                        "future_block_updater_family": collapse_updater(future_winner),
                        "loocv_updater_model": loocv_winner,
                        "loocv_updater_family": collapse_updater(loocv_winner),
                        "bic_aicc_agree": int(bic_winner == aicc_winner),
                        "aicc_future_agree": int(aicc_winner == future_winner),
                        "bic_winner_weight": cinfo.get("bic_winner_weight", np.nan),
                        "near_tie_models_delta_bic_2": cinfo.get("near_tie_models_delta_bic_2", np.nan),
                        "bic_delta_to_runner_up": cinfo.get("bic_delta_to_runner_up", np.nan),
                        "aicc_winner_weight": cinfo.get("aicc_winner_weight", np.nan),
                        "near_tie_models_delta_aicc_2": cinfo.get("near_tie_models_delta_aicc_2", np.nan),
                        "aicc_delta_to_runner_up": cinfo.get("aicc_delta_to_runner_up", np.nan),
                        "aicc_winner_uncertain": cinfo.get("aicc_winner_uncertain", np.nan),
                        "nonzero_updates": cinfo.get("nonzero_updates", np.nan),
                        "sum_abs_delta_model_panel": cinfo.get("sum_abs_delta", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def build_task_summary(trial_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, task), g in trial_panel.sort_values("trialOrdinal").groupby(["participant_id", "taskType"], sort=False):
        g = g.sort_values("trialOrdinal")
        b0 = float(g["belief_lag"].iloc[0])
        final_b = float(g["belief_post"].iloc[-1])
        obj = float(g["objective_ai_accuracy"].iloc[0])
        rows.append(
            {
                "participant_id": pid,
                "taskType": task,
                "taskPosition": int(g["taskPosition"].iloc[0]),
                "updater_model": g["updater_model"].iloc[0],
                "updater_family": g["updater_family"].iloc[0],
                "bic_updater_model": g["bic_updater_model"].iloc[0],
                "bic_updater_family": g["bic_updater_family"].iloc[0],
                "aicc_updater_model": g["aicc_updater_model"].iloc[0],
                "aicc_updater_family": g["aicc_updater_family"].iloc[0],
                "future_block_updater_model": g["future_block_updater_model"].iloc[0],
                "future_block_updater_family": g["future_block_updater_family"].iloc[0],
                "loocv_updater_model": g["loocv_updater_model"].iloc[0],
                "loocv_updater_family": g["loocv_updater_family"].iloc[0],
                "bic_aicc_agree": int(g["bic_aicc_agree"].iloc[0]),
                "aicc_future_agree": int(g["aicc_future_agree"].iloc[0]),
                "bic_winner_weight": float(g["bic_winner_weight"].iloc[0]),
                "near_tie_models_delta_bic_2": float(g["near_tie_models_delta_bic_2"].iloc[0]),
                "bic_delta_to_runner_up": float(g["bic_delta_to_runner_up"].iloc[0]),
                "aicc_winner_weight": float(g["aicc_winner_weight"].iloc[0]),
                "near_tie_models_delta_aicc_2": float(g["near_tie_models_delta_aicc_2"].iloc[0]),
                "aicc_delta_to_runner_up": float(g["aicc_delta_to_runner_up"].iloc[0]),
                "aicc_winner_uncertain": float(g["aicc_winner_uncertain"].iloc[0]),
                "initial_belief": b0,
                "final_belief": final_b,
                "objective_ai_accuracy": obj,
                "calibration_error": final_b - obj,
                "abs_calibration_error": abs(final_b - obj),
                "delegation_rate": float(g["delegation"].mean()),
                "mean_belief_lag": float(g["belief_lag"].mean()),
                "mean_conf_lag": float(g["conf_lag"].mean()),
                "mean_abs_delta": float(g["delta_belief"].abs().mean()),
                "nonzero_updates_observed": int((g["delta_belief"].abs() > 1e-12).sum()),
                "first_ai_correct": float(g["first_ai_correct"].iloc[0]),
                "first_belief_delta": float(g["first_belief_delta"].iloc[0]),
                "final_accuracy": float(g["final_was_correct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_transition_panel(task_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in task_summary.sort_values("taskPosition").groupby("participant_id"):
        g = g.sort_values("taskPosition").reset_index(drop=True)
        for idx in range(1, len(g)):
            curr = g.iloc[idx]
            prev = g.iloc[idx - 1]
            rows.append(
                {
                    "participant_id": pid,
                    "curr_taskType": curr["taskType"],
                    "curr_taskPosition": curr["taskPosition"],
                    "b0_curr": curr["initial_belief"],
                    "prev_taskType": prev["taskType"],
                    "bT_prev": prev["final_belief"],
                    "prev_objective_ai_accuracy": prev["objective_ai_accuracy"],
                    "prev_updater_model": prev["updater_model"],
                    "prev_updater_family": prev["updater_family"],
                    "prev_bic_updater_family": prev["bic_updater_family"],
                    "prev_aicc_updater_family": prev["aicc_updater_family"],
                    "prev_future_block_updater_family": prev["future_block_updater_family"],
                    "prev_abs_calibration_error": prev["abs_calibration_error"],
                    "prev_delegation_rate": prev["delegation_rate"],
                }
            )
    return pd.DataFrame(rows)


def objective_accuracy_validation(trial_panel: pd.DataFrame) -> pd.DataFrame:
    expected = {"grammar": 0.30, "travel": 0.60, "vqa": 0.90}
    rows = []
    for task, g in trial_panel.groupby("taskType", dropna=False):
        observed = float(g["ai_correct"].mean())
        target = expected.get(str(task), np.nan)
        rows.append(
            {
                "taskType": task,
                "observed_ai_accuracy": observed,
                "expected_ai_accuracy": target,
                "abs_diff": abs(observed - target) if np.isfinite(target) else np.nan,
                "n_trials": int(len(g)),
                "passes_tolerance_001": bool(not np.isfinite(target) or abs(observed - target) <= 0.01),
            }
        )
    return pd.DataFrame(rows).sort_values("taskType")


def add_update_gate_variables(trial_panel: pd.DataFrame) -> pd.DataFrame:
    out = trial_panel.copy()
    delta = out["delta_belief"].astype(float)
    out["visible_update"] = np.where(delta.notna(), (delta.abs() > 1e-12).astype(int), np.nan)
    out["substantive_update_05"] = np.where(delta.notna(), (delta.abs() >= 0.05).astype(int), np.nan)
    out["signed_feedback_gap"] = out["ai_correct"].astype(float) - out["belief_lag"].astype(float)
    out["abs_feedback_gap"] = out["signed_feedback_gap"].abs()
    out["positive_feedback"] = out["ai_correct"].astype(float)
    return out


def design_matrix(df: pd.DataFrame, continuous: list[str], categorical: list[str], interactions: list[tuple[str, str]] | None = None):
    X_parts = [np.ones((len(df), 1))]
    names = ["Intercept"]
    for col in continuous:
        X_parts.append(df[[col]].astype(float).to_numpy())
        names.append(col)
    cat_levels = {}
    for col in categorical:
        levels = sorted([x for x in df[col].dropna().unique()])
        cat_levels[col] = levels
        for level in levels[1:]:
            X_parts.append((df[col] == level).astype(float).to_numpy().reshape(-1, 1))
            names.append(f"{col}={level}")
    if interactions:
        for cont, cat in interactions:
            levels = cat_levels.get(cat, sorted([x for x in df[cat].dropna().unique()]))
            for level in levels[1:]:
                X_parts.append((df[cont].astype(float) * (df[cat] == level).astype(float)).to_numpy().reshape(-1, 1))
                names.append(f"{cont}:{cat}={level}")
    return np.column_stack(X_parts), names


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


def ols_fit(
    df: pd.DataFrame,
    y_col: str,
    continuous: list[str],
    categorical: list[str],
    interactions: list[tuple[str, str]] | None = None,
    cluster_col: str = "participant_id",
) -> pd.DataFrame:
    needed = [y_col] + continuous + categorical + ([cluster_col] if cluster_col else [])
    m = df.dropna(subset=needed).copy()
    X, names = design_matrix(m, continuous, categorical, interactions)
    y = m[y_col].astype(float).to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = X.T @ (resid[:, None] ** 2 * X)
    cov = (n / max(n - k, 1)) * xtx_inv @ meat @ xtx_inv
    se_hc1 = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    if cluster_col:
        cluster_cov, n_clusters = cluster_sandwich_cov(xtx_inv, X * resid[:, None], m[cluster_col], n, k)
        se_cluster = np.sqrt(np.clip(np.diag(cluster_cov), 0, np.inf))
    else:
        n_clusters = np.nan
        se_cluster = np.full(k, np.nan)
    return pd.DataFrame(
        {
            "term": names,
            "coef": beta,
            "se_cluster": se_cluster,
            "t_cluster": beta / np.where(se_cluster == 0, np.nan, se_cluster),
            "se_hc1": se_hc1,
            "t_hc1": beta / np.where(se_hc1 == 0, np.nan, se_hc1),
            "n": n,
            "n_clusters": n_clusters,
            "cluster_col": cluster_col or "",
            "r2": 1 - float(np.sum(resid**2) / np.sum((y - y.mean()) ** 2)) if np.sum((y - y.mean()) ** 2) > 0 else np.nan,
        }
    )


def logistic_fit(
    df: pd.DataFrame,
    y_col: str,
    continuous: list[str],
    categorical: list[str],
    interactions: list[tuple[str, str]] | None = None,
    cluster_col: str = "participant_id",
) -> pd.DataFrame:
    needed = [y_col] + continuous + categorical + ([cluster_col] if cluster_col else [])
    m = df.dropna(subset=needed).copy()
    X, names = design_matrix(m, continuous, categorical, interactions)
    y = m[y_col].astype(float).to_numpy()
    beta = np.zeros(X.shape[1])
    ridge = 1e-6
    for _ in range(100):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-6, np.inf)
        z = eta + (y - p) / w
        xtwx = X.T @ (w[:, None] * X) + ridge * np.eye(X.shape[1])
        xtwz = X.T @ (w * z)
        new_beta = np.linalg.solve(xtwx, xtwz)
        if np.max(np.abs(new_beta - beta)) < 1e-7:
            beta = new_beta
            break
        beta = new_beta
    eta = np.clip(X @ beta, -30, 30)
    p = 1 / (1 + np.exp(-eta))
    w = np.clip(p * (1 - p), 1e-6, np.inf)
    bread_inv = np.linalg.pinv(X.T @ (w[:, None] * X))
    se_mle = np.sqrt(np.clip(np.diag(bread_inv), 0, np.inf))
    if cluster_col:
        cluster_cov, n_clusters = cluster_sandwich_cov(bread_inv, X * (y - p)[:, None], m[cluster_col], len(y), X.shape[1])
        se_cluster = np.sqrt(np.clip(np.diag(cluster_cov), 0, np.inf))
    else:
        n_clusters = np.nan
        se_cluster = np.full(X.shape[1], np.nan)
    ll = float(np.sum(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
    return pd.DataFrame(
        {
            "term": names,
            "coef": beta,
            "se_cluster": se_cluster,
            "z_cluster": beta / np.where(se_cluster == 0, np.nan, se_cluster),
            "se_mle": se_mle,
            "z_mle": beta / np.where(se_mle == 0, np.nan, se_mle),
            "odds_ratio": np.exp(np.clip(beta, -20, 20)),
            "n": len(y),
            "n_clusters": n_clusters,
            "cluster_col": cluster_col or "",
            "log_likelihood": ll,
        }
    )


def summarize_by_group(df: pd.DataFrame, group_col: str, out_path: Path) -> pd.DataFrame:
    summary = (
        df.groupby(group_col, dropna=False)
        .agg(
            participant_tasks=("taskType", "size"),
            participants=("participant_id", "nunique"),
            delegation_rate=("delegation_rate", "mean"),
            final_accuracy=("final_accuracy", "mean"),
            mean_initial_belief=("initial_belief", "mean"),
            mean_final_belief=("final_belief", "mean"),
            mean_objective_ai_accuracy=("objective_ai_accuracy", "mean"),
            mean_abs_calibration_error=("abs_calibration_error", "mean"),
            median_abs_calibration_error=("abs_calibration_error", "median"),
            mean_abs_delta=("mean_abs_delta", "mean"),
        )
        .reset_index()
        .sort_values("participant_tasks", ascending=False)
    )
    summary.to_csv(out_path, index=False)
    return summary


def classification_agreement_summary(task_summary: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    out = (
        task_summary.groupby("taskType", as_index=False)
        .agg(
            participant_tasks=("taskType", "size"),
            bic_aicc_agreement=("bic_aicc_agree", "mean"),
            aicc_future_agreement=("aicc_future_agree", "mean"),
            mean_bic_winner_weight=("bic_winner_weight", "mean"),
            mean_near_tie_models_delta_bic_2=("near_tie_models_delta_bic_2", "mean"),
            mean_aicc_winner_weight=("aicc_winner_weight", "mean"),
            mean_near_tie_models_delta_aicc_2=("near_tie_models_delta_aicc_2", "mean"),
            aicc_uncertain_share=("aicc_winner_uncertain", "mean"),
        )
        .sort_values("taskType")
    )
    out.to_csv(out_path, index=False)
    return out


def classification_coverage_summary(task_summary: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    group_specs = [("overall", None), ("taskType", "taskType"), ("taskPosition", "taskPosition")]
    for group_name, group_col in group_specs:
        if group_col is None:
            groups = [("all", task_summary)]
        else:
            groups = list(task_summary.groupby(group_col, dropna=False))
        for group_value, group in groups:
            classified = group["updater_model"] != "not_classified"
            rows.append(
                {
                    "group": group_name,
                    "level": group_value,
                    "participant_tasks": int(len(group)),
                    "classified_tasks": int(classified.sum()),
                    "not_classified_tasks": int((~classified).sum()),
                    "classified_share": float(classified.mean()) if len(group) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, default=CLASSIFICATION)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument(
        "--order-condition-quota",
        type=int,
        default=40,
        help="Keep the earliest N valid sessions per task-order condition. Use 0 to keep all completed export sessions.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    classification = load_classification(args.classification)
    sessions = select_analysis_sessions(load_sessions(INPUT), args.order_condition_quota)
    trial_panel = add_update_gate_variables(build_trial_panel(sessions, classification))
    accuracy_validation = objective_accuracy_validation(trial_panel)
    failed_accuracy = accuracy_validation[~accuracy_validation["passes_tolerance_001"]]
    if not failed_accuracy.empty:
        raise ValueError(
            "Reconstructed AI accuracies do not match expected task accuracies:\n"
            + failed_accuracy.to_string(index=False)
        )
    task_summary = build_task_summary(trial_panel)
    transition_panel = build_transition_panel(task_summary)

    trial_panel.to_csv(args.outdir / "trial_panel_with_updater_types.csv", index=False)
    accuracy_validation.to_csv(args.outdir / "objective_accuracy_validation.csv", index=False)
    task_summary.to_csv(args.outdir / "task_summary_with_updater_types.csv", index=False)
    transition_panel.to_csv(args.outdir / "cross_task_transition_panel.csv", index=False)

    summarize_by_group(task_summary, "updater_model", args.outdir / "task_summary_by_updater_model.csv")
    summarize_by_group(task_summary, "updater_family", args.outdir / "task_summary_by_updater_family.csv")
    summarize_by_group(task_summary, "bic_updater_family", args.outdir / "task_summary_by_bic_updater_family.csv")
    summarize_by_group(task_summary, "aicc_updater_family", args.outdir / "task_summary_by_aicc_updater_family.csv")
    summarize_by_group(task_summary, "future_block_updater_family", args.outdir / "task_summary_by_future_block_updater_family.csv")
    classification_agreement_summary(task_summary, args.outdir / "classification_agreement_by_task.csv")
    classification_coverage_summary(task_summary, args.outdir / "classification_coverage.csv")

    pd.crosstab(task_summary["taskType"], task_summary["updater_family"], normalize="index").to_csv(
        args.outdir / "updater_family_share_by_task.csv"
    )
    pd.crosstab(task_summary["taskType"], task_summary["aicc_updater_family"], normalize="index").to_csv(
        args.outdir / "aicc_updater_family_share_by_task.csv"
    )
    pd.crosstab(task_summary["taskType"], task_summary["future_block_updater_family"], normalize="index").to_csv(
        args.outdir / "future_block_updater_family_share_by_task.csv"
    )
    pd.crosstab(task_summary["taskPosition"], task_summary["updater_family"], normalize="index").to_csv(
        args.outdir / "updater_family_share_by_task_position.csv"
    )

    transition_panel.groupby("prev_updater_family").agg(
        transitions=("participant_id", "size"),
        mean_b0_curr=("b0_curr", "mean"),
        mean_bT_prev=("bT_prev", "mean"),
        mean_prev_abs_calibration_error=("prev_abs_calibration_error", "mean"),
        mean_prev_delegation_rate=("prev_delegation_rate", "mean"),
    ).reset_index().to_csv(args.outdir / "carryover_descriptives_by_prev_updater_family.csv", index=False)

    carry_ols = ols_fit(
        transition_panel,
        y_col="b0_curr",
        continuous=["bT_prev", "prev_objective_ai_accuracy"],
        categorical=["curr_taskType", "curr_taskPosition", "prev_updater_family"],
        interactions=[("bT_prev", "prev_updater_family")],
    )
    carry_ols.to_csv(args.outdir / "carryover_moderation_ols.csv", index=False)

    task_ols = ols_fit(
        task_summary,
        y_col="delegation_rate",
        continuous=["mean_belief_lag", "mean_conf_lag", "abs_calibration_error"],
        categorical=["taskType", "taskPosition", "updater_family"],
    )
    task_ols.to_csv(args.outdir / "task_level_delegation_ols.csv", index=False)

    logit = logistic_fit(
        trial_panel,
        y_col="delegation",
        continuous=["belief_lag", "conf_lag"],
        categorical=["taskType", "trialOrdinal", "updater_family"],
        interactions=[("belief_lag", "updater_family")],
    )
    logit.to_csv(args.outdir / "trial_level_delegation_logit.csv", index=False)

    update_gate = logistic_fit(
        trial_panel,
        y_col="visible_update",
        continuous=["abs_feedback_gap", "belief_lag", "conf_lag"],
        categorical=["taskType", "trialOrdinal", "updater_family"],
    )
    update_gate.to_csv(args.outdir / "update_gate_logit.csv", index=False)

    conditional_movement = ols_fit(
        trial_panel[trial_panel["visible_update"] == 1],
        y_col="delta_belief",
        continuous=["signed_feedback_gap", "belief_lag", "conf_lag"],
        categorical=["taskType", "trialOrdinal", "updater_family"],
    )
    conditional_movement.to_csv(args.outdir / "conditional_update_magnitude_ols.csv", index=False)

    first_impression = ols_fit(
        task_summary.dropna(subset=["first_ai_correct"]),
        y_col="delegation_rate",
        continuous=["first_ai_correct", "first_belief_delta", "initial_belief"],
        categorical=["taskType", "updater_family"],
    )
    first_impression.to_csv(args.outdir / "first_impression_delegation_ols.csv", index=False)

    print("[Behavioral extension] Outputs written to", args.outdir)
    print("Objective AI accuracy validation:")
    print(accuracy_validation.to_string(index=False))
    print("Task summary by updater family:")
    print(pd.read_csv(args.outdir / "task_summary_by_updater_family.csv").to_string(index=False))
    print("\nCarryover moderation key terms:")
    print(carry_ols[carry_ols["term"].str.contains("bT_prev|prev_updater_family", regex=True)].to_string(index=False))
    print("\nDelegation logit key terms:")
    print(logit[logit["term"].str.contains("belief_lag|conf_lag|updater_family", regex=True)].to_string(index=False))
    print("\nUpdate-gate logit key terms:")
    print(update_gate[update_gate["term"].str.contains("abs_feedback_gap|belief_lag|conf_lag|updater_family", regex=True)].to_string(index=False))
    print("\nConditional movement OLS key terms:")
    print(conditional_movement[conditional_movement["term"].str.contains("signed_feedback_gap|belief_lag|conf_lag|updater_family", regex=True)].to_string(index=False))


if __name__ == "__main__":
    main()
