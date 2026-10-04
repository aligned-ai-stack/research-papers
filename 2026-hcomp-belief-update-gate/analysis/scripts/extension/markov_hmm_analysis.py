#!/usr/bin/env python
# coding: utf-8

"""Standalone Markov/HMM analysis for the belief-updating extension.

This script intentionally does not modify the existing extension pipeline. It
reads the trial panel produced by `extension_behavioral_pipeline.py`, fits
pooled input-output HMMs across participant-task sequences, and writes a
separate set of exploratory outputs.

Timeline used here:

    state_t -> pre-trial belief/delegation observation_t
    feedback_t -> state_{t+1}

So emissions use `belief_lag` and `delegation`, while transitions from trial t
to t+1 are conditioned on `ai_correct` at trial t. Each participant-task is a
separate sequence; the model never concatenates across task or participant
boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRIAL_PANEL = ROOT / "data" / "followup_outputs" / "behavioral_extension_pipeline" / "trial_panel_with_updater_types.csv"
DEFAULT_SESSIONS = ROOT / "data" / "intermediate_outputs" / "valid_sessions.json"
DEFAULT_OUTDIR = ROOT / "data" / "followup_outputs" / "markov_hmm_analysis"

OUTPUT_PATTERNS = [
    "model_comparison.csv",
    "manifest.json",
    "observed_delegation_transitions.csv",
    "state_parameters*state.csv",
    "transition_matrices*state.csv",
    "transition_direction_summary*state.csv",
    "feedback_transition_contrast*state.csv",
    "posterior_state_panel*state.csv",
    "state_occupancy_by_task*state.csv",
    "state_occupancy_by_updater_family*state.csv",
    "state_interpretation*state.csv",
]


@dataclass
class Sequence:
    key: tuple[str, str]
    rows: list[dict]
    belief: np.ndarray
    delegation: np.ndarray
    confidence: np.ndarray
    self_correct: np.ndarray
    feedback: np.ndarray
    task_self_accuracy: float
    task_self_correct_count: int


@dataclass
class HMMParams:
    pi: np.ndarray
    transition: np.ndarray
    belief_mean: np.ndarray
    belief_var: np.ndarray
    delegation_prob: np.ndarray
    confidence_mean: np.ndarray | None = None
    confidence_var: np.ndarray | None = None
    self_correct_prob: np.ndarray | None = None


def parse_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def parse_int(value: object) -> int | None:
    try:
        if value == "" or value is None:
            return None
        return int(float(value))
    except Exception:
        return None


def mongo_date_value(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("$date", ""))
    return str(value or "")


def ids_from_balanced_order_conditions(sessions_path: Path, quota: int) -> set[str]:
    if quota <= 0:
        return set()
    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    by_condition: dict[object, list[dict]] = {}
    for session in sessions:
        key = session.get("orderCondition")
        if key is None:
            key = tuple(session.get("taskOrder", []) or [])
        by_condition.setdefault(key, []).append(session)

    keep: set[str] = set()
    for _, group in sorted(by_condition.items(), key=lambda item: str(item[0])):
        ordered = sorted(
            group,
            key=lambda s: (
                mongo_date_value(s.get("completedAt")),
                mongo_date_value(s.get("createdAt")),
                str(s.get("participant_id", "")),
            ),
        )
        keep.update(str(s.get("participant_id", "")) for s in ordered[:quota])
    return keep


def load_sequences(
    path: Path,
    min_len: int = 2,
    allowed_ids: set[str] | None = None,
    use_confidence: bool = False,
    use_self_correct: bool = False,
) -> list[Sequence]:
    groups: dict[tuple[str, str], list[dict]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row.get("participant_id", ""))
            if allowed_ids is not None and pid not in allowed_ids:
                continue
            task = str(row.get("taskType", ""))
            trial = parse_int(row.get("trialOrdinal"))
            belief = parse_float(row.get("belief_lag"))
            delegation = parse_float(row.get("delegation"))
            feedback = parse_float(row.get("ai_correct"))
            confidence = parse_float(row.get("conf_lag"))
            self_correct = parse_float(row.get("self_was_correct"))
            if trial is None or not all(math.isfinite(x) for x in [belief, delegation, feedback]):
                continue
            if delegation not in {0.0, 1.0} or feedback not in {0.0, 1.0}:
                continue
            if use_confidence and not math.isfinite(confidence):
                continue
            row["_trialOrdinal"] = trial
            row["_belief_lag"] = min(max(belief, 0.0), 1.0)
            row["_delegation"] = int(delegation)
            row["_feedback"] = int(feedback)
            row["_confidence"] = min(max(confidence, 0.0), 1.0) if math.isfinite(confidence) else float("nan")
            row["_self_correct"] = int(self_correct) if math.isfinite(self_correct) and self_correct in {0.0, 1.0} else -1
            groups.setdefault((pid, task), []).append(row)

    sequences: list[Sequence] = []
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda r: int(r["_trialOrdinal"]))
        if len(rows) < min_len:
            continue
        belief = np.asarray([float(r["_belief_lag"]) for r in rows], dtype=float)
        delegation = np.asarray([int(r["_delegation"]) for r in rows], dtype=int)
        confidence = np.asarray([float(r["_confidence"]) for r in rows], dtype=float)
        self_correct = np.asarray([int(r["_self_correct"]) for r in rows], dtype=int)
        feedback = np.asarray([int(r["_feedback"]) for r in rows[:-1]], dtype=int)
        valid_self = self_correct[self_correct >= 0]
        task_self_accuracy = float(np.mean(valid_self)) if len(valid_self) else float("nan")
        task_self_correct_count = int(valid_self.sum()) if len(valid_self) else -1
        sequences.append(
            Sequence(
                key=key,
                rows=rows,
                belief=belief,
                delegation=delegation,
                confidence=confidence,
                self_correct=self_correct,
                feedback=feedback,
                task_self_accuracy=task_self_accuracy,
                task_self_correct_count=task_self_correct_count,
            )
        )
    return sequences


def split_sequences(
    sequences: list[Sequence],
    holdout_share: float,
    seed: int,
    split_unit: str = "participant",
) -> tuple[list[Sequence], list[Sequence]]:
    if holdout_share <= 0:
        return sequences, []
    rng = np.random.default_rng(seed)
    if split_unit == "participant":
        units = sorted({seq.key[0] for seq in sequences})
        order = rng.permutation(len(units))
        n_test = max(1, int(round(len(units) * holdout_share)))
        test_units = {units[int(i)] for i in order[:n_test]}
        train = [seq for seq in sequences if seq.key[0] not in test_units]
        test = [seq for seq in sequences if seq.key[0] in test_units]
        return train, test
    if split_unit != "sequence":
        raise ValueError(f"Unknown split unit: {split_unit}")
    order = rng.permutation(len(sequences))
    n_test = max(1, int(round(len(sequences) * holdout_share)))
    test_idx = {int(x) for x in order[:n_test]}
    train = [seq for i, seq in enumerate(sequences) if i not in test_idx]
    test = [seq for i, seq in enumerate(sequences) if i in test_idx]
    return train, test


def normalize_rows(matrix: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    out = np.maximum(matrix, floor)
    return out / out.sum(axis=1, keepdims=True)


def transition_index(feedback: int, transition_mode: str) -> int:
    if transition_mode == "pooled":
        return 0
    return int(feedback)


def emission_matrix(seq: Sequence, params: HMMParams) -> np.ndarray:
    k = len(params.pi)
    b = seq.belief[:, None]
    d = seq.delegation[:, None]
    var = np.maximum(params.belief_var, 1e-5)[None, :]
    mean = params.belief_mean[None, :]
    p = np.clip(params.delegation_prob, 1e-5, 1.0 - 1e-5)[None, :]
    normal = np.exp(-0.5 * ((b - mean) ** 2) / var) / np.sqrt(2.0 * math.pi * var)
    bern = np.where(d == 1, p, 1.0 - p)
    out = normal * bern
    if params.confidence_mean is not None and params.confidence_var is not None:
        c = seq.confidence[:, None]
        c_var = np.maximum(params.confidence_var, 1e-5)[None, :]
        c_mean = params.confidence_mean[None, :]
        conf_normal = np.exp(-0.5 * ((c - c_mean) ** 2) / c_var) / np.sqrt(2.0 * math.pi * c_var)
        out *= conf_normal
    if params.self_correct_prob is not None:
        s = seq.self_correct[:, None]
        p_self = np.clip(params.self_correct_prob, 1e-5, 1.0 - 1e-5)[None, :]
        self_bern = np.where(s == 1, p_self, np.where(s == 0, 1.0 - p_self, 1.0))
        out *= self_bern
    return np.maximum(out, 1e-300).reshape(len(seq.belief), k)


def forward_backward(seq: Sequence, params: HMMParams, transition_mode: str = "conditioned") -> tuple[float, np.ndarray, np.ndarray]:
    bmat = emission_matrix(seq, params)
    t_len, k = bmat.shape
    alpha = np.zeros((t_len, k), dtype=float)
    beta = np.ones((t_len, k), dtype=float)
    scales = np.zeros(t_len, dtype=float)

    raw = np.maximum(params.pi, 1e-12) * bmat[0]
    scales[0] = max(float(raw.sum()), 1e-300)
    alpha[0] = raw / scales[0]

    for t in range(1, t_len):
        a = params.transition[transition_index(int(seq.feedback[t - 1]), transition_mode)]
        raw = (alpha[t - 1] @ a) * bmat[t]
        scales[t] = max(float(raw.sum()), 1e-300)
        alpha[t] = raw / scales[t]

    for t in range(t_len - 2, -1, -1):
        a = params.transition[transition_index(int(seq.feedback[t]), transition_mode)]
        beta[t] = (a @ (bmat[t + 1] * beta[t + 1])) / scales[t + 1]

    gamma = alpha * beta
    gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

    xi = np.zeros((max(t_len - 1, 0), k, k), dtype=float)
    for t in range(t_len - 1):
        a = params.transition[transition_index(int(seq.feedback[t]), transition_mode)]
        raw_xi = alpha[t, :, None] * a * bmat[t + 1, None, :] * beta[t + 1, None, :]
        xi[t] = raw_xi / max(float(raw_xi.sum()), 1e-300)

    return float(np.log(scales).sum()), gamma, xi


def sequence_loglik(seq: Sequence, params: HMMParams, transition_mode: str = "conditioned") -> float:
    ll, _, _ = forward_backward(seq, params, transition_mode=transition_mode)
    return ll


def initialize_params(
    sequences: list[Sequence],
    k: int,
    rng: np.random.Generator,
    min_var: float,
    use_confidence: bool = False,
    use_self_correct: bool = False,
) -> HMMParams:
    belief = np.concatenate([seq.belief for seq in sequences])
    delegation = np.concatenate([seq.delegation for seq in sequences])
    confidence = np.concatenate([seq.confidence for seq in sequences]) if use_confidence else None
    self_correct = np.concatenate([seq.self_correct for seq in sequences]) if use_self_correct else None
    if self_correct is not None:
        self_correct = self_correct[self_correct >= 0]
    if k == 1:
        return HMMParams(
            pi=np.ones(1),
            transition=np.ones((2, 1, 1)),
            belief_mean=np.asarray([float(np.mean(belief))]),
            belief_var=np.asarray([max(float(np.var(belief)), min_var)]),
            delegation_prob=np.asarray([float(np.clip(np.mean(delegation), 1e-4, 1.0 - 1e-4))]),
            confidence_mean=np.asarray([float(np.mean(confidence))]) if use_confidence and confidence is not None else None,
            confidence_var=np.asarray([max(float(np.var(confidence)), min_var)]) if use_confidence and confidence is not None else None,
            self_correct_prob=np.asarray([float(np.clip(np.mean(self_correct), 1e-4, 1.0 - 1e-4))]) if use_self_correct and self_correct is not None and len(self_correct) else None,
        )

    qs = np.linspace(0.15, 0.85, k)
    means = np.quantile(belief, qs)
    means = np.clip(means + rng.normal(0.0, 0.03, size=k), 0.02, 0.98)
    order = np.argsort(means)
    means = means[order]
    var = np.repeat(max(float(np.var(belief)) / max(k, 1), min_var), k)
    base_del = float(np.clip(np.mean(delegation), 0.05, 0.95))
    deltas = np.linspace(-0.15, 0.15, k)
    delegation_prob = np.clip(base_del + deltas + rng.normal(0.0, 0.03, size=k), 0.02, 0.98)
    confidence_mean = None
    confidence_var = None
    if use_confidence and confidence is not None:
        base_conf = float(np.clip(np.mean(confidence), 0.05, 0.95))
        confidence_mean = np.clip(base_conf + np.linspace(-0.10, 0.10, k) + rng.normal(0.0, 0.02, size=k), 0.02, 0.98)
        confidence_var = np.repeat(max(float(np.var(confidence)) / max(k, 1), min_var), k)
    self_correct_prob = None
    if use_self_correct and self_correct is not None and len(self_correct):
        base_self = float(np.clip(np.mean(self_correct), 0.05, 0.95))
        self_correct_prob = np.clip(base_self + np.linspace(-0.12, 0.12, k) + rng.normal(0.0, 0.02, size=k), 0.02, 0.98)
    pi = np.ones(k) / k
    transition = np.zeros((2, k, k), dtype=float)
    for feedback in [0, 1]:
        for state in range(k):
            row = np.ones(k) * 0.08
            row[state] = 0.70
            if feedback == 1 and state + 1 < k:
                row[state + 1] += 0.15
            if feedback == 0 and state - 1 >= 0:
                row[state - 1] += 0.15
            row += rng.uniform(0.0, 0.03, size=k)
            transition[feedback, state] = row / row.sum()
    return HMMParams(
        pi=pi,
        transition=transition,
        belief_mean=means,
        belief_var=var,
        delegation_prob=delegation_prob,
        confidence_mean=confidence_mean,
        confidence_var=confidence_var,
        self_correct_prob=self_correct_prob,
    )


def reorder_params(params: HMMParams) -> HMMParams:
    order = np.argsort(params.belief_mean)
    inv = np.zeros_like(order)
    inv[order] = np.arange(len(order))
    transition = params.transition[:, order, :][:, :, order]
    return HMMParams(
        pi=params.pi[order],
        transition=transition,
        belief_mean=params.belief_mean[order],
        belief_var=params.belief_var[order],
        delegation_prob=params.delegation_prob[order],
        confidence_mean=params.confidence_mean[order] if params.confidence_mean is not None else None,
        confidence_var=params.confidence_var[order] if params.confidence_var is not None else None,
        self_correct_prob=params.self_correct_prob[order] if params.self_correct_prob is not None else None,
    )


def fit_hmm(
    sequences: list[Sequence],
    k: int,
    seed: int,
    starts: int,
    max_iter: int,
    tol: float,
    min_var: float,
    transition_mode: str,
    use_confidence: bool = False,
    use_self_correct: bool = False,
) -> tuple[HMMParams, float, int]:
    best_params: HMMParams | None = None
    best_ll = -float("inf")
    best_iter = 0
    for start in range(starts):
        rng = np.random.default_rng(seed + 1009 * start + 17 * k)
        params = initialize_params(sequences, k, rng, min_var, use_confidence=use_confidence, use_self_correct=use_self_correct)
        prev_ll = -float("inf")
        iterations = 0
        for it in range(1, max_iter + 1):
            pi_acc = np.zeros(k, dtype=float)
            trans_acc = np.zeros((2, k, k), dtype=float)
            gamma_sum = np.zeros(k, dtype=float)
            belief_sum = np.zeros(k, dtype=float)
            belief_sq_sum = np.zeros(k, dtype=float)
            delegation_sum = np.zeros(k, dtype=float)
            confidence_sum = np.zeros(k, dtype=float) if use_confidence else None
            confidence_sq_sum = np.zeros(k, dtype=float) if use_confidence else None
            self_correct_sum = np.zeros(k, dtype=float) if use_self_correct else None
            self_correct_denom = np.zeros(k, dtype=float) if use_self_correct else None
            total_ll = 0.0

            for seq in sequences:
                ll, gamma, xi = forward_backward(seq, params, transition_mode=transition_mode)
                total_ll += ll
                pi_acc += gamma[0]
                for t in range(len(seq.feedback)):
                    trans_acc[transition_index(int(seq.feedback[t]), transition_mode)] += xi[t]
                gamma_sum += gamma.sum(axis=0)
                belief_sum += gamma.T @ seq.belief
                belief_sq_sum += gamma.T @ (seq.belief ** 2)
                delegation_sum += gamma.T @ seq.delegation
                if confidence_sum is not None and confidence_sq_sum is not None:
                    confidence_sum += gamma.T @ seq.confidence
                    confidence_sq_sum += gamma.T @ (seq.confidence ** 2)
                if self_correct_sum is not None and self_correct_denom is not None:
                    valid_self = seq.self_correct >= 0
                    if np.any(valid_self):
                        self_correct_sum += gamma[valid_self].T @ seq.self_correct[valid_self]
                        self_correct_denom += gamma[valid_self].sum(axis=0)

            new_pi = np.maximum(pi_acc, 1e-9)
            new_pi = new_pi / new_pi.sum()
            new_transition = np.zeros_like(trans_acc)
            if transition_mode == "pooled":
                pooled = normalize_rows(trans_acc[0] + 1e-3)
                new_transition[0] = pooled
                new_transition[1] = pooled
            else:
                for feedback in [0, 1]:
                    new_transition[feedback] = normalize_rows(trans_acc[feedback] + 1e-3)

            denom = np.maximum(gamma_sum, 1e-9)
            new_mean = np.clip(belief_sum / denom, 0.0, 1.0)
            new_var = np.maximum(belief_sq_sum / denom - new_mean ** 2, min_var)
            new_del = np.clip(delegation_sum / denom, 1e-5, 1.0 - 1e-5)
            new_conf_mean = None
            new_conf_var = None
            if confidence_sum is not None and confidence_sq_sum is not None:
                new_conf_mean = np.clip(confidence_sum / denom, 0.0, 1.0)
                new_conf_var = np.maximum(confidence_sq_sum / denom - new_conf_mean ** 2, min_var)
            new_self = (
                np.clip(self_correct_sum / np.maximum(self_correct_denom, 1e-9), 1e-5, 1.0 - 1e-5)
                if self_correct_sum is not None and self_correct_denom is not None
                else None
            )
            params = reorder_params(HMMParams(new_pi, new_transition, new_mean, new_var, new_del, new_conf_mean, new_conf_var, new_self))
            iterations = it

            if abs(total_ll - prev_ll) < tol:
                break
            prev_ll = total_ll

        final_ll = sum(sequence_loglik(seq, params, transition_mode=transition_mode) for seq in sequences)
        if final_ll > best_ll:
            best_ll = final_ll
            best_params = params
            best_iter = iterations

    if best_params is None:
        raise RuntimeError("HMM fitting failed.")
    return best_params, best_ll, best_iter


def parameter_count(
    k: int,
    transition_mode: str = "conditioned",
    use_confidence: bool = False,
    use_self_correct: bool = False,
) -> int:
    transition_blocks = 1 if transition_mode == "pooled" else 2
    emission_params = 3 * k
    if use_confidence:
        emission_params += 2 * k
    if use_self_correct:
        emission_params += k
    return (k - 1) + transition_blocks * k * (k - 1) + emission_params


def evaluate(sequences: list[Sequence], params: HMMParams, transition_mode: str = "conditioned") -> tuple[float, int, int]:
    ll = sum(sequence_loglik(seq, params, transition_mode=transition_mode) for seq in sequences)
    n_obs = sum(len(seq.belief) for seq in sequences)
    n_trans = sum(len(seq.feedback) for seq in sequences)
    return float(ll), int(n_obs), int(n_trans)


def write_model_outputs(
    outdir: Path,
    transition_mode: str,
    k: int,
    params: HMMParams,
    train_ll: float,
    train_n: int,
    test_ll: float,
    test_n: int,
    iterations: int,
) -> None:
    use_confidence = params.confidence_mean is not None and params.confidence_var is not None
    use_self_correct = params.self_correct_prob is not None
    k_params = parameter_count(k, transition_mode=transition_mode, use_confidence=use_confidence, use_self_correct=use_self_correct)
    aic = -2.0 * train_ll + 2.0 * k_params
    bic = -2.0 * train_ll + math.log(max(train_n, 2)) * k_params
    with (outdir / "model_comparison.csv").open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "states",
                "transition_mode",
                "parameters",
                "iterations",
                "train_log_likelihood",
                "train_observations",
                "train_avg_log_likelihood",
                "test_log_likelihood",
                "test_observations",
                "test_avg_log_likelihood",
                "aic",
                "bic",
            ],
        )
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(
            {
                "states": k,
                "transition_mode": transition_mode,
                "parameters": k_params,
                "iterations": iterations,
                "train_log_likelihood": train_ll,
                "train_observations": train_n,
                "train_avg_log_likelihood": train_ll / max(train_n, 1),
                "test_log_likelihood": test_ll,
                "test_observations": test_n,
                "test_avg_log_likelihood": test_ll / max(test_n, 1) if test_n else "",
                "aic": aic,
                "bic": bic,
            }
        )

    state_paths = [outdir / f"state_parameters_{transition_mode}_{k}state.csv"]
    if transition_mode == "conditioned":
        state_paths.append(outdir / f"state_parameters_{k}state.csv")
    for state_path in state_paths:
        with state_path.open("w", newline="", encoding="utf-8") as f:
            state_fields = ["states", "transition_mode", "state", "initial_prob", "belief_mean", "belief_sd", "delegation_probability"]
            if use_confidence:
                state_fields.extend(["confidence_mean", "confidence_sd"])
            if use_self_correct:
                state_fields.append("self_correct_probability")
            writer = csv.DictWriter(
                f,
                fieldnames=state_fields,
            )
            writer.writeheader()
            for state in range(k):
                row = {
                    "states": k,
                    "transition_mode": transition_mode,
                    "state": state,
                    "initial_prob": params.pi[state],
                    "belief_mean": params.belief_mean[state],
                    "belief_sd": math.sqrt(float(params.belief_var[state])),
                    "delegation_probability": params.delegation_prob[state],
                }
                if use_confidence:
                    row["confidence_mean"] = params.confidence_mean[state]
                    row["confidence_sd"] = math.sqrt(float(params.confidence_var[state]))
                if use_self_correct:
                    row["self_correct_probability"] = params.self_correct_prob[state]
                writer.writerow(row)

    transition_paths = [outdir / f"transition_matrices_{transition_mode}_{k}state.csv"]
    if transition_mode == "conditioned":
        transition_paths.append(outdir / f"transition_matrices_{k}state.csv")
    for transition_path in transition_paths:
        with transition_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["states", "transition_mode", "feedback", "from_state", "to_state", "probability"])
            writer.writeheader()
            for feedback in [0, 1]:
                for i in range(k):
                    for j in range(k):
                        writer.writerow(
                            {
                                "states": k,
                                "transition_mode": transition_mode,
                                "feedback": feedback,
                                "from_state": i,
                                "to_state": j,
                                "probability": params.transition[feedback, i, j],
                            }
                        )

    direction_paths = [outdir / f"transition_direction_summary_{transition_mode}_{k}state.csv"]
    if transition_mode == "conditioned":
        direction_paths.append(outdir / f"transition_direction_summary_{k}state.csv")
    for direction_path in direction_paths:
        with direction_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "states",
                    "transition_mode",
                    "feedback",
                    "from_state",
                    "stay_probability",
                    "up_probability",
                    "down_probability",
                    "expected_next_state",
                ],
            )
            writer.writeheader()
            state_values = np.arange(k, dtype=float)
            for feedback in [0, 1]:
                for i in range(k):
                    row = params.transition[feedback, i]
                    writer.writerow(
                        {
                            "states": k,
                            "transition_mode": transition_mode,
                            "feedback": feedback,
                            "from_state": i,
                            "stay_probability": row[i],
                            "up_probability": float(row[np.arange(k) > i].sum()),
                            "down_probability": float(row[np.arange(k) < i].sum()),
                            "expected_next_state": float(row @ state_values),
                        }
                    )

    if transition_mode == "conditioned" and k > 1:
        contrast_paths = [outdir / f"feedback_transition_contrast_{transition_mode}_{k}state.csv"]
        contrast_paths.append(outdir / f"feedback_transition_contrast_{k}state.csv")
        for contrast_path in contrast_paths:
            with contrast_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "states",
                        "transition_mode",
                        "from_state",
                        "negative_stay_probability",
                        "positive_stay_probability",
                        "positive_minus_negative_stay",
                        "negative_up_probability",
                        "positive_up_probability",
                        "positive_minus_negative_up",
                        "negative_down_probability",
                        "positive_down_probability",
                        "negative_minus_positive_down",
                    ],
                )
                writer.writeheader()
                state_idx = np.arange(k)
                for i in range(k):
                    neg = params.transition[0, i]
                    pos = params.transition[1, i]
                    neg_up = float(neg[state_idx > i].sum())
                    pos_up = float(pos[state_idx > i].sum())
                    neg_down = float(neg[state_idx < i].sum())
                    pos_down = float(pos[state_idx < i].sum())
                    writer.writerow(
                        {
                            "states": k,
                            "transition_mode": transition_mode,
                            "from_state": i,
                            "negative_stay_probability": neg[i],
                            "positive_stay_probability": pos[i],
                            "positive_minus_negative_stay": pos[i] - neg[i],
                            "negative_up_probability": neg_up,
                            "positive_up_probability": pos_up,
                            "positive_minus_negative_up": pos_up - neg_up,
                            "negative_down_probability": neg_down,
                            "positive_down_probability": pos_down,
                            "negative_minus_positive_down": neg_down - pos_down,
                        }
                    )


def posterior_panel(sequences: list[Sequence], params: HMMParams, transition_mode: str = "conditioned") -> list[dict]:
    rows = []
    for seq in sequences:
        _, gamma, _ = forward_backward(seq, params, transition_mode=transition_mode)
        for idx, source in enumerate(seq.rows):
            out = {
                "participant_id": source.get("participant_id", ""),
                "taskType": source.get("taskType", ""),
                "taskPosition": source.get("taskPosition", ""),
                "trialOrdinal": source.get("trialOrdinal", ""),
                "belief_lag": source.get("belief_lag", ""),
                "conf_lag": source.get("conf_lag", ""),
                "delegation": source.get("delegation", ""),
                "self_was_correct": source.get("self_was_correct", ""),
                "task_self_accuracy": seq.task_self_accuracy,
                "task_self_correct_count": seq.task_self_correct_count,
                "ai_correct": source.get("ai_correct", ""),
                "updater_family": source.get("updater_family", ""),
                "most_likely_state": int(np.argmax(gamma[idx])),
            }
            for state in range(gamma.shape[1]):
                out[f"state_{state}_prob"] = float(gamma[idx, state])
            rows.append(out)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clear_previous_outputs(outdir: Path) -> None:
    for pattern in OUTPUT_PATTERNS:
        for path in outdir.glob(pattern):
            if path.is_file():
                path.unlink()


def summarize_posterior(rows: list[dict], k: int, group_col: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_col, "")), []).append(row)
    out = []
    for key, group in sorted(groups.items()):
        item = {group_col: key, "observations": len(group)}
        for state in range(k):
            item[f"mean_state_{state}_prob"] = float(np.mean([float(r[f"state_{state}_prob"]) for r in group]))
        out.append(item)
    return out


def summarize_state_interpretation(rows: list[dict], k: int) -> list[dict]:
    out = []
    numeric_cols = [
        ("belief_lag", "belief_mean"),
        ("delegation", "delegation_probability"),
        ("conf_lag", "confidence_mean"),
        ("self_was_correct", "self_correct_probability"),
        ("task_self_accuracy", "task_self_accuracy_mean"),
        ("task_self_correct_count", "task_self_correct_count_mean"),
    ]
    for state in range(k):
        weights = np.asarray([float(row[f"state_{state}_prob"]) for row in rows], dtype=float)
        mass = float(weights.sum())
        item = {"state": state, "posterior_mass": mass, "posterior_share": mass / max(len(rows), 1)}
        for source_col, out_col in numeric_cols:
            vals = []
            ws = []
            for row, weight in zip(rows, weights):
                value = parse_float(row.get(source_col))
                if math.isfinite(value):
                    vals.append(value)
                    ws.append(weight)
            if ws and sum(ws) > 0:
                item[out_col] = float(np.average(np.asarray(vals, dtype=float), weights=np.asarray(ws, dtype=float)))
        out.append(item)
    return out


def observed_delegation_transitions(sequences: list[Sequence]) -> list[dict]:
    counts = np.zeros((2, 2), dtype=float)
    for seq in sequences:
        for t in range(len(seq.delegation) - 1):
            counts[int(seq.delegation[t]), int(seq.delegation[t + 1])] += 1
    rows = []
    for i in [0, 1]:
        denom = max(float(counts[i].sum()), 1.0)
        for j in [0, 1]:
            rows.append({"from_delegation": i, "to_delegation": j, "count": int(counts[i, j]), "probability": counts[i, j] / denom})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-panel", type=Path, default=DEFAULT_TRIAL_PANEL)
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--states", default="1,2,3,4", help="Comma-separated state counts to fit.")
    parser.add_argument(
        "--transition-modes",
        default="conditioned,pooled",
        help="Comma-separated transition modes. conditioned estimates separate matrices by feedback; pooled estimates one shared matrix.",
    )
    parser.add_argument(
        "--order-condition-quota",
        type=int,
        default=40,
        help="Keep earliest N valid-session IDs per task-order condition before building HMM sequences. Use 0 to keep all IDs in the trial panel.",
    )
    parser.add_argument("--holdout-share", type=float, default=0.20)
    parser.add_argument(
        "--split-unit",
        choices=["participant", "sequence"],
        default="participant",
        help="Holdout unit for predictive checks. participant keeps all tasks for a person together; sequence splits participant-task blocks.",
    )
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--min-var", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--emissions",
        default="belief,delegation",
        help=(
            "Comma-separated emissions. belief and delegation are always used; "
            "optional values: confidence, self_correct."
        ),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    clear_previous_outputs(args.outdir)
    comparison = args.outdir / "model_comparison.csv"
    if comparison.exists():
        comparison.unlink()

    emission_features = {x.strip() for x in args.emissions.split(",") if x.strip()}
    emission_features.update({"belief", "delegation"})
    unknown_features = emission_features - {"belief", "delegation", "confidence", "self_correct"}
    if unknown_features:
        raise ValueError(f"Unknown emission feature(s): {sorted(unknown_features)}")
    use_confidence = "confidence" in emission_features
    use_self_correct = "self_correct" in emission_features

    allowed_ids = ids_from_balanced_order_conditions(args.sessions, args.order_condition_quota)
    sequences = load_sequences(
        args.trial_panel,
        allowed_ids=allowed_ids or None,
        use_confidence=use_confidence,
        use_self_correct=use_self_correct,
    )
    train, test = split_sequences(sequences, args.holdout_share, args.random_seed, split_unit=args.split_unit)
    states = list(dict.fromkeys(int(x.strip()) for x in args.states.split(",") if x.strip()))
    transition_modes = list(dict.fromkeys(x.strip() for x in args.transition_modes.split(",") if x.strip()))
    unknown_modes = [mode for mode in transition_modes if mode not in {"conditioned", "pooled"}]
    if unknown_modes:
        raise ValueError(f"Unknown transition mode(s): {unknown_modes}")

    manifest = {
        "trial_panel": str(args.trial_panel),
        "sessions": str(args.sessions),
        "outdir": str(args.outdir),
        "order_condition_quota": args.order_condition_quota,
        "allowed_participants": len(allowed_ids) if allowed_ids else "all_trial_panel_ids",
        "participants": len({seq.key[0] for seq in sequences}),
        "sequences": len(sequences),
        "train_sequences": len(train),
        "test_sequences": len(test),
        "observations": sum(len(seq.belief) for seq in sequences),
        "transitions": sum(len(seq.feedback) for seq in sequences),
        "holdout_share": args.holdout_share,
        "split_unit": args.split_unit,
        "train_participants": len({seq.key[0] for seq in train}),
        "test_participants": len({seq.key[0] for seq in test}),
        "states": states,
        "transition_modes": transition_modes,
        "emissions": sorted(emission_features),
        "random_seed": args.random_seed,
        "timeline": (
            "emissions use belief_lag, delegation, and any optional emissions; "
            "transitions use current-trial ai_correct to next trial"
        ),
    }

    fitted: dict[tuple[str, int], HMMParams] = {}
    for transition_mode in transition_modes:
        for k in states:
            params, train_ll, iterations = fit_hmm(
                train,
                k=k,
                seed=args.random_seed,
                starts=args.starts,
                max_iter=args.max_iter,
                tol=args.tol,
                min_var=args.min_var,
                transition_mode=transition_mode,
                use_confidence=use_confidence,
                use_self_correct=use_self_correct,
            )
            train_ll, train_n, _ = evaluate(train, params, transition_mode=transition_mode)
            test_ll, test_n, _ = evaluate(test, params, transition_mode=transition_mode) if test else (float("nan"), 0, 0)
            write_model_outputs(args.outdir, transition_mode, k, params, train_ll, train_n, test_ll, test_n, iterations)
            fitted[(transition_mode, k)] = params

    model_rows = []
    with comparison.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model_rows.append(row)
    best_by_test = max(
        model_rows,
        key=lambda r: float(r["test_avg_log_likelihood"]) if r["test_avg_log_likelihood"] != "" else -float("inf"),
    )
    best_mode = str(best_by_test["transition_mode"])
    best_k = int(best_by_test["states"])
    best_params = fitted[(best_mode, best_k)]
    post = posterior_panel(sequences, best_params, transition_mode=best_mode)
    write_rows(args.outdir / f"posterior_state_panel_{best_mode}_{best_k}state.csv", post)
    write_rows(args.outdir / f"state_occupancy_by_task_{best_mode}_{best_k}state.csv", summarize_posterior(post, best_k, "taskType"))
    write_rows(args.outdir / f"state_interpretation_{best_mode}_{best_k}state.csv", summarize_state_interpretation(post, best_k))
    write_rows(
        args.outdir / f"state_occupancy_by_updater_family_{best_mode}_{best_k}state.csv",
        summarize_posterior(post, best_k, "updater_family"),
    )
    if best_mode == "conditioned":
        write_rows(args.outdir / f"posterior_state_panel_{best_k}state.csv", post)
        write_rows(args.outdir / f"state_occupancy_by_task_{best_k}state.csv", summarize_posterior(post, best_k, "taskType"))
        write_rows(args.outdir / f"state_interpretation_{best_k}state.csv", summarize_state_interpretation(post, best_k))
        write_rows(args.outdir / f"state_occupancy_by_updater_family_{best_k}state.csv", summarize_posterior(post, best_k, "updater_family"))
    if ("conditioned", 3) in fitted:
        interp_post = posterior_panel(sequences, fitted[("conditioned", 3)], transition_mode="conditioned")
        write_rows(args.outdir / "posterior_state_panel_conditioned_3state.csv", interp_post)
        write_rows(args.outdir / "state_occupancy_by_task_conditioned_3state.csv", summarize_posterior(interp_post, 3, "taskType"))
        write_rows(args.outdir / "state_interpretation_conditioned_3state.csv", summarize_state_interpretation(interp_post, 3))
        write_rows(
            args.outdir / "state_occupancy_by_updater_family_conditioned_3state.csv",
            summarize_posterior(interp_post, 3, "updater_family"),
        )
        write_rows(args.outdir / "posterior_state_panel_3state.csv", interp_post)
        write_rows(args.outdir / "state_occupancy_by_task_3state.csv", summarize_posterior(interp_post, 3, "taskType"))
        write_rows(args.outdir / "state_interpretation_3state.csv", summarize_state_interpretation(interp_post, 3))
        write_rows(args.outdir / "state_occupancy_by_updater_family_3state.csv", summarize_posterior(interp_post, 3, "updater_family"))
    write_rows(args.outdir / "observed_delegation_transitions.csv", observed_delegation_transitions(sequences))

    manifest["best_states_by_test_avg_log_likelihood"] = best_k
    manifest["best_transition_mode_by_test_avg_log_likelihood"] = best_mode
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Loaded {len(sequences)} participant-task sequences with {manifest['observations']} observations.")
    print(f"Best held-out average log likelihood: {best_mode}, {best_k} states.")
    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
