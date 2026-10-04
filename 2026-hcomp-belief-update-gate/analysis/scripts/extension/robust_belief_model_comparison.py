#!/usr/bin/env python
# coding: utf-8

"""Robust behavioral model comparison for belief-updating trajectories.

This script is deliberately stricter than the exploratory
`divisible_updating.py` pass:

1. All models are scored on the same outcome: post-trial reported belief
   (`b_post`) on the probability scale. The paper-facing all-candidate run
   includes both trajectory-generating models and one-step conditional delta
   models, with model-scope labels retained so conditional winners are
   interpreted as robustness/descriptive checks.
2. Parameter grids are written to disk as part of the output, so the paper can
   report how coarse Bayes, lambda, thresholds, and asymmetries were
   parameterized.
3. The canonical panel gets conservative BIC, predictive AICc/Akaike weights,
   leave-one-trial-out RMSE, and leave-future-out RMSE.
4. Model shares are reported for the full sample and active-updater subsets.
5. Sensitivity is rerun across the H2 strict/lenient/hybrid panels.
6. Optional model-recovery simulations reuse the exact CHI study panel layout:
   same tasks, trial counts, priors, n0 values, evidence sequences, and a
   configurable response-expression layer.

The goal is not to claim that these simple grid models are the final paper
model. They are a robust, transparent bridge between the original H2 slope and
the eventual hierarchical/latent model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
H2_DIR = ROOT / "data" / "hypothesis_outputs" / "H2"
DEFAULT_OUTDIR = ROOT / "data" / "followup_outputs" / "robust_belief_model_comparison"

CANONICAL_PANEL = H2_DIR / "h2_panel_hybrid_S10.csv"
PANEL_FILES = {
    "strict": H2_DIR / "h2_panel_strict.csv",
    "lenient": H2_DIR / "h2_panel_lenient.csv",
    "hybrid_S5": H2_DIR / "h2_panel_hybrid_S5.csv",
    "hybrid_S10": H2_DIR / "h2_panel_hybrid_S10.csv",
    "hybrid_S20": H2_DIR / "h2_panel_hybrid_S20.csv",
}


PRIMARY_GENERATIVE_MODELS = {
    "no_update",
    "standard_bayes",
    "divisible_weighted_bayes",
    "discounted_bayes_gamma",
    "discounted_weighted_bayes",
    "coarse_weighted_bayes",
    "sticky_weighted_bayes",
    "threshold_bayes",
    "rescorla_wagner",
    "good_bad_news",
    "confirmatory_misperception",
    "anchoring_to_prior",
    "sublinear_sample_size",
}

CONDITIONAL_UPDATE_MODELS = {
    "partial_step",
    "t1_spike",
    "asymmetric",
    "recency",
    "changepoint",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    k: int
    param_grid: list[dict]
    prediction: Callable[[pd.DataFrame, dict], np.ndarray]
    description: str
    parameterization: str


@dataclass
class Fit:
    model: str
    bic: float
    aic: float
    aicc: float
    sse: float
    rmse: float
    k: int
    params: dict


def bic_from_sse(sse: float, n: int, k: int) -> float:
    sigma2 = max(float(sse) / max(n, 1), 1e-12)
    return n * math.log(sigma2) + k * math.log(max(n, 2))


def aic_from_sse(sse: float, n: int, k: int) -> float:
    sigma2 = max(float(sse) / max(n, 1), 1e-12)
    return n * math.log(sigma2) + 2.0 * k


def aicc_from_sse(sse: float, n: int, k: int) -> float:
    base = aic_from_sse(sse, n, k)
    denom = n - k - 1
    if denom <= 0:
        return float("inf")
    return base + (2.0 * k * (k + 1)) / denom


def make_fit(model: str, sse: float, n: int, k: int, params: dict) -> Fit:
    return Fit(
        model=model,
        bic=bic_from_sse(sse, n, k),
        aic=aic_from_sse(sse, n, k),
        aicc=aicc_from_sse(sse, n, k),
        sse=sse,
        rmse=math.sqrt(float(sse) / max(n, 1)),
        k=k,
        params=params,
    )


def clean_panel(panel: pd.DataFrame) -> pd.DataFrame:
    needed = ["t", "b_prev", "b_post", "delta_b", "norm_step", "f_t", "n0"]
    out = panel.copy()
    for col in needed:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    out["pid_task"] = out["participant_id"].astype(str) + "::" + out["taskType"].astype(str)
    return out


def beta_bayes(g: pd.DataFrame, lam: float = 1.0) -> np.ndarray:
    g = g.sort_values("t")
    b0 = float(g["b_prev"].iloc[0])
    n0 = float(g["n0"].iloc[0])
    f = g["f_t"].astype(float).to_numpy()
    t = g["t"].astype(float).to_numpy()
    successes = np.cumsum(f)
    return (b0 * n0 + lam * successes) / (n0 + lam * t)


def discounted_bayes(g: pd.DataFrame, gamma: float, lam: float = 1.0) -> np.ndarray:
    """Exponentially discounted Bayesian-style evidence accumulation.

    gamma = 1 recovers ordinary cumulative weighting.
    gamma < 1 makes older observations matter less than recent ones, so the
    model is a non-divisible recency/forgetting competitor.
    """
    g = g.sort_values("t")
    b0 = float(g["b_prev"].iloc[0])
    n0 = float(g["n0"].iloc[0])
    f = g["f_t"].astype(float).to_numpy()
    out = []
    discounted_successes = 0.0
    discounted_trials = 0.0
    for obs in f:
        discounted_successes = gamma * discounted_successes + obs
        discounted_trials = gamma * discounted_trials + 1.0
        out.append((b0 * n0 + lam * discounted_successes) / (n0 + lam * discounted_trials))
    return np.asarray(out)


def pred_no_update(g: pd.DataFrame, _params: dict) -> np.ndarray:
    g = g.sort_values("t")
    return np.repeat(float(g["b_prev"].iloc[0]), len(g))


def pred_standard_bayes(g: pd.DataFrame, _params: dict) -> np.ndarray:
    return beta_bayes(g, 1.0)


def pred_weighted_bayes(g: pd.DataFrame, params: dict) -> np.ndarray:
    return beta_bayes(g, float(params["lambda"]))


def pred_discounted_bayes_gamma(g: pd.DataFrame, params: dict) -> np.ndarray:
    return discounted_bayes(g, gamma=float(params["gamma"]), lam=1.0)


def pred_discounted_weighted_bayes(g: pd.DataFrame, params: dict) -> np.ndarray:
    return discounted_bayes(g, gamma=float(params["gamma"]), lam=float(params["lambda"]))


def coarse_round(x: np.ndarray, width: float) -> np.ndarray:
    return np.clip(np.round(x / width) * width, 0.0, 1.0)


def pred_coarse_weighted_bayes(g: pd.DataFrame, params: dict) -> np.ndarray:
    return coarse_round(beta_bayes(g, float(params["lambda"])), float(params["width"]))


def pred_threshold_bayes(g: pd.DataFrame, params: dict) -> np.ndarray:
    threshold = float(params["threshold"])
    bayes = beta_bayes(g, 1.0)
    current = float(g.sort_values("t")["b_prev"].iloc[0])
    out = []
    for target in bayes:
        if abs(target - current) >= threshold:
            current = float(target)
        out.append(current)
    return np.asarray(out)


def pred_rescorla_wagner(g: pd.DataFrame, params: dict) -> np.ndarray:
    alpha = float(params["alpha"])
    g = g.sort_values("t")
    b = float(g["b_prev"].iloc[0])
    out = []
    for f in g["f_t"].astype(float).to_numpy():
        b = b + alpha * (f - b)
        out.append(b)
    return np.asarray(out)


def pred_good_bad_news(g: pd.DataFrame, params: dict) -> np.ndarray:
    alpha_good = float(params["alpha_good"])
    alpha_bad = float(params["alpha_bad"])
    g = g.sort_values("t")
    b = float(g["b_prev"].iloc[0])
    out = []
    for f in g["f_t"].astype(float).to_numpy():
        pe = f - b
        alpha = alpha_good if pe >= 0 else alpha_bad
        b = b + alpha * pe
        out.append(b)
    return np.asarray(out)


def pred_anchoring(g: pd.DataFrame, params: dict) -> np.ndarray:
    g = g.sort_values("t")
    b0 = float(g["b_prev"].iloc[0])
    bayes = beta_bayes(g, 1.0)
    return b0 + float(params["adjustment"]) * (bayes - b0)


def pred_sublinear_sample_size(g: pd.DataFrame, params: dict) -> np.ndarray:
    gamma = float(params["gamma"])
    g = g.sort_values("t")
    b0 = float(g["b_prev"].iloc[0])
    n0 = float(g["n0"].iloc[0])
    f = g["f_t"].astype(float).to_numpy()
    t = g["t"].astype(float).to_numpy()
    successes = np.cumsum(f)
    sample_mean = successes / np.maximum(t, 1.0)
    effective_n = np.power(t, gamma)
    return (b0 * n0 + effective_n * sample_mean) / (n0 + effective_n)


def pred_confirmatory(g: pd.DataFrame, params: dict) -> np.ndarray:
    lam = float(params["lambda"])
    q = float(params["q"])
    g = g.sort_values("t")
    b0 = float(g["b_prev"].iloc[0])
    n0 = float(g["n0"].iloc[0])
    alpha = b0 * n0
    beta = (1.0 - b0) * n0
    out = []
    for f in g["f_t"].astype(float).to_numpy():
        b_prev = alpha / (alpha + beta)
        if b_prev >= 0.5 and f < 0.5:
            perceived = q
        elif b_prev < 0.5 and f > 0.5:
            perceived = 1.0 - q
        else:
            perceived = f
        alpha += lam * perceived
        beta += lam * (1.0 - perceived)
        out.append(alpha / (alpha + beta))
    return np.asarray(out)


def pred_sticky_weighted_bayes(g: pd.DataFrame, params: dict) -> np.ndarray:
    """Weighted Bayes latent state with sticky response expression.

    This is an observation-layer competitor: the latent learner can update, but
    the reported belief moves only partially toward that latent value on each
    trial. It captures rounded/flat reports without forcing the latent process
    itself to be no-update.
    """
    stickiness = float(params["stickiness"])
    latent = beta_bayes(g, float(params["lambda"]))
    current = float(g.sort_values("t")["b_prev"].iloc[0])
    out = []
    for target in latent:
        current = stickiness * current + (1.0 - stickiness) * float(target)
        out.append(current)
    return np.asarray(out)


def linear_design(g: pd.DataFrame, model: str) -> tuple[np.ndarray, list[str]]:
    g = g.sort_values("t")
    if model == "partial_step":
        return g[["norm_step"]].to_numpy(), ["sigma"]
    if model == "t1_spike":
        return (g["t"].to_numpy() == 1).astype(float).reshape(-1, 1), ["trial1_shift"]
    if model == "asymmetric":
        x = g["norm_step"].to_numpy()
        return np.column_stack([np.maximum(x, 0.0), np.minimum(x, 0.0)]), ["sigma_positive", "sigma_negative"]
    if model == "recency":
        f = g["f_t"].astype(float).to_numpy()
        t = g["t"].astype(float).to_numpy()
        b_prev = g["b_prev"].astype(float).to_numpy()
        corrects = 0.0
        accbar = []
        for ti, fi in zip(t, f):
            accbar.append(corrects / max(1.0, ti - 1.0) if ti > 1 else 0.0)
            corrects += fi
        return (np.asarray(accbar) - b_prev).reshape(-1, 1), ["sigma"]
    raise ValueError(f"Unknown linear model: {model}")


def pred_linear_delta(g: pd.DataFrame, params: dict) -> np.ndarray:
    model = params["_linear_model"]
    X, names = linear_design(g, model)
    beta = np.asarray([float(params[name]) for name in names])
    delta_hat = float(params["intercept"]) + X @ beta
    return np.clip(g.sort_values("t")["b_prev"].astype(float).to_numpy() + delta_hat, 0.0, 1.0)


def fit_linear_delta(g: pd.DataFrame, model: str, train_mask: np.ndarray | None = None) -> Fit:
    g = g.sort_values("t")
    y = g["b_post"].astype(float).to_numpy()
    target_delta = g["delta_b"].astype(float).to_numpy()
    X, names = linear_design(g, model)
    X = np.column_stack([np.ones(len(g)), X])
    mask = np.ones(len(g), dtype=bool) if train_mask is None else train_mask
    params_arr, *_ = np.linalg.lstsq(X[mask], target_delta[mask], rcond=None)
    params = {"_linear_model": model, "intercept": float(params_arr[0])}
    for name, value in zip(names, params_arr[1:]):
        params[name] = float(value)
    pred = np.clip(g["b_prev"].astype(float).to_numpy() + X @ params_arr, 0.0, 1.0)
    resid = y[mask] - pred[mask]
    sse = float(np.dot(resid, resid))
    k = len(params_arr)
    return make_fit(model, sse, int(mask.sum()), k, params)


def pred_changepoint(g: pd.DataFrame, params: dict) -> np.ndarray:
    g = g.sort_values("t")
    x = g["norm_step"].astype(float).to_numpy()
    early = (g["t"].to_numpy() <= int(params["tau"])).astype(float)
    delta_hat = (
        float(params["intercept"])
        + float(params["sigma_early"]) * x * early
        + float(params["sigma_late"]) * x * (1.0 - early)
    )
    return np.clip(g["b_prev"].astype(float).to_numpy() + delta_hat, 0.0, 1.0)


def fit_changepoint(g: pd.DataFrame, train_mask: np.ndarray | None = None) -> Fit:
    g = g.sort_values("t")
    y = g["b_post"].astype(float).to_numpy()
    target_delta = g["delta_b"].astype(float).to_numpy()
    x = g["norm_step"].astype(float).to_numpy()
    mask = np.ones(len(g), dtype=bool) if train_mask is None else train_mask
    best = None
    for tau in range(1, 10):
        early = (g["t"].to_numpy() <= tau).astype(float)
        X = np.column_stack([np.ones(len(g)), x * early, x * (1.0 - early)])
        params_arr, *_ = np.linalg.lstsq(X[mask], target_delta[mask], rcond=None)
        pred = np.clip(g["b_prev"].astype(float).to_numpy() + X @ params_arr, 0.0, 1.0)
        resid = y[mask] - pred[mask]
        sse = float(np.dot(resid, resid))
        k = len(params_arr) + 1
        fit = make_fit(
            "changepoint",
            sse,
            int(mask.sum()),
            k,
            {
                "intercept": float(params_arr[0]),
                "sigma_early": float(params_arr[1]),
                "sigma_late": float(params_arr[2]),
                "tau": int(tau),
            },
        )
        if best is None or fit.bic < best.bic:
            best = fit
    return best


def model_specs(fast: bool = False) -> list[ModelSpec]:
    lambda_grid = np.linspace(0.0, 5.0, 51 if fast else 101)
    lambda_grid_short = np.linspace(0.0, 3.0, 31 if fast else 61)
    gamma_grid = np.linspace(0.0, 1.0, 21 if fast else 41)
    gamma_grid_focused = np.asarray([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 1.0]) if fast else np.unique(np.concatenate([np.linspace(0.0, 1.0, 21), [0.85, 0.95, 0.99]]))
    widths = [0.01, 0.02, 0.05, 0.10, 0.20] if not fast else [0.02, 0.05, 0.10, 0.20]
    alphas = np.linspace(0.0, 1.0, 11 if fast else 21)
    return [
        ModelSpec("no_update", "inertia", 0, [{}], pred_no_update, "Reported belief stays at task prior.", "No free parameter."),
        ModelSpec("standard_bayes", "bayesian", 0, [{}], pred_standard_bayes, "Full Beta-Bayes benchmark.", "lambda fixed at 1."),
        ModelSpec(
            "divisible_weighted_bayes",
            "bayesian_weighted",
            1,
            [{"lambda": float(x)} for x in lambda_grid],
            pred_weighted_bayes,
            "Path-independent cumulative updating with one evidence weight.",
            f"lambda grid [{lambda_grid.min():.2f}, {lambda_grid.max():.2f}], {len(lambda_grid)} values.",
        ),
        ModelSpec(
            "discounted_bayes_gamma",
            "recency_discounted",
            1,
            [{"gamma": float(x)} for x in gamma_grid],
            pred_discounted_bayes_gamma,
            "Bayesian filter with exponentially discounted older evidence.",
            f"gamma grid [{gamma_grid.min():.2f}, {gamma_grid.max():.2f}], {len(gamma_grid)} values; gamma=1 is no forgetting.",
        ),
        ModelSpec(
            "discounted_weighted_bayes",
            "recency_discounted",
            2,
            [{"lambda": float(lam), "gamma": float(gamma)} for lam in lambda_grid_short for gamma in gamma_grid_focused],
            pred_discounted_weighted_bayes,
            "Hybrid model with global evidence weight lambda and memory retention gamma.",
            f"lambda grid [{lambda_grid_short.min():.2f}, {lambda_grid_short.max():.2f}], gamma grid includes {list(np.round(gamma_grid_focused, 2))}.",
        ),
        ModelSpec(
            "coarse_weighted_bayes",
            "coarse",
            2,
            [{"lambda": float(lam), "width": float(width)} for lam in lambda_grid_short for width in widths],
            pred_coarse_weighted_bayes,
            "Weighted Bayes followed by equal-width belief rounding.",
            f"lambda grid [{lambda_grid_short.min():.2f}, {lambda_grid_short.max():.2f}], widths={widths}.",
        ),
        ModelSpec(
            "sticky_weighted_bayes",
            "response_expression",
            2,
            [
                {"lambda": float(lam), "stickiness": float(stickiness)}
                for lam in lambda_grid_short
                for stickiness in np.linspace(0.0, 0.95, 11 if fast else 21)
            ],
            pred_sticky_weighted_bayes,
            "Weighted Bayesian latent updating with sticky reported-belief expression.",
            "lambda controls latent evidence weight; stickiness controls partial movement of reports toward the latent state.",
        ),
        ModelSpec(
            "threshold_bayes",
            "inattention",
            1,
            [{"threshold": float(x)} for x in np.linspace(0.0, 0.30, 31 if fast else 61)],
            pred_threshold_bayes,
            "Update only when Bayesian move exceeds a threshold.",
            "threshold is an absolute probability-scale move from 0 to 0.30.",
        ),
        ModelSpec(
            "rescorla_wagner",
            "prediction_error",
            1,
            [{"alpha": float(x)} for x in np.linspace(0.0, 1.0, 51 if fast else 101)],
            pred_rescorla_wagner,
            "Prediction-error update toward the binary AI outcome.",
            "alpha learning-rate grid from 0 to 1.",
        ),
        ModelSpec(
            "good_bad_news",
            "asymmetry",
            2,
            [{"alpha_good": float(ag), "alpha_bad": float(ab)} for ag in alphas for ab in alphas],
            pred_good_bad_news,
            "Separate prediction-error rates for AI successes and failures.",
            f"alpha_good and alpha_bad grids from 0 to 1, {len(alphas)} values each.",
        ),
        ModelSpec(
            "confirmatory_misperception",
            "confirmation",
            2,
            [
                {"lambda": float(lam), "q": float(q)}
                for lam in lambda_grid_short
                for q in np.linspace(0.0, 0.50, 6 if fast else 11)
            ],
            pred_confirmatory,
            "Belief-incongruent evidence is partially distorted toward current belief.",
            "lambda controls evidence weight; q controls incongruent-signal distortion.",
        ),
        ModelSpec(
            "anchoring_to_prior",
            "anchoring",
            1,
            [{"adjustment": float(x)} for x in np.linspace(0.0, 2.0, 51 if fast else 101)],
            pred_anchoring,
            "Move partway from task prior toward the Bayesian posterior.",
            "adjustment grid from 0 to 2; values below 1 are conservative.",
        ),
        ModelSpec(
            "sublinear_sample_size",
            "sample_size_distortion",
            1,
            [{"gamma": float(x)} for x in np.linspace(0.0, 1.5, 51 if fast else 76)],
            pred_sublinear_sample_size,
            "Effective sample size grows as t^gamma.",
            "gamma below 1 means evidence accumulates sublinearly.",
        ),
        ModelSpec("partial_step", "linear_norm_step", 2, [], pred_linear_delta, "Linear partial response to Bayesian step.", "OLS delta model: intercept + sigma*norm_step."),
        ModelSpec("t1_spike", "first_impression", 2, [], pred_linear_delta, "A one-time first-trial shift.", "OLS delta model with trial-1 indicator."),
        ModelSpec("asymmetric", "norm_step_asymmetry", 3, [], pred_linear_delta, "Separate slopes for positive and negative Bayesian steps.", "OLS delta model with positive and negative norm_step components."),
        ModelSpec("recency", "heuristic", 2, [], pred_linear_delta, "Move toward running AI accuracy.", "OLS delta model using running accuracy minus current belief."),
        ModelSpec("changepoint", "phase_change", 4, [], pred_changepoint, "Different norm-step slopes before and after a fitted trial cutoff.", "OLS delta model with tau in 1..9."),
    ]


def select_model_set(specs: list[ModelSpec], model_set: str) -> list[ModelSpec]:
    if model_set == "all":
        return specs
    lean = {
        "no_update",
        "standard_bayes",
        "divisible_weighted_bayes",
        "discounted_bayes_gamma",
        "discounted_weighted_bayes",
        "coarse_weighted_bayes",
        "sticky_weighted_bayes",
        "threshold_bayes",
        "good_bad_news",
    }
    gamma_focus = {
        "no_update",
        "standard_bayes",
        "divisible_weighted_bayes",
        "discounted_bayes_gamma",
        "discounted_weighted_bayes",
        "good_bad_news",
    }
    if model_set == "primary":
        keep = PRIMARY_GENERATIVE_MODELS
    elif model_set == "lean":
        keep = lean
    elif model_set == "gamma":
        keep = gamma_focus
    elif model_set == "conditional":
        keep = CONDITIONAL_UPDATE_MODELS
    else:
        raise ValueError("--model-set must be one of: gamma, lean, primary, conditional, all")
    return [spec for spec in specs if spec.name in keep]


def fit_grid_model(g: pd.DataFrame, spec: ModelSpec, train_mask: np.ndarray | None = None) -> Fit:
    g = g.sort_values("t")
    y = g["b_post"].astype(float).to_numpy()
    mask = np.ones(len(g), dtype=bool) if train_mask is None else train_mask
    best = None
    for params in spec.param_grid:
        pred = np.clip(spec.prediction(g, params), 0.0, 1.0)
        resid = y[mask] - pred[mask]
        sse = float(np.dot(resid, resid))
        fit = make_fit(spec.name, sse, int(mask.sum()), spec.k, dict(params))
        if best is None or fit.bic < best.bic:
            best = fit
    return best


def fit_model(g: pd.DataFrame, spec: ModelSpec, train_mask: np.ndarray | None = None) -> Fit:
    if spec.name in {"partial_step", "t1_spike", "asymmetric", "recency"}:
        return fit_linear_delta(g, spec.name, train_mask=train_mask)
    if spec.name == "changepoint":
        return fit_changepoint(g, train_mask=train_mask)
    return fit_grid_model(g, spec, train_mask=train_mask)


def predict_from_fit(g: pd.DataFrame, spec_by_name: dict[str, ModelSpec], fit: Fit) -> np.ndarray:
    if fit.model in {"partial_step", "t1_spike", "asymmetric", "recency"}:
        return pred_linear_delta(g, fit.params)
    if fit.model == "changepoint":
        return pred_changepoint(g, fit.params)
    return spec_by_name[fit.model].prediction(g, fit.params)


def loocv_rmse(g: pd.DataFrame, spec: ModelSpec) -> float:
    g = g.sort_values("t").reset_index(drop=True)
    y = g["b_post"].astype(float).to_numpy()
    errs = []
    for idx in range(len(g)):
        mask = np.ones(len(g), dtype=bool)
        mask[idx] = False
        fit = fit_model(g, spec, train_mask=mask)
        pred = predict_from_fit(g, {spec.name: spec}, fit)
        errs.append(float(y[idx] - pred[idx]) ** 2)
    return math.sqrt(float(np.mean(errs)))


def future_block_rmse(g: pd.DataFrame, spec: ModelSpec, holdout_trials: int) -> float:
    """Fit early trials and score final reports for trajectory-generating models.

    Conditional update-rule checks reconstruct reports from the observed lagged
    report. In a future-block split, that would let later held-out predictions
    condition on earlier held-out reports, so those models are excluded from the
    future-block winner rule rather than treated as strict predictive accounts.
    """
    if spec.name in CONDITIONAL_UPDATE_MODELS:
        return np.nan
    g = g.sort_values("t").reset_index(drop=True)
    if holdout_trials <= 0 or len(g) <= holdout_trials:
        return np.nan
    split = len(g) - holdout_trials
    if split < 2:
        return np.nan
    y = g["b_post"].astype(float).to_numpy()
    mask = np.zeros(len(g), dtype=bool)
    mask[:split] = True
    fit = fit_model(g, spec, train_mask=mask)
    pred = predict_from_fit(g, {spec.name: spec}, fit)
    resid = y[~mask] - pred[~mask]
    return math.sqrt(float(np.mean(np.square(resid))))


def evidence_weights(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return [np.nan for _ in values]
    deltas = np.full_like(arr, np.inf)
    deltas[finite] = arr[finite] - float(np.min(arr[finite]))
    raw = np.zeros_like(arr)
    raw[finite] = np.exp(-0.5 * deltas[finite])
    denom = float(np.sum(raw))
    if denom <= 0:
        return [np.nan for _ in values]
    return [float(x) for x in raw / denom]


def runner_up_delta(values: list[float]) -> float:
    finite = sorted(float(x) for x in values if np.isfinite(x))
    if len(finite) < 2:
        return np.nan
    return float(finite[1] - finite[0])


def classify_one_trajectory(args: tuple) -> tuple[dict | None, list[dict]]:
    pid, task, g, specs, with_loocv, future_holdout_trials = args
    spec_by_name = {s.name: s for s in specs}
    fit_rows = []
    g = g.sort_values("t").reset_index(drop=True)
    if len(g) < 5:
        return None, fit_rows
    fits = []
    task_fit_rows = []
    for spec in specs:
        fit = fit_model(g, spec)
        if with_loocv:
            fit_cv = loocv_rmse(g, spec)
        else:
            fit_cv = np.nan
        fit_lfo = future_block_rmse(g, spec, future_holdout_trials)
        pred = np.clip(predict_from_fit(g, spec_by_name, fit), 0.0, 1.0)
        task_fit_rows.append(
            {
                "participant_id": pid,
                "taskType": task,
                "model": fit.model,
                "bic": fit.bic,
                "aic": fit.aic,
                "aicc": fit.aicc,
                "sse": fit.sse,
                "rmse": fit.rmse,
                "loocv_rmse": fit_cv,
                "future_block_rmse": fit_lfo,
                "k": fit.k,
                "params_json": json.dumps(fit.params, sort_keys=True),
                "mean_abs_error": float(np.mean(np.abs(g["b_post"].to_numpy() - pred))),
            }
        )
        fits.append((fit, fit_cv, fit_lfo))
    bic_values = [x[0].bic for x in fits]
    aicc_values = [x[0].aicc for x in fits]
    bic_weights = evidence_weights(bic_values)
    aicc_weights = evidence_weights(aicc_values)
    min_bic = float(np.min(bic_values))
    min_aicc = float(np.nanmin(aicc_values)) if np.isfinite(aicc_values).any() else np.nan
    bic_runner_up_delta = runner_up_delta(bic_values)
    aicc_runner_up_delta = runner_up_delta(aicc_values)
    for row, bic_weight, aicc_weight in zip(task_fit_rows, bic_weights, aicc_weights):
        row["delta_bic"] = float(row["bic"] - min_bic)
        row["bic_weight"] = bic_weight
        row["delta_aicc"] = float(row["aicc"] - min_aicc) if np.isfinite(min_aicc) else np.nan
        row["aicc_weight"] = aicc_weight
        fit_rows.append(row)
    best_bic = min(fits, key=lambda x: x[0].bic)[0]
    best_aicc = min(fits, key=lambda x: x[0].aicc if np.isfinite(x[0].aicc) else np.inf)[0]
    best_cv = min(fits, key=lambda x: x[1] if np.isfinite(x[1]) else np.inf)[0] if with_loocv else None
    finite_lfo = [x for x in fits if np.isfinite(x[2])]
    best_lfo = min(finite_lfo, key=lambda x: x[2])[0] if future_holdout_trials > 0 and finite_lfo else None
    class_row = {
        "participant_id": pid,
        "taskType": task,
        "taskPosition": int(g["taskPosition"].iloc[0]) if "taskPosition" in g.columns else np.nan,
        "n_trials": int(len(g)),
        "nonzero_updates": int((g["delta_b"].abs() > 1e-12).sum()),
        "sum_abs_delta": float(g["delta_b"].abs().sum()),
        "bic_winner": best_bic.model,
        "bic_winner_bic": best_bic.bic,
        "bic_winner_weight": float(max(bic_weights)) if bic_weights else np.nan,
        "near_tie_models_delta_bic_2": int(np.sum(np.asarray(bic_values) <= min_bic + 2.0)) if np.isfinite(min_bic) else 0,
        "bic_delta_to_runner_up": bic_runner_up_delta,
        "aicc_winner": best_aicc.model,
        "aicc_winner_aicc": best_aicc.aicc,
        "aicc_winner_weight": float(max(aicc_weights)) if aicc_weights else np.nan,
        "near_tie_models_delta_aicc_2": int(np.sum(np.asarray(aicc_values) <= min_aicc + 2.0)) if np.isfinite(min_aicc) else 0,
        "aicc_delta_to_runner_up": aicc_runner_up_delta,
        "aicc_winner_uncertain": int(
            (np.isfinite(aicc_runner_up_delta) and aicc_runner_up_delta <= 2.0)
            or (aicc_weights and np.isfinite(max(aicc_weights)) and max(aicc_weights) < 0.80)
        ),
        "loocv_winner": best_cv.model if best_cv else "",
        "loocv_winner_rmse": min(x[1] for x in fits) if with_loocv else np.nan,
        "future_block_winner": best_lfo.model if best_lfo else "",
        "future_block_winner_rmse": min(x[2] for x in finite_lfo) if best_lfo else np.nan,
        "future_holdout_trials": int(future_holdout_trials),
    }
    return class_row, fit_rows


def classify_panel(
    panel: pd.DataFrame,
    specs: list[ModelSpec],
    with_loocv: bool,
    future_holdout_trials: int = 2,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = clean_panel(panel)
    tasks = [
        (pid, task, g.copy(), specs, with_loocv, future_holdout_trials)
        for (pid, task), g in panel.groupby(["participant_id", "taskType"], sort=False)
    ]
    class_rows = []
    fit_rows = []
    if n_jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = pool.map(classify_one_trajectory, tasks, chunksize=max(1, len(tasks) // (n_jobs * 4)))
            for class_row, rows in results:
                if class_row is not None:
                    class_rows.append(class_row)
                fit_rows.extend(rows)
    else:
        for task_args in tasks:
            class_row, rows = classify_one_trajectory(task_args)
            if class_row is not None:
                class_rows.append(class_row)
            fit_rows.extend(rows)
    return pd.DataFrame(class_rows), pd.DataFrame(fit_rows)


def sample_trajectories(panel: pd.DataFrame, max_trajectories: int | None, seed: int) -> pd.DataFrame:
    if max_trajectories is None:
        return panel
    keys = panel[["participant_id", "taskType"]].drop_duplicates()
    if len(keys) <= max_trajectories:
        return panel
    sampled = keys.sample(n=max_trajectories, random_state=seed)
    return panel.merge(sampled, on=["participant_id", "taskType"], how="inner")


def winner_counts(classification: pd.DataFrame, winner_col: str, label: str) -> pd.DataFrame:
    out = classification[winner_col].value_counts().rename_axis("model").reset_index(name="wins")
    out["share"] = out["wins"] / out["wins"].sum()
    out["winner_rule"] = label
    return out


def subset_summaries(classification: pd.DataFrame, winner_col: str) -> pd.DataFrame:
    subsets = {
        "all": classification,
        "active_ge1": classification[classification["nonzero_updates"] >= 1],
        "active_ge2": classification[classification["nonzero_updates"] >= 2],
        "flat_all_10": classification[classification["nonzero_updates"] == 0],
    }
    rows = []
    for subset_name, df in subsets.items():
        if df.empty:
            continue
        counts = df[winner_col].value_counts()
        for model, wins in counts.items():
            rows.append(
                {
                    "subset": subset_name,
                    "winner_rule": winner_col,
                    "model": model,
                    "wins": int(wins),
                    "share": float(wins / len(df)),
                    "participant_tasks": int(len(df)),
                }
            )
    return pd.DataFrame(rows)


def parameter_summary(fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in fits.iterrows():
        params = json.loads(row["params_json"])
        for key, value in params.items():
            if key.startswith("_"):
                continue
            rows.append({"model": row["model"], "parameter": key, "value": value})
    params = pd.DataFrame(rows)
    if params.empty:
        return params
    return (
        params.groupby(["model", "parameter"])
        .agg(
            mean=("value", "mean"),
            median=("value", "median"),
            p10=("value", lambda s: float(np.quantile(s, 0.10))),
            p90=("value", lambda s: float(np.quantile(s, 0.90))),
        )
        .reset_index()
    )


def trajectory_summary_from_reports(reports: np.ndarray, initial_belief: float) -> dict[str, float]:
    reports = np.asarray(reports, dtype=float)
    prev = np.concatenate([[float(initial_belief)], reports[:-1]])
    updates = reports - prev
    return {
        "mean_b_post": float(np.mean(reports)),
        "final_shift": float(reports[-1] - float(initial_belief)),
        "mean_abs_update": float(np.mean(np.abs(updates))),
        "sd_update": float(np.std(updates, ddof=0)),
        "zero_update_rate_005": float(np.mean(np.abs(updates) <= 0.005)),
        "positive_update_rate": float(np.mean(updates > 0.005)),
        "negative_update_rate": float(np.mean(updates < -0.005)),
    }


def fitted_trajectory_summaries(panel: pd.DataFrame, fits: pd.DataFrame, specs: list[ModelSpec]) -> pd.DataFrame:
    panel = clean_panel(panel)
    spec_by_name = {s.name: s for s in specs}
    rows = []
    for (pid, task), g in panel.groupby(["participant_id", "taskType"], sort=False):
        g = g.sort_values("t").reset_index(drop=True)
        task_fits = fits[(fits["participant_id"].astype(str) == str(pid)) & (fits["taskType"].astype(str) == str(task))]
        if task_fits.empty:
            continue
        initial = float(g["b_prev"].iloc[0])
        observed = trajectory_summary_from_reports(g["b_post"].astype(float).to_numpy(), initial)
        for _, row in task_fits.iterrows():
            params = json.loads(row["params_json"])
            fake_fit = Fit(
                model=row["model"],
                bic=float(row["bic"]),
                aic=float(row["aic"]),
                aicc=float(row["aicc"]),
                sse=float(row["sse"]),
                rmse=float(row["rmse"]),
                k=int(row["k"]),
                params=params,
            )
            pred = np.clip(predict_from_fit(g, spec_by_name, fake_fit), 0.0, 1.0)
            predicted = trajectory_summary_from_reports(pred, initial)
            for metric, obs_value in observed.items():
                pred_value = predicted[metric]
                rows.append(
                    {
                        "participant_id": pid,
                        "taskType": task,
                        "model": row["model"],
                        "metric": metric,
                        "observed": obs_value,
                        "predicted": pred_value,
                        "abs_error": abs(obs_value - pred_value),
                    }
                )
    return pd.DataFrame(rows)


def absolute_fit_summary(trajectory_summaries: pd.DataFrame) -> pd.DataFrame:
    if trajectory_summaries.empty:
        return trajectory_summaries
    return (
        trajectory_summaries.groupby(["model", "metric"], as_index=False)
        .agg(
            mean_observed=("observed", "mean"),
            mean_predicted=("predicted", "mean"),
            mean_abs_error=("abs_error", "mean"),
            p90_abs_error=("abs_error", lambda s: float(np.quantile(s, 0.90))),
        )
        .sort_values(["metric", "mean_abs_error"])
    )


def prediction_for_model(g: pd.DataFrame, spec_by_name: dict[str, ModelSpec], model: str, params: dict) -> np.ndarray:
    if model in {"partial_step", "t1_spike", "asymmetric", "recency"}:
        return pred_linear_delta(g, params)
    if model == "changepoint":
        return pred_changepoint(g, params)
    return spec_by_name[model].prediction(g, params)


def simulate_trajectory(
    g: pd.DataFrame,
    model: str,
    params: dict,
    spec_by_name: dict[str, ModelSpec],
    rng: np.random.Generator,
    noise_sd: float,
    lapse_rate: float,
    response_grid: float,
) -> pd.DataFrame:
    """Generate one synthetic report sequence on the observed study design."""
    g = g.sort_values("t").reset_index(drop=True).copy()
    latent = np.clip(prediction_for_model(g, spec_by_name, model, params), 0.0, 1.0)
    reports = []
    prev = float(g["b_prev"].iloc[0])
    for target in latent:
        report = float(target)
        if noise_sd > 0:
            report += float(rng.normal(0.0, noise_sd))
        if lapse_rate > 0 and float(rng.random()) < lapse_rate:
            report = prev
        if response_grid > 0:
            report = round(report / response_grid) * response_grid
        report = float(np.clip(report, 0.0, 1.0))
        reports.append(report)
        prev = report
    b_prev = np.asarray([float(g["b_prev"].iloc[0])] + reports[:-1], dtype=float)
    b_post = np.asarray(reports, dtype=float)
    g["b_prev"] = b_prev
    g["b_post"] = b_post
    g["delta_b"] = b_post - b_prev
    return g


def run_model_recovery(
    panel: pd.DataFrame,
    fits: pd.DataFrame,
    specs: list[ModelSpec],
    generator_models: list[str],
    reps_per_model: int,
    seed: int,
    noise_sd: float,
    lapse_rate: float,
    response_grid: float,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Design-matched recovery: simulate from fitted models and refit all candidates."""
    panel = clean_panel(panel)
    spec_by_name = {s.name: s for s in specs}
    rng = np.random.default_rng(seed)
    sim_rows = []
    design_lookup = {
        (str(pid), str(task)): g.sort_values("t").reset_index(drop=True)
        for (pid, task), g in panel.groupby(["participant_id", "taskType"], sort=False)
    }
    for model in generator_models:
        if model not in spec_by_name:
            continue
        pool = fits[fits["model"] == model].reset_index(drop=True)
        if pool.empty:
            continue
        sampled_idx = rng.choice(np.arange(len(pool)), size=reps_per_model, replace=True)
        for rep, idx in enumerate(sampled_idx):
            row = pool.iloc[int(idx)]
            key = (str(row["participant_id"]), str(row["taskType"]))
            if key not in design_lookup:
                continue
            params = json.loads(row["params_json"])
            synth = simulate_trajectory(
                design_lookup[key],
                model,
                params,
                spec_by_name,
                rng,
                noise_sd=noise_sd,
                lapse_rate=lapse_rate,
                response_grid=response_grid,
            )
            sim_pid = f"sim_{model}_{rep:04d}"
            synth["participant_id"] = sim_pid
            synth["taskType"] = str(row["taskType"])
            synth["generating_model"] = model
            sim_rows.append(synth)
    if not sim_rows:
        return pd.DataFrame(), pd.DataFrame()
    simulated = pd.concat(sim_rows, ignore_index=True)
    classification, recovery_fits = classify_panel(
        simulated,
        specs,
        with_loocv=False,
        future_holdout_trials=0,
        n_jobs=n_jobs,
    )
    generating = simulated[["participant_id", "taskType", "generating_model"]].drop_duplicates()
    classification = classification.merge(generating, on=["participant_id", "taskType"], how="left")
    recovery_fits = recovery_fits.merge(generating, on=["participant_id", "taskType"], how="left")
    return classification, recovery_fits


def recovery_summary(recovery: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if recovery.empty:
        return pd.DataFrame(rows)
    for rule in ["bic_winner", "aicc_winner"]:
        counts = recovery.groupby(["generating_model", rule]).size().reset_index(name="n")
        totals = counts.groupby("generating_model")["n"].transform("sum")
        counts["share"] = counts["n"] / totals
        counts = counts.rename(columns={rule: "recovered_model"})
        counts["winner_rule"] = rule.replace("_winner", "")
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


def recovery_accuracy_summary(recovery: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if recovery.empty:
        return pd.DataFrame(rows)
    for rule in ["bic_winner", "aicc_winner"]:
        for generating_model, group in recovery.groupby("generating_model", sort=True):
            winners = group[rule].astype(str)
            correct = winners == str(generating_model)
            modal = winners.value_counts().index[0] if not winners.empty else ""
            rows.append(
                {
                    "winner_rule": rule.replace("_winner", ""),
                    "generating_model": generating_model,
                    "n": int(len(group)),
                    "exact_recovery_rate": float(correct.mean()) if len(group) else np.nan,
                    "modal_recovered_model": modal,
                    "modal_recovered_share": float((winners == modal).mean()) if len(group) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def model_scope(model: str) -> str:
    if model in PRIMARY_GENERATIVE_MODELS:
        return "primary_generative"
    if model in CONDITIONAL_UPDATE_MODELS:
        return "conditional_update_robustness"
    return "other"


def write_model_specs(specs: list[ModelSpec], outdir: Path) -> None:
    rows = []
    for spec in specs:
        rows.append(
            {
                "model": spec.name,
                "family": spec.family,
                "scope": model_scope(spec.name),
                "k": spec.k,
                "grid_size": len(spec.param_grid) if spec.param_grid else "closed_form_ols_or_grid_tau",
                "description": spec.description,
                "parameterization": spec.parameterization,
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "model_specifications.csv", index=False)


def run_panel_sensitivity(specs: list[ModelSpec], outdir: Path, fast: bool) -> pd.DataFrame:
    rows = []
    for tag, path in PANEL_FILES.items():
        if not path.exists():
            continue
        panel = pd.read_csv(path)
        classification, _fits = classify_panel(panel, specs, with_loocv=False, future_holdout_trials=0)
        for winner_rule in ["bic_winner"]:
            counts = classification[winner_rule].value_counts()
            for model, wins in counts.items():
                rows.append(
                    {
                        "panel": tag,
                        "model": model,
                        "wins": int(wins),
                        "share": float(wins / len(classification)),
                        "participant_tasks": int(len(classification)),
                        "fast_grid": fast,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "panel_sensitivity_bic_winners.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=CANONICAL_PANEL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--fast", action="store_true", help="Use smaller grids for sensitivity runs.")
    parser.add_argument("--model-set", choices=["gamma", "lean", "primary", "conditional", "all"], default="primary")
    parser.add_argument("--skip-loocv", action="store_true", help="Skip leave-one-trial-out checks.")
    parser.add_argument("--future-holdout-trials", type=int, default=2, help="Final trials held out for leave-future-out RMSE.")
    parser.add_argument("--skip-panel-sensitivity", action="store_true")
    parser.add_argument("--only-panel-sensitivity", action="store_true")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Optional trajectory subsample for expensive LOOCV audits.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel worker processes for participant-task model fits.")
    parser.add_argument("--run-recovery", action="store_true", help="Run design-matched model-recovery simulations.")
    parser.add_argument("--recovery-reps-per-model", type=int, default=25)
    parser.add_argument(
        "--recovery-models",
        default="no_update,divisible_weighted_bayes,discounted_weighted_bayes,coarse_weighted_bayes,sticky_weighted_bayes,threshold_bayes,good_bad_news",
        help="Comma-separated trajectory-generating models for recovery simulations.",
    )
    parser.add_argument("--recovery-noise-sd", type=float, default=0.04, help="Gaussian report noise on the 0-1 belief scale.")
    parser.add_argument("--recovery-lapse-rate", type=float, default=0.05, help="Probability of repeating the previous report in simulations.")
    parser.add_argument("--recovery-response-grid", type=float, default=0.01, help="Response scale granularity for simulated reports.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    specs = select_model_set(model_specs(fast=args.fast), args.model_set)
    write_model_specs(specs, args.outdir)

    if args.only_panel_sensitivity:
        run_panel_sensitivity(select_model_set(model_specs(fast=True), args.model_set), args.outdir, fast=True)
        print("[Robust comparison] panel sensitivity outputs written to", args.outdir)
        return

    panel = sample_trajectories(pd.read_csv(args.panel), args.max_trajectories, args.random_seed)
    classification, fits = classify_panel(
        panel,
        specs,
        with_loocv=not args.skip_loocv,
        future_holdout_trials=args.future_holdout_trials,
        n_jobs=max(1, args.n_jobs),
    )
    classification.to_csv(args.outdir / "canonical_classification.csv", index=False)
    fits.to_csv(args.outdir / "canonical_model_fits.csv", index=False)

    winner_counts(classification, "bic_winner", "bic").to_csv(args.outdir / "canonical_bic_winner_counts.csv", index=False)
    winner_counts(classification, "aicc_winner", "aicc").to_csv(args.outdir / "canonical_aicc_winner_counts.csv", index=False)
    if args.future_holdout_trials > 0:
        winner_counts(classification, "future_block_winner", "future_block_rmse").to_csv(
            args.outdir / "canonical_future_block_winner_counts.csv", index=False
        )
    if not args.skip_loocv:
        winner_counts(classification, "loocv_winner", "loocv_rmse").to_csv(
            args.outdir / "canonical_loocv_winner_counts.csv", index=False
        )
    subset_summaries(classification, "bic_winner").to_csv(args.outdir / "canonical_subset_bic_winners.csv", index=False)
    subset_summaries(classification, "aicc_winner").to_csv(args.outdir / "canonical_subset_aicc_winners.csv", index=False)
    if args.future_holdout_trials > 0:
        subset_summaries(classification, "future_block_winner").to_csv(
            args.outdir / "canonical_subset_future_block_winners.csv", index=False
        )
    if not args.skip_loocv:
        subset_summaries(classification, "loocv_winner").to_csv(args.outdir / "canonical_subset_loocv_winners.csv", index=False)
    parameter_summary(fits).to_csv(args.outdir / "canonical_parameter_summaries.csv", index=False)
    trajectory_summaries = fitted_trajectory_summaries(panel, fits, specs)
    trajectory_summaries.to_csv(args.outdir / "canonical_fitted_trajectory_summaries.csv", index=False)
    absolute_fit_summary(trajectory_summaries).to_csv(args.outdir / "canonical_absolute_fit_summary.csv", index=False)

    if args.run_recovery:
        generator_models = [x.strip() for x in args.recovery_models.split(",") if x.strip()]
        recovery, recovery_fits = run_model_recovery(
            panel,
            fits,
            specs,
            generator_models=generator_models,
            reps_per_model=args.recovery_reps_per_model,
            seed=args.random_seed,
            noise_sd=args.recovery_noise_sd,
            lapse_rate=args.recovery_lapse_rate,
            response_grid=args.recovery_response_grid,
            n_jobs=max(1, args.n_jobs),
        )
        recovery.to_csv(args.outdir / "model_recovery_classification.csv", index=False)
        recovery_fits.to_csv(args.outdir / "model_recovery_fits.csv", index=False)
        recovery_summary(recovery).to_csv(args.outdir / "model_recovery_summary.csv", index=False)
        recovery_accuracy_summary(recovery).to_csv(args.outdir / "model_recovery_accuracy_summary.csv", index=False)

    if not args.skip_panel_sensitivity:
        run_panel_sensitivity(select_model_set(model_specs(fast=True), args.model_set), args.outdir, fast=True)

    print("[Robust comparison] outputs written to", args.outdir)
    print("BIC winners:")
    print(winner_counts(classification, "bic_winner", "bic").to_string(index=False))
    print("AICc winners:")
    print(winner_counts(classification, "aicc_winner", "aicc").to_string(index=False))
    if args.future_holdout_trials > 0:
        print("Leave-future-out RMSE winners:")
        print(winner_counts(classification, "future_block_winner", "future_block_rmse").to_string(index=False))
    if not args.skip_loocv:
        print("LOOCV RMSE winners:")
        print(winner_counts(classification, "loocv_winner", "loocv_rmse").to_string(index=False))


if __name__ == "__main__":
    main()
