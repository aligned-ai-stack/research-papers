#!/usr/bin/env python
# coding: utf-8

"""Follow-up analysis: do observed belief updates look divisible?

The main H2 analysis asks whether moment-to-moment belief changes move in the
Bayesian direction. This follow-up asks a sharper path-independence question:
can a participant-task belief trajectory be represented as repeated applications
of one constant updating rule, so that accumulating evidence sequentially gives
the same posterior as aggregating the same evidence at once?

Operational test used here:
  - Divisible model: weighted Beta-Bayes with one evidence weight lambda per
    participant-task:
        b_t(lambda) = (b0*n0 + lambda*successes_t) / (n0 + lambda*t)
    This is path independent because only cumulative successes and failures
    matter, not how the sequence is partitioned.
  - Competitor models: simple non-divisible or less-constrained alternatives
    fit to observed trial-level changes: partial-step, trial-1 spike,
    changepoint, leaky evidence, asymmetric positive/negative updating, and
    recency.
  - Additional behavioral models for the extension branch: coarse Bayesian
    updating, threshold/inattention, prediction-error learning, good/bad-news
    asymmetry, confirmatory misperception, anchoring to the initial prior, and
    sublinear sample-size sensitivity.

Inputs default to the canonical H2 hybrid panel because it already contains
the inferred n0 values and normative step information used in the paper.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = ROOT / "data" / "hypothesis_outputs" / "H2" / "h2_panel_hybrid_S10.csv"
DEFAULT_OUTDIR = ROOT / "data" / "followup_outputs" / "behavioral_model_comparison"


@dataclass
class FitResult:
    model: str
    bic: float
    sse: float
    n: int
    k: int
    params: dict


def finite_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)


def bic_from_sse(sse: float, n: int, k: int) -> float:
    if n <= 0:
        return np.inf
    sigma2 = max(float(sse) / n, 1e-12)
    return n * math.log(sigma2) + k * math.log(n)


def fit_ols_bic(y: np.ndarray, X: np.ndarray, model: str, param_names: list[str], k_extra: int = 0) -> FitResult:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.column_stack([np.ones(len(y)), X])
    params_arr, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ params_arr
    sse = float(np.dot(resid, resid))
    k = int(len(params_arr) + k_extra)
    params = {"intercept": float(params_arr[0])}
    for name, value in zip(param_names, params_arr[1:]):
        params[name] = float(value)
    return FitResult(model=model, bic=bic_from_sse(sse, len(y), k), sse=sse, n=len(y), k=k, params=params)


def divisible_predictions(g: pd.DataFrame, lam: float) -> np.ndarray:
    g = g.sort_values("t")
    b0 = float(g["b_prev"].iloc[0])
    n0 = float(g["n0"].iloc[0])
    f = g["f_t"].astype(float).to_numpy()
    t = g["t"].astype(float).to_numpy()
    successes = np.cumsum(f)
    denom = n0 + lam * t
    return (b0 * n0 + lam * successes) / denom


def beta_bayes_predictions(g: pd.DataFrame, lam: float = 1.0) -> np.ndarray:
    """Weighted cumulative Beta-Bayes posterior after each observed trial."""
    return divisible_predictions(g, lam)


def fit_prediction_grid(
    g: pd.DataFrame,
    model: str,
    param_grid: list[dict],
    predict_fn,
    k: int,
) -> FitResult:
    g = g.sort_values("t")
    y = g["b_post"].astype(float).to_numpy()
    best = (np.inf, None)
    for params in param_grid:
        pred = np.asarray(predict_fn(g, **params), dtype=float)
        pred = np.clip(pred, 0.0, 1.0)
        sse = float(np.dot(y - pred, y - pred))
        if sse < best[0]:
            best = (sse, params)
    sse, params = best
    return FitResult(model=model, bic=bic_from_sse(sse, len(y), k), sse=sse, n=len(y), k=k, params=dict(params))


def fit_standard_bayes(g: pd.DataFrame) -> FitResult:
    y = g.sort_values("t")["b_post"].astype(float).to_numpy()
    pred = beta_bayes_predictions(g, lam=1.0)
    sse = float(np.dot(y - pred, y - pred))
    return FitResult("standard_bayes", bic_from_sse(sse, len(y), 0), sse, len(y), 0, {"lambda": 1.0})


def fit_no_update(g: pd.DataFrame) -> FitResult:
    g = g.sort_values("t")
    y = g["b_post"].astype(float).to_numpy()
    pred = np.repeat(float(g["b_prev"].iloc[0]), len(g))
    sse = float(np.dot(y - pred, y - pred))
    return FitResult("no_update", bic_from_sse(sse, len(y), 0), sse, len(y), 0, {})


def fit_divisible_weighted_bayes(g: pd.DataFrame, lambda_max: float = 5.0, grid_size: int = 101) -> FitResult:
    g = g.sort_values("t")
    y = g["b_post"].astype(float).to_numpy()
    b0 = float(g["b_prev"].iloc[0])
    n0 = float(g["n0"].iloc[0])
    f = g["f_t"].astype(float).to_numpy()
    t = g["t"].astype(float).to_numpy()
    successes = np.cumsum(f)

    lambdas = np.linspace(0.0, lambda_max, grid_size)
    pred = (b0 * n0 + lambdas[:, None] * successes[None, :]) / (n0 + lambdas[:, None] * t[None, :])
    sse_grid = np.sum((pred - y[None, :]) ** 2, axis=1)
    idx = int(np.argmin(sse_grid))
    sse = float(sse_grid[idx])
    lam = float(lambdas[idx])
    return FitResult(
        model="divisible_weighted_bayes",
        bic=bic_from_sse(sse, len(y), 1),
        sse=sse,
        n=len(y),
        k=1,
        params={"lambda": lam},
    )


def coarse_round(x: np.ndarray, width: float) -> np.ndarray:
    return np.clip(np.round(x / width) * width, 0.0, 1.0)


def pred_coarse_bayes(g: pd.DataFrame, width: float) -> np.ndarray:
    bayes = beta_bayes_predictions(g, lam=1.0)
    return coarse_round(bayes, width)


def fit_coarse_bayes(g: pd.DataFrame) -> FitResult:
    grid = [{"width": w} for w in [0.02, 0.05, 0.10, 0.20, 0.25]]
    return fit_prediction_grid(g, "coarse_bayes", grid, pred_coarse_bayes, k=1)


def pred_coarse_weighted_bayes(g: pd.DataFrame, lam: float, width: float) -> np.ndarray:
    bayes = beta_bayes_predictions(g, lam=lam)
    return coarse_round(bayes, width)


def fit_coarse_weighted_bayes(g: pd.DataFrame) -> FitResult:
    grid = [
        {"lam": lam, "width": width}
        for lam in np.linspace(0.0, 3.0, 31)
        for width in [0.02, 0.05, 0.10, 0.20, 0.25]
    ]
    return fit_prediction_grid(g, "coarse_weighted_bayes", grid, pred_coarse_weighted_bayes, k=2)


def pred_threshold_bayes(g: pd.DataFrame, threshold: float) -> np.ndarray:
    """Update to Bayesian posterior only when the implied move clears a threshold."""
    g = g.sort_values("t")
    bayes = beta_bayes_predictions(g, lam=1.0)
    out = []
    current = float(g["b_prev"].iloc[0])
    for target in bayes:
        if abs(target - current) >= threshold:
            current = float(target)
        out.append(current)
    return np.asarray(out)


def fit_threshold_bayes(g: pd.DataFrame) -> FitResult:
    grid = [{"threshold": th} for th in np.linspace(0.0, 0.30, 31)]
    return fit_prediction_grid(g, "threshold_bayes", grid, pred_threshold_bayes, k=1)


def pred_rescorla_wagner(g: pd.DataFrame, alpha: float) -> np.ndarray:
    g = g.sort_values("t")
    b = float(g["b_prev"].iloc[0])
    out = []
    for f in g["f_t"].astype(float).to_numpy():
        b = b + alpha * (f - b)
        out.append(b)
    return np.asarray(out)


def fit_rescorla_wagner(g: pd.DataFrame) -> FitResult:
    grid = [{"alpha": a} for a in np.linspace(0.0, 1.0, 51)]
    return fit_prediction_grid(g, "rescorla_wagner", grid, pred_rescorla_wagner, k=1)


def pred_good_bad_news(g: pd.DataFrame, alpha_good: float, alpha_bad: float) -> np.ndarray:
    g = g.sort_values("t")
    b = float(g["b_prev"].iloc[0])
    out = []
    for f in g["f_t"].astype(float).to_numpy():
        pe = f - b
        alpha = alpha_good if pe >= 0 else alpha_bad
        b = b + alpha * pe
        out.append(b)
    return np.asarray(out)


def fit_good_bad_news(g: pd.DataFrame) -> FitResult:
    vals = np.linspace(0.0, 1.0, 11)
    grid = [{"alpha_good": ag, "alpha_bad": ab} for ag in vals for ab in vals]
    return fit_prediction_grid(g, "good_bad_news", grid, pred_good_bad_news, k=2)


def pred_confirmatory_misperception(g: pd.DataFrame, lam: float, q: float) -> np.ndarray:
    """Rabin-Schrag-style signal distortion, adapted to AI-reliability beliefs.

    q is the probability a belief-incongruent binary outcome is perceived as
    congruent. The model uses expected perceived evidence, which keeps the fit
    deterministic and comparable by SSE/BIC.
    """
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


def fit_confirmatory_misperception(g: pd.DataFrame) -> FitResult:
    grid = [
        {"lam": lam, "q": q}
        for lam in np.linspace(0.0, 3.0, 31)
        for q in np.linspace(0.0, 0.50, 6)
    ]
    return fit_prediction_grid(g, "confirmatory_misperception", grid, pred_confirmatory_misperception, k=2)


def pred_anchoring_to_prior(g: pd.DataFrame, adjustment: float) -> np.ndarray:
    g = g.sort_values("t")
    b0 = float(g["b_prev"].iloc[0])
    bayes = beta_bayes_predictions(g, lam=1.0)
    return b0 + adjustment * (bayes - b0)


def fit_anchoring_to_prior(g: pd.DataFrame) -> FitResult:
    grid = [{"adjustment": a} for a in np.linspace(0.0, 2.0, 51)]
    return fit_prediction_grid(g, "anchoring_to_prior", grid, pred_anchoring_to_prior, k=1)


def pred_sublinear_sample_size(g: pd.DataFrame, gamma: float) -> np.ndarray:
    """Proxy for nonbelief in the law of large numbers.

    The agent behaves as if the effective sample size after t observations is
    t**gamma. gamma < 1 means later samples accumulate too slowly.
    """
    g = g.sort_values("t")
    b0 = float(g["b_prev"].iloc[0])
    n0 = float(g["n0"].iloc[0])
    f = g["f_t"].astype(float).to_numpy()
    t = g["t"].astype(float).to_numpy()
    successes = np.cumsum(f)
    sample_mean = successes / np.maximum(t, 1.0)
    eff_n = np.power(t, gamma)
    return (b0 * n0 + eff_n * sample_mean) / (n0 + eff_n)


def fit_sublinear_sample_size(g: pd.DataFrame) -> FitResult:
    grid = [{"gamma": gm} for gm in np.linspace(0.0, 1.5, 51)]
    return fit_prediction_grid(g, "sublinear_sample_size", grid, pred_sublinear_sample_size, k=1)


def fit_partial_step(g: pd.DataFrame) -> FitResult:
    return fit_ols_bic(
        g["delta_b"].to_numpy(),
        g[["norm_step"]].to_numpy(),
        "partial_step",
        ["sigma"],
    )


def fit_t1_spike(g: pd.DataFrame) -> FitResult:
    spike = (g["t"].to_numpy() == 1).astype(float).reshape(-1, 1)
    return fit_ols_bic(g["delta_b"].to_numpy(), spike, "t1_spike", ["trial1_shift"])


def fit_changepoint(g: pd.DataFrame) -> FitResult:
    best: FitResult | None = None
    for tau in range(1, 10):
        x = g["norm_step"].to_numpy()
        early = (g["t"].to_numpy() <= tau).astype(float)
        X = np.column_stack([x * early, x * (1.0 - early)])
        fit = fit_ols_bic(g["delta_b"].to_numpy(), X, "changepoint", ["sigma_early", "sigma_late"], k_extra=1)
        fit.params["tau"] = tau
        if best is None or fit.bic < best.bic:
            best = fit
    return best if best is not None else FitResult("changepoint", np.inf, np.inf, 0, 0, {})


def fit_leaky(g: pd.DataFrame) -> FitResult:
    best: FitResult | None = None
    for rho in np.linspace(0.0, 0.75, 31):
        z = g["norm_step"].to_numpy() * np.exp(-rho * (g["t"].to_numpy() - 1))
        fit = fit_ols_bic(g["delta_b"].to_numpy(), z.reshape(-1, 1), "leaky", ["sigma"], k_extra=1)
        fit.params["rho"] = float(rho)
        if best is None or fit.bic < best.bic:
            best = fit
    return best if best is not None else FitResult("leaky", np.inf, np.inf, 0, 0, {})


def fit_asymmetric(g: pd.DataFrame) -> FitResult:
    x = g["norm_step"].to_numpy()
    X = np.column_stack([np.maximum(x, 0.0), np.minimum(x, 0.0)])
    return fit_ols_bic(g["delta_b"].to_numpy(), X, "asymmetric", ["sigma_positive", "sigma_negative"])


def fit_recency(g: pd.DataFrame) -> FitResult:
    g = g.sort_values("t")
    f = g["f_t"].astype(float).to_numpy()
    t = g["t"].astype(float).to_numpy()
    b_prev = g["b_prev"].astype(float).to_numpy()
    corrects = 0.0
    accbar = []
    for ti, fi in zip(t, f):
        accbar.append(corrects / max(1.0, ti - 1.0) if ti > 1 else 0.0)
        corrects += fi
    x = np.asarray(accbar) - b_prev
    return fit_ols_bic(g["delta_b"].to_numpy(), x.reshape(-1, 1), "recency", ["sigma"])


def fit_all_models(g: pd.DataFrame) -> list[FitResult]:
    return [
        fit_no_update(g),
        fit_standard_bayes(g),
        fit_divisible_weighted_bayes(g),
        fit_coarse_bayes(g),
        fit_coarse_weighted_bayes(g),
        fit_threshold_bayes(g),
        fit_rescorla_wagner(g),
        fit_good_bad_news(g),
        fit_confirmatory_misperception(g),
        fit_anchoring_to_prior(g),
        fit_sublinear_sample_size(g),
        fit_partial_step(g),
        fit_t1_spike(g),
        fit_changepoint(g),
        fit_leaky(g),
        fit_asymmetric(g),
        fit_recency(g),
    ]


def classify_participant_tasks(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    model_rows = []
    needed = ["t", "b_prev", "b_post", "delta_b", "norm_step", "f_t", "n0"]
    panel = finite_frame(panel, needed)

    for (pid, task), g in panel.groupby(["participant_id", "taskType"], sort=False):
        g = g.sort_values("t")
        if len(g) < 5 or g["norm_step"].abs().sum() <= 0:
            continue
        fits = fit_all_models(g)
        best = min(fits, key=lambda fit: fit.bic)
        div = next(fit for fit in fits if fit.model == "divisible_weighted_bayes")
        second = sorted(fits, key=lambda fit: fit.bic)[1]
        for fit in fits:
            model_rows.append({
                "participant_id": pid,
                "taskType": task,
                "model": fit.model,
                "bic": fit.bic,
                "sse": fit.sse,
                "n": fit.n,
                "k": fit.k,
                "params_json": json.dumps(fit.params, sort_keys=True),
            })
        rows.append({
            "participant_id": pid,
            "taskType": task,
            "taskPosition": int(g["taskPosition"].iloc[0]) if "taskPosition" in g.columns else np.nan,
            "n_trials": int(len(g)),
            "winner": best.model,
            "winner_bic": best.bic,
            "second_best": second.model,
            "second_best_bic": second.bic,
            "divisible_bic": div.bic,
            "delta_bic_divisible_minus_best": div.bic - best.bic,
            "divisible_lambda": div.params.get("lambda", np.nan),
            "divisible_wins": int(best.model == "divisible_weighted_bayes"),
        })
    return pd.DataFrame(rows), pd.DataFrame(model_rows)


def summarize(classification: pd.DataFrame, model_fits: pd.DataFrame) -> pd.DataFrame:
    if classification.empty:
        return pd.DataFrame()
    winner_counts = (
        classification["winner"]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="n_wins")
    )
    winner_counts["share_wins"] = winner_counts["n_wins"] / winner_counts["n_wins"].sum()

    div = classification["delta_bic_divisible_minus_best"]
    summary_rows = [
        {"metric": "participant_tasks", "value": len(classification)},
        {"metric": "divisible_wins", "value": int(classification["divisible_wins"].sum())},
        {"metric": "divisible_win_share", "value": float(classification["divisible_wins"].mean())},
        {"metric": "median_delta_bic_divisible_minus_best", "value": float(div.median())},
        {"metric": "mean_divisible_lambda", "value": float(classification["divisible_lambda"].mean())},
        {"metric": "median_divisible_lambda", "value": float(classification["divisible_lambda"].median())},
    ]
    summary = pd.DataFrame(summary_rows)

    by_model = model_fits.groupby("model", as_index=False).agg(
        mean_bic=("bic", "mean"),
        median_bic=("bic", "median"),
        mean_sse=("sse", "mean"),
    )
    by_model["metric"] = "model_average"
    by_model["value"] = np.nan
    return summary, winner_counts, by_model


def plot_outputs(classification: pd.DataFrame, outdir: Path) -> None:
    if classification.empty or plt is None:
        return
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.2, 4.6))
    order = classification["winner"].value_counts().index.tolist()
    counts = classification["winner"].value_counts().loc[order]
    plt.bar(order, counts.values)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Participant-task count")
    plt.title("Best-fitting belief-update model by BIC")
    plt.tight_layout()
    plt.savefig(figdir / "winner_counts.png", dpi=300)
    plt.savefig(figdir / "winner_counts.pdf")
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    plt.hist(classification["delta_bic_divisible_minus_best"], bins=30)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("BIC(divisible) - BIC(best)")
    plt.ylabel("Participant-task count")
    plt.title("How far the divisible model is from the best model")
    plt.tight_layout()
    plt.savefig(figdir / "divisible_delta_bic.png", dpi=300)
    plt.savefig(figdir / "divisible_delta_bic.pdf")
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    plt.hist(classification["divisible_lambda"].dropna(), bins=30)
    plt.axvline(1, color="black", linewidth=1)
    plt.xlabel("Divisible evidence weight lambda")
    plt.ylabel("Participant-task count")
    plt.title("Estimated evidence weight under divisible updating")
    plt.tight_layout()
    plt.savefig(figdir / "divisible_lambda.png", dpi=300)
    plt.savefig(figdir / "divisible_lambda.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL, help="H2 panel CSV with b_prev, b_post, f_t, n0.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.panel)
    classification, model_fits = classify_participant_tasks(panel)

    classification.to_csv(args.outdir / "divisible_classification.csv", index=False)
    model_fits.to_csv(args.outdir / "divisible_model_fits.csv", index=False)

    summary, winner_counts, by_model = summarize(classification, model_fits)
    summary.to_csv(args.outdir / "summary.csv", index=False)
    winner_counts.to_csv(args.outdir / "winner_counts.csv", index=False)
    by_model.to_csv(args.outdir / "model_average_fit.csv", index=False)
    plot_outputs(classification, args.outdir)
    if plt is None:
        print("[Divisible updating] matplotlib not available; skipped figures.")

    print("[Divisible updating] participant-task fits:", len(classification))
    print(winner_counts.to_string(index=False))
    print(summary.to_string(index=False))
    print("[Divisible updating] outputs written to", args.outdir)


if __name__ == "__main__":
    main()
