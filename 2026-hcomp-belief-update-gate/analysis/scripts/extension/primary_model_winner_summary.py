#!/usr/bin/env python
# coding: utf-8

"""Derive primary-generative-only winner summaries from existing model fits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FITS = ROOT / "data" / "followup_outputs" / "robust_belief_model_comparison" / "canonical_model_fits.csv"
DEFAULT_OUTDIR = ROOT / "data" / "followup_outputs" / "robust_belief_model_comparison_primary"

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


def weights_from_scores(scores: pd.Series, lower_is_better: bool = True) -> pd.Series:
    vals = scores.astype(float)
    if vals.notna().sum() == 0:
        return pd.Series(np.nan, index=scores.index)
    best = vals.min() if lower_is_better else vals.max()
    delta = vals - best if lower_is_better else best - vals
    raw = np.exp(-0.5 * delta)
    raw = raw.where(vals.notna(), 0.0)
    total = raw.sum()
    return raw / total if total > 0 else pd.Series(np.nan, index=scores.index)


def winner_rows(fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, task), g in fits.groupby(["participant_id", "taskType"], sort=False):
        g = g.copy()
        bic_weights = weights_from_scores(g["bic"])
        aicc_weights = weights_from_scores(g["aicc"])
        g["primary_bic_weight"] = bic_weights
        g["primary_aicc_weight"] = aicc_weights
        bic = g.loc[g["bic"].idxmin()]
        aicc = g.loc[g["aicc"].idxmin()]
        future_candidates = g.dropna(subset=["future_block_rmse"])
        if future_candidates.empty:
            future = None
        else:
            future = future_candidates.loc[future_candidates["future_block_rmse"].idxmin()]
        rows.append(
            {
                "participant_id": pid,
                "taskType": task,
                "primary_bic_winner": bic["model"],
                "primary_bic": float(bic["bic"]),
                "primary_bic_weight": float(bic["primary_bic_weight"]),
                "primary_aicc_winner": aicc["model"],
                "primary_aicc": float(aicc["aicc"]),
                "primary_aicc_weight": float(aicc["primary_aicc_weight"]),
                "primary_future_block_winner": "" if future is None else future["model"],
                "primary_future_block_rmse": np.nan if future is None else float(future["future_block_rmse"]),
            }
        )
    return pd.DataFrame(rows)


def counts(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    out = df[col].replace("", np.nan).dropna().value_counts().rename_axis("model").reset_index(name="wins")
    out["share"] = out["wins"] / out["wins"].sum()
    out["winner_rule"] = label
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(args.fits)
    primary = fits[fits["model"].isin(PRIMARY_GENERATIVE_MODELS)].copy()
    winners = winner_rows(primary)
    winners.to_csv(args.outdir / "primary_classification_from_all_fits.csv", index=False)
    primary.to_csv(args.outdir / "primary_model_fits_from_all_fits.csv", index=False)
    counts(winners, "primary_bic_winner", "bic").to_csv(args.outdir / "primary_bic_winner_counts.csv", index=False)
    counts(winners, "primary_aicc_winner", "aicc").to_csv(args.outdir / "primary_aicc_winner_counts.csv", index=False)
    counts(winners, "primary_future_block_winner", "future_block_rmse").to_csv(
        args.outdir / "primary_future_block_winner_counts.csv", index=False
    )
    manifest = {
        "fits": str(args.fits),
        "outdir": str(args.outdir),
        "method": "Filtered existing all-candidate fit table to primary trajectory-generating models and recomputed winner counts.",
        "primary_models": sorted(PRIMARY_GENERATIVE_MODELS),
        "participant_tasks": int(len(winners)),
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[Primary model summary] outputs written to", args.outdir)
    print("\nBIC winners:")
    print(counts(winners, "primary_bic_winner", "bic").to_string(index=False))
    print("\nAICc winners:")
    print(counts(winners, "primary_aicc_winner", "aicc").to_string(index=False))
    print("\nFuture-block winners:")
    print(counts(winners, "primary_future_block_winner", "future_block_rmse").to_string(index=False))


if __name__ == "__main__":
    main()
