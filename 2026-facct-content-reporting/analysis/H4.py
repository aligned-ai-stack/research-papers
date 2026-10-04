from __future__ import annotations

import warnings
from pathlib import Path
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf


DATA_DIR = Path(os.environ["FACCT_DATA_DIR"])
TAG = "facct_1401"

OUT_DIR = Path(os.environ["FACCT_OUTPUT_DIR"]) / "H4"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
TABLE_DIR = OUT_DIR / "tables"
for d in (OUT_DIR, FIG_DIR, MODEL_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

ANALYSIS_PATH = DATA_DIR / "final_analysis_0301_cleaned.csv"
RATINGS_PATH = DATA_DIR / "combined_coder_ratings.csv"

COND_ORDER = ["explainable", "manual", "evaluative"]

DVS = [
    "element_coverage_mean",
    "proportionality_reasoning_mean",
    "reasoning_depth_mean",
    "perceived_overall_quality_mean",
]


def write_txt(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.labelweight": "bold",
            "axes.edgecolor": "#1A1A1A",
            "axes.linewidth": 1.2,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.3,
            "figure.autolayout": True,
        }
    )


def ensure_valid_mean_1to5(series: pd.Series) -> pd.Series:
    """Keep numeric values in [1,5]; drop others."""
    s = pd.to_numeric(series, errors="coerce")
    return s.where((s >= 1) & (s <= 5))


def _t_test_linear(res, weights: Dict[str, float]) -> Dict[str, float]:
    """
    Linear hypothesis t-test: sum_j weights[j]*beta_j = 0.
    Returns coef, se, t, p, ci_low, ci_high.
    """
    names = list(res.params.index)
    w = np.zeros(len(names))
    for k, v in weights.items():
        if k in names:
            w[names.index(k)] = float(v)

    tt = res.t_test(w)
    b = np.asarray(tt.effect).item()
    se = np.asarray(tt.sd).item()
    t = np.asarray(tt.tvalue).item()
    p = np.asarray(tt.pvalue).item()
    return {
        "coef": b,
        "se": se,
        "t": t,
        "p": p,
        "ci_low": b - 1.96 * se,
        "ci_high": b + 1.96 * se,
    }


set_plot_style()

analysis = pd.read_csv(ANALYSIS_PATH)
ratings = pd.read_csv(RATINGS_PATH)

ratings["id"] = ratings["id"].astype(str)
analysis["id"] = analysis["id"].astype(str)

merged = analysis.merge(ratings, on="id", how="inner")

needed = {"participantId", "condition"} | set(DVS)
missing = needed - set(merged.columns)
if missing:
    raise RuntimeError(f"Missing required columns after merge: {sorted(missing)}")

df0 = merged[["participantId", "condition"] + DVS].copy()
df0["participantId"] = df0["participantId"].astype(str)
df0["condition"] = df0["condition"].astype(str)

df0 = df0[df0["condition"].isin(COND_ORDER)].copy()
df0["condition"] = pd.Categorical(df0["condition"], categories=COND_ORDER, ordered=True)

qc_lines: List[str] = []
qc_lines.append("H4 Panel (reasoning quality measures) — MERGED rated rows only")
qc_lines.append(f"Total rows: {len(df0)} | Participants: {df0['participantId'].nunique()}")
qc_lines.append("\nCondition breakdown:")
qc_lines.append(df0["condition"].value_counts(dropna=False).to_string())

qc_lines.append("\nMissing/invalid per DV (before filtering to [1,5] numeric):")
for dv in DVS:
    dv_num = pd.to_numeric(df0[dv], errors="coerce")
    missing_n = dv_num.isna().sum()
    invalid_n = (((dv_num < 1) | (dv_num > 5)) & dv_num.notna()).sum()
    qc_lines.append(f"  {dv}: {missing_n} NaN, {invalid_n} out-of-range")

qc_lines.append("\nRows-per-participant summary:")
qc_lines.append(df0.groupby("participantId").size().describe().to_string())
write_txt(MODEL_DIR / "H4_panel_checks.txt", "\n".join(qc_lines))
print("\n".join(qc_lines))


contrast_rows: List[Dict[str, Any]] = []
tidy_rows: List[pd.DataFrame] = []

for dv in DVS:
    d = df0.copy()
    d[dv] = ensure_valid_mean_1to5(d[dv])
    d = d[d[dv].notna()].copy()
    if d.empty:
        write_txt(MODEL_DIR / f"H4_{dv}_skipped.txt", f"No valid [1,5] data for {dv}")
        continue

    rows_per_pid = d.groupby("participantId").size()
    use_cluster = int(rows_per_pid.max()) > 1

    formula = f"{dv} ~ C(condition, Treatment(reference='explainable'))"
    model = smf.ols(formula, data=d)

    if use_cluster:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": d["participantId"]})
        se_type = "cluster(participantId)"
    else:
        res = model.fit(cov_type="HC1")
        se_type = "HC1"

    write_txt(
        MODEL_DIR / f"H4_{dv}_OLS.txt",
        res.summary().as_text() + f"\n\nRobust SE: {se_type}\n",
    )

    ci = res.conf_int()
    tidy = pd.DataFrame(
        {
            "dv": dv,
            "term": res.params.index,
            "coef": res.params.values,
            "se": res.bse.values,
            "t": res.tvalues.values,
            "p": res.pvalues.values,
            "ci_low": ci[0].values,
            "ci_high": ci[1].values,
            "robust_se": se_type,
            "n": len(d),
            "participants": d["participantId"].nunique(),
        }
    )
    tidy_rows.append(tidy)

    term_man = "C(condition, Treatment(reference='explainable'))[T.manual]"
    term_eval = "C(condition, Treatment(reference='explainable'))[T.evaluative]"

    for name, weights in [
        ("Evaluative vs Explainable", {term_eval: 1.0}),
        ("Manual vs Explainable", {term_man: 1.0}),
        ("Evaluative vs Manual", {term_eval: 1.0, term_man: -1.0}),
    ]:
        tt = _t_test_linear(res, weights)
        contrast_rows.append(
            {
                "dv": dv,
                "contrast": name,
                **tt,
                "robust_se": se_type,
                "n": int(len(d)),
                "participants": int(d["participantId"].nunique()),
            }
        )

    means = d.groupby("condition")[dv].mean().reindex(COND_ORDER)
    ns = d.groupby("condition")[dv].size().reindex(COND_ORDER)
    ses = d.groupby("condition")[dv].sem().reindex(COND_ORDER)  # iid SEM
    ci_lo = means - 1.96 * ses
    ci_hi = means + 1.96 * ses

    fig, ax = plt.subplots(figsize=(6.8, 4))
    x = np.arange(len(COND_ORDER))
    ax.bar(
        x,
        means.values,
        yerr=(means.values - ci_lo.values, ci_hi.values - means.values),
        capsize=6,
        width=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in COND_ORDER], rotation=0)
    ax.set_ylim(1, 5)
    ax.set_ylabel("Mean rating (1–5)")
    ax.set_title(f"H4: {dv.replace('_mean','').replace('_',' ').title()}")


    plt.tight_layout()
    plt.savefig(FIG_DIR / f"h4_{dv}_means_ci.png", bbox_inches="tight")
    plt.close(fig)

h4_contrasts = pd.DataFrame(contrast_rows).sort_values(["dv", "contrast"]).reset_index(drop=True)
h4_contrasts.to_csv(TABLE_DIR / "H4_OLS_contrasts.csv", index=False)

if tidy_rows:
    pd.concat(tidy_rows, ignore_index=True).to_csv(TABLE_DIR / "H4_OLS_tidy.csv", index=False)

write_txt(MODEL_DIR / "H4_summary.txt", h4_contrasts.to_string(index=False))

print("\n✓ H4 complete (OLS on mean ratings)")
print(f"Models:  {MODEL_DIR.resolve()}")
print(f"Tables:  {TABLE_DIR.resolve()}")
print(f"Figures: {FIG_DIR.resolve()}")
