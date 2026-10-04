from __future__ import annotations

import warnings
from pathlib import Path
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf


DATA_DIR = Path(os.environ["FACCT_DATA_DIR"])
ANALYSIS_PATH = DATA_DIR / "final_analysis_0301_cleaned.csv"

TAG = "facct_0301"
OUT_DIR = Path(os.environ["FACCT_OUTPUT_DIR"]) / "H6"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
TABLE_DIR = OUT_DIR / "tables"
for d in (OUT_DIR, FIG_DIR, MODEL_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

AI_ARMS = ["explainable", "evaluative"]

REMOVAL_ACTIONS = {
    "remove",
    "remove_escalate",
    "suspend_account",
    "deplatform_account",
}

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

def coerce_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().any():
        return x.fillna(0).astype(int).astype(bool)
    m = (
        s.astype(str).str.strip().str.lower()
        .map({"true": 1, "false": 0, "t": 1, "f": 0, "yes": 1, "no": 0})
    )
    return m.fillna(0).astype(int).astype(bool)

def normalize_error_type(x: pd.Series) -> pd.Series:
    m = {
        "Near-Miss": "near_miss", "near_miss": "near_miss", "near-miss": "near_miss", "near miss": "near_miss",
        "Overbreadth": "overbreadth", "overbreadth": "overbreadth", "over-breadth": "overbreadth",
        "Out-of-Scope": "out_of_scope", "out_of_scope": "out_of_scope", "out-of-scope": "out_of_scope", "out of scope": "out_of_scope",
        "No-Error": "no_error", "no_error": "no_error", "no error": "no_error",
    }
    xs = x.astype(str)
    out = xs.map(m)
    out = out.where(out.notna(), xs)
    return out

def tidy_glm(res, model_name: str) -> pd.DataFrame:
    params = res.params
    bse = res.bse
    z = params / bse
    p = np.asarray(res.pvalues, dtype=float)
    ci = res.conf_int()
    ci = pd.DataFrame(ci, index=params.index, columns=["ci_low", "ci_high"])
    out = pd.DataFrame({
        "model": model_name,
        "term": params.index,
        "coef": params.values,
        "se": bse.values,
        "z": z.values,
        "p": p,
        "ci_low": ci["ci_low"].values,
        "ci_high": ci["ci_high"].values,
    })
    out["OR"] = np.exp(out["coef"])
    out["OR_low"] = np.exp(out["ci_low"])
    out["OR_high"] = np.exp(out["ci_high"])
    return out

def glm_cluster_logit(df: pd.DataFrame, formula: str, model_name: str, out_txt: Path, out_csv: Path) -> None:
    m = smf.glm(formula, data=df, family=sm.families.Binomial())
    res = m.fit(cov_type="cluster", cov_kwds={"groups": df["participantId"]})
    write_txt(out_txt, res.summary().as_text())
    tidy_glm(res, model_name).to_csv(out_csv, index=False)

df = pd.read_csv(ANALYSIS_PATH)

needed = {"participantId", "condition", "isErrorTrial", "errorType", "chosenAction", "outOfScopeChoice"}
missing = needed - set(df.columns)
if missing:
    raise RuntimeError(f"Missing required columns in analysis: {sorted(missing)}")

df["participantId"] = df["participantId"].astype(str)
df["condition"] = df["condition"].astype(str)
df["isErrorTrial"] = coerce_bool(df["isErrorTrial"])
df["errorType"] = normalize_error_type(df["errorType"])

df = df[df["condition"].isin(AI_ARMS)].copy()
df["cond2"] = pd.Categorical(df["condition"], categories=["explainable", "evaluative"], ordered=True)

df["outOfScopeChoice"] = coerce_bool(df["outOfScopeChoice"])

h6_ob = df[(df["isErrorTrial"]) & (df["errorType"] == "overbreadth")].copy()

h6_ob["over_removal"] = h6_ob["chosenAction"].astype(str).isin(REMOVAL_ACTIONS).astype(int)

lines = []
lines.append("H6(i) Overbreadth panel")
lines.append(f"rows={len(h6_ob)} | participants={h6_ob['participantId'].nunique()}")
lines.append("\nCounts by condition:")
lines.append(h6_ob["condition"].value_counts(dropna=False).to_string())
lines.append("\nOutcome rate by condition (over_removal):")
lines.append(h6_ob.groupby("condition")["over_removal"].mean().to_string())
write_txt(MODEL_DIR / "H6_overbreadth_panel_checks.txt", "\n".join(lines))

if len(h6_ob) > 0:
    glm_cluster_logit(
        h6_ob,
        "over_removal ~ C(cond2, Treatment(reference='explainable'))",
        "H6_overbreadth_over_removal",
        MODEL_DIR / "H6_overbreadth_GLM.txt",
        TABLE_DIR / "H6_overbreadth_GLM_tidy.csv",
    )

set_plot_style()
if len(h6_ob) > 0:
    fig, ax = plt.subplots(figsize=(5.8, 4))
    rates = h6_ob.groupby("condition")["over_removal"].mean().reindex(["explainable", "evaluative"])
    ax.bar(rates.index, rates.values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Over-removal rate")
    ax.set_xlabel("Condition")
    ax.set_title("H6(i): Over-removal under Overbreadth errors")
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "H6_over_removal_overbreadth_bar.png", bbox_inches="tight")
    plt.close(fig)

h6_oos = df[(df["isErrorTrial"]) & (df["errorType"] == "out_of_scope")].copy()

h6_oos["misrouting"] = (~h6_oos["outOfScopeChoice"]).astype(int)
if h6_oos.groupby("condition", observed=True)["misrouting"].nunique().lt(2).any():
    write_txt(
        MODEL_DIR / "H6_oos_note.txt",
        "At least one condition has no outcome variation (separation). "
        "Model not estimable; reporting descriptives only."
    )
    run_oos_model = False
else:
    run_oos_model = True
lines = []
lines.append("H6(ii) Out-of-Scope panel")
lines.append(f"rows={len(h6_oos)} | participants={h6_oos['participantId'].nunique()}")
lines.append("\nCounts by condition:")
lines.append(h6_oos["condition"].value_counts(dropna=False).to_string())
lines.append("\nOutcome rate by condition (misrouting):")
lines.append(h6_oos.groupby("condition")["misrouting"].mean().to_string())
write_txt(MODEL_DIR / "H6_oos_panel_checks.txt", "\n".join(lines))

if len(h6_oos) > 0 and run_oos_model:
    glm_cluster_logit(
        h6_oos,
        "misrouting ~ C(cond2, Treatment(reference='explainable'))",
        "H6_oos_misrouting",
        MODEL_DIR / "H6_oos_GLM.txt",
        TABLE_DIR / "H6_oos_GLM_tidy.csv",
    )

set_plot_style()
if len(h6_oos) > 0:
    fig, ax = plt.subplots(figsize=(5.8, 4))
    rates = h6_oos.groupby("condition")["misrouting"].mean().reindex(["explainable", "evaluative"])
    ax.bar(rates.index, rates.values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Misrouting rate")
    ax.set_title("H6(ii): Misrouting under Out-of-Scope errors")
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "H6_misrouting_oos_bar.png", bbox_inches="tight")
    plt.close(fig)

print("✓ H6 complete")
print("Models ->", MODEL_DIR.resolve())
print("Tables ->", TABLE_DIR.resolve())
print("Figures ->", FIG_DIR.resolve())
