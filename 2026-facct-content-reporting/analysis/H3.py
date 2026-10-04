from __future__ import annotations

import math
import warnings
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


DATA_DIR = Path(os.environ["FACCT_DATA_DIR"])
TAG = "facct_1401"
OUT_DIR = Path(os.environ["FACCT_OUTPUT_DIR"]) / "H3"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
TABLE_DIR = OUT_DIR / "tables"
for d in (OUT_DIR, FIG_DIR, MODEL_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

ANALYSIS_PATH = DATA_DIR / "final_analysis_0301.csv"
SESSIONS_PATH = DATA_DIR / "final_sessions_0301.csv"  # optional

def write_txt(path: Path, content: str):
    path.write_text(content, encoding="utf-8")

def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.labelweight": "bold",
        "axes.edgecolor": "#1A1A1A",
        "axes.linewidth": 1.2,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.3,
        "figure.autolayout": True,
    })

def coerce_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int).astype(bool)
    m = (
        s.astype(str).str.strip().str.lower()
        .map({"true": 1, "false": 0, "t": 1, "f": 0, "yes": 1, "no": 0})
    )
    return m.fillna(0).astype(int).astype(bool)

def choose_case_column_safe(df: pd.DataFrame) -> str | None:
    """
    Safe case FE columns (design control).
    Avoid postId as FE by default (often too granular / near-unique).
    """
    for c in ["legalCase", "caseId", "case", "correctProvision"]:
        if c in df.columns:
            return c
    return None

def robust_fit_glm(model, df: pd.DataFrame, pid_col="participantId"):
    """
    GLM robust covariance at fit-time.
    """
    rows_per = df.groupby(pid_col).size()
    max_r = int(rows_per.max()) if len(rows_per) else 1
    if max_r <= 1:
        return model.fit(cov_type="HC1"), "HC1"
    return model.fit(cov_type="cluster", cov_kwds={"groups": df[pid_col]}), "cluster(participantId)"

def robust_fit_ols(res, df: pd.DataFrame, pid_col="participantId"):
    """
    OLS robust covariance via get_robustcov_results.
    """
    rows_per = df.groupby(pid_col).size()
    max_r = int(rows_per.max()) if len(rows_per) else 1
    if max_r <= 1:
        return res.get_robustcov_results(cov_type="HC1"), "HC1"
    return res.get_robustcov_results(cov_type="cluster", groups=df[pid_col]), "cluster(participantId)"

def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha/2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat) + z**2/(4*n))/n)) / denom
    return (center - half, center + half)

analysis = pd.read_csv(ANALYSIS_PATH)
sessions = pd.read_csv(SESSIONS_PATH) if SESSIONS_PATH.exists() else pd.DataFrame()

analysis["participantId"] = analysis["participantId"].astype(str)
analysis["condition"] = analysis["condition"].astype(str)

req = ["participantId", "condition", "isErrorTrial", "accuracyProvision", "decisionTimeMs"]
missing = [c for c in req if c not in analysis.columns]
if missing:
    raise RuntimeError(f"Missing required columns in analysis: {missing}")

analysis["isErrorTrial"] = coerce_bool(analysis["isErrorTrial"])

USE_COMPLETED_ONLY = False
if USE_COMPLETED_ONLY and (not sessions.empty) and ("status" in sessions.columns) and ("participantId" in sessions.columns):
    sessions["participantId"] = sessions["participantId"].astype(str)
    completed_ids = set(sessions.loc[sessions["status"].astype(str) == "completed", "participantId"])
    if len(completed_ids) > 0:
        analysis = analysis[analysis["participantId"].isin(completed_ids)].copy()

ai = analysis[analysis["condition"].isin(["evaluative", "explainable"])].copy()
h3 = ai[~ai["isErrorTrial"]].copy()  # no-error trials only

acc = pd.to_numeric(h3["accuracyProvision"], errors="coerce")
if acc.isna().any():
    acc_bool = h3["accuracyProvision"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    acc = acc.fillna(acc_bool)
h3["acc_bin"] = pd.to_numeric(acc, errors="coerce")
h3 = h3[h3["acc_bin"].isin([0, 1])].copy()
h3["acc_bin"] = h3["acc_bin"].astype(int)

h3["decisionTimeMs"] = pd.to_numeric(h3["decisionTimeMs"], errors="coerce")
h3 = h3[h3["decisionTimeMs"].notna() & (h3["decisionTimeMs"] > 0)].copy()

h3["cond2"] = pd.Categorical(h3["condition"], categories=["explainable", "evaluative"])

case_col = choose_case_column_safe(h3)

rows_per_pid = h3.groupby("participantId").size()
checks = []
checks.append(f"H3 panel (no-error, AI arms): rows={len(h3)}, participants={h3['participantId'].nunique()}")
checks.append("\nRows-per-participant summary:\n" + rows_per_pid.describe().to_string())
checks.append("\nCounts by condition:\n" + h3["condition"].value_counts().to_string())
checks.append(f"\nCase FE available (robustness): {case_col if case_col else '(none)'}")
write_txt(MODEL_DIR / "H3_panel_checks.txt", "\n".join(checks))
print("\n".join(checks))

acc_by = h3.groupby("condition")["acc_bin"].agg(["mean", "sum", "count"])
acc_by["pct"] = (acc_by["mean"] * 100).round(1)
write_txt(MODEL_DIR / "H3a_descriptives.txt", "H3a — Accuracy by condition:\n" + acc_by.to_string())

glm0 = smf.glm("acc_bin ~ C(cond2)", data=h3, family=sm.families.Binomial())
res0, se_type0 = robust_fit_glm(glm0, h3)

txt0 = res0.summary().as_text() + f"\n\nRobust SE: {se_type0}\n"
term = "C(cond2)[T.evaluative]"
if term in res0.params.index:
    b = float(res0.params[term])
    se = float(res0.bse[term])
    OR = float(np.exp(b))
    lo = float(np.exp(b - 1.96 * se))
    hi = float(np.exp(b + 1.96 * se))
    p = float(res0.pvalues[term])
    txt0 += f"\nEvaluative vs Explainable: OR={OR:.3f} [{lo:.3f}, {hi:.3f}], p={p:.4f}\n"
write_txt(MODEL_DIR / "H3a_primary_no_caseFE_GLM.txt", txt0)

if case_col is not None:
    glm1 = smf.glm(f"acc_bin ~ C(cond2) + C({case_col})", data=h3, family=sm.families.Binomial())
    res1, se_type1 = robust_fit_glm(glm1, h3)
    txt1 = res1.summary().as_text() + f"\n\nRobust SE: {se_type1}\n"
    write_txt(MODEL_DIR / "H3a_robust_caseFE_GLM.txt", txt1)
else:
    write_txt(MODEL_DIR / "H3a_robust_caseFE_GLM.txt", "Skipped: no safe case column available.")

delta = 0.05
p_eval = float(h3.loc[h3["condition"] == "evaluative", "acc_bin"].mean())
p_expl = float(h3.loc[h3["condition"] == "explainable", "acc_bin"].mean())
n_eval = int((h3["condition"] == "evaluative").sum())
n_expl = int((h3["condition"] == "explainable").sum())
diff = p_eval - p_expl

p_pool = (p_eval * n_eval + p_expl * n_expl) / max(1, (n_eval + n_expl))
se = math.sqrt(max(1e-12, p_pool * (1 - p_pool) * (1/n_eval + 1/n_expl)))
z_lower = (diff + delta) / se
z_upper = (diff - delta) / se
p_lower = 1 - stats.norm.cdf(z_lower)
p_upper = stats.norm.cdf(z_upper)
equivalent = (p_lower < 0.05) and (p_upper < 0.05)

tost_txt = (
    "H3a — TOST equivalence test (difference in proportions)\n"
    f"Margin: ±{delta:.2f}\n"
    f"Eval p={p_eval:.3f} (n={n_eval}); Expl p={p_expl:.3f} (n={n_expl}); diff={diff:.3f}\n"
    f"Lower test (diff > -{delta:.2f}): z={z_lower:.3f}, p={p_lower:.4f}\n"
    f"Upper test (diff < +{delta:.2f}): z={z_upper:.3f}, p={p_upper:.4f}\n"
    f"Equivalent at α=.05? {equivalent}\n"
)
write_txt(MODEL_DIR / "H3a_TOST_equivalence.txt", tost_txt)

set_plot_style()
fig, ax = plt.subplots(figsize=(5.5, 4))
order = ["explainable", "evaluative"]
means = h3.groupby("condition")["acc_bin"].mean().reindex(order)
counts = h3.groupby("condition")["acc_bin"].agg(["sum", "count"]).reindex(order)
cis = [wilson_ci(int(counts.loc[c, "sum"]), int(counts.loc[c, "count"])) for c in order]
bars = ax.bar(order, means.values, alpha=0.9)
ax.set_ylim(0, 1)
ax.set_ylabel("Provision accuracy")
ax.set_xlabel("Condition")
ax.set_title("H3a: Accuracy on no-error trials")
for i, b in enumerate(bars):
    lo, hi = cis[i]
    ax.errorbar(
        b.get_x() + b.get_width() / 2,
        means.iloc[i],
        yerr=[[means.iloc[i] - lo], [hi - means.iloc[i]]],
        fmt="none",
        capsize=4,
    )
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{b.get_height():.2f}",
            ha="center", va="bottom")
plt.tight_layout()
plt.savefig(FIG_DIR / "H3a_accuracy_bar_wilson.png", bbox_inches="tight")
plt.close(fig)

h3b = h3.copy()

h3b["log_dt"] = np.log(h3b["decisionTimeMs"])

ols0 = smf.ols("log_dt ~ C(cond2)", data=h3b)
res_ols0 = ols0.fit()
res_ols0r, se_ols0 = robust_fit_ols(res_ols0, h3b)
write_txt(MODEL_DIR / "H3b_primary_no_caseFE_OLS_logdt.txt", res_ols0r.summary().as_text() + f"\n\nRobust SE: {se_ols0}\n")

if case_col is not None:
    ols1 = smf.ols(f"log_dt ~ C(cond2) + C({case_col})", data=h3b)
    res_ols1 = ols1.fit()
    res_ols1r, se_ols1 = robust_fit_ols(res_ols1, h3b)
    write_txt(MODEL_DIR / "H3b_robust_caseFE_OLS_logdt.txt", res_ols1r.summary().as_text() + f"\n\nRobust SE: {se_ols1}\n")
else:
    write_txt(MODEL_DIR / "H3b_robust_caseFE_OLS_logdt.txt", "Skipped: no safe case column available.")

lo_w, hi_w = h3b["decisionTimeMs"].quantile([0.01, 0.99])
h3b["dt_wins"] = h3b["decisionTimeMs"].clip(lo_w, hi_w)
h3b["log_dt_wins"] = np.log(h3b["dt_wins"])

ols2 = smf.ols("log_dt_wins ~ C(cond2)", data=h3b)
res_ols2 = ols2.fit()
res_ols2r, se_ols2 = robust_fit_ols(res_ols2, h3b)
write_txt(MODEL_DIR / "H3b_robust_winsor_OLS_logdt.txt", res_ols2r.summary().as_text() + f"\n\nRobust SE: {se_ols2}\n")

dt_eval = h3b.loc[h3b["condition"] == "evaluative", "decisionTimeMs"]
dt_expl = h3b.loc[h3b["condition"] == "explainable", "decisionTimeMs"]
if len(dt_eval) and len(dt_expl):
    U, p = stats.mannwhitneyu(dt_expl, dt_eval, alternative="less")
    write_txt(
        MODEL_DIR / "H3b_MWU_one_tailed.txt",
        "H3b — Mann–Whitney U (one-tailed: Explainable faster)\n"
        f"n_expl={len(dt_expl)}, n_eval={len(dt_eval)}\n"
        f"median_expl={dt_expl.median():.1f} ms, median_eval={dt_eval.median():.1f} ms\n"
        f"U={U:.1f}, p={p:.4f}\n"
    )

set_plot_style()
fig, ax = plt.subplots(figsize=(6, 4))
for cond in ["explainable", "evaluative"]:
    x = np.sort(h3b.loc[h3b["condition"] == cond, "decisionTimeMs"].values)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, label=cond)
ax.set_xscale("log")
ax.set_xlabel("Decision time (ms, log scale)")
ax.set_ylabel("ECDF")
ax.set_title("H3b: Decision time distribution (ECDF)")
ax.legend(title="Condition")
plt.tight_layout()
plt.savefig(FIG_DIR / "H3b_dt_ecdf_logx.png", bbox_inches="tight")
plt.close(fig)

print("\n=== H3 completed ===")
print(f"Saved models to:  {MODEL_DIR.resolve()}")
print(f"Saved tables to:  {TABLE_DIR.resolve()}")
print(f"Saved figures to: {FIG_DIR.resolve()}")

# Paper-matching specification from the earlier H3 notebook: 1/99% winsorization,
# correctProvision fixed effects, HC1 covariance and t-reference inference.
lo, hi = h3b.decisionTimeMs.quantile([.01, .99])
reported = h3b.assign(log_dt_reported=np.log(h3b.decisionTimeMs.clip(lo, hi)))
reported_fit = smf.ols("log_dt_reported ~ C(cond2) + C(correctProvision)", data=reported).fit()
reported_fit = reported_fit.get_robustcov_results(cov_type="HC1")
write_txt(MODEL_DIR / "H3b_reported_winsor_caseFE.txt", reported_fit.summary().as_text())
timing_rows=[]
for label, fit in [("reported_winsor_caseFE", reported_fit),
                   ("companion_raw_no_caseFE", res_ols0r), ("companion_raw_caseFE", res_ols1r)]:
    names=fit.model.exog_names; j=names.index("C(cond2)[T.evaluative]")
    ci=np.asarray(fit.conf_int())[j]
    timing_rows.append(dict(specification=label,coef=float(fit.params[j]),se=float(fit.bse[j]),
                            p=float(fit.pvalues[j]),ci_low=float(ci[0]),ci_high=float(ci[1])))
pd.DataFrame(timing_rows).to_csv(TABLE_DIR / "H3b_timing_specifications.csv",index=False)
