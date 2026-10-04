import json
import ast
from pathlib import Path
import os
from textwrap import dedent
import warnings


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


DATA_DIR = Path(os.environ["FACCT_DATA_DIR"])
OUT_DIR = Path(os.environ["FACCT_OUTPUT_DIR"]) / "H1"
FIG_DIR  = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
TABLE_DIR = OUT_DIR / "tables"

for d in [OUT_DIR, FIG_DIR, MODEL_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ANALYSIS_PATH = DATA_DIR / "final_analysis_0301.csv"
SURVEY_PATH   = DATA_DIR / "final_survey_0301.csv"   # optional; can be missing

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
    })

def coerce_01(series: pd.Series) -> pd.Series:
    """
    Coerce 0/1, '0'/'1', True/False, 'true'/'false' to 0/1 int.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    s = series.copy()

    if s.dtype == object:
        s2 = s.astype(str).str.strip().str.lower()
        s = s2.replace({"true": 1, "false": 0, "yes": 1, "no": 0})

    s = pd.to_numeric(s, errors="coerce")
    return s

def normalize_error_type(s: pd.Series) -> pd.Series:
    m = {
        "near-miss":"near_miss", "near miss":"near_miss", "near_miss":"near_miss", "near miss ":"near_miss",
        "over-breadth":"overbreadth", "over breadth":"overbreadth", "overbreadth":"overbreadth",
        "out-of-scope":"out_of_scope", "out of scope":"out_of_scope", "out_of_scope":"out_of_scope",
    }
    return s.astype(str).str.strip().str.lower().replace(m)

def extract_covariates(survey_df: pd.DataFrame, keep_participants=None) -> pd.DataFrame:
    """
    Per-participant means for MAILS / NCS6 / Legal / PTA from pre_experiment survey.

    Expects:
      - participantId
      - responses: dict or JSON-string or python-dict-string
      - surveyType: optional; if present, filters pre_experiment
    """
    cols = ["participantId","MAILS_overall_mean","NCS6_mean","legal_mean","PTA_mean"]
    if survey_df is None or survey_df.empty:
        return pd.DataFrame(columns=cols)

    sdf = survey_df.copy()
    sdf["participantId"] = sdf["participantId"].astype(str)

    if "surveyType" in sdf.columns:
        sdf = sdf[sdf["surveyType"].astype(str) == "pre_experiment"].copy()

    if keep_participants is not None:
        keep = set(map(str, keep_participants))
        sdf = sdf[sdf["participantId"].isin(keep)].copy()

    if sdf.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, r in sdf.iterrows():
        pid = str(r["participantId"])
        resp = r.get("responses", {}) or {}

        if isinstance(resp, str):
            resp_str = resp.strip()
            try:
                resp = json.loads(resp_str)
            except Exception:
                try:
                    resp = ast.literal_eval(resp_str)
                except Exception:
                    resp = {}

        if not isinstance(resp, dict):
            resp = {}

        def grab(prefix):
            vals = []
            for k, v in resp.items():
                if str(k).startswith(prefix) and v is not None:
                    try:
                        vals.append(float(v))
                    except Exception:
                        pass
            return vals

        mails = grab("mails_")
        ncs6  = grab("ncs6_")
        legal = grab("legal_")
        pta   = grab("pta_")

        rows.append({
            "participantId": pid,
            "MAILS_overall_mean": float(np.mean(mails)) if mails else np.nan,
            "NCS6_mean":          float(np.mean(ncs6))  if ncs6  else np.nan,
            "legal_mean":         float(np.mean(legal)) if legal else np.nan,
            "PTA_mean":           float(np.mean(pta))   if pta   else np.nan,
        })

    return pd.DataFrame(rows)

def tidy_result(res, label: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "model": label,
        "term": res.params.index,
        "coef": res.params.values,
        "se": res.bse.values,
        "z": getattr(res, "tvalues", np.nan),
        "p": res.pvalues.values,
    })
    ci = res.conf_int()
    out["ci_low"]  = ci[0].values
    out["ci_high"] = ci[1].values
    return out

def odds_ratio_row(tidy: pd.DataFrame, label: str, term: str) -> dict:
    r = tidy[tidy["term"] == term]
    if r.empty:
        return {"model": label, "term": term, "OR": np.nan, "CI_low": np.nan, "CI_high": np.nan, "p": np.nan}
    b = float(r["coef"].iloc[0])
    se = float(r["se"].iloc[0])
    OR = np.exp(b)
    lo = np.exp(b - 1.96*se)
    hi = np.exp(b + 1.96*se)
    p  = float(r["p"].iloc[0])
    return {"model": label, "term": term, "OR": OR, "CI_low": lo, "CI_high": hi, "p": p}


analysis = pd.read_csv(ANALYSIS_PATH)
analysis["participantId"] = analysis["participantId"].astype(str)

survey = pd.read_csv(SURVEY_PATH) if SURVEY_PATH.exists() else pd.DataFrame()
if not survey.empty and "participantId" in survey.columns:
    survey["participantId"] = survey["participantId"].astype(str)

df = analysis[analysis["condition"].isin(["evaluative","explainable"])].copy()

df["isErrorTrial"] = coerce_01(df["isErrorTrial"]).fillna(0).astype(int)
df["errorType"] = normalize_error_type(df["errorType"])
df["accuracy_bin"] = coerce_01(df["accuracyProvision"])

df = df[df["accuracy_bin"].notna()].copy()
df["accuracy_bin"] = df["accuracy_bin"].astype(int)

h1 = df[df["isErrorTrial"] == 1].copy()
valid_levels = ["near_miss", "overbreadth", "out_of_scope"]

h1["errorType"] = h1["errorType"].where(h1["errorType"].isin(valid_levels), np.nan)
h1 = h1.dropna(subset=["errorType"]).copy()

h1["errorType"] = pd.Categorical(h1["errorType"], categories=valid_levels)
h1["cond2"] = pd.Categorical(h1["condition"], categories=["explainable", "evaluative"])

h1["cond2"] = pd.Categorical(h1["condition"], categories=["explainable","evaluative"])

case_col = None
for c in ["postId", "caseId", "legalCase", "correctProvision"]:
    if c in h1.columns:
        case_col = c
        break

if case_col is None:
    raise RuntimeError("No case identifier found. Add one of: postId, caseId/legalCase, or correctProvision.")

balance = []
balance.append(f"N rows (H1 panel): {len(h1)} | participants: {h1['participantId'].nunique()}")
balance.append("\nCounts by condition:\n" + h1["condition"].value_counts().to_string())

balance.append("\nCounts by condition × errorType:\n" +
               pd.crosstab(h1["condition"], h1["errorType"]).to_string())

balance.append(f"\nCase FE column used: {case_col}")
if case_col in h1.columns:
    balance.append("\nCounts by condition × case:\n" +
                   pd.crosstab(h1["condition"], h1[case_col]).to_string())

write_txt(MODEL_DIR / "H1_panel_checks.txt", "\n".join(balance))

if case_col:
    formula = f"accuracy_bin ~ C(cond2) * C(errorType) + C({case_col})"
else:
    formula = "accuracy_bin ~ C(cond2) * C(errorType)"

m1 = smf.glm(formula, data=h1, family=sm.families.Binomial())
r1 = m1.fit(cov_type="HC1")
terms = r1.params.index.tolist()

def lin_or(model_res, term_weights, label):
    L = np.zeros((1, len(terms)))
    for term, w in term_weights.items():
        if term in terms:
            L[0, terms.index(term)] = w
    tt = model_res.t_test(L)
    b = np.asarray(tt.effect).item()
    se = np.asarray(tt.sd).item()
    return {
        "contrast": label,
        "OR": float(np.exp(b)),
        "CI_low": float(np.exp(b - 1.96*se)),
        "CI_high": float(np.exp(b + 1.96*se)),
        "p": np.asarray(tt.pvalue).item(),
    }

b_eval = "C(cond2)[T.evaluative]"
b_int_ob = "C(cond2)[T.evaluative]:C(errorType)[T.overbreadth]"
b_int_os = "C(cond2)[T.evaluative]:C(errorType)[T.out_of_scope]"

rows = [
    lin_or(r1, {b_eval: 1}, "Evaluative vs Explainable | near_miss"),
    lin_or(r1, {b_eval: 1, b_int_ob: 1}, "Evaluative vs Explainable | overbreadth"),
    lin_or(r1, {b_eval: 1, b_int_os: 1}, "Evaluative vs Explainable | out_of_scope"),
]

pd.DataFrame(rows).to_csv(TABLE_DIR / "H1_OR_by_errorType.csv", index=False)
write_txt(MODEL_DIR / "H1_primary_GLM_HC1.txt", r1.summary().as_text())

r1c = m1.fit(cov_type="cluster", cov_kwds={"groups": h1["participantId"]})
write_txt(MODEL_DIR / "H1_robustA_GLM_cluster.txt", r1c.summary().as_text())

cov_df = extract_covariates(survey, keep_participants=set(h1["participantId"]))
h1c = h1.merge(cov_df, on="participantId", how="left")

covars = ["MAILS_overall_mean","NCS6_mean","legal_mean","PTA_mean"]
for c in covars:
    if c in h1c.columns:
        h1c[c] = pd.to_numeric(h1c[c], errors="coerce")
        if h1c[c].notna().any():
            h1c[c] = h1c[c] - h1c[c].mean()

cov_terms = " + ".join([c for c in covars if c in h1c.columns])
formula_cov = formula + ((" + " + cov_terms) if cov_terms else "")

m2 = smf.glm(formula_cov, data=h1c, family=sm.families.Binomial())
r2 = m2.fit(cov_type="HC1")
write_txt(MODEL_DIR / "H1_exploratory_GLM_covariates_HC1.txt", r2.summary().as_text())

t1  = tidy_result(r1,  "Primary GLM (HC1)")
t1c = tidy_result(r1c, "Robust GLM (cluster pid)")
t2  = tidy_result(r2,  "Exploratory GLM + covariates (HC1)")

t1.to_csv(TABLE_DIR / "H1_primary_GLM_HC1_tidy.csv", index=False)
t1c.to_csv(TABLE_DIR / "H1_robustA_GLM_cluster_tidy.csv", index=False)
t2.to_csv(TABLE_DIR / "H1_exploratory_GLM_covariates_HC1_tidy.csv", index=False)


desc = (h1.groupby("condition")["accuracy_bin"]
        .agg(n="count", mean="mean", std="std")
        .reset_index())
desc.to_csv(TABLE_DIR / "H1_descriptives_by_condition.csv", index=False)

desc_et = (h1.groupby(["condition","errorType"])["accuracy_bin"]
           .agg(n="count", mean="mean", std="std")
           .reset_index())
desc_et.to_csv(TABLE_DIR / "H1_descriptives_by_condition_errorType.csv", index=False)

p_eval = float(h1.loc[h1["condition"]=="evaluative","accuracy_bin"].mean())
p_expl = float(h1.loc[h1["condition"]=="explainable","accuracy_bin"].mean())
n_eval = int((h1["condition"]=="evaluative").sum())
n_expl = int((h1["condition"]=="explainable").sum())
p_pool = (p_eval*n_eval + p_expl*n_expl) / max(1, (n_eval+n_expl))
se = np.sqrt(max(1e-12, p_pool*(1-p_pool)*(1/n_eval + 1/n_expl)))
z = (p_eval - p_expl) / se
p_one = 1 - stats.norm.cdf(z)

write_txt(
    MODEL_DIR / "H1_descriptive_ztest_one_tailed.txt",
    dedent(f"""
    Descriptive z-test (one-tailed; ignores clustering/FE)
    z = {z:.3f}, p = {p_one:.4f}
    eval = {p_eval:.3f} (n={n_eval}) vs expl = {p_expl:.3f} (n={n_expl})
    """).strip()
)

set_plot_style()

from statsmodels.stats.proportion import proportion_confint

def wilson_ci(k, n, alpha=0.05):
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return lo, hi


overall = (
    h1.groupby("condition")["accuracy_bin"]
      .agg(k="sum", n="count")
      .reindex(["explainable", "evaluative"])
)

overall["p"] = overall["k"] / overall["n"]
overall[["lo", "hi"]] = overall.apply(
    lambda r: pd.Series(wilson_ci(int(r.k), int(r.n))), axis=1
)

by_err = (
    h1.groupby(["errorType", "condition"])["accuracy_bin"]
      .agg(k="sum", n="count")
      .reset_index()
)

by_err["p"] = by_err["k"] / by_err["n"]
by_err[["lo", "hi"]] = by_err.apply(
    lambda r: pd.Series(wilson_ci(int(r.k), int(r.n))), axis=1
)

order_err = ["near_miss", "overbreadth", "out_of_scope"]
by_err["errorType"] = pd.Categorical(by_err["errorType"], categories=order_err)


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

x = np.arange(len(overall))
axes[0].bar(
    x,
    overall["p"],
    yerr=[overall["p"] - overall["lo"], overall["hi"] - overall["p"]],
    capsize=5
)
axes[0].set_xticks(x)
axes[0].set_xticklabels(["Explainable", "Evaluative"])
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Provision Accuracy")
axes[0].set_title("(a) Overall")

width = 0.35
x = np.arange(len(order_err))

for i, cond in enumerate(["evaluative", "explainable"]):
    d = by_err[by_err["condition"] == cond].sort_values("errorType")
    axes[1].bar(
        x + (i - 0.5) * width,
        d["p"],
        width,
        yerr=[d["p"] - d["lo"], d["hi"] - d["p"]],
        capsize=4,
        label=cond
    )

axes[1].set_xticks(x)
axes[1].set_xticklabels(["Near-Miss", "Overbreadth", "Out-of-Scope"])
axes[1].set_ylim(0, 1)
axes[1].set_title("(b) By Error Type")
axes[1].legend(title="Condition")

plt.tight_layout()
plt.savefig(FIG_DIR / "H1_combined.png", bbox_inches="tight")
plt.close(fig)


print("\n=== H1 Completed ===")
print(f"H1 participants: {h1['participantId'].nunique()} | rows: {len(h1)}")
print(f"Case FE source: {case_col}")
print("Saved outputs to:", OUT_DIR.resolve())
