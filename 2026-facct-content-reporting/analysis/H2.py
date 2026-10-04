from __future__ import annotations

import json
import ast
import warnings
from pathlib import Path
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.sandwich_covariance import cov_cluster


DATA_DIR = Path(os.environ["FACCT_DATA_DIR"])
TAG = "facct_1401"

OUT_DIR = Path(os.environ["FACCT_OUTPUT_DIR"]) / "H2"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
TABLE_DIR = OUT_DIR / "tables"
for d in (OUT_DIR, FIG_DIR, MODEL_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

ANALYSIS_PATH = DATA_DIR / "final_analysis_0301.csv"
SESSIONS_PATH = DATA_DIR / "final_sessions_0301.csv"
SURVEY_PATH = DATA_DIR / "final_survey_0301.csv"


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
    """Robust bool coercion for {0,1}, {"0","1"}, {"True","False"}, True/False."""
    if pd.api.types.is_bool_dtype(s):
        return s
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int).astype(bool)
    m = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "t": 1, "f": 0, "yes": 1, "no": 0})
    )
    return m.fillna(0).astype(int).astype(bool)


def normalize_error_type(s: pd.Series) -> pd.Series:
    err_map = {
        "near_miss": "near_miss",
        "near-miss": "near_miss",
        "Near-Miss": "near_miss",
        "near miss": "near_miss",
        "overbreadth": "overbreadth",
        "Overbreadth": "overbreadth",
        "over-breadth": "overbreadth",
        "out_of_scope": "out_of_scope",
        "out-of-scope": "out_of_scope",
        "Out-of-Scope": "out_of_scope",
        "out of scope": "out_of_scope",
    }
    return s.astype(str).map(err_map).fillna(s.astype(str))


def ensure_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """OrderedModel expects numeric exog without NaNs."""
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.fillna(0.0)
    return out.astype(float)


def ordered_logit_with_cluster(y: np.ndarray, X: pd.DataFrame, clusters: np.ndarray) -> Any:
    """
    Fit OrderedModel(logit). Attach participant-cluster robust covariance if possible.
    """
    y = np.asarray(y, dtype=int)
    Xn = ensure_numeric_df(X)
    clusters = np.asarray(clusters).flatten()

    model = OrderedModel(y, Xn, distr="logit")
    res = model.fit(method="bfgs", disp=False)

    try:
        n_unique = len(np.unique(clusters))
        if n_unique < len(clusters):
            C = cov_cluster(res, clusters)
            res.cov_params_default = C
            res._cache = {}
    except Exception:
        pass

    return res


def tidy_ordered(res: Any, model_name: str) -> pd.DataFrame:
    params = res.params.copy()
    cov = res.cov_params()
    se = np.sqrt(np.diag(cov))
    z = params / se
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    ci_lo = params - 1.96 * se
    ci_hi = params + 1.96 * se
    out = pd.DataFrame(
        {
            "model": model_name,
            "term": params.index,
            "coef": params.values,
            "se": se,
            "z": z,
            "p": p,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
        }
    )
    out["is_threshold"] = out["term"].astype(str).str.contains(r"/", regex=True)
    return out


def parse_responses_cell(x) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    if isinstance(x, str):
        xs = x.strip()
        if not xs:
            return {}
        try:
            return json.loads(xs)
        except Exception:
            try:
                return ast.literal_eval(xs)
            except Exception:
                return {}
    return {}


def extract_covariates(survey_df: pd.DataFrame, keep_ids: set[str]) -> pd.DataFrame:
    """
    Extract per-participant means from `responses` dict (pre_experiment only).
    Returns: participantId, MAILS_overall_mean, NCS6_mean, legal_mean, PTA_mean
    """
    cols = ["participantId", "MAILS_overall_mean", "NCS6_mean", "legal_mean", "PTA_mean"]
    needed = {"participantId", "surveyType", "responses"}
    if survey_df.empty or not needed.issubset(set(survey_df.columns)):
        return pd.DataFrame(columns=cols)

    pre = survey_df[survey_df["surveyType"].astype(str) == "pre_experiment"].copy()
    pre["participantId"] = pre["participantId"].astype(str)
    pre = pre[pre["participantId"].isin(keep_ids)]
    if pre.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, Any]] = []
    for _, r in pre.iterrows():
        pid = str(r["participantId"])
        resp = parse_responses_cell(r.get("responses"))

        mails = [resp[k] for k in resp if str(k).startswith("mails_") and resp.get(k) is not None]
        ncs6 = [resp[k] for k in resp if str(k).startswith("ncs6_") and resp.get(k) is not None]
        legal = [resp[k] for k in resp if str(k).startswith("legal_") and resp.get(k) is not None]
        pta = [resp[k] for k in resp if str(k).startswith("pta_") and resp.get(k) is not None]

        rows.append(
            {
                "participantId": pid,
                "MAILS_overall_mean": float(np.mean(mails)) if mails else np.nan,
                "NCS6_mean": float(np.mean(ncs6)) if ncs6 else np.nan,
                "legal_mean": float(np.mean(legal)) if legal else np.nan,
                "PTA_mean": float(np.mean(pta)) if pta else np.nan,
            }
        )

    return pd.DataFrame(rows)


analysis = pd.read_csv(ANALYSIS_PATH)
sessions = pd.read_csv(SESSIONS_PATH) if SESSIONS_PATH.exists() else pd.DataFrame()
survey = pd.read_csv(SURVEY_PATH) if SURVEY_PATH.exists() else pd.DataFrame()

req_cols = {"participantId", "condition", "isErrorTrial", "confusionDistance", "errorType"}
missing = req_cols - set(analysis.columns)
if missing:
    raise ValueError(f"`analysis` missing required columns: {sorted(missing)}")

analysis["participantId"] = analysis["participantId"].astype(str)
analysis["condition"] = analysis["condition"].astype(str)

if not sessions.empty:
    if "participantId" not in sessions.columns:
        raise ValueError("`sessions` exists but is missing `participantId`")
    sessions["participantId"] = sessions["participantId"].astype(str)

if not survey.empty and "participantId" in survey.columns:
    survey["participantId"] = survey["participantId"].astype(str)


analysis["isErrorTrial"] = coerce_bool(analysis["isErrorTrial"])

h2 = analysis[
    (analysis["isErrorTrial"])
    & (analysis["condition"].isin(["evaluative", "explainable"]))
].copy()

h2["confusionDistance"] = pd.to_numeric(h2["confusionDistance"], errors="coerce")
h2 = h2[h2["confusionDistance"].isin([0, 1, 2, 3])].copy()
h2["confusionDistance"] = h2["confusionDistance"].astype(int)

h2["errorType"] = normalize_error_type(h2["errorType"])

valid = ["near_miss","overbreadth","out_of_scope"]
h2["errorType"] = h2["errorType"].where(h2["errorType"].isin(valid), np.nan)
h2 = h2.dropna(subset=["errorType"]).copy()
h2["errorType"] = pd.Categorical(h2["errorType"], categories=valid)


case_col: Optional[str] = "correctProvision" if "correctProvision" in h2.columns else None

if not sessions.empty and "order" in sessions.columns:
    h2 = h2.merge(sessions[["participantId", "order"]], on="participantId", how="left")
    h2["order"] = h2["order"].fillna("unknown").astype(str)

desc = []
desc.append(f"N (H2 panel, error trials only): rows={len(h2)} | participants={h2['participantId'].nunique()}")
desc.append("\nCounts by condition:")
desc.append(h2["condition"].value_counts(dropna=False).to_string())
desc.append("\nCounts by condition × errorType:")
desc.append(pd.crosstab(h2["condition"], h2["errorType"], dropna=False).to_string())
desc.append(f"\ncase_col used for design-control robustness: {case_col}")
desc.append(f"order available: {'order' in h2.columns}")
write_txt(MODEL_DIR / "H2_panel_checks.txt", "\n".join(desc))
print("\n".join(desc))


h2["cond2"] = pd.Categorical(h2["condition"], categories=["explainable", "evaluative"])

X = pd.get_dummies(h2[["cond2", "errorType"]], drop_first=True)

X["cond2_evaluative:errorType_overbreadth"] = X.get("cond2_evaluative", 0) * X.get("errorType_overbreadth", 0)
X["cond2_evaluative:errorType_out_of_scope"] = X.get("cond2_evaluative", 0) * X.get("errorType_out_of_scope", 0)

y = h2["confusionDistance"].to_numpy(dtype=int)
clusters = h2["participantId"].to_numpy(dtype=str)

X_base = X.copy()
res_h2a = ordered_logit_with_cluster(y, X_base, clusters)
write_txt(MODEL_DIR / "H2a_primary_prereg_no_caseFE.txt", res_h2a.summary().as_text())
tidy_ordered(res_h2a, "H2a_primary_prereg_no_caseFE").to_csv(
    TABLE_DIR / "H2a_primary_prereg_no_caseFE_tidy.csv", index=False
)

key_lines = []
if "cond2_evaluative" in res_h2a.params.index:
    b = float(res_h2a.params["cond2_evaluative"])
    cov = res_h2a.cov_params()
    se = float(np.sqrt(np.diag(cov))[list(res_h2a.params.index).index("cond2_evaluative")])
    z = b / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    key_lines.append(f"H2a (no case FE): cond2_evaluative={b:.3f}, SE={se:.3f}, z={z:.3f}, p={p:.4f}")
key_lines.append("Negative coefficients shift probability mass toward smaller distances.")
write_txt(MODEL_DIR / "H2a_keyline_primary_no_caseFE.txt", "\n".join(key_lines))


if case_col is not None:
    cc = h2[case_col].fillna("missing_case").astype(str)
    X_case = pd.concat([X_base, pd.get_dummies(cc, prefix="case", drop_first=True)], axis=1)
    res_h2a_case = ordered_logit_with_cluster(y, X_case, clusters)
    write_txt(MODEL_DIR / "H2a_robust_design_caseFE.txt", res_h2a_case.summary().as_text())
    tidy_ordered(res_h2a_case, "H2a_robust_design_caseFE").to_csv(
        TABLE_DIR / "H2a_robust_design_caseFE_tidy.csv", index=False
    )
else:
    write_txt(MODEL_DIR / "H2a_robust_design_caseFE.txt", "Skipped: correctProvision not available, so no case FE robustness.")


if "order" in h2.columns:
    X_order = pd.concat([X_base, pd.get_dummies(h2["order"].astype(str), prefix="order", drop_first=True)], axis=1)
    res_h2a_order = ordered_logit_with_cluster(y, X_order, clusters)
    write_txt(MODEL_DIR / "H2a_robust_order_FE.txt", res_h2a_order.summary().as_text())
    tidy_ordered(res_h2a_order, "H2a_robust_order_FE").to_csv(
        TABLE_DIR / "H2a_robust_order_FE_tidy.csv", index=False
    )
else:
    write_txt(MODEL_DIR / "H2a_robust_order_FE.txt", "Skipped: sessions.order not available.")


h2b = h2[h2["errorType"].isin(["near_miss", "overbreadth"])].copy()
h2b["cond2"] = pd.Categorical(h2b["condition"], categories=["explainable", "evaluative"])
h2b["error2"] = pd.Categorical(h2b["errorType"], categories=["near_miss", "overbreadth"])

Xb = pd.get_dummies(h2b[["cond2", "error2"]], drop_first=True)
Xb["cond2_evaluative:error2_overbreadth"] = Xb.get("cond2_evaluative", 0) * Xb.get("error2_overbreadth", 0)

yb = h2b["confusionDistance"].to_numpy(dtype=int)
clusters_b = h2b["participantId"].to_numpy(dtype=str)

Xb_base = Xb.copy()
res_h2b = ordered_logit_with_cluster(yb, Xb_base, clusters_b)
write_txt(MODEL_DIR / "H2b_primary_prereg_no_caseFE.txt", res_h2b.summary().as_text())
tidy_ordered(res_h2b, "H2b_primary_prereg_no_caseFE").to_csv(
    TABLE_DIR / "H2b_primary_prereg_no_caseFE_tidy.csv", index=False
)

if case_col is not None:
    cc_b = h2b[case_col].fillna("missing_case").astype(str)
    Xb_case = pd.concat([Xb_base, pd.get_dummies(cc_b, prefix="case", drop_first=True)], axis=1)
    res_h2b_case = ordered_logit_with_cluster(yb, Xb_case, clusters_b)
    write_txt(MODEL_DIR / "H2b_robust_design_caseFE.txt", res_h2b_case.summary().as_text())
    tidy_ordered(res_h2b_case, "H2b_robust_design_caseFE").to_csv(
        TABLE_DIR / "H2b_robust_design_caseFE_tidy.csv", index=False
    )
else:
    write_txt(MODEL_DIR / "H2b_robust_design_caseFE.txt", "Skipped: correctProvision not available, so no case FE robustness.")


if not survey.empty:
    cov = extract_covariates(survey, keep_ids=set(h2["participantId"]))
    if cov.empty:
        write_txt(MODEL_DIR / "H2_robust_covariates.txt", "Skipped: no usable survey covariates found.")
    else:
        h2c = h2.merge(cov, on="participantId", how="left")

        covars = ["MAILS_overall_mean", "NCS6_mean", "legal_mean", "PTA_mean"]
        for c in covars:
            if c in h2c.columns:
                h2c[c] = pd.to_numeric(h2c[c], errors="coerce")
                if h2c[c].notna().any():
                    h2c[c] = h2c[c] - h2c[c].mean()

        Xc = X_base.copy()
        cov_present = [c for c in covars if c in h2c.columns and h2c[c].notna().any()]
        if cov_present:
            Xc = pd.concat([Xc, h2c[cov_present].fillna(0.0)], axis=1)

        yc = h2c["confusionDistance"].to_numpy(dtype=int)
        clusters_c = h2c["participantId"].to_numpy(dtype=str)

        res_h2c = ordered_logit_with_cluster(yc, Xc, clusters_c)
        write_txt(MODEL_DIR / "H2_robust_covariates.txt", res_h2c.summary().as_text())
        tidy_ordered(res_h2c, "H2_robust_covariates").to_csv(
            TABLE_DIR / "H2_robust_covariates_tidy.csv", index=False
        )
else:
    write_txt(MODEL_DIR / "H2_robust_covariates.txt", "Skipped: survey file not found / empty.")


if "benignFlagged" in analysis.columns:
    benign_by_pid = (
        analysis.groupby("participantId")["benignFlagged"]
        .max()
        .fillna(0)
    )
    benign_by_pid = pd.to_numeric(benign_by_pid, errors="coerce").fillna(0).astype(int)

    h2_tmp = h2.copy()
    h2_tmp["benign_reporter"] = h2_tmp["participantId"].map(benign_by_pid).fillna(0).astype(int)

    h2_drop = h2_tmp[h2_tmp["benign_reporter"] == 0].copy()
    if len(h2_drop) > 0:
        y_drop = h2_drop["confusionDistance"].to_numpy(dtype=int)
        cl_drop = h2_drop["participantId"].to_numpy(dtype=str)

        X_drop = pd.get_dummies(
            pd.DataFrame(
                {
                    "cond2": pd.Categorical(h2_drop["condition"], categories=["explainable", "evaluative"]),
                    "errorType": h2_drop["errorType"].astype(str),
                }
            ),
            drop_first=True,
        )
        X_drop["cond2_evaluative:errorType_overbreadth"] = X_drop.get("cond2_evaluative", 0) * X_drop.get("errorType_overbreadth", 0)
        X_drop["cond2_evaluative:errorType_out_of_scope"] = X_drop.get("cond2_evaluative", 0) * X_drop.get("errorType_out_of_scope", 0)

        res_drop = ordered_logit_with_cluster(y_drop, X_drop, cl_drop)
        write_txt(MODEL_DIR / "H2_robust_drop_benign.txt", res_drop.summary().as_text())
    else:
        write_txt(MODEL_DIR / "H2_robust_drop_benign.txt", "Skipped: no rows after dropping benign reporters.")

    X_ben = pd.concat([X_base, h2_tmp[["benign_reporter"]]], axis=1)
    res_ben = ordered_logit_with_cluster(y, X_ben, clusters)
    write_txt(MODEL_DIR / "H2_robust_with_benigncov.txt", res_ben.summary().as_text())
else:
    write_txt(MODEL_DIR / "H2_robust_drop_benign.txt", "Skipped: benignFlagged not in analysis.")
    write_txt(MODEL_DIR / "H2_robust_with_benigncov.txt", "Skipped: benignFlagged not in analysis.")


mwu_lines = []
a = h2.loc[h2["condition"] == "evaluative", "confusionDistance"].astype(int)
b = h2.loc[h2["condition"] == "explainable", "confusionDistance"].astype(int)
if len(a) and len(b):
    U, p = stats.mannwhitneyu(a, b, alternative="less")  # Eval < Expl
    mwu_lines.append(f"[MWU overall] U={U:.1f}, p(one-tailed Eval<Expl)={p:.4f}, n_eval={len(a)}, n_expl={len(b)}")

for et in ["near_miss", "overbreadth"]:
    aa = h2[(h2["condition"] == "evaluative") & (h2["errorType"] == et)]["confusionDistance"].astype(int)
    bb = h2[(h2["condition"] == "explainable") & (h2["errorType"] == et)]["confusionDistance"].astype(int)
    if len(aa) and len(bb):
        U, p = stats.mannwhitneyu(aa, bb, alternative="less")
        mwu_lines.append(f"[MWU {et}] U={U:.1f}, p(one-tailed Eval<Expl)={p:.4f}, n_eval={len(aa)}, n_expl={len(bb)}")

write_txt(MODEL_DIR / "H2_robust_MWU.txt", "\n".join(mwu_lines))


set_plot_style()

plot_df = h2.copy()
plot_df["dist"] = plot_df["confusionDistance"].astype(int)

dist_counts = plot_df.groupby(["condition", "dist"]).size().reset_index(name="count")
dist_counts["prop"] = dist_counts["count"] / dist_counts.groupby("condition")["count"].transform("sum")

mat = dist_counts.pivot(index="condition", columns="dist", values="prop").fillna(0.0)
mat = mat.reindex(["explainable", "evaluative"])
for lvl in [0, 1, 2, 3]:
    if lvl not in mat.columns:
        mat[lvl] = 0.0
mat = mat[[0, 1, 2, 3]]

fig, ax = plt.subplots(figsize=(6, 4))
mat.plot(kind="bar", stacked=True, ax=ax)
ax.set_ylabel("Proportion of error trials")
ax.set_xlabel("Condition")
ax.set_title("H2a: Misclassification distance (0–3) by condition")
ax.legend(title="Distance", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4)
plt.tight_layout()
plt.savefig(FIG_DIR / "h2a_distance_stacked_by_condition.png", bbox_inches="tight")
plt.close(fig)

err_order = ["near_miss", "overbreadth", "out_of_scope"]
cond_order = ["explainable", "evaluative"]

g = plot_df.groupby(["errorType", "condition", "dist"]).size().reset_index(name="count")
g["prop"] = g["count"] / g.groupby(["errorType", "condition"])["count"].transform("sum")

mat2 = (
    g.pivot_table(index=["errorType", "condition"], columns="dist", values="prop", fill_value=0.0)
    .reset_index()
)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for i, et in enumerate(err_order):
    ax = axes[i]
    sub = mat2[mat2["errorType"] == et].set_index("condition").reindex(cond_order)
    for lvl in [0, 1, 2, 3]:
        if lvl not in sub.columns:
            sub[lvl] = 0.0
    sub = sub[[0, 1, 2, 3]]
    sub.plot(kind="bar", stacked=True, ax=ax, legend=False)
    ax.set_title(et.replace("_", "-").title())
    ax.set_xlabel("")
    if i == 0:
        ax.set_ylabel("Proportion")
    ax.set_xticklabels(cond_order, rotation=0)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Distance", loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4)
fig.suptitle("H2: Misclassification distance by error type and condition", y=1.12)
plt.tight_layout()
plt.savefig(FIG_DIR / "h2_distance_stacked_by_errorType_condition.png", bbox_inches="tight")
plt.close(fig)

print("\n=== H2 completed ===")
print(f"Saved models to:  {MODEL_DIR.resolve()}")
print(f"Saved tables to:  {TABLE_DIR.resolve()}")
print(f"Saved figures to: {FIG_DIR.resolve()}")

V = cov_cluster(res_h2a, clusters)
robust_se = np.sqrt(np.diag(V))
from scipy.stats import norm
companion = pd.DataFrame({"term": res_h2a.params.index, "coef": res_h2a.params.values,
    "se": robust_se, "p": 2 * norm.sf(np.abs(res_h2a.params.values / robust_se)),
    "ci_low": res_h2a.params.values - norm.ppf(.975) * robust_se,
    "ci_high": res_h2a.params.values + norm.ppf(.975) * robust_se})
companion.to_csv(TABLE_DIR / "H2a_companion_participant_robust.csv", index=False)
