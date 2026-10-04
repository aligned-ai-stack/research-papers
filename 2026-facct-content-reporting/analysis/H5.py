from __future__ import annotations

import warnings
from pathlib import Path
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy.stats import norm


DATA_DIR = Path(os.environ["FACCT_DATA_DIR"])
ANALYSIS_PATH = DATA_DIR / "final_analysis_0301_cleaned.csv"
RATINGS_CSV = DATA_DIR / "combined_coder_ratings.csv"
RATINGS_XLSX = DATA_DIR / "combined_coder_ratings.xlsx"

OUT_TAG = "facct_1401_cont_primary"
OUT_DIR = Path(os.environ["FACCT_OUTPUT_DIR"]) / "H5"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
TABLE_DIR = OUT_DIR / "tables"
for d in (OUT_DIR, FIG_DIR, MODEL_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)


AI_ARMS = ["explainable", "evaluative"]
PAIR_ERR_ORDER = ["near_miss", "overbreadth", "out_of_scope"]  # ref = near_miss
COND_ORDER = ["explainable", "evaluative"]  # ref = explainable

EOR_MEAN = [
    "element_coverage_mean",
    "proportionality_reasoning_mean",
    "reasoning_depth_mean",
    "perceived_overall_quality_mean",
]

ORD_METHOD = "round"  # "round" or "floor" or "ceil"


def write_txt(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def set_plot_style() -> None:
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


def coerce_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().any():
        return x.fillna(0).astype(int).astype(bool)
    m = (s.astype(str).str.strip().str.lower()
         .map({"true": 1, "false": 0, "t": 1, "f": 0, "yes": 1, "no": 0}))
    return m.astype("float").fillna(0).astype(int).astype(bool)


def normalize_error_type(x: pd.Series) -> pd.Series:
    m = {
        "near-miss": "near_miss", "near miss": "near_miss", "near_miss": "near_miss",
        "overbreadth": "overbreadth", "over-breadth": "overbreadth", "over breadth": "overbreadth",
        "out-of-scope": "out_of_scope", "out of scope": "out_of_scope", "out_of_scope": "out_of_scope",
        "no-error": "no_error", "no error": "no_error", "no_error": "no_error",
    }
    xs = x.astype(str).str.strip().str.lower()
    out = xs.replace(m)
    out = out.replace({"nan": np.nan, "none": np.nan, "": np.nan})
    return out


def ensure_binary01(x: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(x):
        return x.astype(int)
    s = pd.to_numeric(x, errors="coerce")
    if s.isna().any():
        s2 = x.astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
        s = s.fillna(s2)
    return s.where(s.isin([0, 1]))


def ensure_mean_1to5(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    return s.where((s >= 1) & (s <= 5))


def mean_to_ordinal(x: pd.Series, method: str = "round") -> pd.Series:
    s = ensure_mean_1to5(x)
    if method == "round":
        o = np.rint(s)
    elif method == "floor":
        o = np.floor(s)
    elif method == "ceil":
        o = np.ceil(s)
    else:
        raise ValueError(f"Unknown method={method}")
    o = o.clip(1, 5)
    return o.astype("Int64")


def tidy_glm_cluster(res, model_name: str) -> pd.DataFrame:
    ci = res.conf_int()
    ci = pd.DataFrame(ci, index=res.params.index, columns=["ci_low", "ci_high"])
    out = pd.DataFrame({
        "model": model_name,
        "term": res.params.index,
        "coef": res.params.values,
        "se": res.bse.values,
        "z": (res.params / res.bse).values,
        "p": np.asarray(res.pvalues, dtype=float),
        "ci_low": ci["ci_low"].values,
        "ci_high": ci["ci_high"].values,
    })
    out["OR"] = np.exp(out["coef"])
    out["OR_low"] = np.exp(out["ci_low"])
    out["OR_high"] = np.exp(out["ci_high"])
    return out


def wald_test_linear(res, weights: Dict[str, float]) -> Tuple[float, float]:
    """Wald test for sum_i w_i * beta_i = 0."""
    names = list(res.params.index)
    w = np.zeros(len(names))
    for k, v in weights.items():
        if k in names:
            w[names.index(k)] = float(v)
    tt = res.t_test(w)
    return np.asarray(tt.tvalue).item(), np.asarray(tt.pvalue).item()


analysis = pd.read_csv(ANALYSIS_PATH)

if RATINGS_CSV.exists():
    ratings = pd.read_csv(RATINGS_CSV)
elif RATINGS_XLSX.exists():
    ratings = pd.read_excel(RATINGS_XLSX)
else:
    raise FileNotFoundError("Could not find combined_coder_ratings.csv or .xlsx in ../data/data_facct")

analysis["id"] = analysis["id"].astype(str)
ratings["id"] = ratings["id"].astype(str)

if analysis["id"].duplicated().any():
    raise RuntimeError("analysis has duplicate id rows; expected unique ids.")
if ratings["id"].duplicated().any():
    dup = ratings.loc[ratings["id"].duplicated(), "id"].iloc[0]
    raise RuntimeError(f"ratings has duplicate id rows (e.g., {dup}); aggregate first.")

df = analysis.merge(ratings, on="id", how="left")

need = {"participantId", "condition", "isErrorTrial", "accuracyProvision", "errorType"}
missing = need - set(df.columns)
if missing:
    raise RuntimeError(f"Missing required columns: {sorted(missing)}")

df["participantId"] = df["participantId"].astype(str)
df["condition"] = df["condition"].astype(str)
df["isErrorTrial"] = coerce_bool(df["isErrorTrial"])
df["errorType"] = normalize_error_type(df["errorType"])

df = df[df["condition"].isin(AI_ARMS)].copy()
df["cond2"] = pd.Categorical(df["condition"], categories=COND_ORDER, ordered=True)

df["acc_bin"] = ensure_binary01(df["accuracyProvision"])
df = df[df["acc_bin"].notna()].copy()
df["acc_bin"] = df["acc_bin"].astype(int)

for m in EOR_MEAN:
    if m not in df.columns:
        df[m] = np.nan
    df[m] = ensure_mean_1to5(df[m])

df["post"] = df["isErrorTrial"].astype(int)

df["err_sub"] = np.where(df["post"].eq(1), df["errorType"], np.nan)
df = df[(df["post"].eq(0)) | (df["err_sub"].isin(PAIR_ERR_ORDER))].copy()

err_map = (
    df[df["post"].eq(1)]
    .groupby("participantId")["err_sub"]
    .agg(lambda s: s.dropna().unique().tolist())
)

bad_multi = err_map[err_map.apply(len) != 1]
if len(bad_multi) > 0:
    write_txt(
        MODEL_DIR / "H5_bad_participants_multiple_errorTypes.txt",
        f"Participants with !=1 unique err_sub in error trial: {len(bad_multi)}\n"
        + bad_multi.astype(str).to_string()
    )

err_map_single = err_map[err_map.apply(len) == 1].apply(lambda lst: lst[0]).to_dict()
df["pair_errorType"] = df["participantId"].map(err_map_single)

counts = df.groupby("participantId")["post"].agg(["count", "sum"])
keep_p = counts[(counts["count"].eq(2)) & (counts["sum"].eq(1))].index
drop_p = counts.index.difference(keep_p)
if len(drop_p) > 0:
    write_txt(
        MODEL_DIR / "H5_dropped_participants_not_paired.txt",
        f"Dropped participants not forming a clean pair (expected count=2 and sum(post)=1): {len(drop_p)}"
    )

dfp = df[df["participantId"].isin(keep_p)].copy()
dfp["pair_errorType"] = pd.Categorical(dfp["pair_errorType"], categories=PAIR_ERR_ORDER, ordered=True)

lines: List[str] = []
lines.append("H5 panel checks (AI arms only; paired DiD) [UPDATED continuous-primary]")
lines.append(f"Rows: {len(dfp)} | Participants: {dfp['participantId'].nunique()} (expected rows=2×participants)")
lines.append("\nCounts by condition:\n" + dfp["cond2"].value_counts().to_string())
lines.append("\nCounts by post (0=no-error, 1=error):\n" + dfp["post"].value_counts().to_string())
lines.append("\nCounts by pair_errorType (participants):\n" +
             dfp.drop_duplicates("participantId")["pair_errorType"].value_counts().to_string())
lines.append("\nCondition × pair_errorType (participants):\n" +
             pd.crosstab(
                 dfp.drop_duplicates("participantId")["cond2"],
                 dfp.drop_duplicates("participantId")["pair_errorType"]
             ).to_string())
write_txt(MODEL_DIR / "H5_panel_checks_primary.txt", "\n".join(lines))

desc_cols = ["acc_bin"] + EOR_MEAN
desc = (
    dfp.groupby(["cond2", "pair_errorType", "post"])[desc_cols]
       .agg(["mean", "std", "count"])
       .round(3)
)
desc.to_csv(TABLE_DIR / "H5_descriptives_condition_pairErrorType_post.csv")


acc_formula = (
    "acc_bin ~ C(cond2, Treatment(reference='explainable'))"
    " + post * C(pair_errorType, Treatment(reference='near_miss'))"
    " + C(cond2, Treatment(reference='explainable')):post * C(pair_errorType, Treatment(reference='near_miss'))"
)
glm_acc = smf.glm(acc_formula, data=dfp, family=sm.families.Binomial())
res_acc = glm_acc.fit(cov_type="cluster", cov_kwds={"groups": dfp["participantId"]})

write_txt(MODEL_DIR / "H5_accuracy_GLM_DiD.txt", res_acc.summary().as_text())
tidy_glm_cluster(res_acc, "H5_accuracy_GLM_DiD").to_csv(TABLE_DIR / "H5_accuracy_GLM_DiD_tidy.csv", index=False)

term_post_ob = "post:C(pair_errorType, Treatment(reference='near_miss'))[T.overbreadth]"
term_post_oos = "post:C(pair_errorType, Treatment(reference='near_miss'))[T.out_of_scope]"

stat_oos_vs_nm, p_oos_vs_nm = wald_test_linear(res_acc, {term_post_oos: 1.0})
stat_oos_vs_ob, p_oos_vs_ob = wald_test_linear(res_acc, {term_post_oos: 1.0, term_post_ob: -1.0})

write_txt(
    MODEL_DIR / "H5a_accuracy_wald_contrasts.txt",
    "Accuracy Wald contrasts (DiD; explainable baseline)\n"
    f"OOS vs Near-miss (tests post:OOS = 0): stat={stat_oos_vs_nm:.3f}, p={p_oos_vs_nm:.6f}\n"
    f"OOS vs Overbreadth (tests post:OOS - post:OB = 0): stat={stat_oos_vs_ob:.3f}, p={p_oos_vs_ob:.6f}\n"
)

term_eval_post_oos = (
    "C(cond2, Treatment(reference='explainable'))[T.evaluative]:"
    "post:C(pair_errorType, Treatment(reference='near_miss'))[T.out_of_scope]"
)
if term_eval_post_oos in res_acc.params.index:
    stat_att, p_att = wald_test_linear(res_acc, {term_eval_post_oos: 1.0})
else:
    stat_att, p_att = np.nan, np.nan

write_txt(
    MODEL_DIR / "H5b_accuracy_attenuation_test_oos.txt",
    f"Attenuation test (accuracy): {term_eval_post_oos}\n"
    f"stat={stat_att:.3f}, p={p_att:.6f}\n"
)

grid = []
for cond in COND_ORDER:
    for pe in PAIR_ERR_ORDER:
        for post in [0, 1]:
            grid.append({"cond2": cond, "pair_errorType": pe, "post": post})
pred_grid = pd.DataFrame(grid)
pred_grid["cond2"] = pd.Categorical(pred_grid["cond2"], categories=COND_ORDER, ordered=True)
pred_grid["pair_errorType"] = pd.Categorical(pred_grid["pair_errorType"], categories=PAIR_ERR_ORDER, ordered=True)

pred = res_acc.get_prediction(pred_grid).summary_frame()
pred_out = pred_grid.copy()
pred_out["pred"] = pred["mean"].values
pred_out["pred_lo"] = pred["mean_ci_lower"].values
pred_out["pred_hi"] = pred["mean_ci_upper"].values
pred_out.to_csv(TABLE_DIR / "H5_accuracy_margins.csv", index=False)


wide_cols = ["participantId", "cond2", "pair_errorType"]
wide_base = dfp.drop_duplicates("participantId")[wide_cols].copy()

noerr = dfp[dfp["post"].eq(0)].set_index("participantId")
err = dfp[dfp["post"].eq(1)].set_index("participantId")

common = wide_base["participantId"].tolist()
noerr = noerr.loc[common]
err = err.loc[common]

delta_df = wide_base.set_index("participantId").copy()
delta_df["cond2"] = pd.Categorical(delta_df["cond2"].astype(str), categories=COND_ORDER, ordered=True)
delta_df["pair_errorType"] = pd.Categorical(delta_df["pair_errorType"].astype(str), categories=PAIR_ERR_ORDER, ordered=True)

for m in EOR_MEAN:
    delta_df[f"{m}_noerr"] = noerr[m].values
    delta_df[f"{m}_err"] = err[m].values
    delta_df[f"delta_{m}"] = delta_df[f"{m}_err"] - delta_df[f"{m}_noerr"]

delta_df.reset_index().to_csv(TABLE_DIR / "H5_EoR_first_difference_panel.csv", index=False)

eor_fd_rows: List[pd.DataFrame] = []

for m in EOR_MEAN:
    dv = f"delta_{m}"
    d = delta_df.dropna(subset=[dv]).copy()
    if d.empty:
        write_txt(MODEL_DIR / f"H5_{dv}_FD_SKIPPED.txt", "No non-missing delta outcomes.")
        continue

    formula = (
        f"{dv} ~ C(cond2, Treatment(reference='explainable'))"
        " + C(pair_errorType, Treatment(reference='near_miss'))"
        " + C(cond2, Treatment(reference='explainable')):C(pair_errorType, Treatment(reference='near_miss'))"
    )
    mod = smf.ols(formula, data=d.reset_index())
    res = mod.fit(cov_type="HC1")  # one row per participant

    write_txt(MODEL_DIR / f"H5_EoR_FD_{m}_OLS_HC1.txt", res.summary().as_text())

    ci = res.conf_int()
    tidy = pd.DataFrame({
        "dv": dv,
        "model": "H5_EoR_FD_OLS_HC1",
        "term": res.params.index,
        "coef": res.params.values,
        "se": res.bse.values,
        "t": res.tvalues.values,
        "p": res.pvalues.values,
        "ci_low": ci[0].values,
        "ci_high": ci[1].values,
        "n_participants": int(d.shape[0]),
    })
    eor_fd_rows.append(tidy)

    term_ob = "C(pair_errorType, Treatment(reference='near_miss'))[T.overbreadth]"
    term_oos = "C(pair_errorType, Treatment(reference='near_miss'))[T.out_of_scope]"
    term_int_oos = (
        "C(cond2, Treatment(reference='explainable'))[T.evaluative]:"
        "C(pair_errorType, Treatment(reference='near_miss'))[T.out_of_scope]"
    )
    term_int_ob = (
        "C(cond2, Treatment(reference='explainable'))[T.evaluative]:"
        "C(pair_errorType, Treatment(reference='near_miss'))[T.overbreadth]"
    )

    stat1, p1 = wald_test_linear(res, {term_oos: 1.0}) if term_oos in res.params.index else (np.nan, np.nan)
    stat2, p2 = wald_test_linear(res, {term_oos: 1.0, term_ob: -1.0}) if (term_oos in res.params.index and term_ob in res.params.index) else (np.nan, np.nan)
    stat3, p3 = wald_test_linear(res, {term_int_oos: 1.0}) if term_int_oos in res.params.index else (np.nan, np.nan)

    write_txt(
        MODEL_DIR / f"H5_EoR_FD_{m}_wald_contrasts.txt",
        "EoR First-Difference Wald contrasts (HC1; one row per participant)\n"
        f"OOS vs Near-miss (baseline explainable): stat={stat1:.3f}, p={p1:.6f}\n"
        f"OOS vs Overbreadth (baseline explainable): stat={stat2:.3f}, p={p2:.6f}\n"
        f"Attenuation (Evaluative × OOS): stat={stat3:.3f}, p={p3:.6f}\n"
    )

if eor_fd_rows:
    pd.concat(eor_fd_rows, ignore_index=True).to_csv(TABLE_DIR / "H5_EoR_FD_OLS_tidy_all.csv", index=False)


def build_ordered_exog(d: pd.DataFrame) -> pd.DataFrame:
    """
    Exog matrix with no intercept. Reference levels:
      - condition: explainable
      - pair_errorType: near_miss
    """
    cond_eval = (d["cond2"].astype(str) == "evaluative").astype(int)
    pe_ob = (d["pair_errorType"].astype(str) == "overbreadth").astype(int)
    pe_oos = (d["pair_errorType"].astype(str) == "out_of_scope").astype(int)
    post = d["post"].astype(int)

    X = pd.DataFrame({
        "cond_eval": cond_eval,
        "post": post,
        "pe_overbreadth": pe_ob,
        "pe_out_of_scope": pe_oos,
        "post_x_ob": post * pe_ob,
        "post_x_oos": post * pe_oos,
        "cond_eval_x_post": cond_eval * post,
        "cond_eval_x_post_x_ob": cond_eval * post * pe_ob,
        "cond_eval_x_post_x_oos": cond_eval * post * pe_oos,
    })

    const_cols = X.columns[X.nunique(dropna=False) <= 1].tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        raise ValueError("Exog is rank-deficient (collinearity). Check pairing or categories.")
    return X


def fit_ordered_logit_cluster(d: pd.DataFrame, dv_ord: str, model_tag: str) -> pd.DataFrame:
    """
    Fit OrderedModel (logit link) and compute cluster-robust SE via cov_cluster.
    We report only beta coefficients (not thresholds) in the tidy table.
    """
    y = d[dv_ord].astype(int)
    X = build_ordered_exog(d)

    mod = OrderedModel(y, X, distr="logit")
    res = mod.fit(method="bfgs", disp=False)

    V = cov_cluster(res, d["participantId"].to_numpy())
    se_all = np.sqrt(np.diag(V))

    beta_names = list(X.columns)
    k_beta = len(beta_names)

    beta = np.asarray(res.params[:k_beta], dtype=float)
    se_beta = np.asarray(se_all[:k_beta], dtype=float)

    rows = []
    for nm, coef, se in zip(beta_names, beta, se_beta):
        z = coef / se if se > 0 else np.nan
        p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        rows.append({
            "dv": dv_ord,
            "model": model_tag,
            "term": nm,
            "coef": float(coef),
            "se_cluster": float(se),
            "z": float(z),
            "p": float(p),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "cumulative_OR": float(np.exp(coef)),
            "cumulative_OR_low": float(np.exp(ci_low)),
            "cumulative_OR_high": float(np.exp(ci_high)),
            "n": int(len(d)),
            "participants": int(d["participantId"].nunique()),
        })

    write_txt(
        MODEL_DIR / f"{model_tag}_{dv_ord}_ordered_logit_DiD.txt",
        res.summary().as_text() + "\n\nCluster-robust covariance computed via cov_cluster(res, participantId).\n"
    )
    return pd.DataFrame(rows)


ordered_rows: List[pd.DataFrame] = []
notes: List[str] = [f"Ordinal construction method: {ORD_METHOD}"]

for m in EOR_MEAN:
    dv_ord = f"{m}_ord"
    dfp[dv_ord] = mean_to_ordinal(dfp[m], method=ORD_METHOD)
    notes.append(f"{dv_ord}: non-missing n={int(dfp[dv_ord].notna().sum())}")

    d = dfp.dropna(subset=[dv_ord]).copy()
    try:
        ordered_rows.append(fit_ordered_logit_cluster(d, dv_ord=dv_ord, model_tag="H5_EoR_Sensitivity_OrderedLogit"))
    except Exception as e:
        write_txt(MODEL_DIR / f"H5_EoR_Sensitivity_{dv_ord}_FAILED.txt", f"Failed: {repr(e)}")

write_txt(MODEL_DIR / "H5_ordered_logit_construction_notes.txt", "\n".join(notes))

if ordered_rows:
    pd.concat(ordered_rows, ignore_index=True).to_csv(
        TABLE_DIR / "H5_EoR_sensitivity_ordered_logit_tidy_all.csv",
        index=False
    )


set_plot_style()
w = 0.38

acc_means = dfp.groupby(["cond2", "pair_errorType", "post"])["acc_bin"].mean().reset_index()
acc_means.to_csv(TABLE_DIR / "H5_accuracy_means_by_pairErrorType_post.csv", index=False)

fig, ax = plt.subplots(figsize=(8.2, 4.2))
labels, v_expl, v_eval = [], [], []
for pe in PAIR_ERR_ORDER:
    for post in [0, 1]:
        labels.append(f"{pe}\n{'no-error' if post==0 else 'error'}")
        row = acc_means[(acc_means["pair_errorType"] == pe) & (acc_means["post"] == post)]
        v_expl.append(float(row[row["cond2"] == "explainable"]["acc_bin"]) if (row["cond2"] == "explainable").any() else np.nan)
        v_eval.append(float(row[row["cond2"] == "evaluative"]["acc_bin"]) if (row["cond2"] == "evaluative").any() else np.nan)

x = np.arange(len(labels))
ax.bar(x - w/2, v_expl, width=w, label="Conventional XAI")
ax.bar(x + w/2, v_eval, width=w, label="Evaluative AI")
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy (proportion correct)")
ax.set_xlabel("Pair error type × trial")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=True)
ax.grid(axis="y", alpha=0.25)
ax.set_title("H5: Accuracy by pair error type and trial (paired DiD panel)")
plt.tight_layout()
plt.savefig(FIG_DIR / "H5_accuracy_by_pairErrorType_post_condition.png", bbox_inches="tight")
plt.close(fig)

for m in EOR_MEAN:
    d = dfp.dropna(subset=[m]).copy()
    if d.empty:
        continue
    means = d.groupby(["cond2", "pair_errorType", "post"])[m].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    labels, v1, v2 = [], [], []
    for pe in PAIR_ERR_ORDER:
        for post in [0, 1]:
            labels.append(f"{pe}\n{'no-error' if post==0 else 'error'}")
            row = means[(means["pair_errorType"] == pe) & (means["post"] == post)]
            v1.append(float(row[row["cond2"] == "explainable"][m]) if (row["cond2"] == "explainable").any() else np.nan)
            v2.append(float(row[row["cond2"] == "evaluative"][m]) if (row["cond2"] == "evaluative").any() else np.nan)

    x = np.arange(len(labels))
    ax.bar(x - w/2, v1, width=w, label="Conventional XAI")
    ax.bar(x + w/2, v2, width=w, label="Evaluative AI")
    ax.set_ylim(1, 5)
    ax.set_ylabel(f"{m.replace('_mean','').replace('_',' ').title()} (mean 1--5)")
    ax.set_xlabel("Pair error type × trial")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=True)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"H5: {m.replace('_mean','').replace('_',' ').title()} (paired DiD panel)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"H5_{m}_by_pairErrorType_post_condition.png", bbox_inches="tight")
    plt.close(fig)

print("✓ H5 complete (UPDATED continuous-primary for EoR)")
print("Models ->", MODEL_DIR.resolve())
print("Tables ->", TABLE_DIR.resolve())
print("Figures ->", FIG_DIR.resolve())
