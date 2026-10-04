#!/usr/bin/env python
# coding: utf-8

# In[4]:


import seaborn as sns
import chi_style_sns
chi_style_sns.set_chi_style_sns()
HAS_SNS = True


# In[5]:


# H2 (Robust): Bayesian normative steps & updating behavior
# - Policies: strict / lenient / hybrid
# - Hybrid fallback guard w/ threshold on share of tasks allowed to use fallback
# - Sensitivity to fallback S in {5, 10, 20}
# - Full QC accounting + per-task slopes + histogram + scatter
#
# Outputs:
#   ../data/results/h2_panel_<policy>_S<val>.csv
#   ../data/results/h2_pooled_regression_<policy>_S<val>.txt
#   ../data/results/h2_individual_slopes_<policy>_S<val>.csv
#   ../data/hypothesis_outputs/H2/h2_hist_<policy>_S<val>.pdf/png
#   ../data/hypothesis_outputs/H2/h2_scatter_<policy>_S<val>.pdf/png
#   ../data/results/h2_cf_qc_<policy>_S<val>.csv
#
# Notes:
# - “strict”: require monotonicity & not degenerate; keep task if at least one consistent side yields n0
# - “lenient”: ignore monotonicity; keep task if at least one consistent side yields n0
# - “hybrid”: prefer CF n0 if reliable; otherwise allow fallback S only while fallback_share < threshold, else drop

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -------------------------
# Paths
# -------------------------
INPUT = Path(__file__).resolve().parents[1] / Path("data/intermediate_outputs/valid_sessions.json")


# RESULTS_DIR = Path(__file__).resolve().parents[1] / Path("data/results/")
HYP_DIR = Path(__file__).resolve().parents[1] / Path("data/hypothesis_outputs/H2/")
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HYP_DIR.mkdir(parents=True, exist_ok=True)


MODELS_DIR = Path(__file__).resolve().parents[1] / Path("data/hypothesis_outputs/H2/models/")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


FIGURES_DIR = Path(__file__).resolve().parents[1] / Path("data/hypothesis_outputs/H2/figures/")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Helpers
# -------------------------
def load_json_any(path: Path):
    txt = path.read_text(encoding="utf-8")
    try:
        return json.loads(txt)
    except Exception:
        rows = []
        for line in txt.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                pass
        if rows:
            return rows
        raise ValueError(f"Unable to parse JSON/NDJSON: {path}")

def get_wrong_ordinals(aiWrongMap: dict, taskType: str) -> set:
    wrong = set()
    m = (aiWrongMap or {}).get(taskType, {}) or {}
    for k, v in m.items():
        try:
            if v:
                wrong.add(int(k))
        except Exception:
            continue
    return wrong

# ---- Counterfactual QC & n0 estimation ----
def counterfactual_flags(b0, b0_plus, b0_minus, eps=1e-9):
    """Range, monotonicity, degeneracy checks for counterfactual priors."""
    vals = [v for v in (b0, b0_plus, b0_minus) if v is not None]
    cf_in_range = all((0.0 - eps) <= v <= (1.0 + eps) for v in vals)

    mono_ok = True
    if b0_minus is not None and b0 is not None:
        mono_ok &= (b0_minus <= b0 + eps)
    if b0_plus is not None and b0 is not None:
        mono_ok &= (b0 <= b0_plus + eps)

    zero_sensitivity = (
        (b0_plus is not None and b0_minus is not None and b0 is not None) and
        (abs(b0_plus - b0) < eps and abs(b0 - b0_minus) < eps)
    )
    return {
        "cf_in_range": bool(cf_in_range),
        "mono_ok": bool(mono_ok),
        "zero_sensitivity": bool(zero_sensitivity),
    }

def robust_n0_from_cfs(b0, b0_plus, b0_minus, eps=1e-12, clip_lo=1.0, clip_hi=500.0):
    """
    Use only directionally-consistent sides to estimate n0:
      plus side: b0_plus > b0 -> n0p = (1 - b0_plus)/(b0_plus - b0)
      minus side: b0_minus < b0 -> n0m = b0_minus/(b0 - b0_minus)
    Returns (n0_est, used_sides) where used_sides in {'plus','minus'}.
    """
    used = set()
    cand = []

    if b0 is not None and b0_plus is not None and (b0_plus - b0) > eps:
        n0p = (1.0 - b0_plus) / (b0_plus - b0)
        if np.isfinite(n0p) and n0p > 0:
            cand.append(n0p); used.add("plus")

    if b0 is not None and b0_minus is not None and (b0 - b0_minus) > eps:
        n0m = b0_minus / (b0 - b0_minus)
        if np.isfinite(n0m) and n0m > 0:
            cand.append(n0m); used.add("minus")

    if not cand:
        return np.nan, used

    n0_est = float(np.median(cand))
    return float(np.clip(n0_est, clip_lo, clip_hi)), used

def reliability_check(b0, b0_plus, b0_minus, used_sides, delta_tau=0.10):
    """
    A CF-based n0 is 'reliable' if:
      - both sides were usable; OR
      - a single side shows a sizable directional difference (> delta_tau absolute change).
    """
    if len(used_sides) >= 2:
        return True
    if len(used_sides) == 1:
        if "plus" in used_sides and (b0 is not None and b0_plus is not None):
            return abs(b0_plus - b0) > delta_tau
        if "minus" in used_sides and (b0 is not None and b0_minus is not None):
            return abs(b0 - b0_minus) > delta_tau
    return False

def compute_task_rows(trials, b0_prop, n0, wrong_ordinals, pid, ttype, task_pos):
    """
    Walk a task's trials sequentially, producing rows with:
    delta_b, norm_step, mB_prev, etc., given (b0, n0).
    """
    rows = []
    alpha = b0_prop * n0
    beta = (1.0 - b0_prop) * n0

    # make a quick index by ordinal
    ord_idx = {int(tr.get("trialOrdinal", 0)): tr for tr in trials}

    for t_ord in sorted(ord_idx.keys()):
        tr = ord_idx[t_ord]
        mB_prev = alpha / (alpha + beta) if (alpha + beta) > 0 else np.nan
        f_t = 0 if t_ord in wrong_ordinals else 1

        # build b_prev and b_post in proportions
        if t_ord == 1:
            b_prev = tr.get("preBeliefAI")
            b_prev = None if b_prev is None else float(b_prev)/100.0
        else:
            prev = ord_idx.get(t_ord - 1)
            b_prev = None if prev is None else prev.get("postBeliefAI")
            b_prev = None if b_prev is None else float(b_prev)/100.0

        b_post = tr.get("postBeliefAI")
        b_post = None if b_post is None else float(b_post)/100.0

        # store row iff both observed
        if b_prev is not None and b_post is not None and np.isfinite(mB_prev):
            delta_b = b_post - b_prev
            normative_step = (f_t - mB_prev) / (n0 + t_ord)  # beta-binomial Bayes step size
            rows.append({
                "participant_id": pid,
                "taskType": ttype,
                "taskPosition": task_pos,
                "t": t_ord,
                "b_prev": b_prev,
                "b_post": b_post,
                "delta_b": delta_b,
                "mB_prev": mB_prev,
                "norm_step": normative_step,
                "f_t": f_t,
                "n0": n0,
            })

        # update posterior counts regardless (to preserve normative path)
        alpha += f_t
        beta += (1 - f_t)

    return rows

def build_h2_panel_robust(
    sessions,
    policy="hybrid",
    fallback_S=10.0,
    fallback_threshold=0.30,
    reliability_delta_tau=0.10,
):
    """
    Build panel under a chosen policy and fallback setting.
    Returns (panel_df, qc_task_df, qc_summary_dict)
    """
    rows = []
    cf_stats = {
        "tasks_total": 0,
        "kept": 0,
        "dropped_out_of_range": 0,
        "dropped_mono_violation": 0,
        "dropped_degenerate": 0,
        "dropped_n0_nan": 0,
        "dropped_hybrid_threshold": 0,
        "cf_reliable": 0,
        "cf_unreliable": 0,
        "fallback_used": 0,
    }
    task_qc_rows = []

    for s in sessions:
        # validity filters (same as your earlier pipeline)
        if not s.get("completed", False):
            continue
        if s.get("failedComprehension", False):
            continue
        sac = s.get("solutionAccessCount", {}) or {}
        if isinstance(sac, dict) and any((v or 0) > 1 for v in sac.values()):
            continue

        pid = s.get("participant_id", "")
        order = s.get("taskOrder", []) or []
        order_idx = {t: i + 1 for i, t in enumerate(order)}
        aiWrongMap = s.get("aiWrongMap", {}) or {}

        # group & sort trials within each task
        by_task = defaultdict(list)
        for r in s.get("responses", []):
            by_task[r.get("taskType")].append(r)
        for t in list(by_task.keys()):
            by_task[t].sort(key=lambda x: int(x.get("trialOrdinal", 0)))

        for ttype in order:
            trials = by_task.get(ttype, [])
            if not trials:
                continue

            cf_stats["tasks_total"] += 1

            # priors from trial 1
            t1 = next((tr for tr in trials if int(tr.get("trialOrdinal", 0)) == 1), None)
            if t1 is None:
                continue

            # proportions
            b0 = t1.get("preBeliefAI")
            b0_plus = t1.get("preB0Plus")
            b0_minus = t1.get("preB0Minus")
            b0 = None if b0 is None else float(b0)/100.0
            b0_plus = None if b0_plus is None else float(b0_plus)/100.0
            b0_minus = None if b0_minus is None else float(b0_minus)/100.0

            cf = counterfactual_flags(b0, b0_plus, b0_minus)
            if not cf["cf_in_range"]:
                cf_stats["dropped_out_of_range"] += 1
                task_qc_rows.append({
                    "participant_id": pid, "taskType": ttype, "taskPosition": order_idx.get(ttype),
                    "n0": np.nan, "n0_source": "drop_out_of_range",
                    "cf_mono_ok": cf["mono_ok"], "cf_zero_sensitivity": cf["zero_sensitivity"],
                    "used_plus": 0, "used_minus": 0
                })
                continue

            # Estimate from CFs
            n0_cf, used_sides = robust_n0_from_cfs(b0, b0_plus, b0_minus)
            n0_from_cf = np.isfinite(n0_cf)
            cf_reliable = reliability_check(b0, b0_plus, b0_minus, used_sides, delta_tau=reliability_delta_tau)

            wrong_ordinals = get_wrong_ordinals(aiWrongMap, ttype)

            # Policy gate
            keep = False
            n0 = np.nan
            n0_source = "drop"

            if policy == "strict":
                # must be monotone + not degenerate + have a CF-based n0
                if (cf["mono_ok"] and (not cf["zero_sensitivity"]) and n0_from_cf):
                    keep = True; n0 = n0_cf; n0_source = "cf_strict"
                else:
                    if not cf["mono_ok"]:
                        cf_stats["dropped_mono_violation"] += 1
                    elif cf["zero_sensitivity"]:
                        cf_stats["dropped_degenerate"] += 1
                    else:
                        cf_stats["dropped_n0_nan"] += 1

            elif policy == "lenient":
                # ignore monotonicity; keep if at least one side yields CF n0
                if n0_from_cf:
                    keep = True; n0 = n0_cf; n0_source = "cf_lenient"
                else:
                    cf_stats["dropped_n0_nan"] += 1

            elif policy == "hybrid":
                if n0_from_cf and cf_reliable:
                    keep = True; n0 = n0_cf; n0_source = "cf_reliable"
                    cf_stats["cf_reliable"] += 1
                else:
                    cf_stats["cf_unreliable"] += 1
                    # Allow fallback only if we haven't exceeded fallback threshold so far
                    denom = max(1, cf_stats["cf_reliable"] + cf_stats["cf_unreliable"])
                    current_fallback_share = cf_stats["fallback_used"] / denom
                    if current_fallback_share < fallback_threshold:
                        keep = True; n0 = float(fallback_S); n0_source = f"fallback_S{int(fallback_S)}"
                        cf_stats["fallback_used"] += 1
                    else:
                        keep = False; n0 = np.nan; n0_source = "drop_hybrid_threshold"
                        cf_stats["dropped_hybrid_threshold"] += 1

            else:
                raise ValueError("policy must be one of: 'strict' | 'lenient' | 'hybrid'")

            if not keep:
                task_qc_rows.append({
                    "participant_id": pid, "taskType": ttype, "taskPosition": order_idx.get(ttype),
                    "n0": np.nan, "n0_source": n0_source,
                    "cf_mono_ok": cf["mono_ok"], "cf_zero_sensitivity": cf["zero_sensitivity"],
                    "used_plus": int("plus" in used_sides), "used_minus": int("minus" in used_sides),
                })
                continue

            # Keep task -> compute rows
            task_rows = compute_task_rows(
                trials=trials,
                b0_prop=b0,
                n0=n0,
                wrong_ordinals=wrong_ordinals,
                pid=pid,
                ttype=ttype,
                task_pos=order_idx.get(ttype),
            )
            rows.extend(task_rows)
            cf_stats["kept"] += 1
            task_qc_rows.append({
                "participant_id": pid, "taskType": ttype, "taskPosition": order_idx.get(ttype),
                "n0": n0, "n0_source": n0_source,
                "cf_mono_ok": cf["mono_ok"], "cf_zero_sensitivity": cf["zero_sensitivity"],
                "used_plus": int("plus" in used_sides), "used_minus": int("minus" in used_sides),
            })

    panel = pd.DataFrame(rows)
    qc_df = pd.DataFrame(task_qc_rows)
    return panel, qc_df, cf_stats

def run_regression_and_figs(panel, tag, out_dir_results, out_dir_figs):
    """
    Demeaned OLS: delta_dm ~ norm_dm, clustered by participant_id.
    Also write per-task slopes; histogram & scatter plots.
    """
    if panel.empty:
        print(f"[H2 {tag}] Panel empty; skipping regression.")
        return

    panel = panel.copy()
    panel["pid_task"] = panel["participant_id"].astype(str) + "·" + panel["taskType"].astype(str)

    # Drop rows with missing regression fields
    reg_df = panel.dropna(subset=["delta_b", "norm_step", "pid_task", "participant_id"]).copy()
    # Within (participant × task) demeaning
    reg_df["delta_dm"] = reg_df["delta_b"] - reg_df.groupby("pid_task")["delta_b"].transform("mean")
    reg_df["norm_dm"]  = reg_df["norm_step"] - reg_df.groupby("pid_task")["norm_step"].transform("mean")

    # Sanity: means ~ 0
    print(f"[H2 {tag}] mean(delta_dm)={reg_df['delta_dm'].mean():.3e}, mean(norm_dm)={reg_df['norm_dm'].mean():.3e}")

    # Pooled OLS with clustered SE by participant
    model = sm.OLS(reg_df["delta_dm"], sm.add_constant(reg_df[["norm_dm"]], has_constant="add"))
    res = model.fit(cov_type="cluster", cov_kwds={"groups": reg_df["participant_id"]})

    with open(out_dir_results / f"h2_pooled_regression_{tag}.txt", "w") as fh:
        fh.write(res.summary().as_text())
    print(f"[H2 {tag}] pooled slope = {res.params.get('norm_dm', np.nan):.3f} (SE={res.bse.get('norm_dm', np.nan):.3f})  N={len(reg_df)}")

    # Per-task slopes (demeaned within pid_task already)
    per_task = []
    for ttype, g in reg_df.groupby("taskType"):
        if g["norm_dm"].abs().sum() == 0 or len(g) < 10:
            continue
        r = sm.OLS(g["delta_dm"], sm.add_constant(g[["norm_dm"]], has_constant="add")).fit(
            cov_type="cluster", cov_kwds={"groups": g["participant_id"]}
        )
        per_task.append({
            "taskType": ttype,
            "slope": float(r.params.get("norm_dm", np.nan)),
            "se": float(r.bse.get("norm_dm", np.nan)),
            "z": float(r.tvalues.get("norm_dm", np.nan)),   # for OLS w/ cluster, t≈z
            "N": int(len(g))
        })
    per_task_df = pd.DataFrame(per_task)
    per_task_df.to_csv(out_dir_results / f"h2_per_task_slopes_{tag}.csv", index=False)

    # Individual slopes by pid_task
    indiv_rows = []
    for pid_task, g in reg_df.groupby("pid_task"):
        if g["norm_dm"].abs().sum() == 0 or len(g) < 3:
            continue
        r = sm.OLS(g["delta_dm"], sm.add_constant(g[["norm_dm"]], has_constant="add")).fit()
        indiv_rows.append({
            "pid_task": pid_task,
            "participant_id": g["participant_id"].iloc[0],
            "taskType": g["taskType"].iloc[0],
            "n_obs": int(len(g)),
            "sigma_hat": float(r.params.get("norm_dm", np.nan))
        })
    indiv_df = pd.DataFrame(indiv_rows)
    if not indiv_df.empty:
        indiv_df.to_csv(out_dir_results / f"h2_individual_slopes_{tag}.csv", index=False)

        # Histogram
        plt.figure(figsize=(6.8, 4.6))
        plt.hist(indiv_df["sigma_hat"].dropna(), bins=24)
        plt.xlabel("Individual updating slope (σ̂)")
        plt.ylabel("Count")
        plt.title(f"Distribution of σ̂ ({tag})")
        plt.tight_layout()
        plt.savefig(out_dir_figs / f"h2_hist_{tag}.pdf")
        plt.savefig(out_dir_figs / f"h2_hist_{tag}.png", dpi=300)
        plt.close()

    # Scatter: observed vs normative steps (sample up to 3000)
    sample = reg_df.sample(min(3000, len(reg_df)), random_state=42)
    plt.figure(figsize=(6.8, 4.6))
    plt.scatter(sample["norm_step"], sample["delta_b"], s=8, alpha=0.5)
    xs = np.linspace(sample["norm_step"].min(), sample["norm_step"].max(), 120)
    plt.plot(xs, xs, linewidth=2)  # σ = 1 reference
    plt.xlabel("Normative step")
    plt.ylabel("Observed change Δb")
    plt.title(f"Observed updates vs. normative steps (sample, {tag})")
    plt.tight_layout()
    plt.savefig(out_dir_figs / f"h2_scatter_{tag}.pdf")
    plt.savefig(out_dir_figs / f"h2_scatter_{tag}.png", dpi=300)
    plt.close()

    return res, per_task_df, indiv_df

def write_qc(panel, qc_df, cf_stats, tag, out_dir_results):
    panel_path = out_dir_results / f"h2_panel_{tag}.csv"
    qc_path = out_dir_results / f"h2_cf_qc_{tag}.csv"

    panel.to_csv(panel_path, index=False)
    qc_df.to_csv(qc_path, index=False)

    # Row-level missing diagnostic for regression fields
    miss_delta = int(panel["delta_b"].isna().sum()) if "delta_b" in panel.columns else 0
    miss_norm  = int(panel["norm_step"].isna().sum()) if "norm_step" in panel.columns else 0
    n_rows = len(panel)
    keep_rows = int(panel.dropna(subset=["delta_b","norm_step"]).shape[0]) if n_rows else 0

    print(f"[H2 {tag}] Panel rows={n_rows}; usable rows (Δb & norm_step present)={keep_rows}; "
          f"missing Δb={miss_delta}, missing norm_step={miss_norm}")

    # Task-level QC summary
    kept = cf_stats["kept"]
    print(f"[H2 {tag}] QC Summary:")
    for k, v in cf_stats.items():
        print(f"  {k:>26}: {v}")
    print("-"*72)

# -------------------------
# Run (strict / lenient / hybrid) × fallback S
# -------------------------
if __name__ == "__main__":
    sessions = load_json_any(INPUT)

    # Evaluate three policies; for hybrid, test multiple S fallbacks
    policies = ["strict", "lenient", "hybrid"]
    fallback_grid = {
        "strict": [None],     # no fallback used
        "lenient": [None],    # no fallback used
        "hybrid": [5, 10, 20] # sensitivity
    }

    for policy in policies:
        for S in fallback_grid[policy]:
            tag = policy if S is None else f"{policy}_S{S}"
            print(f"\n=== Building H2 panel with policy='{policy}'"
                  + ("" if S is None else f", fallback S={S}") + " ===")

            panel, qc_df, cf_stats = build_h2_panel_robust(
                sessions=sessions,
                policy=policy,
                fallback_S=(10.0 if S is None else float(S)),
                fallback_threshold=0.30,        # allow up to 30% fallback under hybrid
                reliability_delta_tau=0.10,     # single-side difference must exceed 10pp to count reliable
            )

            # Save QC + panel, print summary
            write_qc(panel, qc_df, cf_stats, tag, HYP_DIR)

            # Regression + figs
            run_regression_and_figs(panel, tag, MODELS_DIR, FIGURES_DIR)

    print("\n[H2 Robust] Done.")


# In[2]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

def bic_from_ols(y, X, k_extra=0):
    # X should already have a constant if you want an intercept
    res = sm.OLS(y, X).fit()
    n = len(y)
    k = len(res.params) + k_extra  # add any grid-search counted params (e.g., τ, ρ)
    bic = n * np.log(res.ssr / n) + k * np.log(n)
    return bic, res

def fit_M1_partial_bayes(g):
    y = g["delta_b"].values
    X = sm.add_constant(g[["norm_step"]].values)
    return bic_from_ols(y, X)

def fit_M2a_oneshot_t1(g):
    y = g["delta_b"].values
    t = g["t"].values
    spike = (t == 1).astype(float)
    X = sm.add_constant(spike.reshape(-1,1))
    return bic_from_ols(y, X)

def fit_M2b_changepoint(g):
    # grid τ=1..9; piecewise slopes (σ1 before/at τ, σ2 after τ)
    best = (np.inf, None, None)
    for tau in range(1,10):
        mask1 = (g["t"] <= tau)
        X = np.column_stack([
            g["norm_step"] * mask1,
            g["norm_step"] * (~mask1)
        ]).astype(float)
        X = sm.add_constant(X)
        bic, res = bic_from_ols(g["delta_b"].values, X, k_extra=1)  # count τ as a parameter
        if bic < best[0]:
            best = (bic, res, tau)
    return best  # (bic, res, tau)

def fit_M3_leaky(g, rhos=np.linspace(0,0.5,11)):
    best = (np.inf, None, None)
    y = g["delta_b"].values
    x = g["norm_step"].values
    t = g["t"].values
    for rho in rhos:
        z = x * np.exp(-rho*(t-1))
        X = sm.add_constant(z)
        bic, res = bic_from_ols(y, X, k_extra=1)  # count rho
        if bic < best[0]:
            best = (bic, res, rho)
    return best  # (bic, res, rho)

def fit_M4_asym(g):
    y = g["delta_b"].values
    x = g["norm_step"].values
    X = np.column_stack([np.maximum(x,0), np.minimum(x,0)])
    X = sm.add_constant(X)
    return bic_from_ols(y, X)

def fit_M5_recency(g):
    # running accuracy up to t-1
    y = g["delta_b"].values
    f = g["f_t"].values
    bprev = g["b_prev"].values
    t = g["t"].values
    accbar = []
    corrects = 0
    for ti, fi in zip(t, f):
        accbar.append(corrects/max(1,ti-1) if ti>1 else 0.0)
        corrects += fi
    accbar = np.array(accbar)
    X = sm.add_constant(accbar - bprev)
    return bic_from_ols(y, X)

def classify_update_style(panel_h2):
    out = []
    for (pid, task), g in panel_h2.groupby(["participant_id","taskType"]):
        g = g.dropna(subset=["delta_b","norm_step","t","f_t","b_prev"])
        if len(g) < 5:  # need enough trials
            continue
        # Fit all
        bic_M1, res_M1 = fit_M1_partial_bayes(g)
        bic_M2a, res_M2a = fit_M2a_oneshot_t1(g)
        bic_M2b, res_M2b, tau = fit_M2b_changepoint(g)
        bic_M3,  res_M3,  rho = fit_M3_leaky(g)
        bic_M4,  res_M4 = fit_M4_asym(g)
        bic_M5,  res_M5 = fit_M5_recency(g)

        bic_map = {
            ("M1_partial", None): bic_M1,
            ("M2a_t1spike", None): bic_M2a,
            ("M2b_changep", tau): bic_M2b,
            ("M3_leaky", rho): bic_M3,
            ("M4_asym", None): bic_M4,
            ("M5_recency", None): bic_M5,
        }
        (winner, aux), best_bic = min(bic_map.items(), key=lambda kv: kv[1])

        out.append({
            "participant_id": pid,
            "taskType": task,
            "n_trials": len(g),
            "winner": winner,
            "aux_param": aux,          # τ for M2b, ρ for M3; None otherwise
            "bic": best_bic,
            "sigma_M1": res_M1.params[1] if res_M1 is not None and len(res_M1.params)>1 else np.nan,
            "sigma_pos_M4": res_M4.params[1] if res_M4 is not None and len(res_M4.params)>2 else np.nan,
            "sigma_neg_M4": res_M4.params[2] if res_M4 is not None and len(res_M4.params)>2 else np.nan,
        })
    return pd.DataFrame(out)


# In[ ]:





# In[3]:


if __name__ == "__main__":
    classify_update_style(panel)

