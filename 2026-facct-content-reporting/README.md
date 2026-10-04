# AI at the Front Lines of Platform Governance (FAccT 2026)

Aggregate data and executable analyses for [Sekwenz, Biswas, Hermann-Gsenger and Gadiraju (2026)](https://doi.org/10.1145/3805689.3812301), *AI at the Front Lines of Platform Governance: Using LLMs to Support Illegal Content Reporting under the Digital Services Act*.

The study compares unaided reporting, conventional explainable AI (XAI), and evaluative AI (EvalAI). The analysis cohort contains 450 participants, 150 per condition, and 900 illegal-content reporting trials. On AI error trials, provision accuracy is 74% for EvalAI and 46% for XAI. Explanation-quality improvements were not established.

This package preserves reported specifications and labels companion analyses explicitly. The original project and manuscript are unchanged. Public files contain aggregates and model results; participant records remain excluded.

## Run

Verified with Python 3.12. From this directory:

```sh
python -m venv .venv
# Activate the environment using your shell, then:
python -m pip install -r requirements.txt
python plots.py --data data --output local/figures
python -m unittest discover -s tests -v
python release.py --output dist/facct-content-reporting-0.1.0.zip
```

Public figures regenerate without private data. To rerun the six models with approved participant inputs:

```sh
python run.py --data review/data --output local/new-run
```

`review/data` exists only in the author's local copy. The runner requires a new output directory and validates trial uniqueness, conditions, pairing, ranges and coder means before fitting. Model output and logs can contain participant-level diagnostics; keep them local. The public data cannot be used to reconstruct or fully refit individual-level models.

## Files and inputs

| File | Contents |
| --- | --- |
| `run.py`, `analysis/H1.py`–`H6.py` | Validated entry point and six model workflows |
| `plots.py` | Public aggregate figure regeneration |
| `data/accuracy.csv`, `distance.csv`, `explanation_quality.csv` | Group summaries, ordinal counts and coder-mean summaries |
| `data/models/` | Aggregate coefficients, contrasts, intervals and sensitivity results |
| `data/input_dictionary.csv`, `table_dictionary.csv` | Input schema and exported table fields |
| `data/provenance.json` | Sources, cohort, runtime and verification limits |
| `deposit-metadata.json` | Portable metadata draft for 4TU.ResearchData |
| `release.py` | Explicit public-file allowlist with SHA-256 archive manifest |

The local input bundle has two identical minimized trial files (`final_analysis_0301.csv` and `final_analysis_0301_cleaned.csv`) to support the historical script interfaces; `combined_coder_ratings.csv` joins one-to-one by trial `id`; `final_survey_0301.csv` contains pre-task scale responses only; `final_sessions_0301.csv` supplies randomized order. Participant and trial IDs are opaque replacements. Survey responses were reverse-coded by the original interface and must not be reversed again. Historical covariate averages include attention-check items; those items are constant among the retained participants, and are retained here for reproduction.

Manual trials sometimes carry `isErrorTrial=True` in the source export, but manual participants received no AI error manipulation. H1, H2, H3, H5 and H6 filter to AI arms before interpreting that flag. H4 uses all rated reports across all three conditions.

## Reproduction and documented discrepancies

The six model scripts ran successfully from a minimized, re-keyed and shuffled local copy. Main treatment coefficients match the source-script reruns within numerical tolerance. All 900 trials join uniquely to complete coder ratings. The `_cleaned` source file changes identifier representation, not the observations or analysis fields.

| Analysis | Reproduction and release treatment |
| --- | --- |
| H1: accuracy under error | The paper's intervals and p-values match the participant-clustered model (`H1_robustA_GLM_cluster_tidy.csv`), despite the text describing HC1. Both tables are retained. The reference-condition EvalAI odds ratio is 10.96; it is not a pooled odds ratio across error types. |
| H2: misclassification distance | The original helper skips robust covariance when every participant appears once, so the reported estimates use model-based standard errors. These remain in the primary table. `H2a_companion_participant_robust.csv` adds the advertised covariance explicitly. The EvalAI × out-of-scope p-value changes from .00736 to .00982. No substantive predictor crosses .05; the first threshold parameter does. |
| H3: correct-AI trials | The paper's timing coefficient .13204 and p=.06409 reproduce with 1%/99% winsorization, legal-category controls, HC1 covariance and t-reference inference. The newer raw-log-time model without case controls gives .14341, p=.04580; raw times with case controls give p=.05775. All three are labelled in `H3b_timing_specifications.csv`. The published Mann–Whitney result also reproduces (U=10012, one-sided p≈.0498). The timing claim is specification-sensitive. |
| H4: explanation quality | Participant-clustered OLS on two-coder means reproduces the reported null contrasts. Individual ordinal ratings are averaged as in the paper. |
| H5: error penalties | Paired accuracy logit, participant first-difference explanation models and discretized ordered-logit sensitivity outputs are retained. Source rounding and model specifications are documented by the executable code. |
| H6: downstream risk | Overbreadth regression is retained. Out-of-scope outcomes are 50/50 misrouted in EvalAI and 42/44 in XAI. The old guard tested only pooled variance and incorrectly fitted a separated regression. The guard now checks each condition; out-of-scope results are descriptive-only, matching the paper. |

The analysis modules retain the source filenames' distinction between `primary` and `robust` outputs; the table above identifies the actual paper-matching files. Runtime compatibility repairs replace NumPy array-to-scalar casts, externalize input/output paths, and make warnings visible. They do not alter observations or model formulas. Figures in `figures/` are regenerated from public tables. Refitting also produces local diagnostic plots.

Inference is based on the model specifications above; p-values across hypotheses are not a familywise error correction. The experimental task and measurements are research instruments, not legal advice or a validated compliance service.

## Provenance, privacy and deposit

Comparison used the [public author manuscript carrying the ACM DOI](https://arxiv.org/html/2605.23676v1). Direct ACM and Overleaf access was unavailable; local PDFs were earlier anonymous drafts. The raw database export and the full original 489-to-450 exclusion sequence have not been independently reconstructed. This is a reproduction from the final analysis cohort, not a certification of the original collection pipeline or an end-to-end deployment of the study app.

The public release excludes participant IDs, item-response vectors, exact timings per person, free-text reports, demographic records, payment data, database exports and credentials. The local input copy is minimized and de-identified; it is not certified anonymous. Do not force-add `review/` or `local/` or publish the entire working directory. Release archives include only the explicit allowlist.

Aggregate data and figures use **CC BY 4.0**. Code and software documentation use **MIT**. See `LICENSE-DATA`, `LICENSE-CODE` and `CITATION.cff`. Neither the full manuscript nor third-party stimulus and questionnaire wording is redistributed.

No 4TU deposit has been submitted. Confirm dataset creators/contact, choose the repository category and supply the unsigned consent form for deposit review. Keep the paper DOI as a related publication identifier; the dataset receives its own DOI. This metadata is a preparation draft, not a repository-specific submission payload.
