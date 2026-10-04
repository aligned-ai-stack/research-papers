# The Belief Update Gate (HCOMP 2026)

Data and analysis for [Biswas, Erlei and Gadiraju (2026), “The Belief Update Gate: Separating Inertia from Learning in Human-AI Interaction”](https://doi.org/10.1145/3834580.3838740).

The study reanalyzes 240 participants, 720 participant-task blocks and 7,200 trials across grammar, travel and visual question answering. It separates whether a belief report changes from the size of change when it does. No visible movement does not establish an absence of latent learning.

## What can be uploaded

`dist/belief-update-gate-public-0.1.0.zip` contains analysis code, 22 aggregate result tables, study totals, a variable dictionary, citation and deposit metadata. It contains no participant records. Use this archive for the public code-and-summary deposit.

The local `review/` directory is excluded from both Git and the public archive. It contains a separate minimized reproduction dataset and historical results for access review. The detailed trajectories are **de-identified, not certified anonymous**. Do not upload the entire working folder. No repository deposit has been submitted and no dataset DOI has been assigned.

The public archive alone does not allow independent participant-level model refitting. Completing the thesis data deposit requires an agreed access/preservation route for those underlying records, or a documented reason for withholding them.

## Run

Python 3.12 is used for verification. `requirements-lock.txt` records the tested environment; it is not a reconstruction of the original authors' environment.

```sh
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements-lock.txt
python verify.py
python release.py --output dist/belief-update-gate-public-0.1.0.zip
```

`verify.py` checks the public tables without participant data. It does not refit models. The archive builder uses an explicit file list and checksums; adding a file to the working folder does not add it to the archive.

For the author's local review copy, or researchers given approved access:

```sh
python verify.py --reproduction review/reproduction
cd review/reproduction
python analysis/scripts/extension/paper_consistency_check.py
python analysis/scripts/extension/extension_behavioral_pipeline.py --outdir local/behavioral
python analysis/scripts/extension/update_gate_robustness.py --outdir local/update-gate
```

To refit all five benchmark panels and both HMM variants, work on a **copy** of `review/reproduction`, since the pipeline overwrites its output tables:

```sh
python analysis/scripts/extension/run_extension_analysis_pipeline.py --model-set all --n-jobs 4
```

This full grid is computationally expensive. Verification for this package reconstructs all five H2 panels, checks the behavioral panel, reruns the behavioral regressions and the 1,000-draw update-gate bootstrap, and checks HMM split integrity. It does not refit the entire trajectory-model or HMM grid. See `validation-summary.json` for measured agreement and limits.

## Files and interpretation

| Path | Contents |
| --- | --- |
| `analysis/scripts/H2.py` | Counterfactual-prior benchmark construction and pooled slopes |
| `analysis/scripts/extension/` | Trajectory model comparison, behavioral regressions, HMMs and robustness diagnostics |
| `analysis/data/followup_outputs/` | Selected aggregate outputs from the paper's analysis snapshot |
| `data/study-summary.json` | Cohort sizes and visible-update totals |
| `data/dictionary.json` | Core input fields, scales, missing values and derived variables |
| `deposit-metadata.json` | Preparation metadata, publication relation and remaining access decisions |
| `CITATION.cff` | Paper citation; the publication DOI is not a dataset DOI |

The minimized session data retain only task order, validity checks, counterfactual beliefs, trial-level beliefs/confidence, delegation, correctness and AI-feedback maps. Exact timestamps, database IDs, original UUIDs/worker IDs, demographics, questionnaires, free text, solutions and durations are removed. No re-identification key is distributed.

Raw belief and confidence ratings use 0–100; analysis panels use 0–1. Blank/null values mean unavailable, not zero. Each participant contributes 10 trials per task. The hybrid benchmark retains 656 participant-task blocks, including 216 fallback-prior blocks; 64 blocks remain in the 720-block behavioral panel without a trajectory-model label. AICc, BIC and future-block winners are different criteria and must not be interchanged.

The 0.494 overall and 0.949 moving-row slopes describe different subsets. The latter conditions on observed movement and is not evidence that participants are nearly Bayesian. Model labels describe trajectory motifs, not permanent participant types. Participant-clustered errors and participant-level bootstraps account for repeated observations.

## Reproducibility corrections and historical outputs

The current source JSON had been reordered after the saved H2 analysis. Its running fallback cap makes order matter: a naïve rerun yields 652 hybrid blocks instead of 656. The reproduction copy restores the order from all five saved QC tables, independently confirmed against the older valid-session backup. Preserve this order when reproducing the published benchmark. This is a feature of the historical allocation rule, not a newly randomized analysis.

Original UUIDs are replaced consistently with `participant_id` values P0001–P0240. Their ordering preserves the original pre-UUID worker-ID sort order, recovering the published seeded bootstrap draws. Sorting the later UUIDs produces different draws and intervals despite identical point estimates. The original identifiers are not distributed, but the detailed copy retains potentially linkable ordering and trajectories. P0241–P0243 occur only in historical sensitivity outputs. Four stored behavioral sensitivity folders used 243 participants; rerunning the supplied pipeline uses the final 240. These historical tables are labelled in the local audit and must not be represented as newly verified 240-participant results.

The deposit code changes the identifier field name, makes H2 input paths relative to the script, and guards a leftover notebook-only call so importing H2 works. Model definitions and statistical formulas are preserved. Notebooks, caches, slides, manuscript build artifacts and Git history are excluded. The original study's web application/stimulus assets are not part of this extension package; the published methods identify the reused study. Archiving that original experiment is a separate dataset/software record to link to this one.

## Deposit and permissions

The author confirms coverage by the existing DMP 175515, “Multitask Choice Independence and Bayesian Rationality” (last modified 8 July 2025). It specifies 4TU.ResearchData, CC BY for shared data, MIT for code, and Ujwal Gadiraju as the post-project data custodian. No new DMP is created here.

The supplied consent document permits anonymized answers to be published. It describes a loan task, while this dataset and the application consent configuration use travel. Preserve the appropriate deployed consent/version evidence for curator review. The DMP and consent do not by themselves certify that detailed trajectories cannot be linked back to people.

Code uses MIT; the public aggregate data use CC BY 4.0. See `LICENSE-CODE` and `LICENSE-DATA`. These licences do not authorize redistribution of the local detailed review data or third-party assets.

TU Delft requires thesis-supporting data/code to be archived before graduation for PhD candidates starting on or after 1 January 2019, subject to justified exceptions. The supervisor should check archiving before Form B. GitHub alone is insufficient; use 4TU.ResearchData or a suitable DOI-granting archive and register an external archive DOI in Pure. See the official [archiving checklist](https://phdsupervisors.tudl.tudelft.nl/knowledge-base/research-data-archiving-checklist/) and [publishing guidance](https://phdsupervisors.tudl.tudelft.nl/wp-content/uploads/sites/30/2024/10/Publishing-requirements-for-data-and-code-for-TU-Delft-PhD-candidates.pdf), checked 4 October 2026.

Before final deposit, resolve the detailed-data access decision and consent-version evidence, confirm dataset creators/contact and subject category, then add the assigned dataset DOI. [4TU supports restricted access](https://data.4tu.nl/info/fileadmin/user_upload/Documenten/4TU.ResearchData_Restricted_Data_2022.pdf), subject to repository acceptance and appropriate access terms. Keep confidential source data and linkage keys in approved institutional storage.
