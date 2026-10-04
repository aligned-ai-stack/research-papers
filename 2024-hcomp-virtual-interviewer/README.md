# Virtual interviewer study (HCOMP 2024)

Aggregate data and executable analysis for [Biswas et al. (2024), “Hi. I’m Molly, Your Virtual Interviewer!”](https://doi.org/10.1609/hcomp.v12i1.31596). The study examines participant and virtual-interviewer demographic differences in perceived fairness, social perception, privacy/creepiness and impression management.

This package preserves the published scoring as `historical` and provides an explicitly labelled `corrected` companion. It contains 218-participant aggregate results. It does not contain participant records. The original project and paper are unchanged.

## Run

Python 3.12 was used for verification. From this directory:

```sh
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -r requirements.txt
python analysis.py plots --tables data --output local/figures
python -m unittest discover -s tests -v
python release.py --output dist/virtual-interviewer-0.1.0.zip
```

Plot generation works entirely from the included aggregate tables. Full statistical refitting requires approved participant-level inputs, which are not in the public release. On the author's local review copy:

```sh
python analysis.py run --participants review/data/analysis_deidentified.csv --items review/data/survey_items_deidentified.csv --profile historical --output local/historical
python analysis.py run --participants review/data/analysis_deidentified.csv --items review/data/survey_items_deidentified.csv --profile corrected --output local/corrected
```

Each run requires a new output directory and writes participant scores there. Keep these outputs local. `study_id` must uniquely link the two input tables; input item responses must be numeric, complete and pre-reversal. No rows are silently removed. The script uses participant input order for the bootstrap.

## Contents

| Path | Contents |
| --- | --- |
| `analysis.py` | Item validation, scoring, Kruskal–Wallis, Games–Howell, mediation and plots |
| `data/descriptives.csv` | Group sizes, means and sample SDs by participant or interviewer group, separately |
| `data/kruskal.csv` | Unadjusted omnibus tests for four outcomes and both demographic factors |
| `data/posthoc.csv` | Games–Howell pairwise comparisons following significant participant-group omnibus tests |
| `data/mediation.csv` | Frozen source-order model estimates and bootstrap intervals for both scoring profiles |
| `data/*dictionary.csv` | Variables, item positions, reverse keys and table fields |
| `data/provenance.json` | Cohort, profile definitions and reconstruction evidence |
| `figures/` | Repaired group-mean and indirect-effect figures, PNG and SVG |
| `deposit-metadata.json` | Portable 4TU preparation metadata; not a submitted repository record |
| `release.py` | Builds an archive using an explicit public-file allowlist and SHA-256 manifest |

`review/` contains local audit history and de-identified participant data. `review/`, `local/`, environments and generated archives are ignored by Git and excluded from the release allowlist. Do not upload the entire working folder or force-add ignored files.

## Scoring and corrections

The original cohort starts with 236 records: 12 fail the attention check and six further records fall outside the four participant demographic groups used by the paper. This historical restriction is retained, not newly recommended. The analysis cohort has 218 participants. MASI source fields are post-survey; the paper's timing descriptions are inconsistent.

SPP remains the equal-weight mean of the warmth, competence and social-presence component means. The implementation and published group means support this definition. It is not changed to an equal-weight average of all eight items.

| Measure | Historical | Corrected |
| --- | --- | --- |
| ATI | Nine-item mean divided by nine again | Nine-item mean on the 1–6 scale |
| Privacy/creepiness (PER; column `privacy_concerns`) | Ten-item mean, safe-storage item unreversed | Reverse safe-storage item before the ten-item mean |
| Other measures, cohort and models | Original definitions | Unchanged |

ATI reverses items 3, 6 and 8; MASI reverses items 4 and 6 of the 30-item export. The privacy correction reverses `privacy_concerns_item_10` using `8 - response`. The reverse key is documented in [Langer's original instrument, Table 6-3, printed p. 113](https://publikationen.sulb.uni-saarland.de/bitstream/20.500.11880/27165/3/Diss_ML_Ende_deutschAbstract.pdf#page=113).

These corrections change values: ATI mean 0.461207 → 4.150866 and PER mean 3.432569 → 3.365596. In the audited source-order fits, none of the 60 mediation-path p-values crosses the .05 threshold after these corrections. This does not mean the estimates are identical. The male-Black PER indirect coefficient changes from −0.309923 to −0.341331. The historical female-Black PER indirect interval barely excludes zero despite bootstrap p=.072; the corrected interval includes zero. Preserve both intervals and p-values rather than claiming perfect agreement between these decision rules.

The corrected PER mean is computed directly from the integer item responses. An earlier audit adjusted an already-computed mean; floating-point differences split equal ranks and yielded slightly different Kruskal–Wallis results. Direct item scoring gives participant-group p=.936584 and interviewer-group p=.865211, replacing the preliminary .938648 and .875791. Both remain nonsignificant. The release tables use direct item scoring.

The plots derive means directly from tables, fixing the reversed male-group impression-management values in the original plotting script. Interval endpoints are drawn directly, preserving asymmetric bootstrap intervals. Mediation uses integer dummy variables, avoiding the original boolean-dummy runtime issue. The clean workflow does not execute the old notebooks or use `eval`.

## Statistical interpretation and reproducibility

Kruskal–Wallis tests are unadjusted; they do not control for age, ATI or MASI. Games–Howell p-values adjust pairwise comparisons within an outcome, not across all outcomes. Mediation uses one participant group versus everyone else, SPP as mediator, age/ATI/MASI as covariates, 500 bootstrap samples and seed 42. The four group comparisons are separate overlapping models, not coefficients from one jointly dummy-coded model. Mediation is an associational analysis, not causal identification.

The frozen historical estimates and interval endpoints reproduced the original stored mediation results within 4e-13 in the original row order. The local de-identified export deliberately shuffles participants. Refit coefficients agree to numerical precision, but identical bootstrap seeds sample row positions, so refitting shuffled rows changes intervals and bootstrap p-values slightly. The largest audited p-value difference from shuffling was .012; none of the indirect p<.05 classifications changed. Aggregate files retain the source-order reference results and do not reveal the original participant order. Exact participant-level refitting is consequently not possible from this public aggregate-only package.

A bootstrap p-value of zero in the table is a finite Monte Carlo estimate from 500 resamples, not a true zero probability. State the resample count when reporting it; additional resamples are needed for greater precision. The original article contains mediation-summary wording that needs checking against signed coefficients; these tables are the numerical reference. No paper text has been silently rewritten.

## Data access, citation and deposit

Public tables contain separate marginal group summaries and model estimates, with no IDs, exact ages, free text or item-response vectors. Direct identifiers were removed from the separate local analysis copy, but its exact ages and demographic combinations remain potentially identifying. That copy is de-identified, not certified anonymous, and is not covered by this release's data licence.

Cite the paper using `CITATION.cff`. Aggregate data, dictionaries and derived figures use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); code and software documentation use MIT. See `LICENSE-DATA` and `LICENSE-CODE`. The paper and third-party questionnaire wording are not redistributed here.

The release ZIP is portable to GitHub or a 4TU.ResearchData draft. No 4TU deposit has been submitted. Before deposit, confirm dataset creators/contact and subject category and supply the unsigned participant consent form requested by [4TU's metadata review guidance](https://data.4tu.nl/s/documents/Metadata_review_guidelines_June_2021.pdf). A dataset DOI will be assigned by the repository; the related paper DOI must not be used as the dataset DOI. Each future paper should have its own self-contained folder, licences, citation, provenance and release archive within this collection.
