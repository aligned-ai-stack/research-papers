#!/usr/bin/env python
# coding: utf-8

"""Run the paper-facing extension analysis pipeline end to end.

This orchestrates the analysis pieces needed after the model-comparison
upgrade:

1. Regenerate H2 source panels for strict, lenient, and hybrid policies.
2. Fit the 18-model all-candidate behavioral model set on each source panel,
   while retaining model-scope labels that distinguish trajectory-generating
   models from conditional one-step robustness checks.
3. Run design-matched model recovery on the canonical hybrid_S10 panel.
4. Run H2/no-update diagnostics against robust classifications.
5. Rebuild downstream extension panels and regressions for each source panel.
6. Fit the sequential reliance-state HMM and an augmented confidence/correctness
   HMM on the canonical hybrid_S10 trial panel, unless explicitly skipped.
7. Write a compact manifest/summary for paper drafting.

The default intentionally uses the paper-facing all-candidate model set, AICc,
and leave-future-out predictive checks for every panel. It skips exhaustive
random-trial LOOCV, which is expensive for the large all-model grid. A separate
LOOCV audit can still be run from
robust_belief_model_comparison.py when needed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT_DIR = ROOT / "scripts"
H2_DIR = ROOT / "data" / "hypothesis_outputs" / "H2"
FOLLOWUP_DIR = ROOT / "data" / "followup_outputs"

SOURCE_PANELS = {
    "strict": H2_DIR / "h2_panel_strict.csv",
    "lenient": H2_DIR / "h2_panel_lenient.csv",
    "hybrid_S5": H2_DIR / "h2_panel_hybrid_S5.csv",
    "hybrid_S10": H2_DIR / "h2_panel_hybrid_S10.csv",
    "hybrid_S20": H2_DIR / "h2_panel_hybrid_S20.csv",
}


def run_step(cmd: list[str], cwd: Path = ROOT) -> None:
    print("\n[Extension pipeline] running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def robust_outdir(tag: str) -> Path:
    if tag == "hybrid_S10":
        return FOLLOWUP_DIR / "robust_belief_model_comparison"
    return FOLLOWUP_DIR / f"robust_belief_model_comparison_all_{tag}"


def extension_outdir(tag: str) -> Path:
    if tag == "hybrid_S10":
        return FOLLOWUP_DIR / "behavioral_extension_pipeline"
    return FOLLOWUP_DIR / "behavioral_extension_pipeline_by_panel" / tag


def diagnostics_outdir(tag: str, winner_col: str) -> Path:
    return FOLLOWUP_DIR / "h2_no_update_diagnostics_by_panel" / tag / winner_col


def write_run_summary(tags: list[str], out_path: Path, manifest: dict) -> None:
    rows = []
    for tag in tags:
        outdir = robust_outdir(tag)
        classification_path = outdir / "canonical_classification.csv"
        if not classification_path.exists():
            continue
        classification = pd.read_csv(classification_path)
        for winner_col in ["bic_winner", "aicc_winner", "future_block_winner"]:
            if winner_col not in classification.columns:
                continue
            counts = classification[winner_col].value_counts(dropna=False)
            for model, wins in counts.items():
                rows.append(
                    {
                        "source_panel": tag,
                        "winner_rule": winner_col.replace("_winner", ""),
                        "model": model,
                        "wins": int(wins),
                        "share": float(wins / len(classification)),
                        "participant_tasks": int(len(classification)),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_path / "model_winner_summary_by_panel.csv", index=False)
    (out_path / "extension_pipeline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-h2", action="store_true", help="Reuse existing H2 source panels.")
    parser.add_argument("--model-set", choices=["gamma", "lean", "primary", "conditional", "all"], default="all")
    parser.add_argument("--future-holdout-trials", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--skip-recovery", action="store_true")
    parser.add_argument("--skip-hmm", action="store_true", help="Skip the sequential reliance-state HMM analysis.")
    parser.add_argument("--recovery-reps-per-model", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--tags", default="strict,lenient,hybrid_S5,hybrid_S10,hybrid_S20")
    args = parser.parse_args()

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    unknown = [tag for tag in tags if tag not in SOURCE_PANELS]
    if unknown:
        raise ValueError(f"Unknown source panel tag(s): {unknown}")
    if not args.skip_hmm and "hybrid_S10" not in tags:
        raise ValueError("Sequential HMM analysis requires hybrid_S10 in --tags. Use --skip-hmm for panel-only reruns.")

    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "model_set": args.model_set,
        "future_holdout_trials": args.future_holdout_trials,
        "n_jobs": args.n_jobs,
        "recovery_reps_per_model": 0 if args.skip_recovery else args.recovery_reps_per_model,
        "run_hmm": not args.skip_hmm,
        "random_seed": args.random_seed,
        "source_panels": {tag: str(SOURCE_PANELS[tag]) for tag in tags},
        "outputs": {},
    }

    if not args.skip_h2:
        run_step([sys.executable, "H2.py"], cwd=BASE_SCRIPT_DIR)

    for tag in tags:
        panel = SOURCE_PANELS[tag]
        if not panel.exists():
            raise FileNotFoundError(f"Missing H2 source panel for {tag}: {panel}")
        outdir = robust_outdir(tag)
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "robust_belief_model_comparison.py"),
            "--panel",
            str(panel),
            "--outdir",
            str(outdir),
            "--model-set",
            args.model_set,
            "--skip-loocv",
            "--skip-panel-sensitivity",
            "--future-holdout-trials",
            str(args.future_holdout_trials),
            "--random-seed",
            str(args.random_seed),
            "--n-jobs",
            str(args.n_jobs),
        ]
        if tag == "hybrid_S10" and not args.skip_recovery:
            cmd.extend(["--run-recovery", "--recovery-reps-per-model", str(args.recovery_reps_per_model)])
        run_step(cmd)
        manifest["outputs"][f"robust_{tag}"] = str(outdir)

        for winner_col in ["aicc_winner", "bic_winner"]:
            diag_out = diagnostics_outdir(tag, winner_col)
            qc_tag = tag if not tag.startswith("hybrid_") else tag
            run_step(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "h2_no_update_diagnostics.py"),
                    "--panel",
                    str(panel),
                    "--qc",
                    str(H2_DIR / f"h2_cf_qc_{qc_tag}.csv"),
                    "--classification",
                    str(outdir / "canonical_classification.csv"),
                    "--winner-col",
                    winner_col,
                    "--outdir",
                    str(diag_out),
                ]
            )
            manifest["outputs"][f"diagnostics_{tag}_{winner_col}"] = str(diag_out)

        ext_out = extension_outdir(tag)
        run_step(
            [
                sys.executable,
                str(SCRIPT_DIR / "extension_behavioral_pipeline.py"),
                "--classification",
                str(outdir / "canonical_classification.csv"),
                "--outdir",
                str(ext_out),
            ]
        )
        manifest["outputs"][f"extension_{tag}"] = str(ext_out)

    hmm_input = extension_outdir("hybrid_S10") / "trial_panel_with_updater_types.csv"
    hmm_out = FOLLOWUP_DIR / "markov_hmm_analysis"
    if not args.skip_hmm:
        if not hmm_input.exists():
            raise FileNotFoundError(
                "Sequential HMM analysis requires the canonical hybrid_S10 trial panel. "
                f"Expected {hmm_input}. Include hybrid_S10 in --tags or rerun with --skip-hmm."
            )
        run_step(
            [
                sys.executable,
                str(SCRIPT_DIR / "markov_hmm_analysis.py"),
                "--trial-panel",
                str(hmm_input),
                "--outdir",
                str(hmm_out),
                "--random-seed",
                str(args.random_seed),
            ]
        )
        manifest["outputs"]["markov_hmm_analysis"] = str(hmm_out)
        augmented_hmm_out = FOLLOWUP_DIR / "markov_hmm_augmented_analysis"
        run_step(
            [
                sys.executable,
                str(SCRIPT_DIR / "markov_hmm_analysis.py"),
                "--trial-panel",
                str(hmm_input),
                "--outdir",
                str(augmented_hmm_out),
                "--emissions",
                "belief,delegation,confidence,self_correct",
                "--random-seed",
                str(args.random_seed),
            ]
        )
        manifest["outputs"]["markov_hmm_augmented_analysis"] = str(augmented_hmm_out)

    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    write_run_summary(tags, FOLLOWUP_DIR, manifest)
    print("\n[Extension pipeline] complete. Summary written to", FOLLOWUP_DIR / "model_winner_summary_by_panel.csv")


if __name__ == "__main__":
    main()
