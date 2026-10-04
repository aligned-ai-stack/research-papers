"""Print paper-facing consistency facts from current extension outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analysis"
OUT = ANALYSIS / "data" / "followup_outputs"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f(value: str) -> float:
    return float(value)


def unique_count(data: list[dict[str, str]], *cols: str) -> int:
    return len({tuple(row[col] for col in cols) for row in data})


def main() -> None:
    with (ANALYSIS / "data" / "intermediate_outputs" / "valid_sessions.json").open(encoding="utf-8") as f_in:
        valid_sessions = json.load(f_in)
    print(f"valid_sessions={len(valid_sessions)}")

    trial = rows(OUT / "behavioral_extension_pipeline" / "trial_panel_with_updater_types.csv")
    print(f"trial_rows={len(trial)}")
    print(f"trial_participants={unique_count(trial, 'participant_id')}")
    print(f"trial_participant_tasks={unique_count(trial, 'participant_id', 'taskType')}")
    print(f"trial_tasks={dict(Counter(row['taskType'] for row in trial))}")
    zero = sum(abs(f(row["delta_belief"])) < 1e-12 for row in trial)
    lt5 = sum(abs(f(row["delta_belief"])) < 0.05 for row in trial)
    print(f"full_trial_zero_share={zero / len(trial):.6f} ({zero}/{len(trial)})")
    print(f"full_trial_abs_delta_lt_05_share={lt5 / len(trial):.6f} ({lt5}/{len(trial)})")

    task_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in trial:
        task_groups[(row["participant_id"], row["taskType"])].append(row)
    flat_tasks = sum(all(abs(f(row["delta_belief"])) < 1e-12 for row in group) for group in task_groups.values())
    print(f"full_panel_all_zero_tasks={flat_tasks}/{len(task_groups)} ({flat_tasks / len(task_groups):.6f})")

    h2 = rows(ANALYSIS / "data" / "hypothesis_outputs" / "H2" / "h2_panel_hybrid_S10.csv")
    h2_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in h2:
        h2_groups[(row["participant_id"], row["taskType"])].append(row)
    fallback = sum(abs(f(group[0]["n0"]) - 10.0) < 1e-12 for group in h2_groups.values())
    print(f"hybrid_s10_rows={len(h2)}")
    print(f"hybrid_s10_participant_tasks={len(h2_groups)}")
    print(f"hybrid_s10_fixed_s10_blocks={fallback}")
    print(f"hybrid_s10_empirical_n0_blocks={len(h2_groups) - fallback}")

    for name in [
        "classification_coverage.csv",
        "objective_accuracy_validation.csv",
        "task_level_delegation_ols.csv",
        "carryover_moderation_ols.csv",
    ]:
        print(f"\n[{name}]")
        for row in rows(OUT / "behavioral_extension_pipeline" / name):
            print(row)

    for name in [
        "canonical_bic_winner_counts.csv",
        "canonical_aicc_winner_counts.csv",
        "canonical_future_block_winner_counts.csv",
        "model_recovery_accuracy_summary.csv",
    ]:
        print(f"\n[{name}]")
        for row in rows(OUT / "robust_belief_model_comparison" / name):
            print(row)

    hmm = OUT / "markov_hmm_analysis"
    for name in [
        "model_comparison.csv",
        "state_parameters_conditioned_3state.csv",
        "feedback_transition_contrast_conditioned_3state.csv",
    ]:
        print(f"\n[{name}]")
        for row in rows(hmm / name):
            print(row)


if __name__ == "__main__":
    main()
