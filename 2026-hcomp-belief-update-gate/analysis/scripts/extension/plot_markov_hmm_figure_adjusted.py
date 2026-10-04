#!/usr/bin/env python
# coding: utf-8

"""Create a paper-ready HMM belief-delegation state figure from saved HMM outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ANALYSIS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ANALYSIS_ROOT.parent
HMM_OUT = ANALYSIS_ROOT / "data" / "followup_outputs" / "markov_hmm_analysis"
PAPER_IMG = REPO_ROOT / "paper" / "img"


COLORS = ["#4C78A8", "#59A14F", "#E15759"]
STATE_LABELS = ["Skeptical\nmonitoring", "Cautious\nreliance", "Consolidated\nbelief"]
SHORT_STATE_LABELS = ["Skeptical", "Cautious", "Consolidated"]

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_state_panel(ax, state_rows: list[dict[str, str]], contrast_rows: list[dict[str, str]]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    xs = [0.18, 0.50, 0.82]
    y = 0.52
    box_w = 0.23
    box_h = 0.24
    for idx, row in enumerate(state_rows):
        belief = float(row["belief_mean"])
        delegation = float(row["delegation_probability"])
        patch = FancyBboxPatch(
            (xs[idx] - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.2,
            edgecolor=COLORS[idx],
            facecolor=COLORS[idx],
            alpha=0.16,
        )
        ax.add_patch(patch)
        ax.text(xs[idx], y + 0.060, STATE_LABELS[idx], ha="center", va="center", fontsize=9.2, fontweight="bold", linespacing=0.9)
        ax.text(xs[idx], y - 0.025, f"belief {belief:.2f}", ha="center", va="center", fontsize=8.5)
        ax.text(xs[idx], y - 0.090, f"delegate {delegation:.2f}", ha="center", va="center", fontsize=8.5)

    c = {int(row["from_state"]): row for row in contrast_rows}
    up0 = float(c[0]["positive_minus_negative_up"])
    up1 = float(c[1]["positive_minus_negative_up"])
    down1 = float(c[1]["negative_minus_positive_down"])
    down2 = float(c[2]["negative_minus_positive_down"])

    def arrow(x0, x1, yy, label, color, rad):
        ax.add_patch(
            FancyArrowPatch(
                (x0, yy),
                (x1, yy),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
            )
        )
        ax.text((x0 + x1) / 2, yy + (0.075 if yy > y else -0.075), label, ha="center", va="center", fontsize=7.8)

    arrow(xs[0] + box_w / 2, xs[1] - box_w / 2, 0.77, f"+ feedback: to higher state +{up0:.2f}", "#2F7D32", -0.18)
    arrow(xs[1] + box_w / 2, xs[2] - box_w / 2, 0.77, f"+ feedback: to higher state +{up1:.2f}", "#2F7D32", -0.18)
    arrow(xs[1] - box_w / 2, xs[0] + box_w / 2, 0.27, f"- feedback: to lower state +{down1:.2f}", "#A23B3B", -0.18)
    arrow(xs[2] - box_w / 2, xs[1] + box_w / 2, 0.27, f"- feedback: to lower state +{down2:.2f}", "#A23B3B", -0.18)

    ax.text(0.0, 0.98, "A. Feedback-conditioned movement between latent states", fontsize=11, fontweight="bold", va="top")
    # ax.text(0.5, 0.055, "Note: arrows summarize movement to any higher/lower state, not only adjacent transitions.", ha="center", va="center", fontsize=7.4, color="#555555")


def make_parameter_panel(ax, state_rows: list[dict[str, str]]) -> None:
    x = np.arange(len(state_rows))
    belief = [float(row["belief_mean"]) for row in state_rows]
    delegation = [float(row["delegation_probability"]) for row in state_rows]
    width = 0.36
    ax.bar(x - width / 2, belief, width, color="#4C78A8", label="Mean belief")
    ax.bar(x + width / 2, delegation, width, color="#F28E2B", label="Delegation probability")
    ax.set_xticks(x, SHORT_STATE_LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probability")
    ax.set_title("B. Belief and delegation profile by state", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def make_occupancy_panel(ax, occupancy_rows: list[dict[str, str]]) -> None:
    tasks = [row["taskType"] for row in occupancy_rows]
    y = np.arange(len(tasks))
    left = np.zeros(len(tasks))
    for state in range(3):
        vals = np.asarray([float(row[f"mean_state_{state}_prob"]) for row in occupancy_rows])
        ax.barh(y, vals, left=left, color=COLORS[state], label=SHORT_STATE_LABELS[state])
        left += vals
    ax.set_yticks(y, tasks)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Posterior state occupancy")
    ax.set_title("C. State occupancy by task", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def main() -> None:
    PAPER_IMG.mkdir(parents=True, exist_ok=True)
    state_rows = read_rows(HMM_OUT / "state_parameters_conditioned_3state.csv")
    contrast_rows = read_rows(HMM_OUT / "feedback_transition_contrast_conditioned_3state.csv")
    occupancy_rows = read_rows(HMM_OUT / "state_occupancy_by_task_conditioned_3state.csv")

    fig = plt.figure(figsize=(7.0, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    make_state_panel(ax0, state_rows, contrast_rows)
    make_parameter_panel(ax1, state_rows)
    make_occupancy_panel(ax2, occupancy_rows)

    fig.suptitle("Hidden Markov model of belief–delegation states", fontsize=12.5, fontweight="bold")
    for ext in ["pdf", "png"]:
        path = PAPER_IMG / f"hmm_reliance_states.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(path)


if __name__ == "__main__":
    main()
