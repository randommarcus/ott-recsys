"""Figures for the final report, made ONLY from results already saved in the repository. No model is trained, nothing is re-run, and the test
portion is never read (the split dates below come from the committed audit report and PROGRESS.md).

    python scripts/make_report_figures.py            # all figures -> reports/final/figures/
    python scripts/make_report_figures.py fig3_1     # one figure
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "final" / "figures"
BLUE, ORANGE, GREY, GREEN, RED = "#2f6db5", "#e08a2c", "#9a9a9a", "#3a9a5b", "#c0392b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def fig3_1():
    """Two splits. Calendar cutoffs from reports/module1_audit_ml-25m_u20000_s42.txt; per-user shares from the split settings."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 4.2), gridspec_kw={"height_ratios": [1, 1.25]})
    d = lambda s: datetime.strptime(s, "%Y-%m-%d")
    start, c1, c2, end = d("1996-01-29"), d("2015-02-03"), d("2017-03-28"), d("2019-11-21")
    for (a, b, color, label) in ((start, c1, BLUE, "train (up to 3 Feb 2015)"), (c1, c2, ORANGE, "validation"), (c2, end, GREY, "test (to 21 Nov 2019)")):
        ax1.barh(0, mdates.date2num(b) - mdates.date2num(a), left=mdates.date2num(a), color=color, height=0.5, label=label)
    ax1.set_yticks([])
    ax1.xaxis_date()
    ax1.set_title("A. One calendar cutoff for everyone (used for new releases and the bandit)", loc="left", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    ax1.set_ylim(-0.5, 1.0)
    users = [("user A", 0.0, 1.0), ("user B", 0.35, 0.55), ("user C", 0.7, 0.98)]
    for i, (name, a, b) in enumerate(users):
        span = b - a
        for (fa, fb, color) in ((0.0, 0.7, BLUE), (0.7, 0.8, ORANGE), (0.8, 1.0, GREY)):
            ax2.barh(i, span * (fb - fa), left=a + span * fa, color=color, height=0.5)
        ax2.text(a - 0.01, i, name, ha="right", va="center", fontsize=9)
    ax2.set_xlim(-0.12, 1.02)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlabel("each user's own rating history, oldest to newest (70% train, 10% validation, 20% test)")
    ax2.set_title("B. Per-user time split (used for personalization results)", loc="left", fontsize=10)
    ax2.invert_yaxis()
    fig.tight_layout()
    return fig


FIGURES = {"fig3_1": (fig3_1, "fig3_1_splits.png")}


def main(argv=None) -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    names = (argv if argv is not None else sys.argv[1:]) or list(FIGURES)
    for n in names:
        make, fname = FIGURES[n]
        fig = make()
        fig.savefig(FIG / fname, dpi=200)
        plt.close(fig)
        print(f"wrote reports/final/figures/{fname}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
