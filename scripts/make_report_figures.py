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


# ---------------------------------------------------------------------------------------------------- saved results
import json
import re

TEST = ROOT / "reports" / "module8_test_results.txt"
LAT = ROOT / "reports" / "module9_latency_results.json"
# Level study (Module 9, checkpoint 3): run by the project owner on 2026-10-27; numbers as recorded in PROGRESS.md (that table is the only record).
LEVEL_STUDY = [(1, 87.0, 86.7, 89.9), (3, 87.0, 88.0, 89.9), (5, 91.8, 92.6, 92.9), (7, 91.8, 93.3, 92.9), (10, 91.8, 93.4, 92.9)]


def test_rows(section_start: str, n_cols: int) -> dict:
    """Rows 'name  mean [lo-hi] ...' of a block in the saved test report (first n_cols intervals per row)."""
    text = TEST.read_text()
    block = text[text.index(section_start):]
    block = block[:block.index("\n\n")] if "\n\n" in block else block
    out = {}
    for line in block.splitlines()[2:]:
        nums = re.findall(r"([-\d.]+) \[([-\d.]+)-([-\d.]+)\]", line)
        if len(nums) >= n_cols:
            name = re.split(r"\s{2,}", line.strip())[0]
            out[name] = [tuple(float(x) for x in t) for t in nums[:n_cols]]
    return out


def headline() -> dict:
    out = {}
    for line in TEST.read_text().splitlines():
        m = re.match(r"^(P\d[ab]?)\s.*?:\s+([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", line)
        if m:
            out[m.group(1)] = tuple(float(m.group(i)) for i in (2, 3, 4))
    return out


def box(ax, x, y, w, h, text, color, fs=8.5):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="#444444", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#444444", lw=1))


def fig4_1():
    """System architecture: offline (laptop, TensorFlow) and online (AWS Lambda) parts."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.1, 5.75, "OFFLINE (laptop, TensorFlow allowed)", fontsize=10, fontweight="bold")
    ax.text(5.3, 5.75, "ONLINE (AWS eu-central-1, numpy only)", fontsize=10, fontweight="bold")
    ax.plot([5.1, 5.1], [0.2, 5.9], color=GREY, ls="--", lw=1)
    lb, ln, lg = "#dce8f7", "#fde9d4", "#e3f1e6"
    box(ax, 0.1, 4.6, 2.2, 0.8, "MovieLens 25M sample\n+ TMDB descriptions", lb)
    box(ax, 2.6, 4.6, 2.3, 0.8, "two-tower (TF/Keras)\nSBERT embeddings", lb)
    box(ax, 0.1, 3.3, 2.2, 0.8, "NCF (TF/Keras)\nfold-in for new history", lb)
    box(ax, 2.6, 3.3, 2.3, 0.8, "XGBoost ranker M2\n(and synthetic M3b)", lb)
    box(ax, 0.1, 1.9, 4.8, 0.9, "export: item vectors, ranker trees as numpy arrays,\n230-slot candidate lists and user records", lb)
    arrow(ax, 2.3, 5.0, 2.6, 5.0)
    arrow(ax, 1.2, 4.6, 1.2, 4.1)
    arrow(ax, 3.75, 4.6, 3.75, 4.1)
    arrow(ax, 2.5, 3.3, 2.5, 2.8)
    box(ax, 5.4, 4.6, 2.0, 0.8, "client (laptop)\nIAM-signed request", lg)
    box(ax, 7.8, 4.4, 2.1, 1.2, "Lambda function\nfeatures + M2 trees\n+ UCB bandit", ln)
    box(ax, 5.4, 2.9, 2.0, 0.9, "DynamoDB\nott-users, ott-bandit", ln)
    box(ax, 7.8, 2.9, 2.1, 0.9, "Zilliz (Milvus) vector DB\n(live mode only)", ln)
    box(ax, 5.4, 1.5, 2.0, 0.9, "SSM parameter\n(Zilliz key, encrypted)", ln)
    box(ax, 7.8, 1.5, 2.1, 0.9, "CloudWatch logs\n(7-day retention)", ln)
    arrow(ax, 7.4, 5.0, 7.8, 5.0)
    arrow(ax, 8.2, 4.4, 6.8, 3.8)
    arrow(ax, 8.85, 4.4, 8.85, 3.8)
    arrow(ax, 8.0, 4.4, 6.9, 2.4)
    arrow(ax, 9.4, 4.4, 9.4, 2.4)
    arrow(ax, 4.9, 2.35, 5.4, 3.2)
    ax.text(5.15, 0.55, "Precomputed mode reads the stored 230 candidates from DynamoDB; live mode builds them with two Zilliz searches.\n"
            "Both modes rank with the same M2 trees and fill 2 of 20 slots with new releases chosen by UCB (c = 0.2).", ha="center", fontsize=8.5)
    fig.tight_layout()
    return fig


def fig4_2():
    """From catalog to screen."""
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    stages = [("whole catalog", 33392, BLUE), ("candidates: 200 two-tower\n+ 30 new-release lane", 230, ORANGE), ("shown: 18 ranked by M2\n+ 2 chosen by UCB", 20, GREEN)]
    for i, (label, n, color) in enumerate(stages):
        ax.barh(i, n, color=color, height=0.55, log=True)
        ax.text(n * 1.25, i, f"{n:,}  {label}", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(10, 10 ** 7)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("number of movies (log scale)")
    fig.tight_layout()
    return fig


def fig5_1():
    """Headline results P1-P5 with 95% intervals (saved test report)."""
    h = headline()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    rows = [("P1  M2 ranker vs two-tower order\n(NDCG@10)", "P1"), ("P2  M0 ranker vs two-tower order\n(NDCG@10)", "P2"),
            ("P3  context gain, M2 vs M0\n(NDCG@10)", "P3"), ("P4  two-tower vs item-kNN\n(Recall@200)", "P4")]
    for i, (label, k) in enumerate(rows):
        m, lo, hi = h[k]
        a1.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=BLUE, capsize=4)
        a1.text(hi + 0.002, i, f"{m:+.4f}", va="center", fontsize=8.5)
    a1.axvline(0, color=GREY, lw=1)
    a1.set_yticks(range(len(rows)))
    a1.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    a1.invert_yaxis()
    a1.set_xlim(-0.005, 0.062)
    a1.set_xlabel("difference (95% bootstrap interval)")
    a1.set_title("Test set, 19,530 users", loc="left", fontsize=10)
    rows2 = [("P5a  UCB c=0.2\nvs UCB1", "P5a"), ("P5b  UCB c=0.2\nvs Thompson", "P5b")]
    for i, (label, k) in enumerate(rows2):
        m, lo, hi = h[k]
        a2.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=ORANGE, capsize=4)
        a2.text(hi + 6, i, f"{m:+.1f}", va="center", fontsize=8.5)
    a2.axvline(0, color=GREY, lw=1)
    a2.set_yticks(range(2))
    a2.set_yticklabels([r[0] for r in rows2], fontsize=8.5)
    a2.invert_yaxis()
    a2.set_ylim(1.6, -0.6)
    a2.set_xlim(-20, 300)
    a2.set_xlabel("likes per 1,000 impressions")
    a2.set_title("SIMULATION, not an A/B test", loc="left", fontsize=10, color=RED)
    fig.tight_layout()
    return fig


def fig5_2():
    """Re-ranking the 230-slot lists on the test set: end-to-end NDCG@10 (secondary table of the saved test report)."""
    r = test_rows("== RE-RANKING THE 230-SLOT LISTS", 1)
    names = [("random", "random"), ("pop (popularity order, train+val)", "popularity"), ("NCF order, fold-in (primary)", "NCF"),
             ("tt (two-tower's own order)", "two-tower order"), ("50/50 blend tt+NCF fold-in", "50/50 blend"), ("M0 ranker, no context (fold-in NCF)", "M0 ranker"),
             ("M1 + hour/day (fold-in)", "M1 (+hour, day)"), ("M2 + session (fold-in NCF) [primary]", "M2 (+session)")]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for i, (key, label) in enumerate(names):
        m, lo, hi = r[key][0]
        color = GREEN if label.startswith("M") else (BLUE if label in ("two-tower order", "50/50 blend") else GREY)
        ax.barh(i, m, color=color, height=0.6)
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="none", color="#333333", capsize=3)
        ax.text(hi + 0.002, i, f"{m:.4f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[1] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("end-to-end NDCG@10 on the test set (95% interval)")
    ax.set_xlim(0, 0.2)
    fig.tight_layout()
    return fig


def fig5_3():
    """Retrieval over the whole catalog on the test set: Recall@200, warm and cold."""
    text = TEST.read_text()
    block = text[text.index("== RETRIEVAL OVER THE WHOLE CATALOG"):].splitlines()[2:6]
    names, all200, cold200 = [], [], []
    for line in block:
        parts = re.split(r"\s{2,}", line.strip())
        names.append(parts[0])
        vals = [float(x) for x in parts[1:7]]
        all200.append(vals[2])
        cold200.append(vals[5])
    fig, ax = plt.subplots(figsize=(8, 3.0))
    y = range(len(names))
    ax.barh([i - 0.18 for i in y], all200, height=0.35, color=BLUE, label="all test likes")
    ax.barh([i + 0.18 for i in y], cold200, height=0.35, color=ORANGE, label="likes on movies with no training likes (cold)")
    for i in y:
        ax.text(all200[i] + 0.01, i - 0.18, f"{all200[i]:.3f}", va="center", fontsize=8.5)
        ax.text(cold200[i] + 0.01, i + 0.18, f"{cold200[i]:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("Recall@200 over the whole catalog, test set")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def fig5_4():
    """Bandit simulation on the test window, 629-arm set (SIMULATION, NOT AN A/B TEST)."""
    text = TEST.read_text()
    block = text[text.index("-- 629 arms"):]
    block = block[:block.index("oracle")]
    rows = []
    for line in block.splitlines()[2:]:
        m = re.match(r"^(.+?)\s{2,}([\d.]+) \[([\d.]+)-([\d.]+)\]", line)
        if m:
            rows.append((m.group(1).strip(), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    oracle = float(re.search(r"oracle \(expected\)\s+([\d.]+)", text[text.index("-- 629 arms"):]).group(1))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for i, (name, m, lo, hi) in enumerate(rows):
        ax.barh(i, m, color=ORANGE if "0.2" in name else GREY, height=0.6)
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="none", color="#333333", capsize=3)
        ax.text(hi + 8, i, f"{m:.0f}", va="center", fontsize=8.5)
    ax.axvline(oracle, color=GREEN, ls="--", lw=1)
    ax.text(oracle + 5, len(rows) - 0.6, f"oracle {oracle:.0f}", color=GREEN, fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0].replace(" (pre-registered)", " (pre-registered)") for r in rows], fontsize=8.5)
    ax.set_xlim(0, 950)
    ax.set_xlabel("simulated likes per 1,000 impressions (95% interval over 30 seeds)")
    n_arms = re.search(r"-- 629 arms \(>=5 ratings\) \((\d+) arms\)", text).group(1)
    ax.set_title(f"SIMULATION, NOT AN A/B TEST: {n_arms} new releases with at least 5 ratings, 43,064 real session starts", loc="left", fontsize=9.5, color=RED)
    fig.tight_layout()
    return fig


def fig5_5():
    """Warm server time by stage, from the saved latency results."""
    r = json.loads(LAT.read_text())
    stages = ["user_read", "search", "features", "ranker", "bandit"]
    labels = {"user_read": "DynamoDB read", "search": "Zilliz search", "features": "features", "ranker": "ranker (351 trees)", "bandit": "bandit"}
    colors = [BLUE, ORANGE, GREEN, RED, GREY]
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    for i, mode in enumerate(["precomputed", "live"]):
        left = 0.0
        for s, c in zip(stages, colors):
            v = r["warm"][mode]["stage_median_ms"][s]
            ax.barh(i, v, left=left, color=c, height=0.5, label=labels[s] if i == 0 else None)
            left += v
        p50 = r["warm"][mode]["server_ms"]["p50"]
        ax.text(left + 2, i, f"stage medians sum {left:.0f} ms; server median {p50:.1f} ms", va="center", fontsize=8.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["precomputed", "live"])
    ax.invert_yaxis()
    ax.set_xlim(0, 210)
    ax.set_xlabel("milliseconds (median per stage, 200 warm requests per mode)")
    ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.38))
    fig.tight_layout()
    return fig


def fig5_6():
    """Zilliz search level study (100 users): recall of the exact candidates and top-18 agreement."""
    lv = [r[0] for r in LEVEL_STUDY]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.plot(lv, [r[1] for r in LEVEL_STUDY], "o-", color=BLUE, label="main-lane recall of the exact 200 candidates")
    ax.plot(lv, [r[2] for r in LEVEL_STUDY], "s-", color=ORANGE, label="new-release lane recall")
    ax.plot(lv, [r[3] for r in LEVEL_STUDY], "^-", color=GREEN, label="top-18 shown to the user, live vs precomputed")
    ax.set_xticks(lv)
    ax.set_xlabel("Zilliz AUTOINDEX search level")
    ax.set_ylabel("%")
    ax.set_ylim(80, 100)
    ax.axvline(5, color=GREY, ls="--", lw=1)
    ax.text(5.15, 81, "chosen: level 5", fontsize=8.5, color="#555555")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


DEMO = ROOT / "reports" / "module9_demo.html"


def demo_section_b():
    """The two ranked lists (tv, mobile) of section B of the saved demo page, with the Drama-without-Comedy marks."""
    from html.parser import HTMLParser
    text = DEMO.read_text()
    part = text[text.index("<h2>B."):text.index("<h2>C.")]

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tables, self.row, self.cell, self.mark = [], None, None, False

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == "table":
                self.tables.append([])
            elif tag == "tr":
                self.row, self.mark = [], a.get("class") == "mark"
            elif tag == "td" and self.row is not None:
                self.cell = ""

        def handle_endtag(self, tag):
            if tag == "td" and self.cell is not None:
                self.row.append(self.cell)
                self.cell = None
            elif tag == "tr" and self.row:
                self.tables[-1].append((self.row, self.mark))
                self.row = None

        def handle_data(self, data):
            if self.cell is not None:
                self.cell += data
    p = P()
    p.feed(part)
    avg = re.search(r"M3b tv <b>([\d.]+)</b>, M3b mobile <b>([\d.]+)</b>, M2 \(ignores device\) <b>([\d.]+)</b>", part).groups()
    return p.tables[:2], avg


def figF_1():
    """Section B of the demo page (synthetic device scenario), re-drawn from the saved page."""
    (tv, mob), avg = demo_section_b()
    fig, axes = plt.subplots(1, 2, figsize=(9, 6.2))
    fig.suptitle("Synthetic device effect, pipeline demonstration, not a finding", color=RED, fontsize=12, fontweight="bold", y=0.985)
    for ax, rows, label in ((axes[0], tv, "SIMULATED device = tv"), (axes[1], mob, "SIMULATED device = mobile")):
        ax.axis("off")
        ax.set_title(f"{label}\n{sum(m for _, m in rows)} of 18 are Drama-without-Comedy", fontsize=9.5, loc="left")
        for i, (cells, marked) in enumerate(rows):
            y = 1 - (i + 1) / 19.5
            if marked:
                ax.add_patch(plt.Rectangle((0, y - 0.022), 1, 0.045, color="#fde2e2", transform=ax.transAxes, clip_on=False))
            ax.text(0.02, y, cells[0], transform=ax.transAxes, fontsize=8.5, va="center")
            ax.text(0.10, y, cells[1][:44], transform=ax.transAxes, fontsize=8.5, va="center")
    fig.text(0.5, 0.015, f"Shaded rows: Drama-without-Comedy titles. One sampled user shown (MovieLens titles).\n"
             f"Average over 40 users: M3b tv {avg[0]}, M3b mobile {avg[1]}, M2 (ignores device) {avg[2]}.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return fig


FIGURES = {"fig3_1": (fig3_1, "fig3_1_splits.png"), "figF_1": (figF_1, "figF_1_demo_m3b.png"), "fig4_1": (fig4_1, "fig4_1_architecture.png"), "fig4_2": (fig4_2, "fig4_2_funnel.png"),
           "fig5_1": (fig5_1, "fig5_1_headline.png"), "fig5_2": (fig5_2, "fig5_2_reranking.png"), "fig5_3": (fig5_3, "fig5_3_retrieval.png"),
           "fig5_4": (fig5_4, "fig5_4_bandit.png"), "fig5_5": (fig5_5, "fig5_5_latency.png"), "fig5_6": (fig5_6, "fig5_6_levels.png")}


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
