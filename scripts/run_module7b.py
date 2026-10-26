"""Module 7b: a fairer UCB comparison - SEMI-SYNTHETIC SIMULATION, NOT AN A/B TEST.
UCB with a tunable constant c (bonus c*sqrt(ln t / n)) and UCB1-Tuned. c is chosen on the FIRST half of the request
stream with selection seeds; everything is evaluated on the SECOND half of full-stream runs with DIFFERENT seeds.
For fairness epsilon-greedy's epsilon is tuned the same way. Reward = "liked given rated" (ignores click probability).
Test stays sealed."""
import dataclasses
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from ott_recsys import content
from ott_recsys.bandit import (draw_coins, draw_truth, eligible_range, make_arms, mean_ci, oracle_expected, select_param, simulate,
                               window_metrics)
from ott_recsys.banditdata import load_bandit_inputs
from ott_recsys.config import Config
from ott_recsys.data import load_movies
from ott_recsys.memguard import Stages

QUICK = os.getenv("OTT_BANDIT_QUICK") == "1"
S_SEL, S_EVAL, SEL_SEED, EVAL_SEED = (3, 4, 101, 202) if QUICK else (20, 30, 101, 202)
assert SEL_SEED != EVAL_SEED, "selection and evaluation must use different seeds"
SQ2 = float(np.sqrt(2.0))
GRID_C = [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
GRID_EPS = [0.01, 0.02, 0.05, 0.1, 0.2]
BANNER = "*** SIMULATION, NOT AN A/B TEST. Reward = 'liked given rated' (ignores click probability). No real person saw anything. ***"
cfg = Config()
stages = Stages()
movies, links = load_movies(cfg), pd.read_parquet(cfg.out_dir / "links.parquet")
catalog = np.sort(movies["movieId"].to_numpy())
emb_c = content.center_embeddings(content.load_embeddings(cfg.out_dir / "item_embeddings.npz", catalog, "emb384"))
inp = load_bandit_inputs(cfg, ROOT, catalog, movies, links, emb_c)
req_t = inp["req_t"][:6000] if QUICK else inp["req_t"]
R = len(req_t)
H = R // 2
W_PRIOR = inp["w_prior"]
stages.done("setup")

lines = ["== CONFIG ==", *(f"{k:26s} {v}" for k, v in dataclasses.asdict(cfg).items()), "", BANNER, "",
         f"PROTOCOL. The request stream has {R} real session starts ({pd.to_datetime(req_t[0], unit='s'):%Y-%m-%d} to {pd.to_datetime(req_t[-1], unit='s'):%Y-%m-%d}); "
         f"the FIRST half is requests 0-{H - 1}, the SECOND half {H}-{R - 1}.",
         f"SELECTION: UCB constant c from {GRID_C} and epsilon from {GRID_EPS}, scored by likes per impression on the first half only, with {S_SEL} selection seeds ({SEL_SEED}); "
         f"chosen separately for each arm set.",
         f"EVALUATION: full-stream runs with {S_EVAL} DIFFERENT seeds ({EVAL_SEED}); learning carries over from the first half, but only second-half rewards, regret and exposure are reported. "
         "Per seed every strategy sees the same hidden like-rates and reward coins (paired).",
         "UCB1 (c = sqrt 2) is the textbook index; the tunable index is mean + c*sqrt(ln t / n) with unseen arms tried first; UCB1-Tuned (Auer et al. 2002) has no constant. "
         f"UCB(c*)+prior adds the content prior ({W_PRIOR:.1f} pseudo-ratings, fitted on {inp['n_prior_movies']} pre-cutoff movies, CV correlation {inp['prior_corr']:.2f}) at the selected c*.",
         "Fairness addition: epsilon-greedy's epsilon is tuned on the same first half (otherwise only UCB would be tuned). Thompson sampling has no parameter.",
         "Intervals are 95% over seeds." + (" [QUICK MODE - smoke test only]" if QUICK else ""), ""]


def ci_str(x, scale=1.0, nd=1):
    m, lo, hi = mean_ci(np.asarray(x) * scale)
    return f"{m:.{nd}f} [{lo:.{nd}f}-{hi:.{nd}f}]"


summary = {}
for label, minr in [("239 arms (>=20 ratings)", 20), ("629 arms (>=5 ratings)", 5), ("2031 arms (>=1 rating)", 1)]:
    arms = make_arms(inp["arm_ids"], inp["rel_ts"], inp["n_arm"], inp["likes_arm"], inp["mu_arm"], inp["global_rate"], 5.0, minr)
    # ---- 1. selection on the first half only
    c_star, c_edge, c_scores = select_param(arms, req_t[:H], "ucb_c", "c", GRID_C, 2, 365, S_SEL, SEL_SEED)
    e_star, e_edge, e_scores = select_param(arms, req_t[:H], "eps_greedy", "eps", GRID_EPS, 2, 365, S_SEL, SEL_SEED)
    sq2 = select_param(arms, req_t[:H], "ucb_c", "c", [SQ2], 2, 365, S_SEL, SEL_SEED)[2][SQ2]
    summary[label] = (c_star, c_edge, e_star, e_edge)
    lines += [BANNER, f"== {label}: SELECTION on the FIRST half (likes per 1,000 impressions, mean [95% CI over {S_SEL} selection seeds]) ==",
              "UCB constant c:"] + [f"  c = {g:<5} {ci_str(c_scores[g], 1000)}" + ("   <-- selected" if g == c_star else "") for g in GRID_C]
    lines += [f"  (for reference, UCB1 c = sqrt 2 on the same first half: {ci_str(sq2, 1000)}; not in the grid)",
              f"  selected c* = {c_star}; at a grid edge: {c_edge or 'no (interior)'}", "epsilon for epsilon-greedy:"]
    lines += [f"  eps = {g:<5} {ci_str(e_scores[g], 1000)}" + ("   <-- selected" if g == e_star else "") for g in GRID_EPS]
    lines += [f"  selected eps* = {e_star}; at a grid edge: {e_edge or 'no (interior)'}", ""]
    stages.done(f"selection {label}")

    # ---- 2. evaluation on the second half, different seeds
    p, coins = draw_truth(arms, S_EVAL, EVAL_SEED), draw_coins(R, 2, S_EVAL, EVAL_SEED)
    orc = oracle_expected(arms, req_t, 2, 365, p)
    lo_h = int(eligible_range(arms["release_ts"], req_t[H], 365)[0])
    hi_last = int(eligible_range(arms["release_ts"], req_t[-1], 365)[1])
    STR = [("random", "random", {}), ("greedy", "greedy", {}), (f"eps-greedy eps*={e_star}", "eps_greedy", {"eps": e_star}),
           ("eps-greedy 0.1 (Module 7)", "eps_greedy", {"eps": 0.1}), ("UCB1 (c = sqrt 2)", "ucb1", {}),
           (f"UCB c*={c_star}", "ucb_c", {"c": c_star}), ("UCB1-Tuned", "ucb_tuned", {}),
           (f"UCB c*={c_star} + content prior", "ucb_c_prior", {"c": c_star, "w": W_PRIOR}), ("Thompson (Beta)", "thompson", {}),
           ("prior only (no learning)", "prior_only", {})]
    M, whole = {}, {}
    for name, strat, kw in STR:
        res = simulate(arms, req_t, strat, 2, 365, S_EVAL, EVAL_SEED, p=p, coins=coins, snapshot_at=H, **kw)
        M[name] = window_metrics(res, orc, H, (lo_h, hi_last))
        whole[name] = res["reward"].sum(0) / res["impressions"].sum(1)
        print(f"  [{label}] {name}: second-half likes/1000 impr {1000 * M[name]['likes'].mean() / M[name]['impressions'].mean():.1f}", flush=True)
    rnd_reg = M["random"]["regret"]
    served = M["random"]["impressions"] / 2
    rows = {}
    for name, m in M.items():
        rows[name] = {"likes per 1,000 impressions (2nd half)": ci_str(1000 * m["likes"] / m["impressions"]),
                      "regret as % of random's": ci_str(100 * m["regret"] / rnd_reg), "exposure Gini": ci_str(m["gini"], nd=2),
                      "impressions on true top-10% arms": ci_str(100 * m["top10pct_share"]) + "%", "arms never shown": ci_str(100 * m["never_shown"]) + "%",
                      "whole-stream likes/1,000 (ref.)": ci_str(whole[name], 1000)}
    rows["oracle (expected)"] = {"likes per 1,000 impressions (2nd half)": ci_str(1000 * orc[H:].sum(0) / (2 * served)), "regret as % of random's": "0"}
    pm = arms["a"] / (arms["a"] + arms["b"])
    lines += [BANNER, f"== {label}: SECOND-HALF RESULTS (k=2, 365-day shelf, {len(arms['a'])} arms, mean hidden like-rate {pm.mean():.3f}; evaluation seeds {EVAL_SEED}) ==",
              pd.DataFrame(rows).T.to_string(), ""]
    key = {"c": f"UCB c*={c_star}", "u1": "UCB1 (c = sqrt 2)", "tu": "UCB1-Tuned", "pr": f"UCB c*={c_star} + content prior", "ts": "Thompson (Beta)",
           "ep": f"eps-greedy eps*={e_star}", "gr": "greedy"}
    lines += ["paired differences, likes per 1,000 impressions in the second half (* = interval excludes 0):"]
    for a, b in [("c", "u1"), ("tu", "u1"), ("c", "tu"), ("pr", "c"), ("ts", "c"), ("ep", "c"), ("c", "gr")]:
        d = 1000 * (M[key[a]]["likes"] / M[key[a]]["impressions"] - M[key[b]]["likes"] / M[key[b]]["impressions"])
        m_, lo_, hi_ = mean_ci(d)
        lines.append(f"  {key[a] + ' - ' + key[b]:62s} {m_:+7.1f} [{lo_:+.1f}, {hi_:+.1f}]" + ("  *" if lo_ > 0 or hi_ < 0 else ""))
    stages.done(f"evaluation {label}")

    # ---- 3. diagnostic (NOT used for selection): how the whole c grid performs on the second half
    diag = {}
    for g in GRID_C:
        res = simulate(arms, req_t, "ucb_c", 2, 365, S_EVAL, EVAL_SEED, p=p, coins=coins, snapshot_at=H, c=g)
        m = window_metrics(res, orc, H, (lo_h, hi_last))
        diag[g] = 1000 * m["likes"] / m["impressions"]
    best_post = max(GRID_C, key=lambda g: diag[g].mean())
    lines += ["", f"diagnostic, NOT used for selection - second-half likes per 1,000 impressions for every c (evaluation seeds); first-half choice c* = {c_star}, post-hoc best on the second half = {best_post}:"]
    lines += [f"  c = {g:<5} {ci_str(diag[g])}" + ("  <-- c* chosen on the first half" if g == c_star else "") + ("  <-- post-hoc best" if g == best_post else "") for g in GRID_C]
    lines += [""]
    stages.done(f"diagnostic {label}")

lines += [BANNER, "== SELECTED PARAMETERS (chosen on the first half of the stream) ==",
          pd.DataFrame({k: {"c*": v[0], "c* at grid edge": v[1] or "no", "eps*": v[2], "eps* at grid edge": v[3] or "no"} for k, v in summary.items()}).T.to_string(),
          f"c grid: {GRID_C}; epsilon grid: {GRID_EPS}.", "", BANNER]
text = "\n".join(lines)
print(text)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / f"module7b_ucb_tuning_{cfg.run_name}{'_QUICK' if QUICK else ''}.txt").write_text(text)
stages.done("report")
