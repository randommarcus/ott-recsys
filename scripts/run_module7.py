"""Module 7: UCB bandit for the new-release lane - SEMI-SYNTHETIC SIMULATION, NOT AN A/B TEST.
Reward = "liked given rated"; it ignores click probability. Test stays sealed (only global train and validation are read)."""
import dataclasses
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from ott_recsys import content
from ott_recsys.bandit import (display_with_reserved, displayed_metrics, draw_coins, draw_truth, fit_prior, make_arms, mean_ci,
                               oracle_expected, run_metrics, session_starts, simulate)
from ott_recsys.config import Config
from ott_recsys.data import load_movies
from ott_recsys.eval import paired_diff
from ott_recsys.memguard import Stages
from ott_recsys.newrelease import release_sets

QUICK = os.getenv("OTT_BANDIT_QUICK") == "1"
SEEDS, SEED0 = (3 if QUICK else 30), 11
BANNER = "*** SIMULATION, NOT AN A/B TEST. Reward = 'liked given rated' (ignores click probability). No real person saw anything. ***"
cfg = Config()
stages = Stages()
RK = cfg.out_dir / "ranker"
movies, links = load_movies(cfg), pd.read_parquet(cfg.out_dir / "links.parquet")
catalog = np.sort(movies["movieId"].to_numpy())
cat_idx = pd.Index(catalog)
train_g, val_g = pd.read_parquet(cfg.out_dir / "train.parquet"), pd.read_parquet(cfg.out_dir / "val.parquet")
emb_c = content.center_embeddings(content.load_embeddings(cfg.out_dir / "item_embeddings.npz", catalog, "emb384"))
tmdb = content.load_tmdb_cache(ROOT / "data" / "raw" / "tmdb", links["tmdbId"])
rel = content.release_dates(movies, links, tmdb)
sets = release_sets(train_g, val_g, rel, months=6)
arm_ids = np.asarray(sets["release_cold"])
rel_ts = np.array([rel.get(i, pd.NaT).timestamp() if pd.notna(rel.get(i, pd.NaT)) else np.nan for i in arm_ids])
vr = val_g[val_g["item_id"].isin(set(arm_ids))]
n_arm = vr.groupby("item_id").size().reindex(arm_ids).fillna(0).to_numpy()
likes_arm = vr.groupby("item_id")["label"].sum().reindex(arm_ids).fillna(0).to_numpy()
global_rate = float(vr["label"].mean())

# content prior: PRE-cutoff movies only (no arm data); empirical-Bayes weight from the same movies
g = train_g.groupby("item_id").agg(n=("label", "size"), likes=("label", "sum"))
g = g[g["n"] >= 20]
model, W_PRIOR, prior_corr = fit_prior(emb_c, cat_idx.get_indexer(g.index), g["n"].to_numpy(), g["likes"].to_numpy())
mu_arm = np.clip(model.predict(emb_c[cat_idx.get_indexer(arm_ids)]), 0.05, 0.95)
req_t = session_starts(val_g)
if QUICK:
    req_t = req_t[:3000]
R = len(req_t)
stages.done("setup")

lines = ["== CONFIG ==", *(f"{k:26s} {v}" for k, v in dataclasses.asdict(cfg).items()), "", BANNER, "",
         "WHY NOT REPLAY: the replay estimator of Li et al. (2010) needs logged actions chosen at random (or with known probabilities) and "
         "keeps only events where the logged action equals the policy's pick. MovieLens ratings are users' own choices steered by the site, "
         f"and with ~{len(arm_ids)} arms nearly every logged event would be thrown away. So: a semi-synthetic simulation.",
         f"arms: {len(arm_ids)} new releases (TMDB release since cutoff-6mo, no ratings before {sets['cutoff']:%Y-%m-%d}); hidden like-rate of an arm = a draw from "
         f"Beta(rate*5 + likes, (1-rate)*5 + n - likes) built from its real ratings in the validation window (global rate {global_rate:.3f}); a fresh draw per run and seed.",
         f"requests: {R} real session starts (30-min rule) from {val_g['user_id'].nunique()} users between {pd.to_datetime(req_t[0], unit='s'):%Y-%m-%d} and "
         f"{pd.to_datetime(req_t[-1], unit='s'):%Y-%m-%d}; a request shows k lane slots; an arm is on the shelf from its TMDB release date for 365 days (180 as sensitivity).",
         f"content prior (centered SBERT -> like-rate, ridge, fitted on {len(g)} PRE-cutoff movies with >= 20 ratings): cross-validated correlation {prior_corr:.3f}; "
         f"empirical-Bayes weight {W_PRIOR:.1f} pseudo-ratings. Prior-only = always show the arms with the best prior.",
         f"{SEEDS} seeds per setting; per seed every strategy sees the SAME hidden like-rates and reward coins (paired). Intervals are 95% over seeds"
         + (" [QUICK MODE: 3 seeds, first 3000 requests - smoke test only]" if QUICK else "") + ".", ""]

STRATS = [("random", "random", 0.1), ("greedy", "greedy", 0.1), ("eps_greedy 0.1", "eps_greedy", 0.1), ("ucb1", "ucb1", 0.1),
          ("ucb1 + content prior", "ucb1_prior", 0.1), ("thompson (Beta)", "thompson", 0.1), ("prior only (no learning)", "prior_only", 0.1)]


def run_setting(arms, k, shelf, strategies, label):
    p, coins = draw_truth(arms, SEEDS, SEED0), draw_coins(R, k, SEEDS, SEED0)
    orc = oracle_expected(arms, req_t, k, shelf, p)
    out = {}
    for name, strat, eps in strategies:
        res = simulate(arms, req_t, strat, k, shelf, SEEDS, SEED0, w=W_PRIOR, eps=eps, p=p, coins=coins)
        out[name] = run_metrics(res, orc)
        print(f"  [{label}] {name}: likes/1000 impr {1000 * out[name]['likes'].mean() / out[name]['impressions'].mean():.1f}", flush=True)
    return out, orc


def fmt_ci(x, scale=1.0, nd=3):
    m, lo, hi = mean_ci(np.asarray(x) * scale)
    return f"{m:.{nd}f} [{lo:.{nd}f}-{hi:.{nd}f}]"


def results_table(out, orc, k):
    rnd_reg = out["random"]["regret"]
    served = out["random"]["impressions"] / k
    rows = {}
    for name, m in out.items():
        found = m["found_at"]
        med = np.median(found)
        rows[name] = {"likes per 1,000 impressions": fmt_ci(1000 * m["likes"] / m["impressions"], nd=1),
                      "regret as % of random's": fmt_ci(100 * m["regret"] / rnd_reg, nd=1),
                      "median request to find >=5 of true top-10": ("never" if np.isinf(med) else f"{med:.0f}") + f" (found in {np.isfinite(found).mean():.0%} of runs)",
                      "exposure Gini": fmt_ci(m["gini"], nd=2), "impressions on true top-10% arms": fmt_ci(100 * m["top10pct_share"], nd=1) + "%",
                      "arms never shown": fmt_ci(100 * m["never_shown"], nd=1) + "%"}
    best = 1000 * orc.sum(0) / (k * served)
    rows["oracle (expected)"] = {"likes per 1,000 impressions": fmt_ci(best, nd=1), "regret as % of random's": "0"}
    return pd.DataFrame(rows).T


def paired_vs(out, ref="ucb1"):
    res = []
    for name, m in out.items():
        if name != ref:
            d = 1000 * (m["likes"] - out[ref]["likes"]) / m["impressions"]
            res.append(f"  {name + ' - ' + ref:44s} likes per 1,000 impressions {fmt_ci(d, nd=1)}")
    return res


ALL = {}
for label, minr in [("239 arms (>=20 ratings)", 20), ("629 arms (>=5 ratings)", 5), ("2031 arms (>=1 rating)", 1)]:
    arms = make_arms(arm_ids, rel_ts, n_arm, likes_arm, mu_arm, global_rate, 5.0, minr)
    out, orc = run_setting(arms, 2, 365, STRATS, label)
    ALL[label] = (arms, out, orc)
    served = out["random"]["impressions"][0] / 2
    pm = arms["a"] / (arms["a"] + arms["b"])
    lines += [BANNER, f"== MAIN RESULTS, {label}: k=2 lane slots, 365-day shelf; {len(arms['a'])} arms, mean hidden like-rate {pm.mean():.3f} (sd of arm means {pm.std():.3f}); "
              f"{int(served)} of {R} requests had at least one arm on the shelf ==", results_table(out, orc, 2).to_string(), "",
              "paired differences vs plain UCB1 (same hidden rates and coins):"] + paired_vs(out) + [""]
    stages.done(f"main {label}")

# ---- checkpoints on the 629 set
arms629, out629, orc629 = ALL["629 arms (>=5 ratings)"]
lines += [BANNER, "== CUMULATIVE LIKES (k=2) AT 10 / 25 / 50 / 100% OF THE REQUEST STREAM, 629-arm set (mean over seeds) =="]
ck = {n: {f"{int(c * 100)}%": float(m["cum_at"][c].mean()) for c in (0.10, 0.25, 0.50, 1.0)} for n, m in out629.items()}
ck["oracle (expected)"] = {f"{int(c * 100)}%": float(orc629.cumsum(0)[max(int(c * R) - 1, 0)].mean()) for c in (0.10, 0.25, 0.50, 1.0)}
lines += [pd.DataFrame(ck).T.round(0).to_string(), ""]

# ---- sensitivity on the 629 set
sens = {"k=3": (3, 365, STRATS), "shelf 180 days": (2, 180, STRATS),
        "eps 0.05 / 0.2": (2, 365, [("random", "random", 0.1), ("eps_greedy 0.05", "eps_greedy", 0.05), ("eps_greedy 0.2", "eps_greedy", 0.2)])}
rows = {}
for lab, (k, shelf, strategies) in sens.items():
    out, orc = run_setting(arms629, k, shelf, strategies, f"629 {lab}")
    lines += [BANNER, f"== SENSITIVITY, 629-arm set, {lab} =="]
    if lab != "eps 0.05 / 0.2":
        lines += [results_table(out, orc, k).iloc[:, :2].to_string(), ""]
        ALL[f"629 {lab}"] = (arms629, out, orc)
    else:
        rr = out["random"]["regret"]
        lines += [pd.DataFrame({n: {"regret as % of random's": fmt_ci(100 * m["regret"] / rr, nd=1),
                                    "likes per 1,000 impressions": fmt_ci(1000 * m["likes"] / m["impressions"], nd=1)} for n, m in out.items()}).T.to_string(),
                  f"for comparison eps_greedy 0.1: regret {fmt_ci(100 * out629['eps_greedy 0.1']['regret'] / out629['random']['regret'], nd=1)}% of random's", ""]
    stages.done(f"sensitivity {lab}")

# ---- cost of reserving 2 or 3 of the 20 slots (Module 6 M2 ranking, held-out users)
fz = np.load(RK / "features.npz")
row_user, row_slot, cand_f, lane_f, yv, users = fz["row_user"].astype(np.int64), fz["row_slot"].astype(np.int64), fz["cand_f"], fz["lane_f"], fz["y"].astype(bool), fz["users"]
U, Wd = cand_f.shape
oof = np.load(RK / "oof_M2.npy")
S2 = np.full((U, Wd), -np.inf)
S2[row_user, row_slot] = oof
S2[lane_f] = -np.inf  # the MAIN lane ranking: lane items are not part of it
Hm = np.zeros((U, Wd), bool)
Hm[row_user, row_slot] = yv
main_hits = np.take_along_axis(Hm, np.argsort(-S2, axis=1, kind="stable")[:, :25], axis=1)
vtruth = pd.read_parquet(cfg.out_dir / "user_split" / "val.parquet", columns=["user_id", "label"])
n_truth = vtruth[vtruth["label"] == 1].groupby("user_id").size().reindex(users).to_numpy()
hold = ~np.isin(users, np.random.default_rng(0).choice(users, cfg.tune_users, replace=False))
base_nd, base_rc = displayed_metrics(main_hits[:, :20], n_truth)
base_likes = main_hits[:, :20].sum(1)
cfgs = {"none (all 20 slots main lane)": [], "2 reserved at the END (19, 20)": [19, 20], "3 reserved at the END (18, 19, 20)": [18, 19, 20],
        "2 reserved INSIDE the top 10 (5, 10)": [5, 10], "3 reserved, positions 5, 10, 15": [5, 10, 15]}
rows, cost = {}, {}
for lab, pos in cfgs.items():
    disp, _ = display_with_reserved(main_hits, pos)
    nd, rc = displayed_metrics(disp, n_truth)
    dn, dr = paired_diff(pd.Series(nd[hold]), pd.Series(base_nd[hold])), paired_diff(pd.Series(rc[hold]), pd.Series(base_rc[hold]))
    lost = 1000 * (base_likes[hold].mean() - disp.sum(1)[hold].mean())
    cost[lab] = lost
    rows[lab] = {"NDCG@10": f"{nd[hold].mean():.4f} (diff {dn[0]:+.4f} [{dn[1]:+.4f}, {dn[2]:+.4f}])",
                 "Recall@20": f"{rc[hold].mean():.4f} (diff {dr[0]:+.4f} [{dr[1]:+.4f}, {dr[2]:+.4f}])",
                 "main-lane likes shown per 1,000 requests": f"{1000 * disp.sum(1)[hold].mean():.0f} (lost {lost:.0f} = {100 * lost / (1000 * base_likes[hold].mean()):.1f}% of the main-lane hits in the top 20)"}
lines += ["== COST TO MAIN-LANE QUALITY OF RESERVING SLOTS (Module 6 M2 out-of-fold ranking, main lane only, 15,530 held-out users; REAL held-out likes) ==",
          pd.DataFrame(rows).T.to_string(), ""]
gain = {}
for lab, k, key in [("2 slots", 2, "629 arms (>=5 ratings)"), ("3 slots", 3, "629 k=3")]:
    o = ALL[key][1]
    gain[lab] = {n: 1000 * o[n]["likes"].mean() / (o[n]["impressions"].mean() / k) for n in ("random", "ucb1", "ucb1 + content prior", "thompson (Beta)")}
lines += [BANNER, "== SIMULATED LANE LIKES PER 1,000 REQUESTS FOR THE RESERVED SLOTS (629-arm set) ==",
          pd.DataFrame(gain).round(0).to_string(),
          "WARNING - DO NOT NET THESE AGAINST THE 'LOST' NUMBERS ABOVE. The lane reward is 'liked given rated' (about 40-50% of impressions, from self-selected raters, no click model); the main-lane "
          "figure counts how often a displayed slot is one of the user's later liked items (about 4% for slots 19-20). Different units and different populations. Compare only RELATIVE statements: "
          "reserving 2 end slots removes the stated % of main-lane top-20 hits; the best learning strategy earns the stated multiple of random's simulated lane reward.", ""]

lines += ["== WHAT COULD MAKE THIS SIMULATION MISLEADING ==",
          "1. Like-rates come from people who chose to rate the film, so they are probably higher than for a viewer shown the film by a lane; the reward ignores click probability.",
          "2. One hidden rate per arm for everybody: real taste varies by user, so a non-contextual bandit undervalues niche films (LinUCB-style methods are the next step).",
          "3. The main arm set needs >= 5 ratings, i.e. films that already attracted viewers; real duds are missing, which makes the problem easier. The 2,031-arm set is heavily shrunk toward the global rate.",
          "4. No position bias, fatigue, repeat exposure or real viewing behaviour; every impression is instantly judged.",
          "5. Rates are stationary within the window; real interest decays after release.",
          "6. The content prior was learned on older films and may transfer imperfectly to new ones.",
          "7. Intervals reflect seeds (truth draws and coin flips), not modelling choices; hidden rates are themselves estimated from thin data.",
          "8. The slot-cost comparison mixes real held-out likes in one evaluation with simulated likes in another.",
          BANNER]
text = "\n".join(lines)
print(text)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / f"module7_bandit_{cfg.run_name}{'_QUICK' if QUICK else ''}.txt").write_text(text)
stages.done("report")
