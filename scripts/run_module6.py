"""Module 6: XGBRanker (rank:ndcg) with context, out-of-fold, and the context ablation ladder.
Resumable: each variant's out-of-fold scores are saved and reused. Test stays sealed.
DEVICE IS SIMULATED everywhere in this module; scenario (b) uses synthetic labels (pipeline demonstration)."""
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from ott_recsys.config import Config
from ott_recsys.context import DEVICES, device_onehot, genre_matrix, shuffled_device, simulate_device, synthetic_truth
from ott_recsys.data import load_movies, load_user_split
from ott_recsys.eval import paired_diff, summarize
from ott_recsys.memguard import Stages
from ott_recsys.rankfeat import BASE, CTX_SESSION, CTX_TIME
from ott_recsys.ranker import PARAMS, group_importance, oof_scores
from ott_recsys.rerank import all_orders, hit_matrix, metrics_from_ranks, rank_positions

pd.set_option("display.width", 230)
cfg = Config()
stages = Stages()
RK = cfg.out_dir / "ranker"
z = np.load(RK / "features.npz")
X, cols = z["X"], [str(c) for c in z["cols"]]
row_user, row_slot, cand_f, lane_f, ncf_f = z["row_user"].astype(np.int64), z["row_slot"].astype(np.int64), z["cand_f"].astype(np.int64), z["lane_f"], z["ncf_f"]
users, y_real, sess_len = z["users"], z["y"].astype(int), np.nan_to_num(z["session_len"])
U, W = cand_f.shape
valid = cand_f >= 0
movies = load_movies(cfg)
catalog = np.sort(movies["movieId"].to_numpy())
cat_idx = pd.Index(catalog)
split = load_user_split(cfg)
train, val = split.train, split.val
counts = np.bincount(cat_idx.get_indexer(train.loc[train["label"] == 1, "item_id"]), minlength=len(catalog)).astype(float)
truth = {u: cat_idx.get_indexer(g["item_id"].to_numpy()) for u, g in val[val["label"] == 1].groupby("user_id")}
t_list = [truth[u] for u in users]
n_truth = np.array([len(t) for t in t_list])
hit = hit_matrix(cand_f, t_list)
assert (hit[row_user, row_slot] == y_real).all()
sel = np.isin(users, np.random.default_rng(0).choice(users, cfg.tune_users, replace=False))  # used for tt/NCF selection earlier
hold = ~sel

# ---- SIMULATED device and the synthetic-label scenario
dev = simulate_device(U, seed=0)
dev_oh, dev_shuf_oh = device_onehot(dev), device_onehot(shuffled_device(dev, seed=1))
gm, gnames = genre_matrix(movies, catalog)
drama_nc = (gm[:, gnames.index("Drama")] == 1) & (gm[:, gnames.index("Comedy")] == 0)
t_syn, removed = synthetic_truth(t_list, dev, drama_nc, 0.6, 2)
n_truth_b = np.array([len(t) for t in t_syn])
hit_b = hit_matrix(cand_f, t_syn)
y_syn = hit_b[row_user, row_slot].astype(int)
DEV_NAMES = [f"dev_{d}" for d in DEVICES]

# ---- variants
CI = {c: i for i, c in enumerate(cols)}
gcols = [c for c in cols if c.startswith("g_")]
f_base = [CI[c] for c in BASE + gcols]
f_time = [CI[c] for c in CTX_TIME]
f_sess = [CI[c] for c in CTX_SESSION]
VARIANTS = [("M0", f_base, None, y_real), ("M1", f_base + f_time, None, y_real), ("M2", f_base + f_time + f_sess, None, y_real),
            ("M3a", f_base + f_time + f_sess, dev_oh, y_real), ("M2b", f_base + f_time + f_sess, None, y_syn),
            ("M3b", f_base + f_time + f_sess, dev_oh, y_syn), ("M3b_shuf", f_base + f_time + f_sess, dev_shuf_oh, y_syn)]
oof, meta = {}, {}
for name, idx, dv, lab in VARIANTS:
    p_s, p_m = RK / f"oof_{name}.npy", RK / f"oof_{name}.json"
    if p_s.exists() and p_m.exists():
        oof[name], meta[name] = np.load(p_s), json.loads(p_m.read_text())
        print(f"{name}: loaded", flush=True)
        continue
    t0 = time.time()
    names = [cols[i] for i in idx] + (DEV_NAMES if dv is not None else [])
    Xv = X[:, idx] if dv is None else np.concatenate([X[:, idx], dv[row_user]], axis=1)
    print(f"{name}: {Xv.shape[1]} features, {int(lab.sum())} positives; training 5 folds", flush=True)
    scores, best, imp = oof_scores(Xv, lab, row_user, U, names, k=5, seed=0, log=lambda m: print(m, flush=True))
    del Xv
    oof[name] = scores
    meta[name] = {"best_iter": best, "importance": imp, "seconds": round(time.time() - t0), "features": names}
    np.save(p_s, scores)
    p_m.write_text(json.dumps(meta[name]))
    stages.done(f"variant {name}")

# ---- orders and metrics
rand = np.random.default_rng(0).random((U, W))
ranks = all_orders(cand_f, hit, counts, {"main": ncf_f}, seed=0)
for name, s in oof.items():
    S = np.full((U, W), np.nan)
    S[row_user, row_slot] = s
    ranks[name] = rank_positions(valid, S, rand)
M = {m: metrics_from_ranks(r, hit, n_truth, users) for m, r in ranks.items()}
ranks_b = dict(ranks)
ranks_b["oracle"] = all_orders(cand_f, hit_b, counts, None, seed=0)["oracle"]
Mb = {m: metrics_from_ranks(r, hit_b, n_truth_b, users) for m, r in ranks_b.items()}
COLS = ["e2e_ndcg10", "e2e_recall20", "rr_ndcg10", "rr_recall20"]


def fmt(df):
    t = summarize(df[COLS], float("nan"), n_boot=1000)
    return {c: f"{t.loc[c, 'mean']:.4f} [{t.loc[c, 'lo']:.3f}-{t.loc[c, 'hi']:.3f}]" for c in COLS} | {"users": int(t.loc[COLS[0], "n_users"])}


def table(MM, mask, order, names):
    t = pd.DataFrame({m: fmt(MM[m][mask]) for m in order}).T
    t.index = [names.get(i, i) for i in t.index]
    return t.to_string()


def pairs(MM, mask, plist, cols=COLS):
    out = []
    for a, b in plist:
        for c in cols:
            m, lo, hi = paired_diff(MM[a][mask][c], MM[b][mask][c])
            out.append(f"  {a + ' - ' + b:26s} {c:13s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("  *" if lo > 0 or hi < 0 else ""))
    return out


NAMES = {"tt": "tt (two-tower's own order)", "pop": "pop (popularity order)", "ncf_main": "ncf (NCF order)", "blend_main": "blend (50/50 tt+ncf)",
         "M0": "M0 ranker, no context", "M1": "M1 + hour/day", "M2": "M2 + session", "M3a": "M3a + SIMULATED device (independent)",
         "M2b": "M2b ranker (synthetic labels)", "M3b": "M3b + SIMULATED device (synthetic labels)", "M3b_shuf": "M3b_shuf: device shuffled (synthetic labels)"}
lines = ["== CONFIG ==", *(f"{k:26s} {v}" for k, v in dataclasses.asdict(cfg).items()), f"ranker params: {PARAMS} (rank:ndcg, 5 user-grouped folds, early stopping on an inner 10% of training users)", "",
         f"== DATA: {U} validation users; 230-slot lists (200 ID-only two-tower + up to 30 new-release-lane) after removing candidates released AFTER the request time ==",
         f"rows {len(row_user)} ({len(row_user) / U:.0f} per user); positives {int(y_real.sum())} ({y_real.mean():.2%}); features: {len(cols)} (+4 SIMULATED device columns in M3 variants)",
         f"held-out users (never used for tt/NCF selection): {int(hold.sum())}; selection users: {int(sel.sum())}. Primary tables use the held-out users.",
         "request time = first validation interaction (its timestamp only); context features use train interactions strictly before it.", ""]

# ---- filter effect (unfiltered Module-5 protocol vs filtered), baselines only
zc = np.load(cfg.out_dir / "candidates" / "candidates_val.npz")
cand_u = zc["cand"].astype(np.int64)
hit_u = hit_matrix(cand_u, t_list)
ncf_u = np.load(cfg.out_dir / "ncf" / "ncf_main" / "scores.npy")
ranks_u = all_orders(cand_u, hit_u, counts, {"main": ncf_u}, seed=0)
Mu = {m: metrics_from_ranks(r, hit_u, n_truth, users) for m, r in ranks_u.items()}
rows = {}
for m in ["random", "pop", "tt", "ncf_main", "blend_main"]:
    rows[NAMES.get(m, m)] = {"unfiltered NDCG@10": Mu[m][hold]["e2e_ndcg10"].mean(), "filtered NDCG@10": M[m][hold]["e2e_ndcg10"].mean(),
                             "unfiltered Recall@20": Mu[m][hold]["e2e_recall20"].mean(), "filtered Recall@20": M[m][hold]["e2e_recall20"].mean()}
lines += ["== RELEASE-BY-REQUEST-TIME FILTER: baselines on unfiltered (Module 5) vs filtered lists (held-out users, end-to-end) ==",
          pd.DataFrame(rows).T.round(4).to_string(),
          f"slots removed: {(cand_u >= 0).sum() - valid.sum()} of {(cand_u >= 0).sum()} ({1 - valid.sum() / (cand_u >= 0).sum():.1%}); likes inside the lists lost: {int(hit_u.sum() - hit.sum())} of {int(hit_u.sum())}", ""]
stages.done("filter effect")

order = ["random", "pop", "tt", "ncf_main", "blend_main", "M0", "M1", "M2", "M3a", "oracle"]
for label, mask in [("HELD-OUT users (primary)", hold), ("ALL users", np.ones(U, bool))]:
    lines += [f"== LADDER, REAL LABELS: {label}: mean [95% CI over users]; out-of-fold ranker scores ==", table(M, mask, order, NAMES), ""]
lines += ["== PAIRED DIFFERENCES, REAL LABELS, held-out users (* = interval excludes 0) =="]
lines += pairs(M, hold, [("M0", "tt"), ("M0", "ncf_main"), ("M0", "blend_main"), ("M0", "pop"), ("M1", "M0"), ("M2", "M1"), ("M3a", "M2"), ("M2", "M0"), ("M3a", "M0")])
lines += ["", f"held-out vs all users, M2 end-to-end NDCG@10: {M['M2'][hold]['e2e_ndcg10'].mean():.4f} vs {M['M2']['e2e_ndcg10'].mean():.4f}", ""]

# ---- scenario (b)
lines += ["=" * 100, "SCENARIO (b): SYNTHETIC LABELS AND SIMULATED DEVICE - A PIPELINE DEMONSTRATION, NOT A FINDING.",
          f"Devices were assigned at random (tv 40%, mobile 35%, web 15%, tablet 10%). On a MOBILE request each validation like of a Drama-without-Comedy title was "
          f"REMOVED with probability 0.6 ({removed} likes removed; {int(n_truth.sum() - n_truth_b.sum())} fewer likes overall). Nothing here says anything about real viewers.",
          "M3b = M2 + device; M3b_shuf = the same device column shuffled across users (gain over M2b should vanish).", "=" * 100]
orderb = ["random", "pop", "tt", "ncf_main", "blend_main", "M2b", "M3b", "M3b_shuf", "oracle"]
lines += [f"HELD-OUT users, SYNTHETIC labels:", table(Mb, hold, orderb, NAMES), "", "PAIRED, synthetic labels, held-out users:"]
lines += pairs(Mb, hold, [("M3b", "M2b"), ("M3b_shuf", "M2b"), ("M3b", "M3b_shuf"), ("M2b", "tt")])
mob = np.array([DEVICES[d] == "mobile" for d in dev])
lines += ["", "the effect should live in MOBILE users only (held-out users, e2e NDCG@10 paired difference):"]
for lab, mk in [("mobile users", hold & mob), ("non-mobile users", hold & ~mob)]:
    for a, b in [("M3b", "M2b"), ("M3b_shuf", "M2b")]:
        m, lo, hi = paired_diff(Mb[a][mk]["e2e_ndcg10"], Mb[b][mk]["e2e_ndcg10"])
        lines.append(f"  {lab:17s} n={int(mk.sum()):5d} {a} - {b}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("  *" if lo > 0 or hi < 0 else ""))
lines += ["", "=" * 100, ""]
stages.done("ladder tables")

# ---- session gain by spree length
lines += ["== SESSION GAIN (M2 - M1, end-to-end NDCG@10, held-out users) BY HOW LONG THE RATING SPREE ALREADY IS AT REQUEST TIME =="]
for lab, mk in [("not in a session (>30 min silence)", sess_len == 0), ("spree of 1-9 ratings", (sess_len >= 1) & (sess_len < 10)),
                ("spree of 10-49", (sess_len >= 10) & (sess_len < 50)), ("spree of 50-199", (sess_len >= 50) & (sess_len < 200)), ("spree of 200+", sess_len >= 200)]:
    mk = mk & hold
    if mk.sum() > 50:
        m, lo, hi = paired_diff(M["M2"][mk]["e2e_ndcg10"], M["M1"][mk]["e2e_ndcg10"])
        lines.append(f"  {lab:36s} n={int(mk.sum()):5d}  {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("  *" if lo > 0 or hi < 0 else ""))
lines += ["(a gain concentrated in long sprees reflects bulk rating - e.g. rating page after page of one genre - and may not exist on a real OTT service)", ""]

# ---- lane group
lines += ["== NEW-RELEASE LANE GROUP (held-out users) =="]
tot_lane_likes = int((hit & lane_f)[hold].sum())
lines.append(f"in-list liked items on lane slots: {tot_lane_likes} (far too few to judge lane quality) of {int(hit[hold].sum())}")
lr = {}
for m in ["tt", "ncf_main", "blend_main", "M0", "M2"]:
    r = ranks[m][hold]
    top10, top20 = (r < 10) & valid[hold], (r < 20) & valid[hold]
    ln = lane_f[hold]
    lr[NAMES.get(m, m)] = {"lane share of top-10 slots": (top10 & ln).sum() / top10.sum(), "lane share of top-20 slots": (top20 & ln).sum() / top20.sum(),
                           "lane likes found in top-20": f"{int((top20 & ln & hit[hold]).sum())} of {tot_lane_likes}"}
lines += [pd.DataFrame(lr).T.to_string(), "A ranker trained on these labels buries new releases; reserved exposure (Module 7 bandit) and the global-cutoff evaluation are the right tools for them.", ""]

# ---- importances
lines += ["== FEATURE IMPORTANCE (normalised gain, mean over 5 folds) =="]
for name in ["M0", "M2", "M3a", "M3b", "M3b_shuf"]:
    imp = meta[name]["importance"]
    lines += [f"{name} by feature group:", pd.Series(group_importance(imp)).sort_values(ascending=False).round(4).to_string(),
              f"{name} top 12 single features:", pd.Series(imp).sort_values(ascending=False).head(12).round(4).to_string(), ""]
lines += ["WARNING: importances are shaky for correlated features. Trees split credit between near-duplicates arbitrarily (two-tower score/rank, NCF logit and",
          "content similarity all measure 'does this user like this kind of movie'), so a small number does NOT mean a feature is useless, and a large number does not",
          "mean it causes anything. The simulated device column is pure noise in M3a, yet it gets non-zero gain because trees can always fit noise. Use the ablation",
          "ladder (retrain without the feature, paired intervals) to judge usefulness; use importances only as a rough map.", ""]
lines += ["== TRAINING =="] + [f"{n:9s} trees per fold {meta[n]['best_iter']}  ({meta[n]['seconds']} s)" for n in meta]
lines += ["", "== WHAT THE PIECES DO (plain language) ==",
          "rank:ndcg objective. A normal classifier asks 'is this movie liked?' for each row on its own. A ranker is trained on whole lists: for each user it looks at pairs of",
          "candidates where one was liked and one was not, and learns to put the liked one higher. NDCG@10 rewards getting liked movies into the first few slots and rewards",
          "rank 1 more than rank 10, and rank:ndcg weights each pair by how much swapping them would change that score. So the model spends its effort on the top of the list and on",
          "users whose list is mostly wrong near the top, instead of on separating hopeless rows at the bottom. It only ever compares candidates of the same user.",
          "Feature groups. TWO-TOWER (score, rank): 'how close is this movie to the user's recent taste' - the strongest single signal because it reads the last 20 likes.",
          "NCF (logit + known flags): a long-term, ID-based taste score that ignores recency; missing (NaN) means NCF knows nothing about this user or movie.",
          "POPULARITY (as of request): how many likes the movie had before this moment - the crowd's view, without peeking at the future.",
          "CONTENT / GENRE MATCH: does the movie's text or genre resemble what the user liked; item GENRE ONE-HOTS let the ranker learn genre-specific effects and are what any context could interact with.",
          "ITEM AGE: days since release at request time. LANE FLAG: whether the candidate came from the new-release lane.",
          "HOUR / DAY: when the request happens (UTC, so a blurry proxy for local time). SESSION: whether, and how deep into a rating spree, the user is, and what the spree has been rating.",
          "DEVICE (SIMULATED): invented labels; in M3a it carries no information by construction, in scenario (b) it carries a planted effect.",
          "", "== DESIGN NOTE FOR THE FINAL TEST RUN (not executed; test remains sealed) ==",
          "Freeze everything trained on train only: ID-only two-tower, NCF, SBERT, and the ranker (trained on all validation users). Update HISTORIES through validation without retraining:",
          "two-tower user vectors from the last 20 likes in train+val, content/genre profiles from train+val likes, seen-item exclusion, as-of popularity, session features from train+val",
          "strictly before the first test interaction. NCF user IDs go stale: design for (1) fold-in - re-optimise only user embeddings on train+val positives with items and MLP frozen, or",
          "(2) stale NCF as a sensitivity check; the decision is deferred to unlock. Risks: feature distributions shift because histories are longer; the fold-in changes NCF's scale;",
          "test windows are twice as long so absolute numbers are not comparable with validation (only method-to-method comparisons are); test requests often continue the same rating spree;",
          "movies first liked in validation are unknown to the frozen ID models; per-user-split calendar leakage remains; and any re-tuning after seeing test would invalidate it (one shot)."]
text = "\n".join(lines)
print(text)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / f"module6_ranker_{cfg.run_name}.txt").write_text(text)
stages.done("report")
