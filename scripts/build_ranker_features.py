"""Build ranker features for the 230-slot validation candidate lists (Module 6).

Inputs are TRAIN interactions, request times (first validation TIMESTAMP per user) and frozen train-only model
scores. Validation items and labels never enter a feature. Real-data leakage checks at the end:
  1. request times are identical when validation items/labels are scrambled;
  2. adding ALL validation rows to the train frame leaves every user-side feature identical (300-user subset);
  3. as-of popularity equals a brute-force recount from the raw frame.
Never touches the test portion."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from ott_recsys import content
from ott_recsys.config import Config
from ott_recsys.context import genre_matrix, request_times
from ott_recsys.data import load_movies, load_user_split
from ott_recsys.memguard import Stages
from ott_recsys.ncf import NCFScorer
from ott_recsys.rankfeat import build_features
from ott_recsys.rerank import hit_matrix
from ott_recsys.twotower import TwoTowerRetrieval

cfg = Config()
stages = Stages()
TT, NC = cfg.out_dir / "two_tower", cfg.out_dir / "ncf"
out_dir = cfg.out_dir / "ranker"
out_dir.mkdir(exist_ok=True)
movies, links = load_movies(cfg), pd.read_parquet(cfg.out_dir / "links.parquet")
catalog = np.sort(movies["movieId"].to_numpy())
cat_idx = pd.Index(catalog)
split = load_user_split(cfg)
train, val = split.train, split.val
z = np.load(cfg.out_dir / "candidates" / "candidates_val.npz")
users, cand, n_main = z["users"], z["cand"].astype(np.int64), int(z["n_main"])
t_req = request_times(val, users)
emb_c = content.center_embeddings(content.load_embeddings(cfg.out_dir / "item_embeddings.npz", catalog, "emb384"))
gm, gnames = genre_matrix(movies, catalog)
tmdb = content.load_tmdb_cache(ROOT / "data" / "raw" / "tmdb", links["tmdbId"])
rel = content.release_dates(movies, links, tmdb).reindex(catalog)
rel_ts = np.array([x.timestamp() if pd.notna(x) else np.nan for x in rel])
counts = np.bincount(cat_idx.get_indexer(train.loc[train["label"] == 1, "item_id"]), minlength=len(catalog))

# frozen train-only scores: ID-only two-tower (main lane, known items) and NCF
V, w = np.load(TT / "a_id_n20" / "item_vecs.npy"), np.load(TT / "a_id_n20" / "user_tower.npz")
retr = TwoTowerRetrieval(V, w["W"], w["b"], 20)
retr.fit(train, catalog)
known_tt = np.load(TT / "a_id_n20" / "known.npy")
tt_score = np.full(cand.shape, np.nan)
for a in range(0, len(users), 2000):
    b = min(a + 2000, len(users))
    rows = np.array([retr.u_index.get(u, -1) for u in users[a:b]])
    s = np.einsum("cwd,cd->cw", V[np.maximum(cand[a:b], 0)], retr.U[np.maximum(rows, 0)])
    ok = (cand[a:b] >= 0) & (np.arange(cand.shape[1]) < n_main)[None, :] & known_tt[np.maximum(cand[a:b], 0)] & (rows >= 0)[:, None]
    tt_score[a:b] = np.where(ok, s, np.nan)
ncf_scorer = NCFScorer.load(NC / "ncf_main" / "weights.npz")
ncf_logit = np.load(NC / "ncf_main" / "scores.npy").astype(float)
ncf_user_known = np.isin(users, ncf_scorer.user_ids)
ncf_item_known = counts > 0
stages.done("inputs")

args = dict(cand=cand, n_main=n_main, catalog=catalog, genre_m=gm, genre_names=gnames, emb_c=emb_c, rel_ts=rel_ts,
            tt_score=tt_score, ncf_logit=ncf_logit, ncf_user_known=ncf_user_known, ncf_item_known_cat=ncf_item_known)
out = build_features(train, users, t_req, **args)
stages.done("features")

# labels (validation likes) are attached AFTER the features exist
truth = {u: cat_idx.get_indexer(g["item_id"].to_numpy()) for u, g in val[val["label"] == 1].groupby("user_id")}
t_list = [truth[u] for u in users]
hit = hit_matrix(out["cand_f"].astype(np.int64), t_list)
y = hit[out["row_user"], out["row_slot"]].astype(np.int8)

# ---- real-data leakage proofs
scr = val.assign(item_id=np.random.default_rng(0).permutation(val["item_id"].to_numpy()),
                 label=np.random.default_rng(1).permutation(val["label"].to_numpy()))
assert (request_times(scr, users) == t_req).all(), "request time depends on validation items/labels"
sub = np.random.default_rng(0).choice(len(users), 300, replace=False)
sub.sort()
sub_args = {**args, "cand": cand[sub], "tt_score": tt_score[sub], "ncf_logit": ncf_logit[sub], "ncf_user_known": ncf_user_known[sub]}
f0 = build_features(train, users[sub], t_req[sub], **sub_args)
f1 = build_features(pd.concat([train, val], ignore_index=True), users[sub], t_req[sub], **sub_args)
pi = f0["cols"].index("pop_asof")
cols_ok = [i for i in range(len(f0["cols"])) if i != pi]
assert np.array_equal(f0["X"][:, cols_ok], f1["X"][:, cols_ok], equal_nan=True), "a user-side feature changed when validation rows were added"
chk = np.random.default_rng(2).choice(len(f0["row_user"]), 1500, replace=False)
# brute-force recount against TRAIN ONLY (validation rows excluded by the strict rule only for this user; recount from train)
tl = train[train["label"] == 1]
by_item = {k: g["timestamp"].to_numpy() for k, g in tl.groupby("item_id")}
bad = 0
for i in chk[:600]:
    u_row, item = f0["row_user"][i], catalog[f0["cand_f"][f0["row_user"][i], f0["row_slot"][i]]]
    ts = by_item.get(item, np.array([], dtype=np.int64))
    bad += int((ts < t_req[sub][u_row]).sum()) != int(f0["X"][i, pi])
assert bad == 0, f"{bad} as-of popularity values disagree with a brute-force recount"
print("leakage proofs passed: request times ignore validation items/labels; adding all validation rows leaves user-side "
      "features identical; as-of popularity matches a brute-force recount (600 rows)", flush=True)

np.savez(out_dir / "features.npz", X=out["X"], cols=np.array(out["cols"]), row_user=out["row_user"], row_slot=out["row_slot"],
         cand_f=out["cand_f"], lane_f=out["lane_f"], ncf_f=out["ncf_f"], users=users, t_req=t_req, session_len=out["session_len"],
         in_session=out["in_session"], hour=out["hour"], dow=out["dow"], y=y)
cs = out["cols"]
X = out["X"]
nr = len(y)
print(f"rows {nr} ({nr / len(users):.0f}/user), removed as not yet released at request time: {out['removed_future']} "
      f"({out['removed_future'] / (cand >= 0).sum():.1%}); positives {int(y.sum())} ({y.mean():.2%}); features {len(cs)}", flush=True)
print("NaN share per feature:", {c: round(float(np.isnan(X[:, i]).mean()), 3) for i, c in enumerate(cs) if np.isnan(X[:, i]).any()}, flush=True)
stages.done("save + checks")
