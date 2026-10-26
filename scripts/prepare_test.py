"""Module 8 preparation for the one-shot test evaluation. NO TEST ACCESS: uses train and validation only.

Builds every frozen artifact the unlock script will load:
  frozen/ (shared)   item-kNN neighbour lists (from train), final rankers M0/M1/M2 refit once on ALL validation users
  real/              history = train + validation: two-tower user vectors, content profiles, kNN history matrix, NCF fold-in,
                     230-slot candidate lists for every user, two-tower / NCF (fold-in and stale) scores on those lists
  rehearsal/         the same with history = train only (so the validation period can act as a fake test period)
A manifest of sha256 hashes is written; the unlock script refuses to run if any artifact changed."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from scipy import sparse

from ott_recsys import content, testprep
from ott_recsys.baselines import ContentRetrieval, knn_neighbors
from ott_recsys.config import Config
from ott_recsys.data import load_movies, load_user_split
from ott_recsys.hashing import hash_files
from ott_recsys.memguard import Stages
from ott_recsys.ncf import NCFScorer, fold_in_users
from ott_recsys.rankfeat import BASE, CTX_SESSION, CTX_TIME
from ott_recsys.twotower import TwoTowerRetrieval

ap = argparse.ArgumentParser()
ap.add_argument("--rehearsal", action="store_true")
ap.add_argument("--manifest-only", action="store_true", help="only (re)write the sha256 manifest of the existing artifacts")
args = ap.parse_args()
cfg = Config()
stages = Stages()
TP = cfg.out_dir / "test_prep"
FZ, MD = TP / "frozen", TP / ("rehearsal" if args.rehearsal else "real")
FZ.mkdir(parents=True, exist_ok=True)
MD.mkdir(parents=True, exist_ok=True)
TT, NC, RK = cfg.out_dir / "two_tower", cfg.out_dir / "ncf", cfg.out_dir / "ranker"
EXTRA_FROZEN = [TT / "a_id_n20" / "item_vecs.npy", TT / "a_id_n20" / "user_tower.npz", NC / "ncf_main" / "weights.npz", cfg.out_dir / "item_embeddings.npz"]


def write_manifest(mode: str) -> int:
    files = sorted(p for d in (FZ, MD) for p in d.glob("*") if p.name not in ("manifest.json", "facts.json")) + EXTRA_FROZEN
    (MD / "manifest.json").write_text(json.dumps({"mode": mode, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "artifacts": hash_files(files)}, indent=1))
    return len(files)


if args.manifest_only:
    print("manifest rewritten for", write_manifest("rehearsal" if args.rehearsal else "real"), "files")
    sys.exit(0)

movies = load_movies(cfg)
catalog = np.sort(movies["movieId"].to_numpy())
cat_idx = pd.Index(catalog)
split = load_user_split(cfg)
train, val = split.train, split.val  # validation is allowed; the test portion is never touched here
history = testprep.history_frame(train, val, args.rehearsal)
emb_c = content.center_embeddings(content.load_embeddings(cfg.out_dir / "item_embeddings.npz", catalog, "emb384"))
train_counts = np.bincount(cat_idx.get_indexer(train.loc[train["label"] == 1, "item_id"]), minlength=len(catalog)).astype(float)
hist_counts = np.bincount(cat_idx.get_indexer(history.loc[history["label"] == 1, "item_id"]), minlength=len(catalog)).astype(float)
known_items = train_counts > 0  # frozen: the ID models only know movies with a train like
facts = {"mode": "rehearsal" if args.rehearsal else "real", "history_rows": int(len(history)), "history_users": int(history["user_id"].nunique())}

# ---------------------------------------------------------------- 1. frozen, shared
if not (FZ / "knn_neighbors.npz").exists():
    st = knn_neighbors(train, catalog, [0.0], 500)  # train only, k=500, shrink 0 (Module 2 choice)
    np.savez(FZ / "knn_neighbors.npz", idx=st["neighbors"][0.0][0], sim=st["neighbors"][0.0][1])
    stages.done("frozen kNN neighbours")
z = np.load(RK / "features.npz")
Xf, cols, yf, ru = z["X"], [str(c) for c in z["cols"]], z["y"], z["row_user"].astype(np.int64)
CI = {c: i for i, c in enumerate(cols)}
f_base = [CI[c] for c in BASE + [c for c in cols if c.startswith("g_")]]
VARIANTS = {"M0": f_base, "M1": f_base + [CI[c] for c in CTX_TIME], "M2": f_base + [CI[c] for c in CTX_TIME] + [CI[c] for c in CTX_SESSION]}
for name, idx in VARIANTS.items():
    path = FZ / f"final_{name}.json"
    if path.exists():
        continue
    import xgboost as xgb
    from ott_recsys.ranker import PARAMS
    n_trees = int(np.median(json.loads((RK / f"oof_R_{name}.json").read_text())["best_iter"]))  # median of the robustness fold best iterations
    params = {k: v for k, v in PARAMS.items() if k not in ("early_stopping_rounds", "n_estimators", "learning_rate")}
    t0 = time.time()
    m = xgb.XGBRanker(objective="rank:ndcg", tree_method="hist", n_estimators=n_trees, learning_rate=0.03, n_jobs=16, random_state=0, **params)
    m.fit(Xf[:, idx], yf, qid=ru)
    b = m.get_booster()
    b.feature_names = [cols[i] for i in idx]
    b.save_model(str(path))
    chk = xgb.Booster()
    chk.load_model(str(path))
    sample = Xf[:2000][:, idx]
    assert np.allclose(chk.predict(xgb.DMatrix(sample, feature_names=[cols[i] for i in idx])), m.predict(sample), atol=1e-5), "saved ranker disagrees with the fitted one"
    facts[f"ranker_{name}"] = {"trees": n_trees, "features": len(idx), "fit_seconds": round(time.time() - t0)}
    stages.done(f"final ranker {name} ({n_trees} trees)")

# ---------------------------------------------------------------- 2. history-dependent artifacts
users_all = np.sort(history["user_id"].unique())
V, w = np.load(TT / "a_id_n20" / "item_vecs.npy"), np.load(TT / "a_id_n20" / "user_tower.npz")
tt = TwoTowerRetrieval(V, w["W"], w["b"], 20)
tt.fit(history, catalog)
tt_ids = np.array(sorted(tt.u_index))
tt_U = tt.U[[tt.u_index[u] for u in tt_ids]]
ct = ContentRetrieval(emb_c)
ct.fit(history, catalog)
ct_ids = np.array(sorted(ct.u_index))
ct_P = ct.profiles[[ct.u_index[u] for u in ct_ids]]
likes = history[history["label"] == 1]
Xh = sparse.csr_matrix((np.ones(len(likes), np.float32), (pd.Index(users_all).get_indexer(likes["user_id"]), cat_idx.get_indexer(likes["item_id"]))),
                       shape=(len(users_all), len(catalog)))
sparse.save_npz(MD / "knn_history.npz", Xh)
np.savez(MD / "history_vectors.npz", users_all=users_all, tt_ids=tt_ids, tt_U=tt_U.astype(np.float32), ct_ids=ct_ids, ct_P=ct_P.astype(np.float32),
         hist_counts=hist_counts, train_counts=train_counts)
stages.done("history vectors")

# ---------------------------------------------------------------- 3. NCF: stale (train only) and fold-in (history positives)
stale = NCFScorer.load(NC / "ncf_main" / "weights.npz")
fold, fh = fold_in_users(stale, likes["user_id"].to_numpy(), likes["item_id"].to_numpy(), neg_k=4, pop_frac=0.5, epochs=3, lr=1e-3, seed=0,
                         log=lambda m: print(m, flush=True))
fold.save(MD / "ncf_foldin.npz")
common = stale.user_ids[np.isin(stale.user_ids, fold.user_ids)]
old_ug, new_ug = stale.a["ug"][stale._u.get_indexer(common)], fold.a["ug"][fold._u.get_indexer(common)]
rel_change = float(np.mean(np.linalg.norm(new_ug - old_ug, axis=1) / np.linalg.norm(old_ug, axis=1)))  # how far fold-in moved the users
facts["ncf_foldin"] = {"epochs": 3, "lr": 1e-3, "loss_epoch1": fh[0]["loss"], "loss_epoch3": fh[-1]["loss"], "users_with_embedding": int(len(fold.user_ids)),
                       "users_new_vs_stale": int(len(set(fold.user_ids) - set(stale.user_ids))), "stale_users": int(len(stale.user_ids)),
                       "mean_relative_change_of_existing_user_embeddings": rel_change,
                       "positives_dropped_unknown_item": int((~np.isin(likes["item_id"].to_numpy(), stale.item_ids)).sum()), "positives_total": int(len(likes))}
stages.done("NCF fold-in")

# ---------------------------------------------------------------- 4. candidate lists and frozen scores for every user
seen = testprep.seen_map(history, catalog)
lane_pool = np.flatnonzero(~known_items)  # movies unknown to the frozen ID models (includes movies first liked in validation)
cand = testprep.candidate_lists(users_all, seen, V, tt_ids, tt_U, ct_ids, ct_P, emb_c, lane_pool, hist_counts)
tt_score = testprep.two_tower_scores(users_all, cand, V, tt_ids, tt_U, known_items)
ncf_stale, ncf_stale_known = testprep.ncf_matrix(stale, users_all, cand, catalog)
ncf_fold, ncf_fold_known = testprep.ncf_matrix(fold, users_all, cand, catalog)
np.savez(MD / "lists.npz", users=users_all, cand=cand.astype(np.int32), tt_score=tt_score.astype(np.float32), ncf_stale=ncf_stale, ncf_fold=ncf_fold,
         ncf_stale_user_known=ncf_stale_known, ncf_fold_user_known=ncf_fold_known, ncf_item_known_cat=known_items, lane_pool=lane_pool)
valid = cand >= 0
facts["lists"] = {"users": int(len(users_all)), "mean_slots": float(valid.sum(1).mean()), "main_slots_per_user": float((valid[:, :200]).sum(1).mean()),
                  "lane_slots_per_user": float((valid[:, 200:]).sum(1).mean()), "share_slots_ncf_nan_stale": float(np.isnan(ncf_stale[valid]).mean()),
                  "share_slots_ncf_nan_foldin": float(np.isnan(ncf_fold[valid]).mean()), "lane_pool_size": int(len(lane_pool)),
                  "users_without_history_vector": int(len(users_all) - len(tt_ids))}
if args.rehearsal:  # strong plumbing check: rebuilt lists must contain exactly the Module 5 items (history = train, same models)
    zc = np.load(cfg.out_dir / "candidates" / "candidates_val.npz")
    pos = pd.Index(users_all).get_indexer(zc["users"])
    mine, ref = cand[pos], zc["cand"]
    set_equal = np.array([set(m[:200]) == set(r[:200]) and set(m[200:]) - {-1} == set(r[200:]) - {-1} for m, r in zip(mine, ref)])
    order_only = int((~(mine == ref).all(1) & set_equal).sum())
    facts["rehearsal_lists_vs_module5"] = {"users": int(len(pos)), "identical_item_sets": int(set_equal.sum()), "identical_order": int((mine == ref).all(1).sum()),
                                           "order_differs_only_among_tied_scores": order_only}
    assert set_equal.all(), f"rebuilt candidate SETS differ from Module 5 for {int((~set_equal).sum())} users"
stages.done("candidate lists + scores")

(MD / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
write_manifest(facts["mode"])
print(json.dumps(facts, indent=1, default=float), flush=True)
stages.done("manifest")
