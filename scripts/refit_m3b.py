"""Module 9 demo model: refit M3b ONCE (M2 features + SIMULATED device columns, trained on SYNTHETIC labels).

SYNTHETIC SCENARIO, PIPELINE DEMONSTRATION, NOT A FINDING: devices are simulated and, on a mobile request, 60% of validation likes of Drama-without-Comedy
titles are removed from the labels (exactly the Module 6 scenario: same seeds). Module 6 only kept out-of-fold scores; the serving demo needs a real model file.

Data: the Module 6 VALIDATION rows (ranker/features.npz) and validation labels. The test portion is never read (static scan in tests/test_module2.py).
Settings: Module 6 ranker parameters, number of trees = median of the Module 6 out-of-fold best iterations (no early stopping, no validation set).
Output: data/processed/<run>/serving/final_M3b.json + final_M3b_facts.json. Not part of the Module 8 frozen manifest."""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import xgboost as xgb

from ott_recsys.config import Config
from ott_recsys.context import DEVICES, device_onehot, genre_matrix, simulate_device, synthetic_truth
from ott_recsys.data import load_movies, load_user_split
from ott_recsys.memguard import Stages
from ott_recsys.rankfeat import BASE, CTX_SESSION, CTX_TIME
from ott_recsys.ranker import PARAMS
from ott_recsys.rerank import hit_matrix

cfg = Config()
stages = Stages()
RK, SV = cfg.out_dir / "ranker", cfg.out_dir / "serving"
SV.mkdir(exist_ok=True)
z = np.load(RK / "features.npz")
X, cols = z["X"], [str(c) for c in z["cols"]]
row_user, row_slot, cand_f, users = z["row_user"].astype(np.int64), z["row_slot"].astype(np.int64), z["cand_f"].astype(np.int64), z["users"]
U = len(users)
movies = load_movies(cfg)
catalog = np.sort(movies["movieId"].to_numpy())
cat_idx = pd.Index(catalog)
val = load_user_split(cfg).val  # validation only
truth = {u: cat_idx.get_indexer(g["item_id"].to_numpy()) for u, g in val[val["label"] == 1].groupby("user_id")}
t_list = [truth[u] for u in users]

# ---- the Module 6 synthetic scenario, same seeds
dev = simulate_device(U, seed=0)
dev_oh = device_onehot(dev)
gm, gnames = genre_matrix(movies, catalog)
drama_nc = (gm[:, gnames.index("Drama")] == 1) & (gm[:, gnames.index("Comedy")] == 0)
t_syn, removed = synthetic_truth(t_list, dev, drama_nc, 0.6, 2)
report = next((ROOT / "reports").glob("module6_ranker_*.txt")).read_text()
expected = int(re.search(r"\((\d+) likes removed", report).group(1))
assert removed == expected, f"scenario differs from Module 6: {removed} vs {expected} likes removed"
y = hit_matrix(cand_f, t_syn)[row_user, row_slot].astype(int)

# ---- features exactly as Module 6's M3b
CI = {c: i for i, c in enumerate(cols)}
idx = [CI[c] for c in BASE + [c for c in cols if c.startswith("g_")]] + [CI[c] for c in CTX_TIME] + [CI[c] for c in CTX_SESSION]
names = [cols[i] for i in idx] + [f"dev_{d}" for d in DEVICES]
meta = json.loads((RK / "oof_M3b.json").read_text())
assert names == meta["features"], "feature list differs from the Module 6 M3b model"
Xv = np.concatenate([X[:, idx], dev_oh[row_user]], axis=1)
n_trees = int(np.median(meta["best_iter"]))
params = {k: v for k, v in PARAMS.items() if k not in ("early_stopping_rounds", "n_estimators")}
stages.done(f"inputs: {Xv.shape[0]} rows x {Xv.shape[1]} features, {int(y.sum())} synthetic positives, {n_trees} trees")

t0 = time.time()
m = xgb.XGBRanker(objective="rank:ndcg", tree_method="hist", n_estimators=n_trees, n_jobs=16, random_state=0, **params)
m.fit(Xv, y, qid=row_user)
b = m.get_booster()
b.feature_names = names
path = SV / "final_M3b.json"
b.save_model(str(path))
chk = xgb.Booster()
chk.load_model(str(path))
sample = Xv[:5000]
assert np.allclose(chk.predict(xgb.DMatrix(sample, feature_names=names)), m.predict(sample), atol=1e-5), "saved model disagrees with the fitted one"

# plumbing check only (NOT performance): the final model's scores should rank like the out-of-fold scores
oof = np.load(RK / "oof_M3b.npy")
sel = np.random.default_rng(0).choice(len(y), 300_000, replace=False)
corr = float(np.corrcoef(m.predict(Xv[sel]), oof[sel])[0, 1])
facts = {"scenario": "SYNTHETIC device effect, pipeline demonstration, not a finding", "rows": int(len(y)), "users": int(U), "positives": int(y.sum()),
         "likes_removed_matches_module6": removed, "trees": n_trees, "features": names, "fit_seconds": round(time.time() - t0),
         "corr_with_module6_oof_scores_sample": round(corr, 3), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
(SV / "final_M3b_facts.json").write_text(json.dumps(facts, indent=1))
print(json.dumps(facts, indent=1), flush=True)
stages.done("refit M3b")
