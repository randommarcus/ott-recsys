# Progress Log

## 2026-10-25 — Module 0: Project setup
- Python 3.12 venv (.venv) via uv; system Python is 3.14, which TensorFlow does not support.
- TensorFlow 2.20 (CPU) + Keras 3 verified. Also installed: xgboost, sentence-transformers, pymilvus, boto3.
- Folder structure, .gitignore, .env.example created.
- MovieLens small (100k ratings) downloaded to data/raw/ (not committed; re-run scripts/download_movielens.py).
- Note: WSL reports ~9 GB RAM, not 16 GB. Keep subsets small; heavy jobs to Colab.

## 2026-10-25 — Module 1: Data understanding and splitting
Code: `src/ott_recsys/config.py` (all dataset settings, env-overridable), `src/ott_recsys/data.py`, `scripts/run_module1.py`. Output: `data/processed/ml-latest-small/{train,val,test,movies,links}.parquet` (not committed).
- **Audit (ml-latest-small):** 100,836 ratings, 610 users, 9,724 rated movies, 1996-03 to 2018-09. Sparsity 98.3%. No duplicates or missing ratings. 8 missing TMDB ids, 34 movies without genres.
- **Long tail:** median user has 70 ratings (min 20, max 2,698); median movie has 3 ratings; 3,446 movies have exactly 1 rating.
- **Interactions:** every rating = a watch; label = rating >= 4 (48.2% positive). The label is "liked", not "watched": low ratings are still real watches.
- **Sessions (30-min gap):** 7,145 sessions, mean length 14.1, median 2; 3,235 single-item sessions.
- **Caveat - bulk bursts:** 57% of ratings sit in bursts (>= 30 ratings within 10 min), and 83% of ratings are in sessions of 20+. Many "sessions" are people back-filling their history in one sitting, not real viewing sessions. Session-length features are weak signals here.
- **Time features:** hour and day-of-week are UTC. MovieLens has no user timezone, so they are not local viewing time. Device is NOT in the data and will be simulated later.
- **Split (global time cutoff, 70/15/15):** train to 2013-08-18, val to 2016-11-19, test to 2018-09-24. Leakage asserts pass.
- **Cold start (key finding):** only 9 of 96 test users appear in train; 95% of test rows come from users unseen in train, and 27% of test rows are on items unseen in train. Cause: 59% of users rate everything within one day, so a calendar cutoff puts each user wholly on one side.
- **Open decision:** a global cutoff cannot evaluate personalisation on this dataset. Proposed: also build a per-user temporal split (each user's last ~15% of ratings in time for test), keeping the global cutoff as a cold-start report.
- Fixed: pyarrow added to requirements.

## 2026-10-25 — Module 2: per-user split, evaluation harness, baselines
Code: `data.py` (per_user_split, UserSplit lock), `eval.py` (harness), `baselines.py`, `experiments.py`, `scripts/build_user_split.py`, `scripts/run_module2.py`. Report: `reports/module2_validation.txt`.
- **Split:** per user, sorted by (timestamp, movieId); last 20% test, 10% before = val, rest train (floor). 610 users kept (min 10 interactions). Rows: train 71,078 / val 9,818 / test 19,940. The global-cutoff split is kept separately for the cold-start report.
- **TEST IS LOCKED.** The user said not to touch test until approved. `UserSplit.test` raises `TestLocked` until `unlock_test()`. Tests: runtime (pipeline runs with a test loader that fails if called), static (AST scan: tuning code may not reference `.test`/`unlock_test`), and a one-off run of the real script with `test.parquet` reads made fatal. Test cold-start numbers are deferred until unlock.
- **Protocol:** truth = items rated >= 4 in the eval portion; rank every catalog movie the user has not seen in fit data; Recall@10/50/200, NDCG@10, Coverage@10; warm/cold views reported separately; 95% bootstrap CI over users. Validation mode = fit on train. (Test mode, later = fit on train+val.)
- **Validation results (558 users with >= 1 val like):** NDCG@10 random 0.0003 / global pop 0.0605 / hour pop 0.0628 / item-kNN 0.1016. Recall@50: 0.004 / 0.190 / 0.188 / 0.298.
- **Hour-of-day finding:** hour-pop beats global pop by a margin well inside the CIs, and the shuffled-hours control (0.0593) is also inside them: the UTC hour carries no usable signal on this data. Best smoothing was the grid edge (lam=5000), i.e. the model prefers to collapse to global popularity.
- **Cold items:** 7.6% of val rows and 279 of 4,464 val likes (6.3%) are on items never seen in train. All baselines score 0 on them by construction.
- **Sample users (validation likes shown):** 0 of 10 hits for all 5 sampled users and all 3 baselines. Popularity gives every user nearly the same list; item-kNN personalises (sci-fi fans get sci-fi, an old-westerns user gets westerns) but still misses.
- Caveat: hyperparameters were tuned on validation, so validation numbers are slightly optimistic; kNN shrink=50 and hour-pop lam=5000 sit on grid edges.

## 2026-10-25 — Module 2b: scale up to MovieLens 25M and re-baseline
Code: `config.py` (sampling, memory limit), `data.load_raw` (user sampling), `eval.py` (chunked scoring, fast top-K, paired_diff), `baselines.knn_neighbors` (block-wise kNN), `memguard.py` + `scripts/guarded_run.py` (memory guard), `.claude/settings.json` (deny rule for reading .env; did not exist before). Reports: `reports/module1_audit_ml-25m_u20000_s42.txt`, `reports/module2_validation_<run>.{txt,json}`.
- **Subset:** seeded (seed 42) random sample of 20,000 of the 162k users (all have >= 20 ratings), keeping every rating of each. 3,101,914 ratings, 33,392 movies, sparsity 99.5%. Sampled ids + seed saved to `data/processed/ml-25m_u20000_s42/sampled_users.json` (gitignored; recreated by the seed). Config printed at the top of each report.
- **Memory guard:** `guarded_run.py` kills a run above 7.5 GB (or when system free memory < 0.8 GB). It never tripped: peaks were 0.66 GB (audit), 1.16 GB (split), 3.48 GB (baselines).
- **Timing (this laptop):** download+unzip 71 s; sample+audit+global split 10 s; per-user split 2 s; tune+evaluate all baselines 6.7 min (wall 404 s).
- **Split (25M subset):** train 2,187,571 / val 301,690 / test sealed. 18,530 of 20,000 users have >= 1 validation like.
- **Grids:** hour-pop lam widened to 5M (best interior: window 3, lam 50,000). kNN widened twice: best k=500 (interior of 100-1500), shrink=0 (natural floor). Small dataset unchanged by widening (k=100, shrink=50; same as the earlier wider grid, where it was interior).
- **Results (validation, NDCG@10 / Recall@50):** 25M: random 0.0004/0.001, global pop 0.0555/0.180, hour pop 0.0559/0.182, kNN 0.0980/0.306. Small: 0.0007/0.005, 0.0605/0.190, 0.0628/0.188, 0.1014/0.298. kNN beats popularity by ~1.8x NDCG on both; paired CI excludes 0.
- **Hour of day:** small data: nothing (all paired CIs include 0). 25M: a real but tiny effect: Recall@10 +0.0016 over both global pop and the shuffled-hours control (CI excludes 0; ~2.5% relative); NDCG@10 +0.0004 (borderline, CI touches 0 vs global pop). Not practically useful; hour is UTC. Shuffled control equals global pop, as it should.
- **Cold items:** the share fell sharply with scale: 0.77% of validation rows and 0.36% of validation likes (509 of 141,159) are on items unseen in train, versus 7.6% / 6.3% on the small data. All baselines score 0 on them. A per-user split does not create "new releases" (every item exists across the whole window), so the bandit/new-release story needs its own evaluation (e.g. a time-ordered simulation).
- Test portion still sealed; the lock tests plus a real-script run with test.parquet reads made fatal passed again.
- **Housekeeping flag:** a TMDB API key was found in `.env.example` (tracked file) - it must live in `.env` only. Not committed.

## Next
- Wait for approval before unlocking test. Next module candidate: SBERT item embeddings + two-tower candidate generation (must beat item-kNN NDCG@10 ~0.098 and give non-zero cold-item recall).
