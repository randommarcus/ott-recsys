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

## 2026-10-25 — Module 3: TMDB descriptions, SBERT embeddings, content retrieval, new-release evaluation
Code: `tmdb.py` + `scripts/fetch_tmdb.py` (fetcher), `content.py` (texts, coverage, release dates, embeddings), `scripts/build_embeddings.py`, `baselines.ContentRetrieval` / `RandomTies`, `newrelease.py`, `scripts/run_module3.py`, `eval.evaluate(candidate_items=...)`. Reports: `reports/module3_embeddings_*.txt`, `reports/module3_content_*.txt`. Raw TMDB cache in `data/raw/tmdb/` (gitignored: TMDB terms do not allow redistribution). This product uses the TMDB API but is not endorsed or certified by TMDB.
- **Fetch:** standalone detached script, log in `logs/fetch_tmdb.log`, 33,319 unique ids at ~11.5 req/s = 36 min. Result: 32,859 ok, 440 not found (404), 0 failed; only 6 transient connection errors (retried). Key never logged or cached (tests cover this).
- **Coverage (33,392 movies):** 32,873 (98.4%) have an overview >= 40 chars; 456 empty + 15 very short + 48 with no tmdbId/404 -> 519 use fallback text; of those only 147 have MovieLens tags and 10 TMDB keywords, so 363 are title + genres only. 32,902 have a release date. Tags used only if dated <= the global train cutoff (2015-02-03).
- **SBERT** all-MiniLM-L6-v2, CPU, batch 64: 549 s for 33,392 texts (61 texts/s; slower than the 270/s benchmark because real overviews are long). 384-d, plus a PCA-64 copy. Neighbours look right (Godfather -> Part II/III, Dark Knight -> Rises/Batman Begins).
- **Content-only retrieval (per-user validation, 18,530 users), NDCG@10 / Recall@50:** raw-384 0.0019/0.010; centered-384 0.0103/0.036; PCA-64 0.0064/0.024; vs global pop 0.0555/0.180 and kNN 0.0980/0.306. As expected, content-only is far worse than collaborative baselines on the overall table.
- **Finding - center the embeddings:** raw SBERT vectors share a large common component, so averaged user profiles point at "generic" items. Removing the mean (what PCA does implicitly) multiplies NDCG@10 by ~5 (paired CI excludes 0). Centered-384 also beats PCA-64 overall, so 64 dims costs a little accuracy. Use centered embeddings in any content tower.
- **Cold view (236 users, wide CIs):** pop and kNN 0 (by construction); centered-384 Recall@200 0.040 [0.019-0.062] vs random 0.004.
- **New-release evaluation (global cutoff 2015-02-03, validation window to 2017-03-28; fit only on ratings <= cutoff):** PRIMARY = TMDB release date from cutoff-6mo on (3,493 movies; 420 evaluable users). Split into COLD part (3,341 movies with no train ratings; 360 users, 3,326 likes) and WARM part (152 movies already rated in train). SECONDARY = first rated after cutoff (6,864 movies; 378 users). Ties broken at random.
  - Cold part, competing only against other new items: Recall@50 centered-384 0.159 [0.135-0.184], PCA-64 0.145, pop 0.017, kNN 0.017, random 0.019 (pop = kNN = random, as constructed).
  - Cold part vs the whole catalog: content ~0.016 Recall@50, pop/kNN 0: new items rarely beat established ones without a hybrid model.
  - Warm part (pool of 152): popularity wins by far (Recall@10 0.60 vs content 0.24): the "primary all" pool table is dominated by these 152 items, so read the cold/warm split, not the combined row.
  - Secondary: centered-384 Recall@50 0.068 vs random 0.009 in the new-only pool.
- Test portion still sealed; lock tests + real-script run with every test.parquet read made fatal passed again. 41 tests pass.
- Timing: embeddings 9.2 min, evaluation 4.6 min, peak memory 2.0 GB.

## 2026-10-26 — Module 4: two-tower candidate generation (TensorFlow/Keras)
Code: `twotower.py` (pairs, loss, model, numpy-only `TwoTowerRetrieval`), `export.py` (Milvus-ready), `scripts/train_two_tower.py`, `scripts/run_module4_all.py` (one detached launch), `scripts/run_module4.py` (report), `eval.union_top/evaluate_lists`, `content.center_embeddings`. Report: `reports/module4_two_tower_ml-25m_u20000_s42.txt`. Predictions written and committed BEFORE the runs: `reports/module4_predictions.md`.
- **Setup:** item tower = [ID emb | projected centered SBERT] -> dense -> L2 (64-d); user tower = mean of item-tower outputs of the last N liked items -> dense -> L2. In-batch sampled softmax, tau 0.07, batch 2048, logQ from train-positive frequencies, accidental hits masked. ID dropout 25% (ID+content only). Select/early-stop on Recall@200 (tie-break Recall@50) on a seeded 3,000-user validation subsample. 1.09M training pairs. No hour/device context.
- **Leakage rule:** history = liked items with timestamp STRICTLY earlier than the target (ties never see each other), newest N. Tests include a future-invariance check.
- **Runs (9, one launch, memory guard never tripped, peak ~1.7 GB, 78 min total):** ID-only, content-only, ID+content at N=20; ID+content N=50; ID+content logQ off; plus four global-cutoff retrains (fixed epochs from the per-user runs) for the new-release evaluation. Training minutes: 4.8 / 13.5 / 10.1 / 11.4 / 7.2 (user), 3.1-9.4 (global).
- **Validation (18,530 users), Recall@200 / Recall@50 / NDCG@10:** kNN 0.584/0.306/0.098; ID-only 0.650/0.377/0.123; ID+content N20 0.635/0.371/0.118; ID+content N50 0.633/0.366/0.116; content-only 0.515/0.288/0.090; ID+content logQ-off 0.263/0.145/0.040 (worse than global popularity 0.381/0.180/0.056). ID+content beats kNN by +0.051 Recall@200 (paired CI [+0.047,+0.055]).
- **ID-only beats ID+content on warm items** (R@200 -0.015, CI excludes 0); content adds nothing for warm users. N=50 is no better than N=20.
- **logQ is essential:** without it the model learns roughly score = log(p(item|user)/popularity), i.e. it ranks by "lift", pushes popular items down, and collapses on the offline metric (coverage@10 0.57 vs 0.13). It does help the per-user cold view (R@200 0.021 vs 0.001) but not the new-release pool.
- **Cold items:** per-user cold view (236 users): ID+content 0.0008, ID-only 0.000, content-only 0.016, centered content retrieval 0.040. The ID+content model ranks never-seen items far below items it has ID vectors for (ID dropout 0.25 was not enough).
- **New releases (global cutoff; cold part, new-items pool, Recall@50):** content retrieval 0.159, ID+content 0.134, content-only 0.125, ID-only 0.025 (= random), logQ-off 0.034, pop/kNN = random. Against the whole catalog every model stays below 0.02 Recall@50 on cold movies.
- **Candidate union (100 two-tower + 100 kNN, filled from the two-tower list):** Recall@200 0.647 vs 0.635 (two-tower alone, +0.012) and 0.584 (kNN alone, +0.063); cold view unchanged at ~0. A third, content-based source is needed for cold items.
- **Users without train likes:** 45 of 20,000 (6 of the 18,530 validation users); they get popularity.
- **Caveats:** selection used the same 3,000 validation users that are inside the final table (small optimism, untestable until test is unsealed); content-only hit the 15-epoch cap (best epoch 13) so it is likely under-trained; new-release results rest on 360-420 users.
- **Export:** `data/processed/<run>/two_tower/export/` (ID+content N=20): item_vectors.parquet (INT64 pk + FLOAT_VECTOR 64 + metadata), milvus_schema.json (HNSW, IP = cosine), user_tower.npz, user_vectors.parquet. 16,942 of 33,392 items are flagged cold. Not loaded into Zilliz yet.
- Test sealed: lock tests, static scan, and a real-script run with every test.parquet read made fatal (train + report scripts) all passed. 62 tests pass.

## Next
- Wait for approval before unlocking test. Decisions for the user: (1) fix cold items in the two-tower (higher ID dropout / content-heavy tower) vs. add content retrieval as a third candidate source; (2) which model to export for Milvus (ID+content for cold coverage, or ID-only for warm accuracy); (3) then Module 5: XGBoost/NCF ranker with hour/device context on the union candidates.
