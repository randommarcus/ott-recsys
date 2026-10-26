# Module 8 protocol: the one-shot test evaluation (pre-registered)

Status: written and committed BEFORE the test portion is read. Test has not been unlocked.
The unlock is performed once, by `scripts/unlock_test.py --confirm UNLOCK-TEST-ONCE`, and only after the project owner says so.

## 1. What is frozen and what is updated through validation

Frozen (fitted on train only, or chosen on validation; never refit before or during the test run):
- ID-only two-tower item vectors and user tower (numpy arrays); NCF item embeddings and MLP; SBERT embeddings and their centering;
  item-kNN neighbour lists (k = 500, shrink 0, computed from train: `test_prep/frozen/knn_neighbors.npz`).
- Rankers M0, M1, M2: refit ONCE on all 18,530 validation users (features built from train-only histories, validation labels), with the
  robustness-check settings (learning rate 0.03, no early stopping). Number of trees = median over folds of the robustness best iterations:
  M0 188, M1 65, M2 351. Files: `test_prep/frozen/final_M{0,1,2}.json`.
- Rules: 30-minute session rule, release-by-request-time filter, 365-day bandit shelf, k = 2 lane slots, UCB constant c = 0.2, epsilon 0.05.

Updated through validation WITHOUT retraining (histories and counts only): history = train + validation interactions.
- Two-tower user vectors (last 20 likes), content profiles, genre shares, item-kNN user matrix, seen-item exclusion, as-of-request popularity,
  session features (strictly before the request time).
- Request time of a test user = timestamp of their first test interaction (timestamp only; its item and label are never used).
- Lane pool = movies unknown to the frozen ID models (no train like). This includes movies first liked during validation.
- Candidate lists: 200 main-lane slots (frozen two-tower, user vector from the history, seen items removed) + 30 appended lane slots
  (centered-SBERT content search inside the lane pool), for every user; the release filter is applied afterwards at the test request time.

## 2. NCF: fold-in is primary, stale is the sensitivity check

- PRIMARY: fold-in. Re-optimise ONLY the user embeddings on train + validation positives; item embeddings and the whole MLP stay frozen
  (asserted in code). Fixed in advance: 3 epochs, learning rate 1e-3, 4 negatives per positive (half popularity-weighted), seed 0. Positives on items the
  frozen model has no embedding for are dropped; users without a trained embedding start from a seeded random vector.
- SENSITIVITY: stale NCF (train-only weights, exactly what the rankers saw while being fitted).
- The rankers' primary numbers (M0, M1, M2) use fold-in NCF features. The row "M2 with STALE NCF features" and the NCF-order rows for both variants are reported
  next to them; the paired difference shows how much the staleness choice matters.

## 3. The lane and the bandit at test time (a SIMULATION, not an A/B test)

- Lane in the lists: as in section 1. The bandit simulation uses the global-cutoff split: the history is global train + global validation, the evaluation window is the
  global TEST window. Arms = movies released from 6 months before that window starts, with no ratings before it. Hidden like-rates come from test-window ratings with the
  usual Beta shrinkage (5 pseudo-ratings toward the global rate). Requests = the window's real session starts (30-minute rule). Arm sets: >= 20, >= 5, >= 1 test-window ratings;
  the 629-type set (>= 5 ratings) is primary. k = 2, 365-day shelf, 30 paired seeds. Reward = "liked given rated" (ignores click probability).
- Strategies fixed in advance: UCB c = 0.2 (primary), UCB1 (c = sqrt 2), UCB1-Tuned, Thompson (Beta), epsilon-greedy 0.05, greedy, random, oracle;
  sensitivity: the per-set c* chosen in Module 7b (0.35 for the 239-type set, 0.1 for the 2,031-type set). No exploration constant is chosen on test data.
  Content-prior variants are dropped (they hurt in Module 7b).
- Slot cost: reserving the last 2 (or 3) of 20 slots for the lane, and 2 slots inside the top 10, on the M2 main-lane ranking.

## 4. Methods compared on the same lists

Every test user with at least one liked test item gets the same 230-slot list (after the release filter). Re-ranking rows: random, popularity (train + validation likes),
two-tower's own order, NCF order (fold-in and stale), 50/50 rank blend (fold-in and stale), rankers M0, M1, M2 (fold-in NCF), M2 with stale NCF, and the oracle order.
Metrics: NDCG@10 and Recall@20, end-to-end (truth = all test likes) and reachable-only, with paired bootstrap intervals over users.
Retrieval over the whole catalog: ID-only two-tower, item-kNN, centered content, popularity (Recall@10/50/200, NDCG@10, warm and cold views).

## 5. Headline claims (the ONLY findings)

| id | Claim | Metric / population |
|---|---|---|
| P1 | M2 ranker (fold-in NCF) vs the two-tower's own order | paired difference, end-to-end NDCG@10, all test users |
| P2 | M0 ranker (fold-in NCF) vs the two-tower's own order | same |
| P3 | Context gain: M2 - M0 | same |
| P4 | Two-tower vs item-kNN | paired difference, Recall@200 over the whole catalog |
| P5 | Bandit simulation, 629-type arm set: UCB c = 0.2 vs UCB1, and vs Thompson | paired difference in likes per 1,000 impressions over the whole request stream |

## 6. Interpretation rule

- Headline claims are P1-P5 only. Everything else is secondary or descriptive, even if an interval excludes 0.
- If a headline result fails its prediction, it is reported as it is.
- No re-running after seeing results. The script runs once.
- Any later bug fix is shown as a LABELLED ADDITION next to the original numbers; it never replaces them.
- Pipeline code may receive bug fixes before the unlock (rehearsal); the analysis choices in this file may not change. The unlock records git commit and source hashes.

## 7. Unlock mechanics and the crash rule

- The unlock log `reports/TEST_UNLOCK_LOG.json` is written BEFORE any test data is read. The script refuses to run if the log exists in the working tree or anywhere in git history.
- It refuses unless the working tree is clean, this protocol and the predictions file are committed and unchanged, and every frozen artifact matches its recorded sha256.
- A rerun after a crash is allowed only with `--after-crash`, only if the log says "started", and only if NO test metric was written or printed before the crash.
  The script sets `test_metrics_emitted` immediately before it first emits a metric; if a crash happens after that point the run is SPENT, no rerun is allowed,
  and the report says so. Attempts are counted in the log and the report.
- Nothing else in the repository may import the unlock script or its guard module, and the script may not call any fitting or selection function
  (repository-wide static tests).
- `--rehearsal` runs the identical pipeline end to end on a fake test set built from validation data (history = train only), with the real `test.parquet` read made fatal,
  a separate log and report. Its numbers are in-sample and meaningless; it exists to find bugs before the single real run.

## 8. What could make test numbers misleading

1. The test window is twice as long as validation (last 20% of each user's history vs 10%): about twice as many likes per user, so Recall@20's denominator roughly doubles and
   absolute Recall falls while NDCG@10 may rise. Absolute test numbers are not comparable with validation numbers; only method-to-method comparisons within test are.
2. Staler frozen ID models (two-tower item vectors, NCF).
3. Feature shift: rankers learned on train-only-history features; test features use train + validation histories.
4. Movies first liked during validation are unknown to the frozen ID models.
5. Rating bursts continue across the validation/test boundary (favours recency-aware methods and session features).
6. The per-user split still lets models learn from other users' later ratings (calendar leakage).
7. The same users appear in validation and test (rankers were fitted on their validation labels): this measures the next period for known users, not new users.
8. Many methods, views and metrics are computed in one shot; only P1-P5 are pre-registered.
9. MovieLens ratings are not viewing; hours are UTC; device is simulated and not tested; TMDB release dates drive the release filter.
10. The bandit part is a simulation on thin test-window data with "liked given rated" rewards.
