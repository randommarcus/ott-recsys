# Module 8: predictions versus actuals

Built from the committed predictions file (`reports/module8_predictions.md`, commit e3d5da3, written before the rehearsal and before any test access) and the single
test run (`reports/module8_test_results.txt`, sha256 `dd475ceab61c8a386f2f96709abfe2e80ee70d38834ed7e5796a123c34fc8c15`, unlock log `reports/TEST_UNLOCK_LOG.json`: status done, attempts 1).
The protocol, predictions and results files were not edited. Nothing was re-run. Per the interpretation rule only P1-P5 are findings; the secondary rows are descriptive.
The bandit rows are a SIMULATION, NOT an A/B test (reward = "liked given rated", ignores click probability).

## Headline predictions (the findings)

| id | Prediction (verbatim from the committed file) | Actual | Verdict |
|---|---|---|---|
| P1 | M2 (fold-in NCF) - two-tower order, e2e NDCG@10: positive (80% confident), +0.006 to +0.022 | +0.0245 [+0.0229, +0.0261] | Direction right. Magnitude ABOVE the predicted range: the whole interval lies above +0.006 to +0.022. |
| P2 | M0 (fold-in NCF) - two-tower order, e2e NDCG@10: positive (80%), +0.005 to +0.020 | +0.0232 [+0.0217, +0.0247] | Direction right. Magnitude ABOVE the predicted range: the whole interval lies above +0.005 to +0.020. |
| P3 | M2 - M0: between -0.002 and +0.004 (70%); context remains roughly null | +0.0013 [+0.0004, +0.0021] | Right: inside the predicted band (-0.002 to +0.004). The interval excludes 0, so the effect is detectable, but it is tiny (about 0.8% of M0's 0.1615). |
| P4 | two-tower - item-kNN, Recall@200 (whole catalog): positive (85%), +0.03 to +0.07 | +0.0473 [+0.0441, +0.0505]  (0.5850 vs 0.5377) | Right: positive and inside +0.03 to +0.07. |
| P5a | bandit simulation, 629-type arm set: UCB c = 0.2 better than UCB1 (90%), by > +100 likes per 1,000 impressions (80%) | +237.1 [+232.2, +242.0] likes per 1,000 impressions (807.3 vs 570.2) | Right on both parts: UCB c = 0.2 is better, and by far more than +100. |
| P5b | UCB c = 0.2 vs Thompson: within +-30 likes per 1,000 impressions (65%); the sign is close to a coin flip (Module 7b gave +26 for UCB) | +31.9 [+27.9, +35.9] (807.3 vs 775.3) | Band NARROWLY MISSED: the point estimate is just outside +-30 (the interval straddles +30). The prediction called the sign a coin flip; UCB came out ahead. |

Scorecard: every directional prediction was right (P1-P4, P5a; for P5b the sign had been called a coin flip, and UCB came out ahead). 3 of 6 magnitude claims fell inside the predicted range (P3, P4, P5a); P1 and P2 came out above their ranges and P5b narrowly outside its band.

## Secondary predictions (descriptive)

| # | Prediction (verbatim from the committed file) | Actual | Verdict |
|---|---|---|---|
| 1 | Mean liked items per test user about 13-17 (validation: 7.6), i.e. roughly double | 14.7 liked items per test user (19,530 users) | Right: inside 13-17. |
| 2 | Two-tower order, e2e: NDCG@10 0.12-0.17; Recall@20 0.15-0.21 (lower than validation's 0.232 because the denominator is larger) | two-tower order: NDCG@10 0.1383, Recall@20 0.1880 | Right on both: inside 0.12-0.17 and 0.15-0.21. |
| 3 | NCF order: fold-in better than stale by +0.002 to +0.010 NDCG@10 (65%); fold-in moved existing users' embeddings by 30% on average, so it is a real update | NCF order, fold-in minus stale: NDCG@10 +0.0075 [+0.0064, +0.0086] | Right: inside +0.002 to +0.010. |
| 4 | Ranker M2 with fold-in vs stale NCF features: within +-0.003 NDCG@10 (60%) | M2 with fold-in minus M2 with stale NCF features: NDCG@10 +0.0027 [+0.0019, +0.0034] | Right on the point estimate (inside +-0.003); the interval reaches +0.0034 and excludes 0. |
| 5 | Release-by-request-time filter removes 4-9% of slots (test requests occur later in calendar time than validation's 7.9%) | not reported | NOT EVALUATED: the unlock script did not report the share of slots removed by the release filter, and I did not re-run or re-read test to compute it (protocol: no re-running). |
| 6 | Cost of reserving 2 end slots on Recall@20: -0.010 to -0.020; NDCG@10 unchanged by construction | 2 reserved end slots: Recall@20 -0.0138 [-0.0146, -0.0130]; NDCG@10 +0.0000 | Right: inside -0.010 to -0.020, and NDCG@10 unchanged. |
| 7 | Retrieval, cold view: item-kNN and popularity score exactly 0 on cold likes; content retrieval > 0 | cold Recall@200: item-kNN 0.0000, popularity 0.0000 (two-tower also 0.0000), content 0.0221 | Right. |
| 8 | Bandit, 239-type and 2,031-type arm sets: UCB c = 0.2 beats UCB1 on both (85%) | UCB c = 0.2 vs UCB1: 727.1 vs 623.8 (239-type set); 871.6 vs 484.2 (2031-type set), likes per 1,000 impressions | Right on both sets (the reported intervals do not overlap). |

Scorecard: 7 right, 1 not evaluable (5). No secondary prediction was wrong.

## Where I was wrong
- The two rankers gained more over the two-tower's own order than I predicted (P1, P2): about +0.024 instead of +0.005 to +0.022.
- The UCB-minus-Thompson margin (P5b) landed just outside the +-30 band I gave it.
- I never made the release-filter prediction checkable (#5): the script did not report that number.

## Plain-language reading of P1-P5

- **P1 and P2 (the learned rankers vs the two-tower's own order).** Re-ordering the 230 candidates with the learned ranker puts the movies people actually liked higher than the two-tower's own order does:
  about +0.024 NDCG@10 on a base of 0.138, i.e. roughly 17% better, for both the context-free ranker (M0) and the one with session features (M2). Both gains were larger than I had predicted.
  Caveat from the protocol: the rankers were fitted on these same users' validation labels, so this measures the next period for KNOWN users, and the test window is twice as long as validation,
  so compare methods only against each other within the test run, not against validation numbers.
- **P3 (does context help?).** Barely. M2 beats M0 by +0.0013 NDCG@10, under 1% relative: distinguishable from zero but far too small to matter on its own. M0, which has no context features, already gets almost all of the gain.
  (In the descriptive rows, adding hour/day alone was slightly worse than no context and the session step was the part that helped; those are not headline findings.)
- **P4 (retrieval).** Using the two-tower to fetch candidates finds more of what users later liked than item-kNN does: 58.5% vs 53.8% of their liked movies appear in the top 200 (+0.047 Recall@200).
- **P5 (bandit, simulation only).** With the exploration constant fixed in advance at c = 0.2 (not tuned on test), UCB earned 807 likes per 1,000 impressions in the simulated new-release lane, against 570 for textbook UCB1 and 775 for Thompson sampling,
  with a perfect-knowledge oracle at 833. So in this simulation tuned UCB beats UCB1 by a lot and Thompson by a little. This says nothing about real viewers: it is a simulation on thin test-window data, not an A/B test.

Overall, every directional headline prediction came out as predicted (P5b's sign was called a coin flip), in a single pre-registered run that followed the stated interpretation rule (no re-running, no tuning).
