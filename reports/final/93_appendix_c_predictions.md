# Appendix C. Predictions Made Before the Results

From Module 4 onward, I committed my expected results to the project history before running anything. Table C.1 lists those files. The detailed predictions-versus-actuals for each module are in the module reports in the repository; the two that matter most for this report are reproduced below.

**Table C.1** *Pre-registration files and the commits that fixed them*

| Module | File | Commit | Date |
|---|---|---|---|
| 4: two-tower model | `reports/module4_predictions.md` | 58f485e | 26 Oct 2026 |
| 5: NCF | `reports/module5_predictions.md` | 9b46c62 | 26 Oct 2026 |
| 6: ranker and context | `reports/module6_predictions.md` | eb9bad9 | 26 Oct 2026 |
| 7: bandit | `reports/module7_predictions.md` | 4e98d3b | 26 Oct 2026 |
| 7b: bandit tuning | `reports/module7b_predictions.md` | def2b66 | 26 Oct 2026 |
| 8: test protocol and predictions | `reports/module8_protocol.md`, `reports/module8_predictions.md` | e3d5da3 | 26 Oct 2026 |
| 9: serving latency | `reports/module9_latency_predictions.md` | 519e241 | 27 Oct 2026 |

**Table C.2** *The headline test claims: prediction against result*

| Claim | Predicted (verbatim range) | Result | Verdict |
|---|---|---|---|
| P1: M2 against the two-tower order, NDCG@10 | positive (80%), +0.006 to +0.022 | +0.0245 [+0.0229, +0.0261] | direction right, above the range |
| P2: M0 against the two-tower order, NDCG@10 | positive (80%), +0.005 to +0.020 | +0.0232 [+0.0217, +0.0247] | direction right, above the range |
| P3: M2 against M0, NDCG@10 | between -0.002 and +0.004 (70%) | +0.0013 [+0.0004, +0.0021] | inside the range |
| P4: two-tower against item-kNN, Recall@200 | positive (85%), +0.03 to +0.07 | +0.0473 [+0.0441, +0.0505] | inside the range |
| P5a: UCB c = 0.2 against UCB1 (SIMULATION) | better (90%), by more than +100 per 1,000 (80%) | +237.1 [+232.2, +242.0] | right on both parts |
| P5b: UCB c = 0.2 against Thompson (SIMULATION) | within plus or minus 30 per 1,000 (65%) | +31.9 [+27.9, +35.9] | just outside the range |

The eight secondary test predictions came out as 7 right and 1 not evaluable (the test script did not report the share of candidates removed by the release-date filter, which one prediction was about).

**Table C.3** *Serving latency: prediction against result*

| Id | Quantity | Predicted | Measured | Verdict |
|---|---|---|---|---|
| L1 | Warm server time, precomputed, median | 8 to 30 ms | 91.1 ms | miss |
| L2 | Warm server time, live, median | 25 to 150 ms | 121.9 ms | inside |
| L3 | Live minus precomputed, medians | 15 to 130 ms | 30.7 ms | inside |
| L4 | Laptop end to end, precomputed, median | 150 to 500 ms | 338.4 ms | inside |
| L5 | Cold start (start-up phase), median | 800 to 3,000 ms | 572.1 ms | miss |
| L6 | Peak memory | 150 to 450 MB | 220 MB | inside |
| L7 | Laptop connection setup to Frankfurt, median | 100 to 400 ms | 492.2 ms | miss |
| L8 | Live-mode fallbacks during the warm run | 0 | 0 | inside |

The latency predictions file also records that they were written after I had seen the smoke-test timings, so they were informed guesses, not blind ones.
