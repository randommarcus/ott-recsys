# Module 8 predictions (written 2026-10-26, BEFORE the test portion is read and BEFORE the rehearsal has been run)

Refined from the plan only with ARTIFACT FACTS from `scripts/prepare_test.py` (listed at the bottom). No rehearsal output and no test data were used.
Validation reference (Modules 5-6, unfiltered-then-filtered lists, held-out users): two-tower order e2e NDCG@10 0.1242 / Recall@20 0.2321; M0 0.1370 / 0.2508; M2 0.1388 / 0.2529;
item-kNN Recall@200 0.584, two-tower 0.650; bandit 629-arm set (7b second half): UCB c*=0.2 836, Thompson 810, UCB1 640 likes per 1,000 impressions.

HEADLINE predictions (P1-P5; these are the findings):

| id | Prediction |
|---|---|
| P1 | M2 (fold-in NCF) - two-tower order, e2e NDCG@10: positive (80% confident), +0.006 to +0.022 |
| P2 | M0 (fold-in NCF) - two-tower order, e2e NDCG@10: positive (80%), +0.005 to +0.020 |
| P3 | M2 - M0: between -0.002 and +0.004 (70%); context remains roughly null |
| P4 | two-tower - item-kNN, Recall@200 (whole catalog): positive (85%), +0.03 to +0.07 |
| P5a | bandit simulation, 629-type arm set: UCB c = 0.2 better than UCB1 (90%), by > +100 likes per 1,000 impressions (80%) |
| P5b | UCB c = 0.2 vs Thompson: within +-30 likes per 1,000 impressions (65%); the sign is close to a coin flip (Module 7b gave +26 for UCB) |

SECONDARY predictions (descriptive; not findings):

| # | Prediction |
|---|---|
| 1 | Mean liked items per test user about 13-17 (validation: 7.6), i.e. roughly double |
| 2 | Two-tower order, e2e: NDCG@10 0.12-0.17; Recall@20 0.15-0.21 (lower than validation's 0.232 because the denominator is larger) |
| 3 | NCF order: fold-in better than stale by +0.002 to +0.010 NDCG@10 (65%); fold-in moved existing users' embeddings by 30% on average, so it is a real update |
| 4 | Ranker M2 with fold-in vs stale NCF features: within +-0.003 NDCG@10 (60%) |
| 5 | Release-by-request-time filter removes 4-9% of slots (test requests occur later in calendar time than validation's 7.9%) |
| 6 | Cost of reserving 2 end slots on Recall@20: -0.010 to -0.020; NDCG@10 unchanged by construction |
| 7 | Retrieval, cold view: item-kNN and popularity score exactly 0 on cold likes; content retrieval > 0 |
| 8 | Bandit, 239-type and 2,031-type arm sets: UCB c = 0.2 beats UCB1 on both (85%) |

ARTIFACT FACTS (from preparation, real mode; not results):
- Final rankers refit once on 18,530 validation users: M0 188 trees (29 features), M1 65 trees (31), M2 351 trees (37); learning rate 0.03.
- NCF fold-in: 3 epochs, lr 1e-3, 6.30M rows per epoch; loss 0.1973 -> 0.1875; 19,961 users have an embedding (6 new vs the 19,955 of the stale model);
  mean relative change of existing users' embeddings 0.296; 1,271 of 1,260,898 history positives dropped (item unknown to the frozen model).
- Candidate lists: all 20,000 users get 230 slots (200 main + 30 lane) before the release filter; 13.2% of slots have NCF = NaN (stale 13.24%, fold-in 13.21%);
  lane pool 16,942 movies; 39 users have no history vector (no likes in train + validation) and get popularity fallback lists.
