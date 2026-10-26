# Module 4 predictions (written 2026-10-26, BEFORE any two-tower run)

Reference numbers from earlier modules (validation, 20k-user MovieLens 25M subset):
item-kNN NDCG@10 0.098 / Recall@50 0.306 / Recall@200 0.584; global popularity 0.0555 / 0.180 / 0.381;
centered content-only 0.0103 / 0.036 / 0.089.

| # | Quantity | Prediction |
|---|---|---|
| 1 | ID+content (N=20) beats item-kNN overall? | Probably not (25-30% chance it does). NDCG@10 0.07-0.09; Recall@50 0.24-0.29; Recall@200 0.50-0.55 |
| 2 | ID+content beats global popularity? | Yes (about 90% confident) |
| 3 | ID-only vs ID+content on warm items | ID+content a little better or equal, within 0.005 NDCG@10 |
| 4 | Content-only on warm items | NDCG@10 0.02-0.04, well above raw content retrieval's 0.0103 |
| 5 | Cold view (236 users, wide CIs), Recall@200 | ID+content 0.03-0.07; content-only similar or better; ID-only near random (<= 0.005) |
| 6 | New releases, cold part, new-items-only pool, Recall@50 | ID+content and content-only 0.10-0.25 (centered content-only was 0.159); ID-only near random (~0.02) |
| 7 | New releases, cold part, whole catalog, Recall@50 | still low (<= 0.05) |
| 8 | logQ off vs on | Without logQ: higher NDCG@10 on warm items (about 65% confident) and lower coverage; with logQ: better on rare/cold items |
| 9 | History 50 vs 20 | small difference, within +-0.003 NDCG@10 |
| 10 | Time per run | 5-8 minutes |
| 11 | Selecting on Recall@200 instead of NDCG@10 | picks the same or a slightly later epoch (best epoch 5-10) |
| 12 | Candidate union (100 two-tower + 100 kNN), Recall@200 | higher than kNN alone by about +0.01 or more (about 80% confident) and higher than two-tower alone |
| 13 | Validation users with no liked items in train | small: 100-250 of 20,000 (0.5-1.3%) |
