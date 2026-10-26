# Module 5 predictions (written 2026-10-26, BEFORE any NCF code was run)

Reference numbers, computed from the saved Module 4 lists before the plan (validation, 18,530 users, ID-only
two-tower top-200; 91% of users have >= 1 validation like inside the list; mean 3.9 of 7.6 likes inside).
Order inside the list, END-TO-END (all users, truth = all validation likes), NDCG@10 / Recall@20:
random 0.028 / 0.065; popularity 0.061 / 0.117; two-tower's own order 0.123 / 0.231; oracle 0.760 / 0.649.

| # | Quantity | Prediction |
|---|---|---|
| 1 | NCF alone, end-to-end NDCG@10 / Recall@20 (230 lists) | about 0.09 (range 0.075-0.11) / about 0.19 (0.16-0.22) |
| 2 | NCF beats the two-tower's own order? | No (about 80% confident); gap -0.02 to -0.04 NDCG@10 |
| 3 | NCF beats popularity order? | Yes (95%), by >= +0.02 NDCG@10 |
| 4 | NCF beats random order? | Yes (99%) |
| 5 | Share of the two-tower's gain over random that NCF captures | 60-80% |
| 6 | Untuned 50/50 rank blend of two-tower and NCF beats the two-tower alone? | Yes (75%), +0.005 to +0.015 NDCG@10 |
| 7 | Uniform-only negatives vs the uniform+popularity mix | Uniform-only worse (80%), by >= 0.01 NDCG@10; K=8 vs K=4 within +-0.005 |
| 8 | Best epoch | 3-8 (overfits quickly) |
| 9 | Personalization check (user embeddings shuffled across users) | NDCG@10 falls to 0.05-0.07, about popularity level |
| 10 | Time | at most 1 minute per epoch; each run under 10 minutes |
| 11 | Selection optimism (3,000 selection users vs 15,530 held-out users) | gap <= 0.005 NDCG@10 |
| 12 | Global-cutoff robustness: ordering | same as validation (two-tower > NCF > popularity > random), about 60% confident; wide intervals (few hundred users) |
| 13 | Share of validation likes inside the 230 that sit on items NCF cannot score (no train likes) | Main lane: <= 0.5%. New-release lane: about 100% by construction (lane = movies with no train likes). Combined over the whole 230: <= 0.5%, because lane hits are rare |
| 14 | Users NCF cannot score (no train likes) | 45 of 20,000 overall; 6 of the 18,530 evaluated users |
