# Module 7 predictions (written 2026-10-26, BEFORE any robustness or bandit run)

THIS IS A SIMULATION, NOT AN A/B TEST. Reward = "liked given rated" (ignores click probability).
Setting: 3,341 new-release movies (released since cutoff-6mo, no ratings before the 2015-02-03 cutoff); arm sets by ratings in the
validation window: 239 arms (>=20 ratings), 629 (>=5), 2,031 (>=1); 31,909 real session starts as the request stream; k=2 lane slots,
365-day new-release shelf; hidden like-rates drawn from Beta posteriors shrunk toward 0.457 (5 pseudo-ratings); 30 seeds.
Content prior (centered SBERT -> like-rate, fitted on PRE-cutoff movies only) has cross-validated correlation 0.49 (R2 0.24).

| # | Quantity | Prediction |
|---|---|---|
| R1 | Trees per fold with lr 0.03, 25% inner set, patience 100 | 200-800 |
| R2 | M0 new vs old (NDCG@10) | small improvement, +0.000 to +0.004 |
| R3 | Context conclusion | unchanged (80%): M2-M0 within +0.000 to +0.004; hour/day step <= 0.0015 |
| B1 | Like rate per impression | random about 0.45; oracle 0.75-0.85 |
| B2 | Cumulative-reward order (629-arm set) | UCB1+prior > UCB1 ~ eps-greedy > greedy > random (65% for the full order) |
| B3 | UCB1 regret | 20-40% of random's regret by the end of the stream |
| B4 | Prior-only (no learning) | beats random (90%) but loses to UCB1+prior (85%) |
| B5 | Time to find the top arms | shorter with the prior than plain UCB1 (80%) |
| B6 | Exposure Gini | random lowest (~0.3), greedy highest (> 0.8), UCB1 in between |
| B7 | Slot cost, reserved at the END of the top 20 | NDCG@10 unchanged by construction; Recall@20 -0.004 to -0.010 (2 slots), -0.006 to -0.015 (3 slots) |
| B8 | Slot cost, reserved INSIDE the top 10 | NDCG@10 -0.005 to -0.015 |
| B9 | Thompson sampling (Beta posterior) | beats plain UCB1 on cumulative reward (85%); within 10% of UCB1+prior's regret (65%) |
| B10 | Greedy | below UCB1 on cumulative reward (80%) because it locks onto early lucky arms |
| B11 | Arm-set sensitivity | gap between best strategy and random is smallest on the 2,031-arm set (less true spread; 80%); strategy ORDER the same on all three sets (70%) |
| B12 | eps-greedy sensitivity | eps 0.05 and 0.2 within +-10% of eps 0.1's regret (70%) |
| B13 | Empirical-Bayes prior weight | 5-20 pseudo-ratings |
